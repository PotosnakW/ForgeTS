import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import itertools

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    Sampler,
)


def _load_df(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix == ".csv":
        return pd.read_csv(p)
    raise ValueError(f"Unsupported file format '{p.suffix}'. Use .parquet or .csv.")


# ─────────────────────────────────────────────────────────────────────────────
# Splitting
# ─────────────────────────────────────────────────────────────────────────────

def _split_df(
    df: pd.DataFrame,
    val_size: int,
    test_size: int,
    per_series_split: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["ds"]):
        df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values(["unique_id", "ds"])
    return _split_per_series(df, val_size, test_size)


def _split_per_series(df, val_size, test_size):
    def label(g):
        n = len(g)
        if val_size + test_size >= n:
            raise ValueError(
                f"Series '{g['unique_id'].iloc[0]}' has {n} timesteps but "
                f"val_size ({val_size}) + test_size ({test_size}) = {val_size + test_size}."
            )
        return g.assign(_split=(
            ["train"] * (n - val_size - test_size)
            + ["val"]  * val_size
            + ["test"] * test_size
        ))
    df = df.groupby("unique_id", group_keys=False).apply(label)
    return (
        df[df["_split"] == "train"].drop(columns="_split"),
        df[df["_split"] == "val"].drop(columns="_split"),
        df[df["_split"] == "test"].drop(columns="_split"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pivot → arrays
# ─────────────────────────────────────────────────────────────────────────────

def _pivot_to_arrays(
    df: pd.DataFrame,
    hist_exog_cols: List[str],
    futr_exog_cols: List[str],
    stat_exog_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray]:
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["ds"]):
        df["ds"] = pd.to_datetime(df["ds"])

    if "available_mask" not in df.columns:
        raise ValueError(
            "Column 'available_mask' is missing from the dataframe. "
            "Please add a column of 1.0 (available) / 0.0 (missing) values. "
            "If all data is available, add: df['available_mask'] = 1.0"
        )

    channel_ids = sorted(df["unique_id"].unique().tolist())

    lengths = df.groupby("unique_id")["ds"].count()
    if lengths.nunique() > 1:
        warnings.warn(
            f"Unequal series lengths {lengths.to_dict()} — forward-filling to align."
        )
        mi = pd.MultiIndex.from_product(
            [channel_ids, df["ds"].drop_duplicates().sort_values()],
            names=["unique_id", "ds"],
        )
        df = (
            df.set_index(["unique_id", "ds"])
            .reindex(mi)
            .groupby("unique_id")
            .ffill()
            .reset_index()
        )

    def _piv(col):
        return (
            df.pivot(index="ds", columns="unique_id", values=col)
            .loc[:, channel_ids]
            .values
            .astype(np.float32)
        )

    T, C = df["ds"].nunique(), len(channel_ids)
    y = _piv("y")                                        # [T, C]
    hist = (
        np.stack([_piv(c) for c in hist_exog_cols], axis=-1)
        if hist_exog_cols else np.zeros((T, C, 0), dtype=np.float32)
    )
    futr = (
        np.stack([_piv(c) for c in futr_exog_cols], axis=-1)
        if futr_exog_cols else np.zeros((T, C, 0), dtype=np.float32)
    )
    stat = (
        df.groupby("unique_id", sort=False)[stat_exog_cols]
        .first().loc[channel_ids].values.astype(np.float32)
        if stat_exog_cols else np.zeros((C, 0), dtype=np.float32)
    )
    available_mask = _piv("available_mask")              # [T, C]

    return y, hist, futr, stat, channel_ids, available_mask


# ─────────────────────────────────────────────────────────────────────────────
# Static metadata
# ─────────────────────────────────────────────────────────────────────────────

class SeriesMetadata:
    """
    Per-series static features — values that don't change over time
    (e.g. region, sensor_type, capacity).

    Shape: [C, n_static_features]

    Kept separate from temporal arrays because static features are indexed
    by channel only, not by (time, channel). Makes the distinction explicit
    and prevents accidental dimension mixing in the model.
    """
    def __init__(self, data: np.ndarray, col_names: List[str], channel_ids: List[str]):
        assert data.shape == (len(channel_ids), len(col_names)), (
            f"Static data shape {data.shape} inconsistent with "
            f"{len(channel_ids)} channels and {len(col_names)} columns."
        )
        self.data = torch.from_numpy(data.astype(np.float32))  # [C, n_stat]
        self.col_names = col_names
        self.channel_ids = channel_ids

    def __repr__(self):
        return (
            f"SeriesMetadata(channels={len(self.channel_ids)}, "
            f"features={self.col_names})"
        )

    @staticmethod
    def empty(channel_ids: List[str]) -> "SeriesMetadata":
        return SeriesMetadata(
            data=np.zeros((len(channel_ids), 0), dtype=np.float32),
            col_names=[],
            channel_ids=channel_ids,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class FullSeriesDataset(Dataset):
    """
    Delivers the full series to the model. fork_sequences handles all windowing.

    available_mask shape: [T, C]
        1 = real data belonging to this split (loss is computed here)
        0 = left-padding, missing values, or out-of-split extension rows
            (encoder may still see these; loss ignores them via outsample_mask)

    Split-specific layout
    ─────────────────────
    train : [train rows (1)] [H-1 val rows (0)]
    val   : [ctx rows (0)]   [val rows (1)]   [H-1 test rows (0)]
    test  : [ctx rows (0)]   [test rows (1)]
    """
    def __init__(
        self,
        y:               np.ndarray,             # [T, C]
        hist:            np.ndarray,             # [T, C, Vh]
        futr:            np.ndarray,             # [T, C, Vf]
        available_mask:  np.ndarray,             # [T, C]
        context_length:  int,
        horizon:         int,
        channel_ids:     List[str] = None,
        metadata:        Optional[SeriesMetadata] = None,
        name:            str = "",
        is_multivariate: bool = False,
    ):
        T, C = y.shape
        min_len = context_length + horizon
        if T < min_len:
            raise ValueError(
                f"Dataset '{name}': series length {T} < "
                f"context_length + horizon ({min_len})."
            )
        self.y = torch.from_numpy(y)
        self.hist = torch.from_numpy(hist)
        self.futr = torch.from_numpy(futr)
        self.available_mask = torch.from_numpy(
            available_mask.astype(np.float32)
        ).contiguous()
        self.channel_ids = channel_ids or [str(i) for i in range(C)]
        self.metadata = metadata
        self.ctx = context_length
        self.horizon = horizon
        self.T = T
        self.name = name
        self.is_multivariate = is_multivariate

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        y_enc = self.y.unsqueeze(-1)                                 # [T, C, 1]
        x_enc = (
            torch.cat([y_enc, self.hist], dim=-1)
            if self.hist.shape[-1] > 0 else y_enc
        )                                                            # [T, C, 1+Vh]
        out = dict(
            x_enc           = x_enc,
            x_futr          = self.futr,
            available_mask  = self.available_mask,
            series_len      = torch.tensor(self.T,       dtype=torch.long),
            horizon         = torch.tensor(self.horizon, dtype=torch.long),
            dataset_name    = self.name,
            channel_ids     = self.channel_ids,
            is_multivariate = self.is_multivariate,
        )
        if self.metadata is not None and self.metadata.data.shape[-1] > 0:
            out["x_stat"] = self.metadata.data
        return out


def _make_dataset(
    y, hist, futr, stat, available_mask, channel_ids, mcfg, horizon, name="",
    is_multivariate=False,
) -> FullSeriesDataset:
    ctx = _ctx(mcfg)
    metadata = (
        SeriesMetadata(
            data = stat,
            col_names = getattr(mcfg, "stat_exog_cols", []) or [],
            channel_ids = channel_ids,
        )
        if stat.shape[-1] > 0 else None
    )
    return FullSeriesDataset(
        y = y,
        hist = hist,
        futr = futr,
        available_mask = available_mask,
        context_length = ctx,
        horizon = horizon,
        channel_ids = channel_ids,
        metadata = metadata,
        name = name,
        is_multivariate = is_multivariate,
    )


def _ctx(mcfg) -> int:
    return getattr(mcfg, "input_size", None) or getattr(mcfg, "context_length", 512)


def _extend_with_next_split(df: pd.DataFrame, next_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Append the first H-1 rows of next_df per series to df, with available_mask=0.
    These act as a buffer so windows near the split boundary can form without
    their horizons overflowing, while contributing zero loss.
    Only applied when horizon > 1 and next_df is non-empty.
    """
    if next_df is None or len(next_df) == 0 or horizon <= 1:
        return df
    extension = (
        next_df.groupby("unique_id", sort=False)
        .head(horizon - 1)
        .copy()
    )
    extension["available_mask"] = 0.0
    return (
        pd.concat([df, extension])
        .sort_values(["unique_id", "ds"])
        .reset_index(drop=True)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Collation — left-pad time, right-pad channels/features
# ─────────────────────────────────────────────────────────────────────────────

def _pad_left(t: Tensor, target_len: int) -> Tensor:
    """Left-pad tensor along dim 0 (time) with zeros."""
    pad = target_len - t.shape[0]
    if pad == 0:
        return t
    return F.pad(t, [0] * (2 * (t.ndim - 1)) + [pad, 0])


def _full_series_collate_fn(batch):
    T_max  = max(s["x_enc"].shape[0]      for s in batch)
    C_max  = max(s["x_enc"].shape[-2]     for s in batch)
    Vh_max = max(s["x_enc"].shape[-1] - 1 for s in batch)

    B            = len(batch)
    channel_mask = torch.zeros(B, C_max,          dtype=torch.float32)
    hist_mask    = torch.zeros(B, max(Vh_max, 1), dtype=torch.float32)

    out_x_enc  = []
    out_mask   = []
    out_x_stat = [] if any("x_stat" in s for s in batch) else None

    for i, s in enumerate(batch):
        T_i  = s["x_enc"].shape[0]
        C_i  = s["x_enc"].shape[-2]
        Vh_i = s["x_enc"].shape[-1] - 1

        x = F.pad(s["x_enc"], (0, Vh_max - Vh_i, 0, C_max - C_i))
        x = _pad_left(x, T_max)
        out_x_enc.append(x)

        m = F.pad(s["available_mask"], (0, C_max - C_i))
        m = _pad_left(m, T_max)
        out_mask.append(m)

        channel_mask[i, :C_i] = 1.0
        if Vh_i > 0:
            hist_mask[i, :Vh_i] = 1.0

        if out_x_stat is not None:
            stat = s.get("x_stat")
            if stat is not None:
                out_x_stat.append(F.pad(stat, (0, 0, 0, C_max - stat.shape[0])))
            else:
                out_x_stat.append(torch.zeros(C_max, 1))

    result = dict(
        x_enc           = torch.stack(out_x_enc),
        available_mask  = torch.stack(out_mask),
        channel_mask    = channel_mask,
        hist_mask       = hist_mask,
        dataset_name    = [s.get("dataset_name", "unknown") for s in batch],
        channel_ids     = [s.get("channel_ids",  [])        for s in batch],
        is_multivariate = batch[0].get("is_multivariate", False),
    )
    for key in ("series_len", "horizon"):
        if key in batch[0]:
            result[key] = torch.stack([s[key] for s in batch])
    if out_x_stat:
        result["x_stat"] = torch.stack(out_x_stat)

    return result

# ─────────────────────────────────────────────────────────────────────────────
# BatchSampler
# ─────────────────────────────────────────────────────────────────────────────

class BatchSampler(Sampler):
    """
    Flat weighted sampler — no horizon bucketing.
    Datasets are mixed into a single pool per multivariate flag.
    Intended for fixed-horizon training (horizon_override) to ablate
    against HorizonBatchSampler's horizon-grouped behaviour.

    Supports the same mixing strategies as HorizonBatchSampler.
    Distributed behaviour matches HorizonBatchSampler: contiguous slices
    with front-padded pool to equalise batch counts across ranks.
    """

    def __init__(
        self,
        datasets,           # list of (dataset, weight, is_multivariate)
        global_offsets,     # list of int, one per dataset
        batch_size,
        batch_mixing_strategy="concat",  # "concat" | "round_robin"
        shuffle=True,
        drop_last=False,
        seed=0,
        rank=0,
        world_size=1,
    ):
        self.datasets        = datasets
        self.global_offsets  = global_offsets
        self.batch_size      = batch_size
        self.batch_mixing_strategy = batch_mixing_strategy
        self.shuffle         = shuffle
        self.drop_last       = drop_last
        self.seed            = seed
        self.rank            = rank
        self.world_size      = world_size
        self._epoch          = 0

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def _build_pool(self, rng, multivariate: bool):
        entries = [
            (ds, w, offset)
            for (ds, w, is_mv), offset in zip(self.datasets, self.global_offsets)
            if is_mv == multivariate
        ]
        if not entries:
            return []

        total  = sum(len(ds) for ds, _, _ in entries)
        w_arr  = np.array([w for _, w, _ in entries], dtype=np.float64)
        w_arr /= w_arr.sum()

        slots_per = (w_arr * total).round().astype(int)
        slots_per[np.argmax(slots_per)] += total - slots_per.sum()

        pool = []
        for (ds, _, offset), slots in zip(entries, slots_per):
            idxs = np.arange(len(ds)) + offset
            if self.shuffle:
                rng.shuffle(idxs)
            it = itertools.cycle(idxs.tolist())
            pool.extend(itertools.islice(it, int(slots)))

        if self.shuffle:
            rng.shuffle(pool)

        if self.world_size > 1:
            total_slots = self.batch_size * self.world_size
            pad  = (-len(pool)) % total_slots
            pool = pool + pool[:pad]
            rank_size = len(pool) // self.world_size
            pool = pool[self.rank * rank_size : (self.rank + 1) * rank_size]

        if self.drop_last:
            pool = pool[: (len(pool) // self.batch_size) * self.batch_size]

        return pool

    def _pool_to_batches(self, pool):
        bs      = self.batch_size
        batches = [pool[i : i + bs] for i in range(0, len(pool) - bs + 1, bs)]
        if not self.drop_last and len(pool) % bs:
            batches.append(pool[-(len(pool) % bs):])
        return batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)

        # Build one batch list per multivariate flag
        pool_batches = {}
        for is_mv in (False, True):
            pool = self._build_pool(rng, multivariate=is_mv)
            if pool:
                pool_batches[is_mv] = self._pool_to_batches(pool)

        if self.batch_mixing_strategy == "round_robin":
            iters  = {k: iter(v) for k, v in pool_batches.items()}
            active = list(pool_batches.keys())
            while active:
                exhausted = []
                for k in active:
                    b = next(iters[k], None)
                    if b is None:
                        exhausted.append(k)
                    else:
                        yield b
                for k in exhausted:
                    active.remove(k)
        else:  # concat
            all_batches = [b for bl in pool_batches.values() for b in bl]
            if self.shuffle:
                order       = rng.permutation(len(all_batches)).tolist()
                all_batches = [all_batches[i] for i in order]
            yield from all_batches

    def __len__(self):
        total = 0
        for ds, _, _ in self.datasets:
            n = len(ds)
            if self.drop_last:
                n = (n // self.batch_size) * self.batch_size
            total += max(1, n // self.batch_size // max(1, self.world_size))
        return total

# ─────────────────────────────────────────────────────────────────────────────
# HorizonBatchSampler
# ─────────────────────────────────────────────────────────────────────────────

class HorizonBatchSampler(Sampler):
    """
    Samples batches grouped by (horizon, is_multivariate) with weighted dataset mixing.
    Batches never mix univariate and multivariate datasets.

    Distributed behaviour
    ─────────────────────
    When world_size > 1 each rank receives a non-overlapping CONTIGUOUS slice
    of every group's pool. Contiguous (not strided) slices are used for better
    cache locality.

    Padding ensures all ranks see the same number of batches — required by
    DDP's barrier synchronisation (unequal batch counts cause deadlock).
    Padding draws from the front of the pool so no rank sees another's data.
    """

    def __init__(
        self,
        group_datasets,
        group_weights,
        global_offsets,
        batch_size,
        batch_mixing_strategy="concat",
        shuffle=True,
        drop_last=False,
        seed=0,
        rank=0,
        world_size=1,
    ):
        self.group_datasets  = group_datasets
        self.group_weights   = group_weights
        self.global_offsets  = global_offsets
        self.batch_size      = batch_size
        self.batch_mixing_strategy = batch_mixing_strategy
        self.shuffle         = shuffle
        self.drop_last       = drop_last
        self.seed            = seed
        self.rank            = rank
        self.world_size      = world_size
        self._epoch          = 0
        self.groups          = sorted(group_datasets.keys())

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def _group_batches(self, group_key, rng):
        datasets = self.group_datasets[group_key]
        weights  = self.group_weights[group_key]
        offsets  = self.global_offsets[group_key]

        per_ds = []
        for ds, offset in zip(datasets, offsets):
            idxs = np.arange(len(ds)) + offset
            if self.shuffle:
                rng.shuffle(idxs)
            per_ds.append(idxs)

        if not self.shuffle and len(set(weights)) == 1:
            pool    = [idx for idxs in per_ds for idx in idxs.tolist()]
            bs      = self.batch_size
            batches = [pool[i : i + bs] for i in range(0, len(pool) - bs + 1, bs)]
            if not self.drop_last and len(pool) % bs:
                batches.append(pool[-(len(pool) % bs):])
            return batches

        total     = sum(len(a) for a in per_ds)
        w_arr     = np.array(weights, dtype=np.float64)
        w_arr     = w_arr / w_arr.sum()
        slots_per = (w_arr * total).round().astype(int)
        slots_per[np.argmax(slots_per)] += total - slots_per.sum()

        pool     = []
        ds_iters = [itertools.cycle(a.tolist()) for a in per_ds]
        for slots, it in zip(slots_per, ds_iters):
            pool.extend(itertools.islice(it, int(slots)))
        if self.shuffle:
            rng.shuffle(pool)

        if self.world_size > 1:
            total_slots = self.batch_size * self.world_size
            pad  = (-len(pool)) % total_slots
            pool = pool + pool[:pad]
            rank_size = len(pool) // self.world_size
            pool = pool[self.rank * rank_size : (self.rank + 1) * rank_size]

        if self.drop_last:
            pool = pool[: (len(pool) // self.batch_size) * self.batch_size]

        bs      = self.batch_size
        batches = [pool[i : i + bs] for i in range(0, len(pool) - bs + 1, bs)]
        if not self.drop_last and len(pool) % bs:
            batches.append(pool[-(len(pool) % bs):])
        return batches

    def __iter__(self):
        rng               = np.random.default_rng(self.seed + self._epoch)
        group_batch_lists = {g: self._group_batches(g, rng) for g in self.groups}

        if self.batch_mixing_strategy == "round_robin":
            iters  = {g: iter(b) for g, b in group_batch_lists.items()}
            active = list(self.groups)
            while active:
                exhausted = []
                for g in active:
                    b = next(iters[g], None)
                    if b is None:
                        exhausted.append(g)
                    else:
                        yield b
                for g in exhausted:
                    active.remove(g)
        else:
            all_batches = [b for bl in group_batch_lists.values() for b in bl]
            if self.shuffle:
                order       = rng.permutation(len(all_batches)).tolist()
                all_batches = [all_batches[i] for i in order]
            yield from all_batches

    def __len__(self):
        total = 0
        for g, datasets in self.group_datasets.items():
            n = sum(len(ds) for ds in datasets)
            if self.drop_last:
                n = (n // self.batch_size) * self.batch_size
            total += max(1, n // self.batch_size // max(1, self.world_size))
        return total


# ─────────────────────────────────────────────────────────────────────────────
# DataLoaderFactory
# ─────────────────────────────────────────────────────────────────────────────

class DataLoaderFactory:
    def __init__(self, mcfg, dcfg):
        self.mcfg = mcfg
        self.dcfg = dcfg
        self._horizon_groups: Dict[tuple, list] = defaultdict(list)
        self._build_train()

        seed = getattr(self.mcfg, "seed", 42)
        self._val_rng = np.random.default_rng(seed)

    def _arrays_from_df(self, df, entry):
        return _pivot_to_arrays(
            df, entry.hist_exog_cols, entry.futr_exog_cols, entry.stat_exog_cols
        )

    def _build_train(self):
        from dataloaders.ts_sharding import ShardedTrainDataset
        for entry in self.dcfg.train:
            is_multivariate = getattr(entry, "multivariate", False)
            group_key = (entry.horizon, is_multivariate)

            if getattr(entry, "sharded_dir", None):
                ds = ShardedTrainDataset(
                    data_dir = entry.sharded_dir,
                    context_length = _ctx(self.mcfg),
                    horizon = entry.horizon,
                    rank = getattr(self, "_rank", 0),
                    world_size = getattr(self, "_world_size", 1),
                )
                self._horizon_groups[group_key].append((ds, entry.weight, entry.name))
                continue

            df = _load_df(entry.path)
            train_df, val_df, _ = _split_df(
                df, entry.val_size, entry.test_size,
                per_series_split=entry.per_series_split,
            )
            # Extend train with H-1 val rows (mask=0) so windows near the
            # train/val boundary can form. Predictions landing in these rows
            # have outsample_mask=0 and don't contribute to the loss.
            train_df = _extend_with_next_split(train_df, val_df, entry.horizon)

            y, hist, futr, stat, channel_ids, available_mask = self._arrays_from_df(
                train_df, entry
            )
            ds = _make_dataset(
                y, hist, futr, stat, available_mask,
                channel_ids, self.mcfg, entry.horizon, entry.name,
                is_multivariate=is_multivariate,
            )
            self._horizon_groups[group_key].append((ds, entry.weight, entry.name))

    def rebuild_for_rank(self, rank: int, world_size: int) -> "DataLoaderFactory":
        self._rank = rank
        self._world_size = world_size
        self._horizon_groups.clear()
        self._build_train()
        return self

    def _build_eval_dataset(self, entry, split: str) -> Dataset:
        is_multivariate = getattr(entry, "multivariate", False)

        if getattr(entry, "sharded_dir", None):
            if split == "val":
                from dataloaders.ts_sharding import ShardedValDataset
                return ShardedValDataset(
                    data_dir = entry.sharded_dir,
                    context_length = _ctx(self.mcfg),
                    horizon = entry.horizon,
                    name = entry.name,
                )
            if split == "test":
                from dataloaders.ts_sharding import ShardedTestDataset
                return ShardedTestDataset(
                    data_dir = entry.sharded_dir,
                    context_length = _ctx(self.mcfg),
                    horizon = entry.horizon,
                    name = entry.name,
                )

        df = _load_df(entry.path)
        train_df, val_df, test_df = _split_df(
            df, entry.val_size, entry.test_size,
            per_series_split=entry.per_series_split,
        )
        eval_df = val_df if split == "val" else test_df
        if eval_df is None or len(eval_df) == 0:
            raise ValueError(f"Dataset '{entry.name}': '{split}' split is empty.")

        # Extend val with H-1 test rows (mask=0) so windows near the val/test
        # boundary can form. Test is not extended — no data beyond it exists.
        if split == "val":
            eval_df = _extend_with_next_split(eval_df, test_df, entry.horizon)

        # Prepend context rows from train so the encoder has lookback at the
        # start of the eval window. ctx_rows get mask=0 — encoder sees them
        # but predictions landing here don't contribute to the loss.
        if entry.use_context_head and train_df is not None and len(train_df) > 0:
            ctx_rows = train_df.groupby("unique_id", sort=False).tail(_ctx(self.mcfg)).copy()
            ctx_rows["available_mask"] = 0.0
            eval_df = (
                pd.concat([ctx_rows, eval_df])
                .sort_values(["unique_id", "ds"])
                .reset_index(drop=True)
            )

        y, hist, futr, stat, channel_ids, available_mask = self._arrays_from_df(
            eval_df, entry
        )
        return _make_dataset(
            y, hist, futr, stat, available_mask,
            channel_ids, self.mcfg, entry.horizon, entry.name,
            is_multivariate=is_multivariate,
        )

    def _make_train_batch_sampler(self, rank=0, world_size=1):
        all_datasets = []
        flat_offset  = 0

        use_flat = getattr(self.mcfg, "batch_sampler", "HorizonBatchSampler") == "BatchSampler"

        if use_flat:
            if not getattr(self.mcfg, "horizon_override", None):
                warnings.warn(
                    "batch_sampler='flat' is set without horizon_override — "
                    "batches may contain mixed horizons with no fixed output size."
                )
            datasets       = []
            global_offsets = []

            for group_key in sorted(self._horizon_groups.keys()):
                _, is_multivariate = group_key
                for ds, weight, _ in self._horizon_groups[group_key]:
                    datasets.append((ds, weight, is_multivariate))
                    global_offsets.append(flat_offset)
                    all_datasets.append(ds)
                    flat_offset += len(ds)

            combined = ConcatDataset(all_datasets)
            sampler  = BatchSampler(
                datasets       = datasets,
                global_offsets = global_offsets,
                batch_size     = self.mcfg.batch_size,
                shuffle        = True,
                drop_last      = self.mcfg.drop_last,
                seed           = getattr(self.mcfg, "seed", 0),
                rank           = rank,
                world_size     = world_size,
            )

        else:
            group_datasets = defaultdict(list)
            group_weights  = defaultdict(list)
            global_offsets = defaultdict(list)

            for group_key in sorted(self._horizon_groups.keys()):
                for ds, weight, _ in self._horizon_groups[group_key]:
                    global_offsets[group_key].append(flat_offset)
                    group_datasets[group_key].append(ds)
                    group_weights[group_key].append(weight)
                    all_datasets.append(ds)
                    flat_offset += len(ds)

            combined = ConcatDataset(all_datasets)
            sampler  = HorizonBatchSampler(
                group_datasets  = group_datasets,
                group_weights   = group_weights,
                global_offsets  = global_offsets,
                batch_size      = self.mcfg.batch_size,
                batch_mixing_strategy = self.mcfg.batch_mixing_strategy,
                shuffle         = True,
                drop_last       = self.mcfg.drop_last,
                seed            = getattr(self.mcfg, "seed", 0),
                rank            = rank,
                world_size      = world_size,
            )

        return combined, sampler

    def train_dataloader(self, distributed=False, rank=0, world_size=1) -> DataLoader:
        if not self._horizon_groups:
            raise RuntimeError("No training datasets configured.")
        combined, batch_sampler = self._make_train_batch_sampler(
            rank       = rank       if distributed else 0,
            world_size = world_size if distributed else 1,
        )
        return DataLoader(
            combined,
            batch_sampler      = batch_sampler,
            num_workers        = self.mcfg.num_workers,
            pin_memory         = True,
            collate_fn         = _full_series_collate_fn,
            persistent_workers = self.mcfg.num_workers > 0,
        )

    def _make_eval_dataloader(self, entries, split: str) -> DataLoader:
        group_datasets: Dict[tuple, list] = defaultdict(list)
        group_weights:  Dict[tuple, list] = defaultdict(list)
        global_offsets: Dict[tuple, list] = defaultdict(list)
        all_datasets = []
        flat_offset  = 0

        for entry in entries:
            ds              = self._build_eval_dataset(entry, split)
            is_multivariate = getattr(entry, "multivariate", False)
            group_key       = (entry.horizon, is_multivariate)

            global_offsets[group_key].append(flat_offset)
            group_datasets[group_key].append(ds)
            group_weights[group_key].append(entry.weight)
            all_datasets.append(ds)
            flat_offset += len(ds)

        combined = all_datasets[0] if len(all_datasets) == 1 else ConcatDataset(all_datasets)
        sampler  = HorizonBatchSampler(
            group_datasets  = group_datasets,
            group_weights   = {g: [1.0] * len(ds_list) for g, ds_list in group_datasets.items()},
            global_offsets  = global_offsets,
            batch_size      = self.mcfg.valid_batch_size,
            batch_mixing_strategy = self.mcfg.batch_mixing_strategy,
            shuffle         = False,
            drop_last       = False,
            rank            = 0,
            world_size      = 1,
        )
        return DataLoader(
            combined,
            batch_sampler      = sampler,
            num_workers        = self.mcfg.num_workers,
            pin_memory         = True,
            collate_fn         = _full_series_collate_fn,
            persistent_workers = self.mcfg.num_workers > 0,
        )

    def val_dataloaders(self, epoch: int = 0) -> Dict[str, DataLoader]:
        strategy = getattr(self.mcfg, "val_strategy", "exhaustive")
        rng      = np.random.default_rng(int(self._val_rng.integers(2**32)) + epoch)

        if strategy == "exhaustive":
            return {"val": self._make_eval_dataloader(self.dcfg.validation, "val")}

        elif strategy == "random_datasets":
            k       = getattr(self.mcfg, "val_max_datasets", len(self.dcfg.validation))
            indices = rng.choice(len(self.dcfg.validation), size=min(k, len(self.dcfg.validation)), replace=False)
            entries = [self.dcfg.validation[i] for i in indices]
            return {"val": self._make_eval_dataloader(entries, "val")}

        elif strategy == "stratified":
            horizon_groups = defaultdict(list)
            for entry in self.dcfg.validation:
                horizon_groups[entry.horizon].append(entry)
            entries = [group[int(rng.integers(len(group)))] for group in horizon_groups.values()]
            return {"val": self._make_eval_dataloader(entries, "val")}

        else:
            raise ValueError(
                f"Unknown val_strategy '{strategy}'. "
                f"Choose from: exhaustive, random_datasets, stratified"
            )

    def test_dataloaders(self, distributed=False, rank=0, world_size=1):
        return {"test": self._make_eval_dataloader(self.dcfg.test, "test")}

    def _eval_loaders(self, entries, split):
        loaders = {}
        for entry in entries:
            ds = self._build_eval_dataset(entry, split)
            loaders[entry.name] = DataLoader(
                ds,
                batch_size         = self.mcfg.valid_batch_size,
                shuffle            = False,
                sampler            = None,
                num_workers        = self.mcfg.num_workers,
                pin_memory         = True,
                drop_last          = False,
                collate_fn         = _full_series_collate_fn,
                persistent_workers = self.mcfg.num_workers > 0,
            )
        return loaders
