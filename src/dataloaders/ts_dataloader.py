"""
ts_dataloader.py
────────────────
Multivariate time series data loading pipeline.

Key design decisions
────────────────────
- available_mask shape is [B, C, T] — per-channel, per-timestep availability.
- Left-padding: real data is right-aligned so position T-1 always means "now"
  regardless of series length. Channel and feature dims are right-padded.
- SeriesMetadata holds per-series static features separately from temporal data.
- available_mask must be present in input data — raises if missing.
- HorizonBatchSampler uses contiguous rank slices for DDP (no duplicate batches).
- ShardedTimeSeriesDataset lives in ts_sharding.py.
"""

from __future__ import annotations

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


# ─────────────────────────────────────────────────────────────────────────────
# IO
# ─────────────────────────────────────────────────────────────────────────────

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
    y    = _piv("y")                                        # [T, C]
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
    # available_mask: [T, C] — per-channel, per-timestep
    available_mask = _piv("available_mask")                 # [T, C]

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
        self.data        = torch.from_numpy(data.astype(np.float32))  # [C, n_stat]
        self.col_names   = col_names
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

    available_mask shape: [C, T] — stored per-channel so heterogeneous_sampler
    can find the first real timestep per channel and pick valid window starts.
    """
    def __init__(
        self,
        y:              np.ndarray,    # [T, C]
        hist:           np.ndarray,    # [T, C, Vh]
        futr:           np.ndarray,    # [T, C, Vf]
        available_mask: np.ndarray,    # [T, C]
        context_length: int,
        horizon:        int,
        channel_ids:    List[str]                = None,
        metadata:       Optional[SeriesMetadata] = None,
        name:           str = "",
    ):
        T, C = y.shape
        min_len = context_length + horizon
        if T < min_len:
            raise ValueError(
                f"Dataset '{name}': series length {T} < "
                f"context_length + horizon ({min_len})."
            )
        self.y              = torch.from_numpy(y)                          # [T, C]
        self.hist           = torch.from_numpy(hist)                       # [T, C, Vh]
        self.futr           = torch.from_numpy(futr)                       # [T, C, Vf]
        # Store as [C, T] so collation pads time on the left naturally
        self.available_mask = torch.from_numpy(
            available_mask.astype(np.float32)
        ).T.contiguous()    
        self.channel_ids = channel_ids or [str(i) for i in range(C)]
        self.metadata       = metadata
        self.ctx            = context_length
        self.horizon        = horizon
        self.T              = T
        self.name           = name

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        y_enc = self.y.unsqueeze(-1)                                       # [T, C, 1]
        x_enc = (
            torch.cat([y_enc, self.hist], dim=-1)
            if self.hist.shape[-1] > 0 else y_enc
        )                                                                  # [T, C, 1+Vh]
        out = dict(
            x_enc          = x_enc,                                        # [T, C, 1+Vh]
            x_futr         = self.futr,                                    # [T, C, Vf]
            available_mask = self.available_mask,                          # [C, T]
            series_len     = torch.tensor(self.T,       dtype=torch.long),
            horizon        = torch.tensor(self.horizon, dtype=torch.long),
            dataset_name   = self.name,
            channel_ids    = self.channel_ids,
        )
        if self.metadata is not None and self.metadata.data.shape[-1] > 0:
            out["x_stat"] = self.metadata.data                             # [C, n_stat]
        return out


def _make_dataset(
    y, hist, futr, stat, available_mask, channel_ids, mcfg, horizon, name=""
) -> FullSeriesDataset:
    ctx      = _ctx(mcfg)
    metadata = (
        SeriesMetadata(
            data        = stat,
            col_names   = getattr(mcfg, "stat_exog_cols", []) or [],
            channel_ids = channel_ids,
        )
        if stat.shape[-1] > 0 else None
    )
    return FullSeriesDataset(
        y              = y,
        hist           = hist,
        futr           = futr,
        available_mask = available_mask,
        context_length = ctx,
        horizon        = horizon,
        channel_ids    = channel_ids,
        metadata       = metadata,
        name           = name,
    )


def _ctx(mcfg) -> int:
    return getattr(mcfg, "input_size", None) or getattr(mcfg, "context_length", 512)


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

        # Right-pad channels and features, then left-pad time
        x = F.pad(s["x_enc"], (0, Vh_max - Vh_i, 0, C_max - C_i))  # [T_i, C_max, 1+Vh_max]
        x = _pad_left(x, T_max)                                       # [T_max, C_max, 1+Vh_max]
        out_x_enc.append(x)

        # available_mask: [C_i, T_i] -> right-pad channels, left-pad time
        m = F.pad(s["available_mask"], (0, 0, 0, C_max - C_i))       # [C_max, T_i]
        m = F.pad(m, (T_max - T_i, 0))                                # [C_max, T_max]
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
        x_enc          = torch.stack(out_x_enc),
        available_mask = torch.stack(out_mask),
        hist_mask      = hist_mask,
        dataset_name   = [s.get("dataset_name", "unknown") for s in batch],
        channel_ids    = [s.get("channel_ids",  [])        for s in batch],
    )
    for key in ("series_len", "horizon"):
        if key in batch[0]:
            result[key] = torch.stack([s[key] for s in batch])
    if out_x_stat:
        result["x_stat"] = torch.stack(out_x_stat) # [B, C_max, n_stat]

    return result


# ─────────────────────────────────────────────────────────────────────────────
# HorizonBatchSampler
# ─────────────────────────────────────────────────────────────────────────────

class HorizonBatchSampler(Sampler):
    """
    Samples batches grouped by horizon with weighted dataset mixing.

    Distributed behaviour
    ─────────────────────
    When world_size > 1 each rank receives a non-overlapping CONTIGUOUS slice
    of every horizon group's pool. Contiguous (not strided) slices are used
    for better cache locality.

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
        mixing_strategy="concat",
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
        self.mixing_strategy = mixing_strategy
        self.shuffle         = shuffle
        self.drop_last       = drop_last
        self.seed            = seed
        self.rank            = rank
        self.world_size      = world_size
        self._epoch          = 0
        self.horizons        = sorted(group_datasets.keys())

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def _group_batches(self, horizon, rng):
        datasets = self.group_datasets[horizon]
        weights  = self.group_weights[horizon]
        offsets  = self.global_offsets[horizon]

        per_ds = []
        for ds, offset in zip(datasets, offsets):
            idxs = np.arange(len(ds)) + offset
            if self.shuffle:
                rng.shuffle(idxs)
            per_ds.append(idxs)

        # Exhaustive mode — every item exactly once, no cycling, no weighting.
        # Triggered when shuffle=False and all weights are equal (eval path).
        if not self.shuffle and len(set(weights)) == 1:
            pool    = [idx for idxs in per_ds for idx in idxs.tolist()]
            bs      = self.batch_size
            batches = [pool[i : i + bs] for i in range(0, len(pool) - bs + 1, bs)]
            if not self.drop_last and len(pool) % bs:
                batches.append(pool[-(len(pool) % bs):])
            return batches

        total = sum(len(a) for a in per_ds)
        w_arr = np.array(weights, dtype=np.float64)
        w_arr = w_arr / w_arr.sum()
        slots_per = (w_arr * total).round().astype(int)
        slots_per[np.argmax(slots_per)] += total - slots_per.sum()

        pool     = []
        ds_iters = [itertools.cycle(a.tolist()) for a in per_ds]
        for slots, it in zip(slots_per, ds_iters):
            pool.extend(itertools.islice(it, int(slots)))
        if self.shuffle:
            rng.shuffle(pool)

        if self.world_size > 1:
            # Pad to multiple of (batch_size * world_size) so every rank gets
            # the same number of complete batches. Draw pad from front of pool.
            total_slots = self.batch_size * self.world_size
            pad  = (-len(pool)) % total_slots
            pool = pool + pool[:pad]
            # Contiguous slice per rank — better cache behaviour than strided
            rank_size = len(pool) // self.world_size
            pool = pool[self.rank * rank_size : (self.rank + 1) * rank_size]

        if self.drop_last:
            pool = pool[: (len(pool) // self.batch_size) * self.batch_size]

        bs = self.batch_size
        batches = [pool[i : i + bs] for i in range(0, len(pool) - bs + 1, bs)]
        if not self.drop_last and len(pool) % bs:
            batches.append(pool[-(len(pool) % bs):])
        return batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        group_batch_lists = {h: self._group_batches(h, rng) for h in self.horizons}

        if self.mixing_strategy == "round_robin":
            iters  = {h: iter(b) for h, b in group_batch_lists.items()}
            active = list(self.horizons)
            while active:
                exhausted = []
                for h in active:
                    b = next(iters[h], None)
                    if b is None:
                        exhausted.append(h)
                    else:
                        yield b
                for h in exhausted:
                    active.remove(h)
        else:
            all_batches = [b for bl in group_batch_lists.values() for b in bl]
            if self.shuffle:
                order       = rng.permutation(len(all_batches)).tolist()
                all_batches = [all_batches[i] for i in order]
            yield from all_batches

    def __len__(self):
        total = 0
        for h, datasets in self.group_datasets.items():
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
        self.mcfg        = mcfg
        self.dcfg        = dcfg
        self._horizon_groups: Dict[int, list]   = defaultdict(list)
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
            if getattr(entry, "sharded_dir", None):
                # rank/world_size injected later via rebuild_for_rank()
                # For single-GPU training, defaults of rank=0, world_size=1 are correct
                ds = ShardedTrainDataset(
                    data_dir       = entry.sharded_dir,
                    context_length = _ctx(self.mcfg),
                    horizon        = entry.horizon,
                    rank           = getattr(self, "_rank", 0),
                    world_size     = getattr(self, "_world_size", 1),
                )
                self._horizon_groups[entry.horizon].append((ds, entry.weight, entry.name))
                continue

            df = _load_df(entry.path)
            train_df, _, _ = _split_df(
                df, entry.val_size, entry.test_size,
                per_series_split=entry.per_series_split,
            )
            y, hist, futr, stat, channel_ids, available_mask = self._arrays_from_df(
                train_df, entry
            )
            ds = _make_dataset(
                y, hist, futr, stat, available_mask,
                channel_ids, self.mcfg, entry.horizon, entry.name,
            )
            self._horizon_groups[entry.horizon].append((ds, entry.weight, entry.name))

    def rebuild_for_rank(self, rank: int, world_size: int) -> "DataLoaderFactory":
        """
        Rebuild sharded train datasets for a specific rank.
        Called by _distributed_worker after rank/world_size are known.
        Non-sharded datasets are unaffected.
        """
        self._rank       = rank
        self._world_size = world_size
        self._horizon_groups.clear()
        self._build_train()
        return self

    def _build_eval_dataset(self, entry, split: str) -> Dataset:
        if getattr(entry, "sharded_dir", None):
            if split == "val":
                from dataloaders.ts_sharding import ShardedValDataset
                return ShardedValDataset(
                    data_dir       = entry.sharded_dir,
                    context_length = _ctx(self.mcfg),
                    horizon        = entry.horizon,
                )
            if split == "test":
                from dataloaders.ts_sharding import ShardedTestDataset
                return ShardedTestDataset(
                    data_dir       = entry.sharded_dir,
                    context_length = _ctx(self.mcfg),
                    horizon        = entry.horizon,
                )

        df = _load_df(entry.path)
        train_df, val_df, test_df = _split_df(
            df, entry.val_size, entry.test_size,
            per_series_split=entry.per_series_split,
        )
        eval_df = val_df if split == "val" else test_df
        if eval_df is None or len(eval_df) == 0:
            raise ValueError(f"Dataset '{entry.name}': '{split}' split is empty.")

        if entry.use_context_head and train_df is not None and len(train_df) > 0:
            ctx_rows = train_df.groupby("unique_id", sort=False).tail(_ctx(self.mcfg))
            eval_df  = pd.concat([ctx_rows, eval_df]).sort_values(["unique_id", "ds"])

        y, hist, futr, stat, channel_ids, available_mask = self._arrays_from_df(
            eval_df, entry
        )
        return _make_dataset(
            y, hist, futr, stat, available_mask,
            channel_ids, self.mcfg, entry.horizon, entry.name,
        )

    def _make_horizon_batch_sampler(self, rank=0, world_size=1):
        all_datasets, group_datasets, group_weights, global_offsets = (
            [], defaultdict(list), defaultdict(list), defaultdict(list)
        )
        flat_offset = 0
        for horizon in sorted(self._horizon_groups.keys()):
            for ds, weight, _ in self._horizon_groups[horizon]:
                global_offsets[horizon].append(flat_offset)
                group_datasets[horizon].append(ds)
                group_weights[horizon].append(weight)
                all_datasets.append(ds)
                flat_offset += len(ds)

        combined = ConcatDataset(all_datasets)
        sampler  = HorizonBatchSampler(
            group_datasets  = group_datasets,
            group_weights   = group_weights,
            global_offsets  = global_offsets,
            batch_size      = self.mcfg.batch_size,
            mixing_strategy = self.mcfg.mixing_strategy,
            shuffle         = True,
            drop_last       = self.mcfg.drop_last,
            rank            = rank,
            world_size      = world_size,
        )
        return combined, sampler

    def train_dataloader(self, distributed=False, rank=0, world_size=1) -> DataLoader:
        if not self._horizon_groups:
            raise RuntimeError("No training datasets configured.")
        combined, batch_sampler = self._make_horizon_batch_sampler(
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
        horizon_groups: Dict[int, list] = defaultdict(list)
        group_weights:  Dict[int, list] = defaultdict(list)
        global_offsets: Dict[int, list] = defaultdict(list)
        all_datasets = []
        flat_offset  = 0

        for entry in entries:
            ds = self._build_eval_dataset(entry, split)
            h  = entry.horizon
            global_offsets[h].append(flat_offset)
            horizon_groups[h].append(ds)
            group_weights[h].append(entry.weight)
            all_datasets.append(ds)
            flat_offset += len(ds)

        combined = ConcatDataset(all_datasets)
        sampler  = HorizonBatchSampler(
            group_datasets  = horizon_groups,
            group_weights   = {h: [1.0] * len(ds_list) for h, ds_list in horizon_groups.items()},
            global_offsets  = global_offsets,
            batch_size      = self.mcfg.valid_batch_size,
            mixing_strategy = self.mcfg.mixing_strategy,
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
        rng = np.random.default_rng(int(self._val_rng.integers(2**32)) + epoch)

        if strategy == "exhaustive":
            return {"val": self._make_eval_dataloader(self.dcfg.validation, "val")}

        elif strategy == "random_datasets":
            k       = getattr(self.mcfg, "val_max_datasets", len(self.dcfg.validation))
            indices = rng.choice(len(self.dcfg.validation), size=min(k, len(self.dcfg.validation)), replace=False)
            entries = [self.dcfg.validation[i] for i in indices]

        elif strategy == "stratified":
            horizon_groups = defaultdict(list)
            for entry in self.dcfg.validation:
                horizon_groups[entry.horizon].append(entry)
            entries = [group[int(rng.integers(len(group)))] for group in horizon_groups.values()]
            return {"val": self._make_eval_dataloader(entries, "val")}

        else:
            raise ValueError(f"Unknown val_strategy '{strategy}'. "
                            f"Choose from: exhaustive, random_datasets, stratified")

    def test_dataloaders(self, distributed=False, rank=0, world_size=1):
        # Test: always single GPU, always outside distributed context.
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
    