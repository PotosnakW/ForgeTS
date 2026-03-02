import itertools
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    DistributedSampler,
    Sampler,
)


# Config dataclasses removed — use config.py:
#   mcfg = load_model_config(base_cfg_path, model_cfg_path)
#   dcfg = load_dataset_config(dataset_cfg_path)
#
# mcfg is a SimpleNamespace with all merged base + model fields.
# dcfg.train / .validation / .test are lists of SimpleNamespace entries.
#
# Attribute aliases: base_config.yaml uses "input_size"; code below
# checks both "input_size" and "context_length" via _ctx().

def _ctx(mcfg) -> int:
    """Return context length — handles both input_size and context_length keys."""
    return getattr(mcfg, "input_size", None) or getattr(mcfg, "context_length", 512)


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
    """
    Chronological train/val/test split.

    per_series_split=False (default)
        Shared timestamp boundaries — all series cut at the same timestamp.
        Best for aligned global datasets (energy, weather, finance).

    per_series_split=True
        Each series split independently at its own last N timesteps.
        Best for datasets with independent timelines (patient stays, sessions).
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["ds"]):
        df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values(["unique_id", "ds"])

    if per_series_split:
        return _split_per_series(df, val_size, test_size)
    return _split_by_timestamp(df, val_size, test_size)


def _split_by_timestamp(
    df: pd.DataFrame,
    val_size: int,
    test_size: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_times = df["ds"].drop_duplicates().sort_values().reset_index(drop=True)
    T = len(all_times)

    if val_size + test_size >= T:
        raise ValueError(
            f"val_size ({val_size}) + test_size ({test_size}) = {val_size + test_size} "
            f">= total timestamps ({T}). Increase series length or reduce split sizes."
        )

    train_end = all_times.iloc[T - val_size - test_size - 1]
    val_end   = all_times.iloc[T - test_size - 1]

    return (
        df[df["ds"] <= train_end],
        df[(df["ds"] > train_end) & (df["ds"] <= val_end)],
        df[df["ds"] > val_end],
    )


def _split_per_series(
    df: pd.DataFrame,
    val_size: int,
    test_size: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def label(g: pd.DataFrame) -> pd.DataFrame:
        n = len(g)
        if val_size + test_size >= n:
            raise ValueError(
                f"Series '{g['unique_id'].iloc[0]}' has {n} timesteps but "
                f"val_size ({val_size}) + test_size ({test_size}) = {val_size + test_size}."
            )
        splits = (
            ["train"] * (n - val_size - test_size)
            + ["val"]  * val_size
            + ["test"] * test_size
        )
        return g.assign(_split=splits)

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

    def _piv(col: str) -> np.ndarray:
        return (
            df.pivot(index="ds", columns="unique_id", values=col)
            .loc[:, channel_ids]
            .values
            .astype(np.float32)
        )

    T, C = df["ds"].nunique(), len(channel_ids)
    y    = _piv("y")
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

    if "available_mask" in df.columns:
        mask_pivot     = _piv("available_mask")
        available_mask = mask_pivot.min(axis=1)
    else:
        available_mask = np.ones(T, dtype=np.float32)

    return y, hist, futr, stat, channel_ids, available_mask


# ─────────────────────────────────────────────────────────────────────────────
# Dataset classes
# ─────────────────────────────────────────────────────────────────────────────

class _TSWindowDataset(Dataset):
    def __init__(
        self,
        y, hist, futr, stat, available_mask,
        context_length, horizon, name="",
    ):
        min_len = context_length + horizon
        if y.shape[0] < min_len:
            raise ValueError(
                f"Dataset '{name}': series length {y.shape[0]} < "
                f"context_length + horizon ({min_len})."
            )
        self.y              = torch.from_numpy(y)
        self.hist           = torch.from_numpy(hist)
        self.futr           = torch.from_numpy(futr)
        self.stat           = torch.from_numpy(stat)
        self.available_mask = torch.from_numpy(available_mask.astype(np.float32))
        self.ctx            = context_length
        self.horizon        = horizon
        self.T              = y.shape[0]
        self.name           = name


class DatasetSpecificWindowDataset(_TSWindowDataset):
    def __len__(self):
        return self.T - self.ctx - self.horizon + 1

    def __getitem__(self, idx):
        h  = self.horizon
        t0, t1, t2 = idx, idx + self.ctx, idx + self.ctx + h
        y_enc = self.y[t0:t1].unsqueeze(-1)
        x_enc = (
            torch.cat([y_enc, self.hist[t0:t1]], dim=-1)
            if self.hist.shape[-1] > 0 else y_enc
        )
        return dict(
            x_enc        = x_enc,
            x_futr       = self.futr[t0:t2],
            x_stat       = self.stat,
            y            = self.y[t1:t2],
            horizon      = torch.tensor(h, dtype=torch.long),
            window_start = torch.tensor(idx, dtype=torch.long),
        )


class MixedWindowDataset(_TSWindowDataset):
    def __len__(self):
        return (self.T - self.ctx - self.horizon + 1) * self.y.shape[1]

    def __getitem__(self, idx):
        h  = self.horizon
        C  = self.y.shape[1]
        c  = idx % C
        t0 = idx // C
        t1, t2 = t0 + self.ctx, t0 + self.ctx + h
        y_enc = self.y[t0:t1, c].unsqueeze(-1).unsqueeze(-1)
        x_enc = (
            torch.cat([y_enc, self.hist[t0:t1, c, :].unsqueeze(1)], dim=-1)
            if self.hist.shape[-1] > 0 else y_enc
        )
        return dict(
            x_enc        = x_enc,
            x_futr       = self.futr[t0:t2, c, :].unsqueeze(1),
            x_stat       = self.stat[c].unsqueeze(0),
            y            = self.y[t1:t2, c].unsqueeze(-1),
            horizon      = torch.tensor(h, dtype=torch.long),
            channel_idx  = torch.tensor(c, dtype=torch.long),
            window_start = torch.tensor(t0, dtype=torch.long),
        )


class FullSeriesDataset(_TSWindowDataset):
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        y_enc = self.y.unsqueeze(-1)
        x_enc = (
            torch.cat([y_enc, self.hist], dim=-1)
            if self.hist.shape[-1] > 0 else y_enc
        )
        return dict(
            x_enc          = x_enc,
            x_futr         = self.futr,
            x_stat         = self.stat,
            available_mask = self.available_mask,
            series_len     = torch.tensor(self.T, dtype=torch.long),
            horizon        = torch.tensor(self.horizon, dtype=torch.long),
        )


def _make_dataset(y, hist, futr, stat, available_mask, mcfg, horizon, name=""):
    ctx = _ctx(mcfg)
    if mcfg.batch_mode == "full_series":
        return FullSeriesDataset(y, hist, futr, stat, available_mask, ctx, horizon, name)
    cls = (
        DatasetSpecificWindowDataset
        if mcfg.batch_mode == "dataset_specific"
        else MixedWindowDataset
    )
    return cls(y, hist, futr, stat, available_mask, ctx, horizon, name)


# ─────────────────────────────────────────────────────────────────────────────
# Collate
# ─────────────────────────────────────────────────────────────────────────────

def _build_channel_and_hist_masks(batch, C_max, Vh_max):
    B = len(batch)
    channel_mask = torch.zeros(B, C_max,          dtype=torch.float32)
    hist_mask    = torch.zeros(B, max(Vh_max, 1), dtype=torch.float32)
    for i, s in enumerate(batch):
        C_i  = s["x_enc"].shape[-2]
        Vh_i = s["x_enc"].shape[-1] - 1
        channel_mask[i, :C_i] = 1.0
        if Vh_i > 0:
            hist_mask[i, :Vh_i] = 1.0
    return channel_mask, hist_mask


def _collate_fn(batch):
    import torch.nn.functional as F
    C_max  = max(s["x_enc"].shape[-2]     for s in batch)
    Vh_max = max(s["x_enc"].shape[-1] - 1 for s in batch)
    out = {}
    for key in batch[0]:
        first = batch[0][key]
        if not isinstance(first, torch.Tensor):
            out[key] = [s[key] for s in batch]; continue
        if key == "x_enc":
            out[key] = torch.stack([
                F.pad(s[key], (0, Vh_max - (s[key].shape[-1] - 1),
                               0, C_max  -  s[key].shape[-2]))
                for s in batch
            ])
        elif key in ("x_futr", "x_stat"):
            out[key] = torch.stack([
                F.pad(s[key], (0, 0, 0, C_max - s[key].shape[-2]))
                for s in batch
            ])
        elif key == "y":
            out[key] = torch.stack([
                F.pad(s[key], (0, C_max - s[key].shape[-1]))
                for s in batch
            ])
        else:
            out[key] = torch.stack([s[key] for s in batch])
    ch_mask, h_mask = _build_channel_and_hist_masks(batch, C_max, Vh_max)
    out["channel_mask"] = ch_mask
    out["hist_mask"]    = h_mask
    return out


def _full_series_collate_fn(batch):
    import torch.nn.functional as F
    T_max  = max(s["x_enc"].shape[0]      for s in batch)
    C_max  = max(s["x_enc"].shape[-2]     for s in batch)
    Vh_max = max(s["x_enc"].shape[-1] - 1 for s in batch)
    out = {}
    for key in batch[0]:
        first = batch[0][key]
        if not isinstance(first, torch.Tensor):
            out[key] = [s[key] for s in batch]; continue
        if key == "x_enc":
            out[key] = torch.stack([
                F.pad(s[key], (0, Vh_max - (s[key].shape[-1] - 1),
                               0, C_max  -  s[key].shape[-2],
                               0, T_max  -  s[key].shape[0]))
                for s in batch
            ])
        elif key == "x_futr":
            out[key] = torch.stack([
                F.pad(s[key], (0, 0,
                               0, C_max - s[key].shape[-2],
                               0, T_max - s[key].shape[0]))
                for s in batch
            ])
        elif key == "available_mask":
            out[key] = torch.stack([
                F.pad(s[key], (0, T_max - s[key].shape[0]))
                for s in batch
            ])
        else:
            out[key] = torch.stack([s[key] for s in batch])
    ch_mask, h_mask = _build_channel_and_hist_masks(batch, C_max, Vh_max)
    out["channel_mask"] = ch_mask
    out["hist_mask"]    = h_mask
    return out


class HorizonBatchSampler(Sampler):
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

    def set_epoch(self, epoch):
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

        total = sum(len(a) for a in per_ds)
        if self.drop_last:
            total = (total // self.batch_size) * self.batch_size

        w_arr = np.array(weights, dtype=np.float64)
        w_arr = w_arr / w_arr.sum()

        pool       = []
        ds_iters   = [itertools.cycle(a.tolist()) for a in per_ds]
        slots_per  = (w_arr * total).round().astype(int)
        slots_per[np.argmax(slots_per)] += total - slots_per.sum()
        for slots, it in zip(slots_per, ds_iters):
            pool.extend(itertools.islice(it, int(slots)))

        if self.shuffle:
            rng.shuffle(pool)

        if self.world_size > 1:
            pad  = (-len(pool)) % (self.batch_size * self.world_size)
            pool = pool + pool[:pad]
            pool = pool[self.rank::self.world_size]

        bs      = self.batch_size
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
                    batch = next(iters[h], None)
                    if batch is None:
                        exhausted.append(h)
                    else:
                        yield batch
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
            total += max(1, n // self.batch_size // self.world_size)
        return total


# ─────────────────────────────────────────────────────────────────────────────
# Normaliser  (z-score per channel, fit on train y and hist)
# ─────────────────────────────────────────────────────────────────────────────

class Normaliser:
    def fit(self, y: np.ndarray, hist: np.ndarray) -> "Normaliser":
        self.y_mean = y.mean(axis=0, keepdims=True)
        self.y_std  = y.std(axis=0, keepdims=True) + 1e-8
        return self

    def transform(
        self, y: np.ndarray, hist: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return (y - self.y_mean) / self.y_std, hist

    def inverse_transform_y(self, y: np.ndarray) -> np.ndarray:
        return y * self.y_std + self.y_mean


# ─────────────────────────────────────────────────────────────────────────────
# DataLoaderFactory
# ─────────────────────────────────────────────────────────────────────────────

class DataLoaderFactory:
    def __init__(self, mcfg, dcfg):
        self.mcfg        = mcfg
        self.dcfg        = dcfg
        self.normalisers: Dict[str, Normaliser] = {}
        self._horizon_groups: Dict[int, List[Tuple[_TSWindowDataset, float, str]]] = (
            defaultdict(list)
        )
        self._build_train()

    def _arrays_from_df(self, df, entry):
        y, hist, futr, stat, _, available_mask = _pivot_to_arrays(
            df, entry.hist_exog_cols, entry.futr_exog_cols, entry.stat_exog_cols
        )
        return y, hist, futr, stat, available_mask

    def _build_train(self):
        for entry in self.dcfg.train:
            df = _load_df(entry.path)
            train_df, _, _ = _split_df(
                df, entry.val_size, entry.test_size,
                per_series_split=entry.per_series_split,
            )
            y, hist, futr, stat, available_mask = self._arrays_from_df(train_df, entry)
            norm = Normaliser()
            if self.mcfg.normalize:
                norm.fit(y, hist)
                y, hist = norm.transform(y, hist)
            self.normalisers[entry.name] = norm
            ds = _make_dataset(
                y, hist, futr, stat, available_mask,
                self.mcfg, entry.horizon, entry.name,
            )
            self._horizon_groups[entry.horizon].append((ds, entry.weight, entry.name))

    def _build_eval_dataset(self, entry, split: str) -> _TSWindowDataset:
        df = _load_df(entry.path)
        train_df, val_df, test_df = _split_df(
            df, entry.val_size, entry.test_size,
            per_series_split=entry.per_series_split,
        )
        eval_df = val_df if split == "val" else test_df

        if eval_df is None or len(eval_df) == 0:
            raise ValueError(
                f"Dataset '{entry.name}': '{split}' split is empty. "
                "Check val_size / test_size against total series length."
            )

        if entry.use_context_head and train_df is not None and len(train_df) > 0:
            ctx_rows = train_df.groupby("unique_id", sort=False).tail(
                _ctx(self.mcfg)
            )
            eval_df = pd.concat([ctx_rows, eval_df]).sort_values(["unique_id", "ds"])

        y, hist, futr, stat, available_mask = self._arrays_from_df(eval_df, entry)

        norm = self.normalisers.get(entry.name)
        if norm is None:
            norm = Normaliser().fit(y, hist)
            self.normalisers[entry.name] = norm
        if self.mcfg.normalize:
            y, hist = norm.transform(y, hist)

        return _make_dataset(
            y, hist, futr, stat, available_mask,
            self.mcfg, entry.horizon, entry.name,
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
            group_datasets   = group_datasets,
            group_weights    = group_weights,
            global_offsets   = global_offsets,
            batch_size       = self.mcfg.batch_size,
            mixing_strategy  = self.mcfg.mixing_strategy,
            shuffle          = True,
            drop_last        = self.mcfg.drop_last,
            rank             = rank,
            world_size       = world_size,
        )
        return combined, sampler

    def train_dataloader(self, distributed=False, rank=0, world_size=1) -> DataLoader:
        if not self._horizon_groups:
            raise RuntimeError("No training datasets configured.")
        combined, batch_sampler = self._make_horizon_batch_sampler(
            rank       = rank       if distributed else 0,
            world_size = world_size if distributed else 1,
        )
        collate = (
            _full_series_collate_fn
            if self.mcfg.batch_mode == "full_series"
            else _collate_fn
        )
        return DataLoader(
            combined,
            batch_sampler      = batch_sampler,
            num_workers        = self.mcfg.num_workers,
            pin_memory         = True,
            collate_fn         = collate,
            persistent_workers = self.mcfg.num_workers > 0,
        )

    def val_dataloaders(self, distributed=False, rank=0, world_size=1):
        return self._eval_loaders(self.dcfg.validation, "val", distributed, rank, world_size)

    def test_dataloaders(self, distributed=False, rank=0, world_size=1):
        return self._eval_loaders(self.dcfg.test, "test", distributed, rank, world_size)

    def _eval_loaders(self, entries, split, distributed, rank, world_size):
        loaders = {}
        collate = (
            _full_series_collate_fn
            if self.mcfg.batch_mode == "full_series"
            else _collate_fn
        )
        for entry in entries:
            ds      = self._build_eval_dataset(entry, split)
            sampler = (
                DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=False)
                if distributed else None
            )
            loaders[entry.name] = DataLoader(
                ds,
                batch_size         = self.mcfg.valid_batch_size,
                shuffle            = False,
                sampler            = sampler,
                num_workers        = self.mcfg.num_workers,
                pin_memory         = True,
                drop_last          = False,
                collate_fn         = collate,
                persistent_workers = self.mcfg.num_workers > 0,
            )
        return loaders
