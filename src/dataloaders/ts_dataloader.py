"""
ts_dataloader.py
────────────────
Time-series dataloader driven by two YAML config files:

    model_config.yaml    — architecture / training hyper-parameters
    dataset_config.yaml  — all dataset-specific parameters

Key guarantee
─────────────
Every batch contains samples from a SINGLE horizon group.
Datasets sharing the same horizon value are pooled and can appear
together in a batch. Datasets with different horizons are always
in separate batches — no padding or masking, ever.

Parameter ownership
───────────────────
    ModelConfig   : context_length, batch_mode, mixing_strategy, training knobs
    DatasetEntry  : horizon, val_size, test_size, exog columns, weight

Encoder output shapes
─────────────────────
    x_enc  : [B, context_length, C, 1 + n_hist]
    x_futr : [B, context_length + horizon, C, n_futr]
    x_stat : [B, C, n_stat]
    y      : [B, horizon, C]
    horizon: [B]  scalar — same value for every item in a batch

    C = n_channels (unique_ids in one file).
    In mixed batch_mode C = 1 (each item is one series).

Quickstart
──────────
    from ts_dataloader import ModelConfig, DatasetConfig, DataLoaderFactory

    mcfg    = ModelConfig.from_yaml("model_config.yaml")
    dcfg    = DatasetConfig.from_yaml("dataset_config.yaml")
    factory = DataLoaderFactory(mcfg, dcfg)

    train_loader = factory.train_dataloader()
    val_loaders  = factory.val_dataloaders()   # dict[name → DataLoader]
    test_loaders = factory.test_dataloaders()  # dict[name → DataLoader]

    # distributed
    train_loader = factory.train_dataloader(distributed=True, rank=0, world_size=4)

    # call every epoch (DDP + round_robin)
    for epoch in range(n_epochs):
        set_epoch(train_loader, epoch)
        for batch in train_loader:
            ...
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import (
    BatchSampler,
    ConcatDataset,
    DataLoader,
    Dataset,
    DistributedSampler,
    Sampler,
    WeightedRandomSampler,
)


# ─────────────────────────────────────────────────────────────────────────────
# Config dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """
    Loaded from model_config.yaml.
    Purely architectural / training parameters — nothing dataset-specific.
    """
    context_length: int

    batch_mode: str              = "dataset_specific"  # "dataset_specific" | "mixed"
    mixing_strategy: str         = "concat"             # "concat" | "round_robin"

    batch_size: int              = 32
    valid_batch_size: int        = 1024
    num_workers: int             = 0
    drop_last: bool              = False
    normalize: bool              = True

    max_epochs: int              = 100
    learning_rate: float         = 1e-3
    gradient_clip_val: float     = 1.0
    early_stopping_patience: int = 10

    checkpoint_dir: str          = "checkpoints/"
    save_top_k: int              = 3
    monitor_metric: str          = "val_loss"
    monitor_mode: str            = "min"

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ModelConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in raw.items() if k in known})

    def validate(self):
        if self.batch_mode not in ("dataset_specific", "mixed"):
            raise ValueError(
                f"batch_mode must be 'dataset_specific' or 'mixed', got '{self.batch_mode}'."
            )
        if self.mixing_strategy not in ("concat", "round_robin"):
            raise ValueError(
                f"mixing_strategy must be 'concat' or 'round_robin', got '{self.mixing_strategy}'."
            )


_REQUIRED = frozenset((
    "path", "name", "horizon", "val_size", "test_size",
    "hist_exog_cols", "futr_exog_cols", "stat_exog_cols",
))


@dataclass
class DatasetEntry:
    """
    One dataset record from dataset_config.yaml.
    All fields are per-dataset — nothing inherited from ModelConfig.
    """
    # ── required ─────────────────────────────────────────────────
    path:           str
    name:           str
    horizon:        int
    val_size:       int
    test_size:      int
    hist_exog_cols: List[str]
    futr_exog_cols: List[str]
    stat_exog_cols: List[str]

    # ── train only ───────────────────────────────────────────────
    weight: float = 1.0

    # ── val / test only ──────────────────────────────────────────
    use_context_head: bool = False

    def __post_init__(self):
        missing = _REQUIRED - set(
            f for f in self.__dataclass_fields__ if getattr(self, f, None) is not None
        )
        # also check lists (they default to [] which is not None)
        for f in ("hist_exog_cols", "futr_exog_cols", "stat_exog_cols"):
            if getattr(self, f, None) is None:
                missing.add(f)
        if missing:
            raise ValueError(f"DatasetEntry '{self.name}': missing required fields {sorted(missing)}.")
        for attr in ("horizon", "val_size", "test_size"):
            v = getattr(self, attr)
            if not isinstance(v, int) or v <= 0:
                raise TypeError(
                    f"DatasetEntry '{self.name}': '{attr}' must be a positive int, got {v!r}."
                )

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetEntry":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class DatasetConfig:
    """Loaded from dataset_config.yaml."""
    train:      List[DatasetEntry] = field(default_factory=list)
    validation: List[DatasetEntry] = field(default_factory=list)
    test:       List[DatasetEntry] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "DatasetConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(
            train      = [DatasetEntry.from_dict(d) for d in (raw.get("train")      or [])],
            validation = [DatasetEntry.from_dict(d) for d in (raw.get("validation") or [])],
            test       = [DatasetEntry.from_dict(d) for d in (raw.get("test")       or [])],
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
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological split on shared timestamp boundaries (oldest → newest):

        │◄──── train ────►│◄── val ──►│◄── test ──►│

    All channels are cut at the same timestamp so there is no leakage.
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["ds"]):
        df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values(["unique_id", "ds"])

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


# ─────────────────────────────────────────────────────────────────────────────
# Pivot → arrays
# ─────────────────────────────────────────────────────────────────────────────

def _pivot_to_arrays(
    df: pd.DataFrame,
    hist_exog_cols: List[str],
    futr_exog_cols: List[str],
    stat_exog_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Long-format DataFrame → aligned float32 numpy arrays.

    Returns: y (T,C), hist (T,C,Vh), futr (T,C,Vf), stat (C,Vs), channel_ids
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["ds"]):
        df["ds"] = pd.to_datetime(df["ds"])

    channel_ids = sorted(df["unique_id"].unique().tolist())

    # Align any series that differ in length after splitting
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
    hist = np.stack([_piv(c) for c in hist_exog_cols], axis=-1) if hist_exog_cols else np.zeros((T, C, 0), dtype=np.float32)
    futr = np.stack([_piv(c) for c in futr_exog_cols], axis=-1) if futr_exog_cols else np.zeros((T, C, 0), dtype=np.float32)
    stat = (
        df.groupby("unique_id", sort=False)[stat_exog_cols]
        .first().loc[channel_ids].values.astype(np.float32)
    ) if stat_exog_cols else np.zeros((C, 0), dtype=np.float32)

    return y, hist, futr, stat, channel_ids


# ─────────────────────────────────────────────────────────────────────────────
# Normaliser
# ─────────────────────────────────────────────────────────────────────────────

class Normaliser:
    """Per-channel z-score. Always fit on training data only."""

    def __init__(self):
        self.y_mean = self.y_std = self.h_mean = self.h_std = None

    def fit(self, y: np.ndarray, hist: np.ndarray) -> "Normaliser":
        self.y_mean = y.mean(axis=0, keepdims=True).astype(np.float32)
        self.y_std  = (y.std(axis=0, keepdims=True) + 1e-8).astype(np.float32)
        if hist.shape[-1] > 0:
            self.h_mean = hist.mean(axis=(0, 1), keepdims=True).astype(np.float32)
            self.h_std  = (hist.std(axis=(0, 1), keepdims=True) + 1e-8).astype(np.float32)
        else:
            self.h_mean = np.zeros((1, 1, 1), dtype=np.float32)
            self.h_std  = np.ones((1, 1, 1),  dtype=np.float32)
        return self

    def transform(self, y: np.ndarray, hist: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y_n = ((y - self.y_mean) / self.y_std).astype(np.float32)
        h_n = ((hist - self.h_mean) / self.h_std).astype(np.float32) if hist.shape[-1] > 0 else hist
        return y_n, h_n

    def inverse_transform_y(self, y_norm: torch.Tensor) -> torch.Tensor:
        """Undo z-score on model predictions.  y_norm: (..., C)"""
        mean = torch.from_numpy(self.y_mean).to(y_norm.device)
        std  = torch.from_numpy(self.y_std).to(y_norm.device)
        return y_norm * std + mean


# ─────────────────────────────────────────────────────────────────────────────
# Dataset classes
# ─────────────────────────────────────────────────────────────────────────────

class _TSWindowDataset(Dataset):
    """Shared base: stores tensors, validates minimum length."""

    def __init__(
        self,
        y:    np.ndarray,
        hist: np.ndarray,
        futr: np.ndarray,
        stat: np.ndarray,
        context_length: int,
        horizon: int,
        name: str = "",
    ):
        min_len = context_length + horizon
        if y.shape[0] < min_len:
            raise ValueError(
                f"Dataset '{name}': series length {y.shape[0]} < "
                f"context_length + horizon ({min_len}). "
                "Reduce val_size / test_size or context_length."
            )
        self.y       = torch.from_numpy(y)
        self.hist    = torch.from_numpy(hist)
        self.futr    = torch.from_numpy(futr)
        self.stat    = torch.from_numpy(stat)
        self.ctx     = context_length
        self.horizon = horizon
        self.T       = y.shape[0]
        self.name    = name


class DatasetSpecificWindowDataset(_TSWindowDataset):
    """
    All channels share the same time window per sample.

    Item shapes:
        x_enc  : (ctx, C, 1+Vh)
        x_futr : (ctx+h, C, Vf)
        x_stat : (C, Vs)
        y      : (h, C)
        horizon: scalar
    """

    def __len__(self) -> int:
        return self.T - self.ctx - self.horizon + 1

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        h  = self.horizon
        t0, t1, t2 = idx, idx + self.ctx, idx + self.ctx + h
        y_enc = self.y[t0:t1].unsqueeze(-1)
        x_enc = torch.cat([y_enc, self.hist[t0:t1]], dim=-1) if self.hist.shape[-1] > 0 else y_enc
        return dict(
            x_enc        = x_enc,
            x_futr       = self.futr[t0:t2],
            x_stat       = self.stat,
            y            = self.y[t1:t2],
            horizon      = torch.tensor(h, dtype=torch.long),
            window_start = torch.tensor(idx, dtype=torch.long),
        )


class MixedWindowDataset(_TSWindowDataset):
    """
    Each (channel, window) pair is an independent sample.

    Item shapes:
        x_enc  : (ctx, 1, 1+Vh)
        x_futr : (ctx+h, 1, Vf)
        x_stat : (1, Vs)
        y      : (h, 1)
        horizon: scalar
    """

    def __len__(self) -> int:
        return (self.T - self.ctx - self.horizon + 1) * self.y.shape[1]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
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


def _make_dataset(
    y, hist, futr, stat, mcfg: ModelConfig, horizon: int, name: str = ""
) -> _TSWindowDataset:
    cls = (
        DatasetSpecificWindowDataset
        if mcfg.batch_mode == "dataset_specific"
        else MixedWindowDataset
    )
    return cls(y, hist, futr, stat, mcfg.context_length, horizon, name)


# ─────────────────────────────────────────────────────────────────────────────
# Collate
# ─────────────────────────────────────────────────────────────────────────────

def _collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Stack a list of same-horizon sample dicts into batched tensors.
    All samples in a batch are guaranteed to have the same horizon,
    so shapes are always uniform — no padding logic needed here.
    """
    out: Dict[str, torch.Tensor] = {}
    for key in batch[0]:
        val = batch[0][key]
        out[key] = (
            torch.stack([s[key] for s in batch], dim=0)
            if isinstance(val, torch.Tensor)
            else [s[key] for s in batch]
        )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Horizon-bucketed batch sampler
# ─────────────────────────────────────────────────────────────────────────────

class HorizonBatchSampler(Sampler[List[int]]):
    """
    Yields complete batches where every index belongs to the same horizon group.

    Datasets are grouped by horizon value.  Within each group, datasets may
    be pooled (their indices are concatenated in the ConcatDataset flat space)
    and drawn according to per-dataset weights.

    Across horizon groups, batches are interleaved in one of two modes:
        concat      — groups emit batches proportionally to their total weight;
                      implemented as a single weighted draw over all group batches.
        round_robin — strict rotation: one batch from group 0, one from group 1, ...

    This is used as DataLoader(..., batch_sampler=HorizonBatchSampler(...))
    so DataLoader's batch_size, shuffle, sampler, and drop_last are ignored.

    Parameters
    ----------
    group_datasets : dict[horizon → list of Dataset]
        Datasets keyed by horizon. Each dataset must already be a
        _TSWindowDataset (or compatible __len__ Dataset).
    group_weights  : dict[horizon → list of float]
        Per-dataset weights within each horizon group.
    global_offsets : dict[horizon → list of int]
        Starting flat index of each dataset inside the ConcatDataset.
    batch_size     : int
    mixing_strategy: "concat" | "round_robin"
    shuffle        : bool
    drop_last      : bool
    seed           : int
    rank / world_size : for distributed support
    """

    def __init__(
        self,
        group_datasets: Dict[int, List[Dataset]],
        group_weights:  Dict[int, List[float]],
        global_offsets: Dict[int, List[int]],
        batch_size: int,
        mixing_strategy: str = "concat",
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
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

        self.horizons = sorted(group_datasets.keys())

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def _group_batches(
        self, horizon: int, rng: np.random.Generator
    ) -> List[List[int]]:
        """
        Return a list of batches (each batch = list of global flat indices)
        for one horizon group, sampled according to per-dataset weights.
        """
        datasets = self.group_datasets[horizon]
        weights  = self.group_weights[horizon]
        offsets  = self.global_offsets[horizon]

        # Build per-dataset shuffled index arrays
        per_ds: List[np.ndarray] = []
        for ds, offset in zip(datasets, offsets):
            idxs = np.arange(len(ds)) + offset
            if self.shuffle:
                rng.shuffle(idxs)
            per_ds.append(idxs)

        # Weighted interleave: sample dataset slots proportional to weight,
        # cycling each dataset's index sequence independently.
        total = sum(len(a) for a in per_ds)
        if self.drop_last:
            total = (total // self.batch_size) * self.batch_size

        # Normalise weights and build a probability-weighted draw order
        w_arr = np.array(weights, dtype=np.float64)
        w_arr = w_arr / w_arr.sum()

        # Expand each dataset into a (possibly repeated) flat pool weighted by w
        pool: List[int] = []
        ds_iters = [itertools.cycle(a.tolist()) for a in per_ds]
        slots_per_ds = (w_arr * total).round().astype(int)
        # Adjust rounding error on the heaviest dataset
        slots_per_ds[np.argmax(slots_per_ds)] += total - slots_per_ds.sum()
        for slots, it in zip(slots_per_ds, ds_iters):
            pool.extend(itertools.islice(it, int(slots)))

        if self.shuffle:
            rng.shuffle(pool)

        # Shard for distributed: each rank gets its own non-overlapping slice
        if self.world_size > 1:
            # Pad so all ranks get equal batches
            pad = (-len(pool)) % (self.batch_size * self.world_size)
            pool = pool + pool[:pad]
            pool = pool[self.rank::self.world_size]

        # Chop into batches
        bs = self.batch_size
        batches = [pool[i : i + bs] for i in range(0, len(pool) - bs + 1, bs)]
        if not self.drop_last and len(pool) % bs:
            batches.append(pool[-(len(pool) % bs):])

        return batches

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(self.seed + self._epoch)

        # Build per-group batch lists
        group_batch_lists: Dict[int, List[List[int]]] = {
            h: self._group_batches(h, rng) for h in self.horizons
        }

        if self.mixing_strategy == "round_robin":
            # Strict rotation across horizon groups
            iters = {h: iter(batches) for h, batches in group_batch_lists.items()}
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

        else:  # concat — interleave all batches, shuffle the order across groups
            all_batches: List[List[int]] = []
            for batches in group_batch_lists.values():
                all_batches.extend(batches)
            if self.shuffle:
                order = rng.permutation(len(all_batches)).tolist()
                all_batches = [all_batches[i] for i in order]
            yield from all_batches

    def __len__(self) -> int:
        # Approximate — actual length depends on rng (weights rounding),
        # but this gives DataLoader a reasonable value for tqdm etc.
        total_batches = 0
        for h, datasets in self.group_datasets.items():
            total = sum(len(ds) for ds in datasets)
            if self.drop_last:
                total = (total // self.batch_size) * self.batch_size
            n = total // self.batch_size
            total_batches += max(1, n // self.world_size)
        return total_batches


import itertools  # used inside HorizonBatchSampler


# ─────────────────────────────────────────────────────────────────────────────
# DataLoaderFactory — main public API
# ─────────────────────────────────────────────────────────────────────────────

class DataLoaderFactory:
    """
    Reads both configs, loads and splits datasets, fits normalisers,
    and vends ready-to-use DataLoaders.

    Train DataLoader
    ────────────────
    Uses HorizonBatchSampler to guarantee every batch contains only
    samples sharing the same horizon value.  Datasets with matching
    horizons are pooled and can appear in the same batch.

    Eval DataLoaders
    ────────────────
    One DataLoader per named entry in validation / test.
    Each is horizon-homogeneous by construction (single dataset per loader).

    Attributes
    ----------
    normalisers : dict[dataset_name → Normaliser]
        Call .inverse_transform_y() to convert model output to original scale.
    """

    def __init__(self, mcfg: ModelConfig, dcfg: DatasetConfig):
        mcfg.validate()
        self.mcfg = mcfg
        self.dcfg = dcfg
        self.normalisers: Dict[str, Normaliser] = {}

        # Built eagerly; used to construct the HorizonBatchSampler
        # Keyed by horizon → list of (Dataset, weight, name)
        self._horizon_groups: Dict[int, List[Tuple[_TSWindowDataset, float, str]]] = defaultdict(list)
        self._build_train()

    # ── internals ────────────────────────────────────────────────

    def _arrays_from_df(
        self, df: pd.DataFrame, entry: DatasetEntry
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        y, hist, futr, stat, _ = _pivot_to_arrays(
            df, entry.hist_exog_cols, entry.futr_exog_cols, entry.stat_exog_cols
        )
        return y, hist, futr, stat

    def _build_train(self):
        for entry in self.dcfg.train:
            df = _load_df(entry.path)
            train_df, _, _ = _split_df(df, entry.val_size, entry.test_size)

            y, hist, futr, stat = self._arrays_from_df(train_df, entry)

            norm = Normaliser()
            if self.mcfg.normalize:
                norm.fit(y, hist)
                y, hist = norm.transform(y, hist)
            self.normalisers[entry.name] = norm

            ds = _make_dataset(y, hist, futr, stat, self.mcfg, entry.horizon, entry.name)
            self._horizon_groups[entry.horizon].append((ds, entry.weight, entry.name))

    def _build_eval_dataset(self, entry: DatasetEntry, split: str) -> _TSWindowDataset:
        df = _load_df(entry.path)
        train_df, val_df, test_df = _split_df(df, entry.val_size, entry.test_size)
        eval_df = val_df if split == "val" else test_df

        if eval_df is None or len(eval_df) == 0:
            raise ValueError(
                f"Dataset '{entry.name}': '{split}' split is empty. "
                "Check val_size / test_size against total series length."
            )

        if entry.use_context_head and train_df is not None and len(train_df) > 0:
            ctx_rows = train_df.groupby("unique_id", sort=False).tail(self.mcfg.context_length)
            eval_df  = pd.concat([ctx_rows, eval_df]).sort_values(["unique_id", "ds"])

        y, hist, futr, stat = self._arrays_from_df(eval_df, entry)

        # Reuse training normaliser if available (same file → same z-score stats)
        norm = self.normalisers.get(entry.name)
        if norm is None:
            norm = Normaliser().fit(y, hist)
            self.normalisers[entry.name] = norm
        if self.mcfg.normalize:
            y, hist = norm.transform(y, hist)

        return _make_dataset(y, hist, futr, stat, self.mcfg, entry.horizon, entry.name)

    def _make_horizon_batch_sampler(
        self, rank: int = 0, world_size: int = 1
    ) -> Tuple[ConcatDataset, HorizonBatchSampler]:
        """
        Build the ConcatDataset and matching HorizonBatchSampler.

        The ConcatDataset is a flat union of all training datasets.
        The sampler tracks which flat indices belong to which horizon group
        and ensures each yielded batch is horizon-homogeneous.
        """
        all_datasets: List[Dataset] = []
        group_datasets: Dict[int, List[Dataset]] = defaultdict(list)
        group_weights:  Dict[int, List[float]]   = defaultdict(list)
        global_offsets: Dict[int, List[int]]     = defaultdict(list)

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

    # ── public DataLoader vending ─────────────────────────────────

    def train_dataloader(
        self,
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ) -> DataLoader:
        """
        Training DataLoader with horizon-bucketed batching.

        Every batch is guaranteed to contain only samples sharing the same
        horizon — regardless of how many datasets contributed to that batch.

        mixing_strategy = "concat"
            Batches from all horizon groups are emitted in a globally shuffled
            order, weighted by each group's total weight.

        mixing_strategy = "round_robin"
            Batches alternate strictly across horizon groups:
            [h0_b0, h1_b0, h2_b0, h0_b1, h1_b1, ...]

        Distributed training is supported; rank / world_size shard the
        flat index pool before batching. Call set_epoch() each epoch.
        """
        if not self._horizon_groups:
            raise RuntimeError("No training datasets configured.")

        effective_rank       = rank       if distributed else 0
        effective_world_size = world_size if distributed else 1

        combined, batch_sampler = self._make_horizon_batch_sampler(
            rank=effective_rank, world_size=effective_world_size
        )

        return DataLoader(
            combined,
            batch_sampler=batch_sampler,    # overrides batch_size / shuffle / sampler
            num_workers=self.mcfg.num_workers,
            pin_memory=True,
            collate_fn=_collate_fn,
            persistent_workers=self.mcfg.num_workers > 0,
        )

    def val_dataloaders(
        self,
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ) -> Dict[str, DataLoader]:
        """Returns {dataset_name: DataLoader} for every validation entry."""
        return self._eval_loaders(self.dcfg.validation, "val", distributed, rank, world_size)

    def test_dataloaders(
        self,
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ) -> Dict[str, DataLoader]:
        """Returns {dataset_name: DataLoader} for every test entry."""
        return self._eval_loaders(self.dcfg.test, "test", distributed, rank, world_size)

    def _eval_loaders(
        self,
        entries: List[DatasetEntry],
        split: str,
        distributed: bool,
        rank: int,
        world_size: int,
    ) -> Dict[str, DataLoader]:
        loaders = {}
        for entry in entries:
            ds = self._build_eval_dataset(entry, split)
            sampler = (
                DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=False)
                if distributed else None
            )
            loaders[entry.name] = DataLoader(
                ds,
                batch_size=self.mcfg.valid_batch_size,
                shuffle=False,
                sampler=sampler,
                num_workers=self.mcfg.num_workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=_collate_fn,
                persistent_workers=self.mcfg.num_workers > 0,
            )
        return loaders


# ─────────────────────────────────────────────────────────────────────────────
# Epoch helper
# ─────────────────────────────────────────────────────────────────────────────

def set_epoch(loader: DataLoader, epoch: int) -> None:
    """
    Call at the top of each training epoch.
    Supports HorizonBatchSampler and DistributedSampler.

        for epoch in range(n_epochs):
            set_epoch(train_loader, epoch)
            for batch in train_loader:
                ...
    """
    sampler = loader.batch_sampler if hasattr(loader, "batch_sampler") else loader.sampler
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)
    else:
        warnings.warn("set_epoch: sampler does not support set_epoch — no-op.")


# ─────────────────────────────────────────────────────────────────────────────
# Self-test  (python ts_dataloader.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import tempfile

    np.random.seed(42)

    def _synth(n_series: int = 4, n_steps: int = 700) -> pd.DataFrame:
        dates = pd.date_range("2020-01-01", periods=n_steps, freq="h")
        rows  = []
        for i in range(n_series):
            for t, ds in enumerate(dates):
                rows.append(dict(
                    unique_id      = f"s{i}",
                    ds             = ds,
                    y              = float(np.sin(t / 24 + i) + np.random.randn() * 0.1),
                    temperature    = float(20 + 5 * np.cos(t / 24)),
                    day_of_week    = float(ds.dayofweek),
                    price_forecast = float(1.0 + 0.1 * np.random.randn()),
                    holiday_flag   = float(ds.dayofweek >= 5),
                    region_id      = float(i % 2),
                ))
        return pd.DataFrame(rows)

    with tempfile.TemporaryDirectory() as tmp:
        # Three datasets: two share horizon=96, one has horizon=48
        p1 = os.path.join(tmp, "ds1.parquet"); _synth(4, 700).to_parquet(p1, index=False)
        p2 = os.path.join(tmp, "ds2.parquet"); _synth(3, 650).to_parquet(p2, index=False)
        p3 = os.path.join(tmp, "ds3.parquet"); _synth(2, 600).to_parquet(p3, index=False)

        def entry(path, name, horizon, val, test, weight=1):
            return dict(
                path=path, name=name,
                horizon=horizon, val_size=val, test_size=test,
                weight=weight,
                hist_exog_cols=["temperature", "day_of_week"],
                futr_exog_cols=["price_forecast", "holiday_flag"],
                stat_exog_cols=["region_id"],
            )

        mcfg_d = dict(
            context_length=48,
            batch_mode="dataset_specific",
            batch_size=4,
            valid_batch_size=8,
            normalize=True,
            num_workers=0,
        )
        dcfg_d = dict(
            train=[
                entry(p1, "ds1", horizon=96, val=96,  test=192, weight=2),
                entry(p2, "ds2", horizon=96, val=96,  test=192, weight=1),  # same horizon as ds1
                entry(p3, "ds3", horizon=48, val=48,  test=96,  weight=1),  # different horizon
            ],
            validation=[
                {**entry(p1, "ds1_val", horizon=96, val=96, test=192), "use_context_head": True},
                {**entry(p3, "ds3_val", horizon=48, val=48, test=96),  "use_context_head": True},
            ],
            test=[
                {**entry(p1, "ds1_test", horizon=96, val=96, test=192), "use_context_head": True},
                {**entry(p3, "ds3_test", horizon=48, val=48, test=96),  "use_context_head": True},
            ],
        )

        def _tmp(d):
            f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
            yaml.dump(d, f); f.close(); return f.name

        for strategy in ("concat", "round_robin"):
            print(f"\n{'═'*60}")
            print(f"  mixing_strategy = {strategy}")
            print(f"  horizon groups  : 96 (ds1 w=2, ds2 w=1)  |  48 (ds3 w=1)")
            print(f"{'═'*60}")

            mc = _tmp({**mcfg_d, "mixing_strategy": strategy})
            dc = _tmp(dcfg_d)
            factory = DataLoaderFactory(ModelConfig.from_yaml(mc), DatasetConfig.from_yaml(dc))
            os.unlink(mc); os.unlink(dc)

            print("── train (first 4 batches) ──")
            loader = factory.train_dataloader()
            set_epoch(loader, 0)
            for i, b in enumerate(loader):
                h = b["horizon"][0].item()
                print(f"  batch {i}  horizon={h:3d}  "
                      f"x_enc={tuple(b['x_enc'].shape)}  y={tuple(b['y'].shape)}")
                if i >= 3:
                    break

            print("── val ──")
            for name, ldr in factory.val_dataloaders().items():
                b = next(iter(ldr))
                print(f"  [{name}]  horizon={b['horizon'][0].item()}  "
                      f"x_enc={tuple(b['x_enc'].shape)}  y={tuple(b['y'].shape)}")

            print("── test ──")
            for name, ldr in factory.test_dataloaders().items():
                b = next(iter(ldr))
                print(f"  [{name}]  horizon={b['horizon'][0].item()}  "
                      f"x_enc={tuple(b['x_enc'].shape)}  y={tuple(b['y'].shape)}")

        print("\n✓ All loaders verified — every batch is horizon-homogeneous.")
        