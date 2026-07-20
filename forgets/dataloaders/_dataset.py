import threading
from typing import List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset

from ._utils import _extend_with_next_split, _pivot_to_arrays, _series_arrays, _split_df, _load_df


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
        loss_mask:       np.ndarray, 
        context_len:     int,
        horizon:         int,
        channel_ids:     List[str] = None,
        metadata:        Optional[SeriesMetadata] = None,
        name:            str = "",
        is_multivariate: bool = False,
    ):
        T, C = y.shape
        min_len = context_len + horizon
        if context_len != -1:
            min_len = context_len + horizon
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
        self.loss_mask = torch.from_numpy(
            loss_mask.astype(np.float32)
            ).contiguous()
        self.channel_ids = channel_ids or [str(i) for i in range(C)]
        self.metadata = metadata
        self.context_len = context_len
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
            loss_mask       = self.loss_mask, 
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
    y, hist, futr, stat, available_mask, loss_mask, channel_ids, mcfg, horizon, name="",
    is_multivariate=False,
) -> FullSeriesDataset:
    ctx = getattr(mcfg, "context_len", -1)
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
        loss_mask = loss_mask,
        context_len = ctx,
        horizon = horizon,
        channel_ids = channel_ids,
        metadata = metadata,
        name = name,
        is_multivariate = is_multivariate,
    )


class PerSeriesDataset(Dataset):
    """
    One independent univariate series per item — the counterpart to
    FullSeriesDataset for non-multivariate sources. Each series keeps its
    own native length (no cross-series date alignment); left-padding to a
    common length within a batch happens in the collate_fn, same as it
    already does for multi-dataset batches (see _pad_left / _full_series_collate_fn).

    The context_len+horizon minimum that FullSeriesDataset enforces doesn't
    apply here: a short series is simply padded further, and ForkingSequences
    already raises a clear error if a whole batch is still too short for the
    configured context_len/fcd_samples/horizon.
    """
    def __init__(
        self,
        series: List[dict],
        context_len: int,
        horizon: int,
        name: str = "",
    ):
        self.series = series
        self.context_len = context_len
        self.horizon = horizon
        self.name = name

    def __len__(self):
        return len(self.series)

    def __getitem__(self, idx):
        s = self.series[idx]
        T = s["y"].shape[0]

        y_enc = torch.from_numpy(s["y"]).view(T, 1, 1)                 # [T, 1, 1]
        hist  = torch.from_numpy(s["hist"]).unsqueeze(1)               # [T, 1, Vh]
        x_enc = torch.cat([y_enc, hist], dim=-1) if hist.shape[-1] > 0 else y_enc
        futr  = torch.from_numpy(s["futr"]).unsqueeze(1)               # [T, 1, Vf]
        available_mask = torch.from_numpy(s["available_mask"]).unsqueeze(-1).contiguous()  # [T, 1]
        loss_mask       = torch.from_numpy(s["loss_mask"]).unsqueeze(-1).contiguous()       # [T, 1]

        out = dict(
            x_enc           = x_enc,
            x_futr          = futr,
            available_mask  = available_mask,
            loss_mask       = loss_mask,
            series_len      = torch.tensor(T,       dtype=torch.long),
            horizon         = torch.tensor(self.horizon, dtype=torch.long),
            dataset_name    = self.name,
            channel_ids     = [s["channel_id"]],
            is_multivariate = False,
        )
        if s["stat"].shape[-1] > 0:
            out["x_stat"] = torch.from_numpy(s["stat"]).unsqueeze(0)   # [1, n_stat]
        return out


def _make_per_series_dataset(series: List[dict], mcfg, horizon, name="") -> PerSeriesDataset:
    return PerSeriesDataset(
        series      = series,
        context_len = getattr(mcfg, "context_len", -1),
        horizon     = horizon,
        name        = name,
    )


class LazyDatasetCache:
    """
    LRU cache of loaded FullSeriesDatasets, shared across all LazyDataset
    wrappers. Only `max_cached` datasets are held in memory at once; the
    least-recently-used dataset is evicted when the cache is full.

    Thread-safe: a lock guards the OrderedDict so multiple DataLoader
    workers don't corrupt the cache. Loading itself happens outside the
    lock so one slow disk read doesn't block other workers.
    """
    def __init__(self, max_cached: int = 10):
        from collections import OrderedDict
        self._cache: "OrderedDict[str, FullSeriesDataset]" = OrderedDict()
        self.max_cached = max_cached
        self._lock = threading.Lock()

    def get(self, entry, mcfg) -> FullSeriesDataset:
        key = entry.name
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]

        # Load outside lock — slow disk I/O shouldn't block other workers
        ds = self._load(entry, mcfg)

        with self._lock:
            # Another worker may have loaded the same dataset while we were
            # reading; prefer the already-cached version to avoid duplicates.
            if key not in self._cache:
                self._cache[key] = ds
                if len(self._cache) > self.max_cached:
                    self._cache.popitem(last=False)  # evict LRU
            self._cache.move_to_end(key)
            return self._cache[key]

    def _load(self, entry, mcfg):
        df = _load_df(entry.path)
        train_df, val_df, _ = _split_df(df, entry.val_size, entry.test_size)
        train_df = _extend_with_next_split(train_df, val_df, entry.horizon)
        train_df["loss_mask"] = train_df["available_mask"]

        if getattr(entry, "multivariate", False):
            y, hist, futr, stat, channel_ids, available_mask, loss_mask = _pivot_to_arrays(
                train_df,
                list(entry.hist_exog_cols or []),
                list(entry.futr_exog_cols or []),
                list(entry.stat_exog_cols or []),
            )
            return _make_dataset(
                y=y, hist=hist, futr=futr, stat=stat,
                available_mask=available_mask, loss_mask=loss_mask,
                channel_ids=channel_ids, mcfg=mcfg,
                horizon=entry.horizon, name=entry.name,
                is_multivariate=True,
            )

        series = _series_arrays(
            train_df,
            list(entry.hist_exog_cols or []),
            list(entry.futr_exog_cols or []),
            list(entry.stat_exog_cols or []),
        )
        return _make_per_series_dataset(series, mcfg, entry.horizon, entry.name)

    def evict(self, name: str):
        """Manually evict a dataset by name (e.g. after an epoch)."""
        with self._lock:
            self._cache.pop(name, None)

    def clear(self):
        with self._lock:
            self._cache.clear()

    def __repr__(self):
        with self._lock:
            keys = list(self._cache.keys())
        return f"LazyDatasetCache(cached={keys}, max={self.max_cached})"


class LazyDataset(Dataset):
    """
    Thin wrapper around a dataset config entry. len=1, matching
    FullSeriesDataset's contract. Data is only loaded when __getitem__
    is first called, and is evicted from the LRU cache once max_cached
    datasets are exceeded.
    """
    def __init__(self, entry, mcfg, cache: LazyDatasetCache):
        self.entry = entry
        self.mcfg  = mcfg
        self.cache = cache

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        ds = self.cache.get(self.entry, self.mcfg)
        return ds[0]


class LazyPerSeriesDataset(Dataset):
    """
    Thin wrapper around a dataset config entry, for the non-multivariate
    (per-series) path. Unlike LazyDataset, length varies per entry (number
    of series in it) rather than being a constant 1, so the first __len__
    call forces a load — real laziness only kicks in for __getitem__ calls
    after that. The underlying LazyDatasetCache still evicts by
    max_cached_datasets across entries.
    """
    def __init__(self, entry, mcfg, cache: LazyDatasetCache):
        self.entry = entry
        self.mcfg  = mcfg
        self.cache = cache

    def __len__(self):
        return len(self.cache.get(self.entry, self.mcfg).series)

    def __getitem__(self, idx):
        ds = self.cache.get(self.entry, self.mcfg)
        return ds[idx]
