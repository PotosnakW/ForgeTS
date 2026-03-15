import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from dataloaders._ts_dataloader import SeriesMetadata, _pivot_to_arrays


def write_sharded_dataset(
    df:               pd.DataFrame,
    out_dir:          str,
    val_size:         int,
    test_size:        int,
    context_length:   int,
    shard_size:       int = 5_000,
    hist_exog_cols:   List[str] = [],
    futr_exog_cols:   List[str] = [],
    stat_exog_cols:   List[str] = [],
    id_col:           str = "unique_id",
    time_col:         str = "ds",
    overwrite:        bool = False,
) -> None:
    """
    Convert a long-format DataFrame into train shards + val.parquet + test.parquet.

    val_size and test_size should match the values in your data config yaml.
    The full series (all splits) is passed in — boundaries are computed here.

    Parameters
    ----------
    df               Full series (train + val + test). Must contain 'available_mask'.
    out_dir          Destination directory. Created if absent.
    val_size         Timesteps for validation  (dcfg entry.val_size).
    test_size        Timesteps for test        (dcfg entry.test_size).
    context_length   L — prepended to val/test as context head.
                     Also the right-edge overlap on each train shard so
                     boundary windows are self-contained.
    shard_size       Train timesteps per shard (not counting overlap).
                     Rule of thumb: shard_size × C × n_features × 4B < 200MB.
    overwrite        Raise if out_dir already has shards, unless True.
    """
    out_dir = Path(out_dir)
    if not overwrite and (out_dir / "metadata.json").exists():
        raise FileExistsError(
            f"'{out_dir}' already contains a sharded dataset. "
            "Pass overwrite=True to replace it."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    # _pivot_to_arrays enforces available_mask and aligns all channels
    y, hist, futr, stat, channel_ids, available_mask = _pivot_to_arrays(
        df, hist_exog_cols, futr_exog_cols, stat_exog_cols
    )
    T_total   = y.shape[0]
    train_end = T_total - val_size - test_size   # exclusive end of train
    val_end   = T_total - test_size              # exclusive end of val

    if train_end <= 0:
        raise ValueError(
            f"val_size ({val_size}) + test_size ({test_size}) = "
            f"{val_size + test_size} >= total timesteps ({T_total}). "
            "No training data left."
        )
    if train_end < context_length:
        raise ValueError(
            f"Train split ({train_end} timesteps) < context_length ({context_length}). "
            "Reduce context_length or use more training data."
        )

    # ── Train shards ──────────────────────────────────────────────────────────
    # Each shard covers [t0, min(t0 + shard_size + L, train_end)).
    # The L-row right-edge overlap means a window starting at the last row
    # of shard N's sample range has a full context window within shard N — 
    # no cross-file reads needed.
    shard_meta = []
    for shard_idx, t0 in enumerate(range(0, train_end, shard_size)):
        t1_data   = min(t0 + shard_size + context_length, train_end)
        t1_sample = min(t0 + shard_size, train_end)
        path      = out_dir / f"shard_{shard_idx:06d}.parquet"
        _write_block(
            path, t0, t1_data,
            y, hist, futr, available_mask,
            channel_ids, hist_exog_cols, futr_exog_cols,
        )
        shard_meta.append({
            "shard":     shard_idx,
            "t0":        int(t0),
            "t1":        int(t1_data),
            "t0_sample": int(t0),        # first valid window_start in this shard
            "t1_sample": int(t1_sample), # last valid window_start
            "path":      str(path),
        })

    # ── Val file ──────────────────────────────────────────────────────────────
    # Prepend last L train timesteps as context head so the first FCD window
    # has a full L steps of real history from the training period.
    val_ctx_start = max(0, train_end - context_length)
    _write_block(
        out_dir / "val.parquet",
        val_ctx_start, val_end,
        y, hist, futr, available_mask,
        channel_ids, hist_exog_cols, futr_exog_cols,
    )

    # ── Test file ─────────────────────────────────────────────────────────────
    # Prepend last L val timesteps as context head so the first FCD window
    # has a full L steps of real history from the validation period.
    test_ctx_start = max(0, val_end - context_length)
    _write_block(
        out_dir / "test.parquet",
        test_ctx_start, T_total,
        y, hist, futr, available_mask,
        channel_ids, hist_exog_cols, futr_exog_cols,
    )

    # ── Static features ───────────────────────────────────────────────────────
    if stat.shape[-1] > 0:
        stat_df = pd.DataFrame(stat, columns=stat_exog_cols)
        stat_df.insert(0, "unique_id", channel_ids)
        stat_df.to_parquet(out_dir / "static.parquet", index=False)

    # ── Metadata ──────────────────────────────────────────────────────────────
    meta = {
        "T_total":          int(T_total),
        "T_train":          int(train_end),
        "T_val":            int(val_size),
        "T_test":           int(test_size),
        "val_end":          int(val_end),
        "val_ctx_start":    int(val_ctx_start),
        "test_ctx_start":   int(test_ctx_start),
        "context_length":   int(context_length),
        "shard_size":       int(shard_size),
        "channel_ids":      channel_ids,
        "hist_exog_cols":   hist_exog_cols,
        "futr_exog_cols":   futr_exog_cols,
        "stat_exog_cols":   stat_exog_cols,
        "n_shards":         len(shard_meta),
        "shards":           shard_meta,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(
        f"Wrote {len(shard_meta)} train shards + val.parquet + test.parquet → {out_dir}\n"
        f"  T_train={train_end}  T_val={val_size}  T_test={test_size}  "
        f"C={len(channel_ids)}  shard_size={shard_size}  L_overlap={context_length}"
    )

def _write_block(
    path:           Path,
    t0:             int,
    t1:             int,
    y:              np.ndarray,   # [T, C]
    hist:           np.ndarray,   # [T, C, Vh]
    futr:           np.ndarray,   # [T, C, Vf]
    available_mask: np.ndarray,   # [T, C]
    channel_ids:    List[str],
    hist_exog_cols: List[str],
    futr_exog_cols: List[str],
) -> None:
    """Write one parquet block covering global timesteps [t0, t1)."""
    rows: Dict[str, np.ndarray] = {
        "__t__": np.arange(t0, t1, dtype=np.int32),
    }
    for ci, cid in enumerate(channel_ids):
        rows[f"y__{cid}"] = y[t0:t1, ci]
    if hist.shape[-1] > 0:
        for fi, fcol in enumerate(hist_exog_cols):
            for ci, cid in enumerate(channel_ids):
                rows[f"hist__{fcol}__{cid}"] = hist[t0:t1, ci, fi]
    if futr.shape[-1] > 0:
        for fi, fcol in enumerate(futr_exog_cols):
            for ci, cid in enumerate(channel_ids):
                rows[f"futr__{fcol}__{cid}"] = futr[t0:t1, ci, fi]
    for ci, cid in enumerate(channel_ids):
        rows[f"mask__{cid}"] = available_mask[t0:t1, ci]

    pd.DataFrame(rows).to_parquet(path, index=False)

def _load_metadata(data_dir: Path) -> dict:
    with open(data_dir / "metadata.json") as f:
        return json.load(f)

def _load_static(data_dir: Path, meta: dict) -> Optional[SeriesMetadata]:
    stat_path = data_dir / "static.parquet"
    if stat_path.exists() and meta.get("stat_exog_cols"):
        stat_df = (
            pd.read_parquet(stat_path)
            .set_index("unique_id")
            .loc[meta["channel_ids"]]
        )
        return SeriesMetadata(
            data        = stat_df[meta["stat_exog_cols"]].values.astype(np.float32),
            col_names   = meta["stat_exog_cols"],
            channel_ids = meta["channel_ids"],
        )
    return SeriesMetadata.empty(meta["channel_ids"])

def _parquet_to_tensors(
    df:          pd.DataFrame,
    channel_ids: List[str],
    hist_cols:   List[str],
    futr_cols:   List[str],
) -> Dict[str, Tensor]:
    """Convert a parquet block DataFrame into model-ready tensors."""
    C = len(channel_ids)
    T = len(df)

    y = np.stack(
        [df[f"y__{cid}"].values for cid in channel_ids], axis=-1
    ).astype(np.float32)                                           # [T, C]

    hist = (
        np.stack([
            np.stack([df[f"hist__{fc}__{cid}"].values
                      for cid in channel_ids], axis=-1)
            for fc in hist_cols
        ], axis=-1).astype(np.float32)                            # [T, C, Vh]
        if hist_cols else np.zeros((T, C, 0), dtype=np.float32)
    )

    futr = (
        np.stack([
            np.stack([df[f"futr__{fc}__{cid}"].values
                      for cid in channel_ids], axis=-1)
            for fc in futr_cols
        ], axis=-1).astype(np.float32)                            # [T, C, Vf]
        if futr_cols else np.zeros((T, C, 0), dtype=np.float32)
    )

    # available_mask: [T, C] → transpose to [C, T] to match FullSeriesDataset
    mask = np.stack(
        [df[f"mask__{cid}"].values for cid in channel_ids], axis=-1
    ).astype(np.float32)                                           # [T, C]

    y_enc = torch.from_numpy(y).unsqueeze(-1)                     # [T, C, 1]
    hist_t = torch.from_numpy(hist)
    x_enc  = (
        torch.cat([y_enc, hist_t], dim=-1)
        if hist_t.shape[-1] > 0 else y_enc
    )                                                              # [T, C, 1+Vh]

    return dict(
        x_enc          = x_enc,                                    # [T, C, 1+Vh]
        x_futr         = torch.from_numpy(futr),                   # [T, C, Vf]
        available_mask = torch.from_numpy(mask).T.contiguous(),    # [C, T]
    )

def _make_item(tensors: Dict[str, Tensor], horizon: int,
               metadata: Optional[SeriesMetadata]) -> Dict[str, Tensor]:
    """Wrap raw tensors into a dict matching FullSeriesDataset.__getitem__."""
    T = tensors["x_enc"].shape[0]
    out = dict(
        x_enc          = tensors["x_enc"],
        x_futr         = tensors["x_futr"],
        available_mask = tensors["available_mask"],
        series_len     = torch.tensor(T,       dtype=torch.long),
        horizon        = torch.tensor(horizon, dtype=torch.long),
    )
    if metadata is not None and metadata.data.shape[-1] > 0:
        out["x_stat"] = metadata.data
    return out

class ShardedTrainDataset(Dataset):
    """
    Training dataset — each rank loads its assigned shard files at startup.

    Shard assignment: files[rank::world_size] — strided for balanced time
    coverage. All assigned shards are concatenated into a single in-memory
    tensor block. heterogeneous_sampler (inside fork_sequences) picks a
    random window_start from within this block on each training step.

    The L-row right-edge overlap baked into each shard means windows near
    shard boundaries are self-contained — no cross-file reads.
    """

    def __init__(
        self,
        data_dir:       str,
        context_length: int,
        horizon:        int,
        rank:           int = 0,
        world_size:     int = 1,
    ):
        self.data_dir   = Path(data_dir)
        self.ctx        = context_length
        self.horizon    = horizon

        meta             = _load_metadata(self.data_dir)
        self.channel_ids = meta["channel_ids"]
        self.hist_cols   = meta["hist_exog_cols"]
        self.futr_cols   = meta["futr_exog_cols"]
        self.metadata    = _load_static(self.data_dir, meta)

        # Assign shards to this rank: strided for balanced time coverage
        all_shards  = meta["shards"]
        rank_shards = all_shards[rank::world_size]

        if not rank_shards:
            raise ValueError(
                f"Rank {rank} has no shards assigned "
                f"(world_size={world_size}, n_shards={len(all_shards)}). "
                "Use fewer ranks or write more shard files."
            )

        # Load and concatenate all assigned shards into one tensor block
        blocks       = []
        n_windows    = 0
        window_size  = context_length + horizon

        for sh in rank_shards:
            df      = pd.read_parquet(sh["path"])
            tensors = _parquet_to_tensors(
                df, self.channel_ids, self.hist_cols, self.futr_cols
            )
            blocks.append(tensors)
            # Count valid window_start positions in this shard's sample range
            sample_len = sh["t1_sample"] - sh["t0_sample"]
            n_windows += max(0, sample_len - window_size + 1)

        self.x_enc          = torch.cat([b["x_enc"]          for b in blocks], dim=0)
        self.x_futr         = torch.cat([b["x_futr"]         for b in blocks], dim=0)
        self.available_mask = torch.cat([b["available_mask"] for b in blocks], dim=1)
        self.T              = self.x_enc.shape[0]
        self._n_windows     = max(1, n_windows)

    def __len__(self) -> int:
        # Valid window positions across this rank's time partitions.
        # Drives how many steps HorizonBatchSampler produces per epoch.
        return self._n_windows

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        # Return the full concatenated block.
        # heterogeneous_sampler inside fork_sequences picks a random
        # valid window_start from it on each call.
        return _make_item(
            dict(
                x_enc          = self.x_enc,
                x_futr         = self.x_futr,
                available_mask = self.available_mask,
            ),
            self.horizon,
            self.metadata,
        )

class _ShardedEvalDataset(Dataset):
    """
    Base for val/test datasets — loads a single parquet file in full.
    fork_sequences with fcd_samples=-1 produces all valid FCD windows.
    """
    def __init__(self, data_dir: str, filename: str,
             context_length: int, horizon: int, name: str = ""):
        self.data_dir = Path(data_dir)
        self.ctx = context_length
        self.horizon = horizon
        self.name = name

        meta             = _load_metadata(self.data_dir)
        self.channel_ids = meta["channel_ids"]
        self.hist_cols   = meta["hist_exog_cols"]
        self.futr_cols   = meta["futr_exog_cols"]
        self.metadata    = _load_static(self.data_dir, meta)

        path    = self.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"'{path}' not found. "
                "Re-run write_sharded_dataset() to generate it."
            )
        df      = pd.read_parquet(path)
        tensors = _parquet_to_tensors(
            df, self.channel_ids, self.hist_cols, self.futr_cols
        )
        self._tensors = tensors
        self.T        = tensors["x_enc"].shape[0]

    def __len__(self) -> int:
        return 1   # whole series as one item; fork_sequences windows it

    def __getitem__(self, idx: int) -> Dict[str, object]:
        out = _make_item(self._tensors, self.horizon, self.metadata)
        out["dataset_name"] = getattr(self, "name", "unknown")
        out["channel_ids"]  = self.channel_ids
        return out

class ShardedValDataset(_ShardedEvalDataset):
    """
    Validation dataset — every rank loads the complete val.parquet.

    val.parquet includes the last L train timesteps as context head so the
    first FCD window has L steps of real history.

    Every rank evaluates the full set (no DistributedSampler). Each rank has
    a different model state because it trained on a different time partition.
    Losses are all-reduced across ranks for the global val metric.
    """
    def __init__(self, data_dir: str, context_length: int, horizon: int, name=""):
        super().__init__(data_dir, "val.parquet", context_length, horizon, name)

class ShardedTestDataset(_ShardedEvalDataset):
    """
    Test dataset — loads test.parquet on a single GPU.

    test.parquet includes the last L val timesteps as context head so the
    first FCD window has L steps of real history from the validation period.

    Always used outside of any distributed context (eval_test() enforces this).
    """
    def __init__(self, data_dir: str, context_length: int, horizon: int, name=""):
        super().__init__(data_dir, "test.parquet", context_length, horizon, name)