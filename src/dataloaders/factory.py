from collections import defaultdict
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from ._utils import (
    _to_cfg,
    _load_df,
    _split_df,
    _extend_with_next_split,
    _pivot_to_arrays,
    _pad_left,
)
from ._dataset import _make_dataset, LazyDataset, LazyDatasetCache
from ._samplers import HorizonBatchSampler


class DataLoaderFactory:
    def __init__(self, mcfg, dcfg):
        self.mcfg = mcfg
        self.dcfg = dcfg
        self._horizon_groups: Dict[tuple, list] = defaultdict(list)

        # Lazy loading cache — only max_cached datasets live in memory at once.
        # Set `max_cached_datasets` in mcfg to tune (default 10).
        # Set `lazy_loading: false` in mcfg to disable and restore eager behaviour.
        max_cached = getattr(self.mcfg, "max_cached_datasets", 10)
        self._lazy  = getattr(self.mcfg, "lazy_loading", True)
        self._dataset_cache = LazyDatasetCache(max_cached=max_cached) if self._lazy else None

        self._build_train()
        self._validate_dataset_compatibility()

        seed = getattr(self.mcfg, "seed", 42)
        self._val_rng = np.random.default_rng(seed)

    def _arrays_from_df(self, df, entry):
        return _pivot_to_arrays(
            df, 
            entry.hist_exog_cols, 
            entry.futr_exog_cols, 
            entry.stat_exog_cols
        )
    
    def _full_series_collate_fn(self, batch):
        ctx       = ctx = getattr(self.mcfg, "context_len", -1)
        patch_len = getattr(self.mcfg, "patch_len", 1)
        min_pad   = ctx - patch_len if ctx != -1 else 0

        T_max  = max(s["x_enc"].shape[0] for s in batch) + min_pad
        C_max  = max(s["x_enc"].shape[-2] for s in batch)
        Vh_max = max(s["x_enc"].shape[-1] - 1 for s in batch)

        B            = len(batch)
        channel_mask = torch.zeros(B, C_max, dtype=torch.float32)
        hist_mask    = torch.zeros(B, max(Vh_max, 1), dtype=torch.float32)

        out_x_enc     = []
        out_mask      = []
        out_loss_mask = []
        out_x_stat    = [] if any("x_stat" in s for s in batch) else None

        for i, s in enumerate(batch):
            C_i  = s["x_enc"].shape[-2]
            Vh_i = s["x_enc"].shape[-1] - 1

            x = F.pad(s["x_enc"], (0, Vh_max - Vh_i, 0, C_max - C_i))
            x = _pad_left(x, T_max)
            out_x_enc.append(x)

            m = F.pad(s["available_mask"], (0, C_max - C_i))
            m = _pad_left(m, T_max)
            out_mask.append(m)

            lm = F.pad(s["loss_mask"], (0, C_max - C_i))
            lm = _pad_left(lm, T_max)
            out_loss_mask.append(lm)

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
            loss_mask       = torch.stack(out_loss_mask),
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

    def _build_train(self):
        from dataloaders.ts_sharding import ShardedTrainDataset
        for entry in self.dcfg.train:
            entry = _to_cfg(entry)
            is_multivariate = getattr(entry, "multivariate", False)
            group_key = (entry.horizon, is_multivariate)

            if getattr(entry, "sharded_dir", None):
                ds = ShardedTrainDataset(
                    data_dir    = entry.sharded_dir,
                    context_len = getattr(mcfg, "context_len", -1),
                    horizon     = entry.horizon,
                    rank        = getattr(self, "_rank", 0),
                    world_size  = getattr(self, "_world_size", 1),
                )
                self._horizon_groups[group_key].append((ds, entry.weight, entry.name))
                continue

            if self._lazy:
                # Lazy path: store a lightweight wrapper; data loads on first __getitem__
                ds = LazyDataset(entry, self.mcfg, self._dataset_cache)
            else:
                # Eager path: load everything into memory now (original behaviour)
                df = _load_df(entry.path)
                train_df, val_df, _ = _split_df(
                    df=df, val_size=entry.val_size, test_size=entry.test_size,
                )
                train_df = _extend_with_next_split(train_df, val_df, entry.horizon)
                train_df["loss_mask"] = train_df["available_mask"]
                y, hist, futr, stat, channel_ids, available_mask, loss_mask = self._arrays_from_df(
                    train_df, entry
                )
                ds = _make_dataset(
                    y=y, hist=hist, futr=futr, stat=stat,
                    available_mask=available_mask, loss_mask=loss_mask,
                    channel_ids=channel_ids, mcfg=self.mcfg,
                    horizon=entry.horizon, name=entry.name,
                    is_multivariate=is_multivariate,
                )

            self._horizon_groups[group_key].append((ds, entry.weight, entry.name))

    def rebuild_for_rank(self, rank: int, world_size: int) -> "DataLoaderFactory":
        self._rank = rank
        self._world_size = world_size
        self._horizon_groups.clear()
        if self._dataset_cache is not None:
            self._dataset_cache.clear()
        self._build_train()
        return self

    
    def _build_eval_dataset(self, entry, split: str) -> Dataset:
        if split == "val":
            return self._build_val_dataset(entry)
        elif split == "test":
            return self._build_test_dataset(entry)
        else:
            raise ValueError(f"Unknown split '{split}'")
    
    def _build_val_dataset(self, entry) -> Dataset:
        entry = _to_cfg(entry)
        is_multivariate = getattr(entry, "multivariate", False)
        ctx = ctx = getattr(self.mcfg, "context_len", -1)

        if getattr(entry, "sharded_dir", None):
            from dataloaders.ts_sharding import ShardedValDataset
            return ShardedValDataset(
                data_dir    = entry.sharded_dir,
                context_len = ctx,
                horizon     = entry.horizon,
                name        = entry.name,
            )

        df = _load_df(entry.path)
        train_df, val_df, test_df = _split_df(
            df=df, val_size=entry.val_size, test_size=entry.test_size,
        )

        if val_df is None or len(val_df) == 0:
            raise ValueError(f"Dataset '{entry.name}': val split is empty.")

        eval_df = _extend_with_next_split(val_df, test_df, entry.horizon)
        eval_df["loss_mask"] = eval_df["available_mask"]

        if ctx == -1:
            prior_df = train_df.copy()
        else:
            prior_df = train_df.groupby("unique_id", sort=False).tail(ctx).copy()
        prior_df["loss_mask"] = 0.0

        eval_df = (
            pd.concat([prior_df, eval_df])
            .sort_values(["unique_id", "ds"])
            .reset_index(drop=True)
        )

        y, hist, futr, stat, channel_ids, available_mask, loss_mask = self._arrays_from_df(
            eval_df, entry
        )
        return _make_dataset(
            y=y, hist=hist, futr=futr, stat=stat,
            available_mask=available_mask, loss_mask=loss_mask,
            channel_ids=channel_ids, mcfg=self.mcfg,
            horizon=entry.horizon, name=entry.name,
            is_multivariate=is_multivariate,
        )

    def _build_test_dataset(self, entry) -> Dataset:
        entry = _to_cfg(entry)
        is_multivariate = getattr(entry, "multivariate", False)
        ctx = ctx = getattr(self.mcfg, "context_len", -1)

        if getattr(entry, "sharded_dir", None):
            from dataloaders.ts_sharding import ShardedTestDataset
            return ShardedTestDataset(
                data_dir    = entry.sharded_dir,
                context_len = ctx,
                horizon     = entry.horizon,
                name        = entry.name,
            )

        df = _load_df(entry.path)
        train_df, val_df, test_df = _split_df(
            df=df, val_size=entry.val_size, test_size=entry.test_size,
        )

        if test_df is None or len(test_df) == 0:
            raise ValueError(f"Dataset '{entry.name}': test split is empty.")

        eval_df = test_df.copy()
        eval_df["loss_mask"] = eval_df["available_mask"]

        prior_df = (
            pd.concat([train_df, val_df])
            .sort_values(["unique_id", "ds"])
            .reset_index(drop=True)
        )

        if ctx == -1:
            prior_df = prior_df.copy()
        else:
            prior_df = prior_df.groupby("unique_id", sort=False).tail(ctx).copy()

        prior_df["loss_mask"] = 0.0

        eval_df = (
            pd.concat([prior_df, eval_df])
            .sort_values(["unique_id", "ds"])
            .reset_index(drop=True)
        )

        y, hist, futr, stat, channel_ids, available_mask, loss_mask = self._arrays_from_df(
            eval_df, entry
        )
        return _make_dataset(
            y=y, hist=hist, futr=futr, stat=stat,
            available_mask=available_mask, loss_mask=loss_mask,
            channel_ids=channel_ids, mcfg=self.mcfg,
            horizon=entry.horizon, name=entry.name,
            is_multivariate=is_multivariate,
        )


    def _make_train_batch_sampler(self, rank=0, world_size=1):
        horizon_override = getattr(self.mcfg, "horizon_override", None)
        group_datasets   = defaultdict(list)
        group_weights    = defaultdict(list)
        global_offsets   = defaultdict(list)
        all_datasets     = []
        flat_offset      = 0

        for group_key in sorted(self._horizon_groups.keys()):
            horizon, is_multivariate = group_key
            effective_key = ("override", is_multivariate) if horizon_override else group_key
            for ds, weight, _ in self._horizon_groups[group_key]:
                global_offsets[effective_key].append(flat_offset)
                group_datasets[effective_key].append(ds)
                group_weights[effective_key].append(weight)
                all_datasets.append(ds)
                flat_offset += len(ds)   

        combined = ConcatDataset(all_datasets)
        sampler  = HorizonBatchSampler(
            group_datasets        = group_datasets,
            group_weights         = group_weights,
            global_offsets        = global_offsets,
            batch_size            = self.mcfg.batch_size,
            batch_mixing_strategy = self.mcfg.batch_mixing_strategy,
            shuffle               = True,
            drop_last             = self.mcfg.drop_last,
            seed                  = getattr(self.mcfg, "seed", 0),
            rank                  = rank,
            world_size            = world_size,
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
            collate_fn         = self._full_series_collate_fn,
            persistent_workers = self.mcfg.num_workers > 0,
        )

    def _make_eval_dataloader(self, entries, split: str) -> DataLoader:
        horizon_override = getattr(self.mcfg, "horizon_override", None)
        group_datasets: Dict[tuple, list] = defaultdict(list)
        group_weights:  Dict[tuple, list] = defaultdict(list)
        global_offsets: Dict[tuple, list] = defaultdict(list)
        all_datasets = []
        flat_offset  = 0

        for entry in entries:
            entry = _to_cfg(entry)
            ds = self._build_eval_dataset(entry, split)
            is_multivariate = getattr(entry, "multivariate", False)
            group_key = (entry.horizon, is_multivariate)
            effective_key = ("override", is_multivariate) if horizon_override else group_key

            global_offsets[effective_key].append(flat_offset)
            group_datasets[effective_key].append(ds)
            group_weights[effective_key].append(entry.weight)
            all_datasets.append(ds)
            flat_offset += len(ds)

        combined = all_datasets[0] if len(all_datasets) == 1 else ConcatDataset(all_datasets)
        sampler  = HorizonBatchSampler(
            group_datasets        = group_datasets,
            group_weights         = {g: [1.0] * len(ds_list) for g, ds_list in group_datasets.items()},
            global_offsets        = global_offsets,
            batch_size            = self.mcfg.valid_batch_size,
            batch_mixing_strategy = self.mcfg.batch_mixing_strategy,
            shuffle               = False,
            drop_last             = False,
            rank                  = 0,
            world_size            = 1,
        )
        return DataLoader(
            combined,
            batch_sampler      = sampler,
            num_workers        = self.mcfg.num_workers,
            pin_memory         = True,
            collate_fn         = self._full_series_collate_fn,
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
                collate_fn         = self._full_series_collate_fn,
                persistent_workers = self.mcfg.num_workers > 0,
            )
        return loaders
    
    def _validate_dataset_compatibility(self):
        all_horizons = set(horizon for horizon, _ in self._horizon_groups.keys())
        horizon_override = getattr(self.mcfg, "horizon_override", None)
        output_patch_len = getattr(self.mcfg, "output_patch_len", None)

        names = [entry.name for entry in self.dcfg.train]
        duplicates = {n for n in names if names.count(n) > 1}

        if len(all_horizons) > 1:
            if horizon_override:
                # fine — all datasets collapsed into one group
                pass
            elif output_patch_len:
                # fine — model rolls out K patches of output_patch_len to meet
                # each dataset's horizon dynamically in forward pass
                pass
            else:
                raise ValueError(
                    f"Multiple horizons detected across datasets: {all_horizons}. "
                    f"Either set horizon_override to fix a single horizon, "
                    f"or set output_patch_len to enable dynamic horizon rollout "
                    f"in the forward pass."
                )
            
        if duplicates:
            raise ValueError(
                f"Duplicate dataset names found in train config: {duplicates}. "
                f"Each dataset must have a unique name."
            )
