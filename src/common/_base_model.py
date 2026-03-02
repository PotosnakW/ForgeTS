"""
base_model.py
─────────────
Plain-PyTorch training infrastructure for multivariate time-series models.

Components
──────────
    BaseModel        — abstract nn.Module; subclasses implement forward(batch)
    Trainer          — step-based loop with val_check_interval, early stopping,
                       top-k checkpointing, and optional DDP support
    LossRegistry     — named loss functions (mse, mae, huber, smape)
    EarlyStopper     — patience counter on validation metric
    CheckpointManager— save / restore top-k checkpoints by monitored metric

Batch contract (from ts_dataloader)
────────────────────────────────────
    x_enc  : [B, context_length, C, 1 + n_hist]   encoder input
    x_futr : [B, context_length + horizon, C, Vf]  future-known covariates
    x_stat : [B, C, Vs]                            static covariates
    y      : [B, horizon, C]                        forecast target
    horizon: [B]                                    scalar (same for all in batch)

Model output contract
─────────────────────
    BaseModel.forward(batch) → Tensor[B, horizon, C]      (point forecast)
                             → Tensor[B, horizon, C*c_out] (probabilistic)

    The default compute_loss compares pred[..., :C] against y using MSE.
    Override compute_loss for custom loss functions or probabilistic heads.

Quickstart
──────────
    from base_model import BaseModel, Trainer
    from ts_dataloader import ModelConfig, DatasetConfig, DataLoaderFactory

    class MyModel(BaseModel):
        def forward(self, batch):
            x = batch["x_enc"][..., 0]          # [B, ctx, C] — target only
            x = x.permute(0, 2, 1)              # [B, C, ctx]
            return self.encoder(x)              # [B, horizon, C]

    mcfg    = ModelConfig.from_yaml("model_config.yaml")
    dcfg    = DatasetConfig.from_yaml("dataset_config.yaml")
    factory = DataLoaderFactory(mcfg, dcfg)

    model   = MyModel(config)
    trainer = Trainer(
        model        = model,
        mcfg         = mcfg,
        train_loader = factory.train_dataloader(),
        val_loaders  = factory.val_dataloaders(),
        normalisers  = factory.normalisers,       # optional — for denorm metrics
    )
    trainer.fit()

    # inference
    preds = trainer.predict(factory.test_dataloaders()["electricity_test"])
"""

from __future__ import annotations

import heapq
import logging
import math
import os
import time
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

import common.losses as losses

logger = logging.getLogger(__name__)


_LOSSES: Dict[str, Callable[[Tensor, Tensor], Tensor]] = {
    "mse":  losses._mse,
    "mae":  losses._mae,
    "quantile_loss": losses._quantile_loss,
}

def get_loss_fn(name: str) -> Callable[[Tensor, Tensor], Tensor]:
    name = name.lower()
    if name not in _LOSSES:
        raise ValueError(f"Unknown loss '{name}'. Choose from {list(_LOSSES)}.")
    return _LOSSES[name]


class BaseModel(nn.module):
    """
    Step-based training loop for BaseModel subclasses.

    Training is counted in optimiser steps, not epochs.  Validation runs every
    `mcfg.val_check_interval` steps.  Early stopping patience counts validation
    *checks*, not steps.

    Parameters
    ----------
    model        : BaseModel subclass (already on the target device)
    mcfg         : ModelConfig (from model_config.yaml)
    train_loader : DataLoader from DataLoaderFactory.train_dataloader()
    val_loaders  : dict[name → DataLoader] from DataLoaderFactory.val_dataloaders()
    optimizer    : torch.optim.Optimizer (default: AdamW from mcfg.learning_rate)
    scheduler    : optional LR scheduler with .step() called after each opt step
    loss_name    : name of the loss function (default "mse"; see LossRegistry)
    normalisers  : optional dict[dataset_name → Normaliser] for original-scale metrics
    device       : torch.device (default: cuda if available)
    dtype        : torch.dtype (default: float32)
    grad_scaler  : optional torch.cuda.amp.GradScaler for mixed-precision training
    """

    def __init__(
        self,
        mcfg,                            # ModelConfig — avoid circular import
        train_loader: DataLoader,
        val_loaders:  Dict[str, DataLoader],
        optimizer:    Optional[torch.optim.Optimizer] = None,
        scheduler:    Optional[object] = None,
        loss_name:    str = "mse",
        normalisers:  Optional[dict] = None,
        device:       Optional[torch.device] = None,
        dtype:        torch.dtype = torch.float32,
        grad_scaler:  Optional[torch.cuda.amp.GradScaler] = None,
    ):
        self.mcfg         = mcfg
        self.train_loader = train_loader
        self.val_loaders  = val_loaders
        self.normalisers  = normalisers or {}
        self.dtype        = dtype
        self.scaler       = grad_scaler

        # Device
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = self.to(self.device)

        # Optimizer
        self.optimizer = optimizer or torch.optim.AdamW(
            self.model.parameters(),
            lr=mcfg.learning_rate,
            weight_decay=1e-2,
        )

        # LR scheduler (optional)
        self.scheduler = scheduler

        # Loss
        self.loss_fn = get_loss_fn(loss_name)

        # Early stopping — patience in number of val checks
        self.early_stopper = EarlyStopper(
            patience=mcfg.early_stopping_patience,
            mode=mcfg.monitor_mode,
        )

        # Checkpoint manager
        self.ckpt_manager = CheckpointManager(
            checkpoint_dir=mcfg.checkpoint_dir,
            save_top_k=mcfg.save_top_k,
            mode=mcfg.monitor_mode,
        )

        # State
        self.global_step   = 0
        self.best_val_metrics: Dict[str, float] = {}

    # ── core steps ───────────────────────────────────────────────

    def _to_device(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Move all tensor values in the batch dict to device + dtype."""
        return {
            k: v.to(device=self.device, dtype=self.dtype)
               if isinstance(v, Tensor) else v
            for k, v in batch.items()
        }

    def train_step(self, batch: Dict[str, Tensor]) -> float:
        """
        Single training step.

        1. Forward pass  → pred [B, horizon, C]
        2. compute_loss  → scalar loss
        3. Backward + gradient clip + optimizer step

        Returns
        -------
        float : detached loss value for logging
        """
        self.model.train()
        batch = self._to_device(batch)

        self.optimizer.zero_grad(set_to_none=True)

        if self.scaler is not None:
            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                pred = self.model(batch)
                loss = self.model.compute_loss(pred, batch, self.loss_fn)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.mcfg.gradient_clip_val
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            pred = self.model(batch)
            loss = self.model.compute_loss(pred, batch, self.loss_fn)
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.mcfg.gradient_clip_val
            )
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return loss.item()

    @torch.no_grad()
    def val_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        """
        Single validation step.

        Returns a dict of metrics computed on this batch:
            loss      : normalised-space loss (same loss_fn as training)
            mae       : normalised MAE
            mae_orig  : original-scale MAE (only if a matching Normaliser is found)
            mse_orig  : original-scale MSE (only if a matching Normaliser is found)
        """
        self.model.eval()
        batch = self._to_device(batch)

        pred = self.model(batch)          # [B, h, C]
        y    = batch["y"]                 # [B, h, C]
        C    = y.shape[-1]
        pred_y = pred[..., :C]

        metrics: Dict[str, float] = {
            "loss": self.model.compute_loss(pred, batch, self.loss_fn).item(),
            "mae":  _mae(pred_y, y).item(),
        }

        # Optional original-scale metrics via the dataset's Normaliser
        # batch may contain a "dataset_name" key if you inject it; otherwise skip
        ds_name = getattr(batch.get("dataset_name"), "__iter__", None) and batch.get("dataset_name")
        if ds_name and ds_name[0] in self.normalisers:
            norm = self.normalisers[ds_name[0]]
            pred_orig = norm.inverse_transform_y(pred_y.cpu())
            y_orig    = norm.inverse_transform_y(y.cpu())
            metrics["mae_orig"] = _mae(pred_orig, y_orig).item()
            metrics["mse_orig"] = _mse(pred_orig, y_orig).item()

        return metrics

    @torch.no_grad()
    def predict_step(self, batch: Dict[str, Tensor]) -> Tensor:
        """
        Single inference step.

        Returns
        -------
        Tensor[B, horizon, C] on CPU in float32.
        Call factory.normalisers[name].inverse_transform_y(pred)
        to convert back to original scale.
        """
        self.model.eval()
        batch = self._to_device(batch)
        pred  = self.model(batch)
        return pred.cpu().float()

    # ── aggregate loops ──────────────────────────────────────────

    @torch.no_grad()
    def validate(self) -> Dict[str, Dict[str, float]]:
        """
        Run val_step over every validation loader.

        Returns
        -------
        dict[loader_name → averaged metric dict]
        """
        self.model.eval()
        results: Dict[str, Dict[str, float]] = {}

        for loader_name, loader in self.val_loaders.items():
            accumulator: Dict[str, float] = {}
            n_batches = 0
            for batch in loader:
                step_metrics = self.val_step(batch)
                for k, v in step_metrics.items():
                    accumulator[k] = accumulator.get(k, 0.0) + v
                n_batches += 1

            if n_batches == 0:
                continue
            results[loader_name] = {k: v / n_batches for k, v in accumulator.items()}

        return results

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> Tensor:
        """
        Run predict_step over a full DataLoader.

        Returns
        -------
        Tensor[N_total, horizon, C] concatenated across all batches, on CPU.
        """
        self.model.eval()
        preds = []
        for batch in loader:
            preds.append(self.predict_step(batch))
        return torch.cat(preds, dim=0)

    # ── training loop ────────────────────────────────────────────

    def fit(self) -> Dict[str, Dict[str, float]]:
        """
        Step-based training loop.

        Loop structure
        ──────────────
            while global_step < max_steps:
                for batch in train_loader:           ← epoch loop (auto-repeating)
                    loss = train_step(batch)
                    global_step += 1

                    if global_step % val_check_interval == 0:
                        val_metrics = validate()
                        monitor_metric = val_metrics[primary_loader][monitor_metric]
                        checkpoint if improved
                        early_stop if patience exhausted

        Returns
        -------
        dict[loader_name → final val metric dict]
        """
        logger.info(
            f"Training for {self.mcfg.max_steps:,} steps  |  "
            f"val every {self.mcfg.val_check_interval} steps  |  "
            f"patience {self.mcfg.early_stopping_patience} checks"
        )

        primary_loader = next(iter(self.val_loaders)) if self.val_loaders else None
        final_metrics: Dict[str, Dict[str, float]] = {}
        train_iter = _InfiniteLoader(self.train_loader)
        t0 = time.time()

        while self.global_step < self.mcfg.max_steps:
            batch     = next(train_iter)
            train_loss = self.train_step(batch)
            self.global_step += 1

            # ── logging ──────────────────────────────────────────
            if self.global_step % max(1, self.mcfg.val_check_interval // 10) == 0:
                elapsed = time.time() - t0
                steps_per_sec = self.global_step / elapsed
                logger.info(
                    f"step {self.global_step:>8,} / {self.mcfg.max_steps:,}  "
                    f"train_loss={train_loss:.4f}  "
                    f"({steps_per_sec:.1f} steps/s)"
                )

            # ── validation ───────────────────────────────────────
            if self.global_step % self.mcfg.val_check_interval == 0:
                val_metrics = self.validate()
                final_metrics = val_metrics
                self._log_val_metrics(val_metrics)

                # Extract monitored metric from primary loader
                if primary_loader and primary_loader in val_metrics:
                    monitor_val = val_metrics[primary_loader].get(
                        self.mcfg.monitor_metric, float("nan")
                    )
                    if math.isnan(monitor_val):
                        warnings.warn(
                            f"monitor_metric '{self.mcfg.monitor_metric}' not found in "
                            f"val metrics for '{primary_loader}'. "
                            f"Available: {list(val_metrics[primary_loader])}."
                        )
                    else:
                        # Checkpoint
                        self.ckpt_manager.save(
                            step=self.global_step,
                            metric=monitor_val,
                            model=self.model,
                            optimizer=self.optimizer,
                            extra={"val_metrics": val_metrics},
                        )

                        # Early stopping
                        if self.early_stopper.step(monitor_val):
                            logger.info(
                                f"Early stopping triggered at step {self.global_step}. "
                                f"Best {self.mcfg.monitor_metric}="
                                f"{self.early_stopper.best:.4f}"
                            )
                            break

        # ── restore best weights ─────────────────────────────────
        best_path = self.ckpt_manager.best_checkpoint()
        if best_path is not None:
            CheckpointManager.load(best_path, self.model, self.optimizer)
            logger.info(f"Restored best checkpoint: {best_path.name}")

        return final_metrics

    def _log_val_metrics(self, val_metrics: Dict[str, Dict[str, float]]):
        for loader_name, metrics in val_metrics.items():
            parts = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            logger.info(
                f"  [val/{loader_name}]  step {self.global_step:,}  {parts}"
            )

    # ── state persistence ─────────────────────────────────────────

    def save_state(self, path: str | Path):
        """Save full trainer state (model, optimizer, step, early stopper)."""
        torch.save({
            "global_step":   self.global_step,
            "model":         self.model.state_dict(),
            "optimizer":     self.optimizer.state_dict(),
            "early_stopper": self.early_stopper.state_dict(),
            "scheduler":     self.scheduler.state_dict() if self.scheduler else None,
        }, path)
        logger.info(f"Trainer state saved → {path}")

    def load_state(self, path: str | Path):
        """Resume training from a saved trainer state."""
        ckpt = torch.load(path, map_location=self.device)
        self.global_step = ckpt["global_step"]
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.early_stopper.load_state_dict(ckpt["early_stopper"])
        if self.scheduler is not None and ckpt.get("scheduler"):
            self.scheduler.load_state_dict(ckpt["scheduler"])
        logger.info(f"Trainer state loaded ← {path}  (step={self.global_step})")

    @abstractmethod
    def forward(self, batch: Dict[str, Tensor]) -> Tensor:
        """
        Parameters
        ----------
        batch : dict as described in the class docstring

        Returns
        -------
        Tensor[B, horizon, C]  or  Tensor[B, horizon, C * c_out]
        """
        ...

    def compute_loss(
        self,
        pred: Tensor,
        batch: Dict[str, Tensor],
        loss_fn: Callable[[Tensor, Tensor], Tensor],
    ) -> Tensor:
        """
        Default loss: compare pred[..., :C] against batch['y'].

        Override this method for:
          • Probabilistic heads (NLL, CRPS, quantile loss)
          • Auxiliary losses (e.g. reconstruction, contrastive)
          • Masked / weighted losses per channel or timestep
        """
        y = batch["y"]                   # [B, h, C]
        C = y.shape[-1]
        return loss_fn(pred[..., :C], y)


# ─────────────────────────────────────────────────────────────────────────────
# Infinite DataLoader wrapper
# ─────────────────────────────────────────────────────────────────────────────

class _InfiniteLoader:
    """
    Wraps a DataLoader to cycle indefinitely, calling set_epoch on
    the batch_sampler (if it exists) each time the loader is exhausted.
    This handles HorizonBatchSampler and DistributedSampler correctly.
    """

    def __init__(self, loader: DataLoader):
        self.loader = loader
        self._iter  = iter(loader)
        self._epoch = 0

    def __next__(self):
        try:
            return next(self._iter)
        except StopIteration:
            self._epoch += 1
            self._reset()
            return next(self._iter)

    def _reset(self):
        sampler = getattr(self.loader, "batch_sampler", None) or getattr(self.loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(self._epoch)
        self._iter = iter(self.loader)


