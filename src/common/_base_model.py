"""
_base_model.py
──────────────
BaseModel is an nn.Module that also owns the training loop.

Two-phase construction
──────────────────────
Subclasses (e.g. MOMENT) only need their own architecture args in __init__.
Training state is attached lazily via setup_training() before calling fit().

    class MOMENT(BaseModel):
        def __init__(self, config):
            super().__init__()          # just nn.Module — no training args needed
            self.encoder = Encoder(config)
            ...

    model = MOMENT(config)
    model.setup_training(mcfg, train_loader, val_loaders)
    model.fit()
"""

from __future__ import annotations

import logging
import random
import time
from abc import abstractmethod
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from ..utils.utils import EarlyStopper

logger = logging.getLogger(__name__)


class CheckpointManager:
    def __init__(self, checkpoint_dir: str, checkpoint_step: int = 1000):
        self.checkpoint_dir  = Path(checkpoint_dir)
        self.checkpoint_step = checkpoint_step
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def step(self, step: int, model: nn.Module, **extra) -> None:
        if step % self.checkpoint_step != 0:
            return
        path = self.checkpoint_dir / f"ckpt_step={step:07d}.pt"
        torch.save({"step": step, "model_state_dict": model.state_dict(), **extra}, path)
        logger.info("Checkpoint saved → %s", path)

    def load(self, path: str, model: nn.Module, map_location: str = "cpu") -> dict:
        payload = torch.load(path, map_location=map_location)
        model.load_state_dict(payload["model_state_dict"])
        logger.info("Checkpoint loaded ← %s", path)
        return payload

class _InfiniteLoader:
    def __init__(self, loader: DataLoader):
        self.loader = loader
        self._iter  = iter(loader)
        self._epoch = 0

    def __next__(self):
        try:
            return next(self._iter)
        except StopIteration:
            self._epoch += 1
            sampler = (
                getattr(self.loader, "batch_sampler", None)
                or getattr(self.loader, "sampler", None)
            )
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(self._epoch)
            self._iter = iter(self.loader)
            return next(self._iter)


# ─────────────────────────────────────────────────────────────────────────────
# BaseModel
# ─────────────────────────────────────────────────────────────────────────────

class BaseModel(nn.Module):
    """
    nn.Module base that also owns the step-based training loop.

    Subclass responsibility
    ───────────────────────
    __init__(self, ...)
        Call super().__init__() — no training args needed here.
        Build your architecture (layers, heads, etc.).

    forward(self, batch) → Tensor
        Implement the forward pass.
        `batch` is already the output of fork_sequences when batch_mode="full_series".

    compute_loss(self, pred, batch) → Tensor    [optional override]
        Default: MSE(pred[..., :C], outsample_y).
        Override for probabilistic losses, auxiliary losses, masking, etc.

    Training
    ────────
    Call setup_training(mcfg, train_loader, val_loaders, ...) before fit().
    """

    def __init__(self):
        super().__init__()
        # Training state is None until setup_training() is called.
        self._training_ready = False

    # ── must override ────────────────────────────────────────────

    @abstractmethod
    def forward(self, batch: Dict[str, Tensor]) -> Tensor:
        ...

    # ── optional override ────────────────────────────────────────

    def compute_loss(self, pred: Tensor, batch: Dict[str, Tensor]) -> Tensor:
        """
        Default point-forecast MSE loss.
        pred  : [B*T, H, c_out*C]
        y     : [B,   T,  H,  C]
        """
        y = batch["outsample_y"]        # [B, T, H, C]
        B, T, H, C = y.shape
        y    = y.reshape(B * T, H, C)
        pred = pred[..., :C]            # drop extra c_out dims for point forecast
        return self.loss_fn(pred, y)

    # ── setup ────────────────────────────────────────────────────

    def setup_training(
        self,
        mcfg,
        train_loader:   DataLoader,
        val_loaders:    Dict[str, DataLoader],
        optimizer:      Optional[torch.optim.Optimizer] = None,
        scheduler=None,
        loss_fn:        Optional[Callable[[Tensor, Tensor], Tensor]] = None,
        device:         Optional[torch.device] = None,
        seed:           int = 42,
    ) -> "BaseModel":
        """
        Attach training state to the model.  Call once before fit().
        Returns self for chaining:

            model = MOMENT(config).setup_training(mcfg, train_loader, val_loaders)
        """
        self.mcfg         = mcfg
        self.train_loader = train_loader
        self.val_loaders  = val_loaders
        self.scheduler    = scheduler
        self.seed         = seed
        self.global_step  = 0

        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.to(self.device)

        self.optimizer = optimizer or torch.optim.AdamW(
            self.parameters(),
            lr=mcfg.learning_rate,
            weight_decay=1e-2,
        )

        self.loss_fn = loss_fn or nn.functional.mse_loss

        self.early_stopper = EarlyStopper(
            patience=mcfg.early_stopping_patience,
            mode=mcfg.monitor_mode,
        )
        self.ckpt_manager = CheckpointManager(
            checkpoint_dir  = mcfg.checkpoint_dir,
            checkpoint_step = getattr(mcfg, "checkpoint_step", 1000),
        )

        self._training_ready = True
        return self

    def _assert_training_ready(self):
        if not self._training_ready:
            raise RuntimeError(
                "Call model.setup_training(mcfg, train_loader, val_loaders) "
                "before fit() / train_step() / validate()."
            )

    # ── internal helpers ─────────────────────────────────────────

    def _to_device(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return {
            k: v.to(self.device) if isinstance(v, Tensor) else v
            for k, v in batch.items()
        }

    def _prepare_batch(self, raw_batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        For full_series mode: call fork_sequences to convert raw series into
        model-ready windows.  Other modes: pass through as-is.
        """
        raw_batch = self._to_device(raw_batch)
        if self.mcfg.batch_mode == "full_series":
            from fork_sequences import fork_sequences
            return fork_sequences(
                batch          = raw_batch,
                context_length = self.mcfg.context_length,
                fcd_samples    = getattr(self.mcfg, "fcd_samples", 8),
                horizon        = int(raw_batch["horizon"][0].item()),
            )
        return raw_batch

    # ── core steps ───────────────────────────────────────────────

    def train_step(self, raw_batch: Dict[str, Tensor]) -> float:
        self._assert_training_ready()
        self.train()
        batch = self._prepare_batch(raw_batch)

        self.optimizer.zero_grad(set_to_none=True)
        pred = self(batch)
        loss = self.compute_loss(pred, batch)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.mcfg.gradient_clip_val)
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return loss.item()

    @torch.no_grad()
    def val_step(self, raw_batch: Dict[str, Tensor]) -> float:
        self._assert_training_ready()
        self.eval()
        batch = self._prepare_batch(raw_batch)
        return self.compute_loss(self(batch), batch).item()

    @torch.no_grad()
    def validate(self) -> Dict[str, Dict[str, float]]:
        """Returns {loader_name: {"loss": mean_loss}}."""
        self._assert_training_ready()
        self.eval()
        results: Dict[str, Dict[str, float]] = {}
        for name, loader in self.val_loaders.items():
            total, n = 0.0, 0
            for batch in loader:
                total += self.val_step(batch)
                n     += 1
            results[name] = {"loss": total / n if n > 0 else float("nan")}
        return results

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> Tensor:
        self.eval()
        preds = []
        for batch in loader:
            batch = self._prepare_batch(self._to_device(batch))
            preds.append(self(batch).cpu().float())
        return torch.cat(preds, dim=0)

    # ── training loop ────────────────────────────────────────────

    def fit(self) -> Dict[str, Dict[str, float]]:
        self._assert_training_ready()

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        logger.info(
            "Training — max_steps=%d  val_every=%d  patience=%d  device=%s",
            self.mcfg.max_steps,
            self.mcfg.val_check_interval,
            self.mcfg.early_stopping_patience,
            self.device,
        )

        primary    = next(iter(self.val_loaders)) if self.val_loaders else None
        train_iter = _InfiniteLoader(self.train_loader)
        final_metrics: Dict[str, Dict[str, float]] = {}
        t0 = time.time()

        while self.global_step < self.mcfg.max_steps:
            train_loss   = self.train_step(next(train_iter))
            self.global_step += 1

            self.ckpt_manager.step(
                self.global_step, self,
                optimizer=self.optimizer.state_dict(),
                loss=train_loss,
            )

            log_every = max(1, self.mcfg.val_check_interval // 10)
            if self.global_step % log_every == 0:
                sps = self.global_step / (time.time() - t0)
                logger.info(
                    "step %8d / %d  train_loss=%.4f  (%.1f steps/s)",
                    self.global_step, self.mcfg.max_steps, train_loss, sps,
                )

            if self.global_step % self.mcfg.val_check_interval == 0:
                val_metrics   = self.validate()
                final_metrics = val_metrics
                self._log_val_metrics(val_metrics)

                if primary and primary in val_metrics:
                    monitor_val = val_metrics[primary].get(
                        self.mcfg.monitor_metric,
                        val_metrics[primary].get("loss", float("nan")),
                    )
                    if self.early_stopper.step(monitor_val):
                        logger.info(
                            "Early stopping at step %d (best=%.4f)",
                            self.global_step, self.early_stopper.best,
                        )
                        break

        return final_metrics

    def _log_val_metrics(self, val_metrics: Dict[str, Dict[str, float]]):
        for name, metrics in val_metrics.items():
            parts = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            logger.info("  [val/%s] step %d  %s", name, self.global_step, parts)

    # ── persistence ──────────────────────────────────────────────

    def save_state(self, path: str | Path):
        """Save full trainer state — model weights + optimizer + step + early stopper."""
        self._assert_training_ready()
        torch.save({
            "global_step":   self.global_step,
            "model":         self.state_dict(),
            "optimizer":     self.optimizer.state_dict(),
            "early_stopper": self.early_stopper.state_dict(),
            "scheduler":     self.scheduler.state_dict() if self.scheduler else None,
        }, path)
        logger.info("Trainer state saved → %s", path)

    def load_train(self, path: str | Path):
        """Resume training — restores step, weights, optimizer, early stopper."""
        self._assert_training_ready()
        ckpt = torch.load(path, map_location=self.device)
        self.global_step = ckpt["global_step"]
        self.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.early_stopper.load_state_dict(ckpt["early_stopper"])
        if self.scheduler and ckpt.get("scheduler"):
            self.scheduler.load_state_dict(ckpt["scheduler"])
        logger.info("Trainer state loaded ← %s  (step=%d)", path, self.global_step)

    @staticmethod
    def load_weights(path: str | Path, model: nn.Module, map_location: str = "cpu") -> nn.Module:
        """Load only model weights into an already-constructed model (no training state needed)."""
        ckpt  = torch.load(path, map_location=map_location)
        state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
        model.load_state_dict(state, strict=True)
        logger.info("Model weights loaded ← %s", path)
        return model
    