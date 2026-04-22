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
import torch.distributed as dist
from tqdm import tqdm

from ._utils import CheckpointManager, EarlyStopper
from ..dataloaders._forking_sequences import ForkingSequences
from ..metrics.torch_losses import get_loss
from ..scalers.torch_scalers import Scaler

logger = logging.getLogger(__name__)


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


class BaseModel(nn.Module):
    """
    nn.Module base that also owns the step-based training loop.

    Subclass responsibilities
    ─────────────────────────
    __init__(self, ...)
        Call super().__init__(). Build architecture only — no training args.

    forward(self, batch) -> Tensor
        Implement the forward pass. batch is the output of forking-sequences.

    compute_loss(self, pred, batch) -> Tensor   [optional]
        Default: MSE against outsample_y. Override for custom losses.
    """

    def __init__(self, config):
        super().__init__()
        self._training_ready = False
        self._rank = 0
        self._world_size = 1

        self.scaler = Scaler(
            scaler_type=config.scaler_type,
            stride=config.stride,
            eps=1e-5
        )
        
        self.fcd_samples = config.fcd_samples
        self._fork_sequences_train = ForkingSequences(
            context_len = config.context_len,
            fcd_samples = config.fcd_samples,
            stride = 1,
            fcd_sampler = config.fcd_sampler,
        )
        self._fork_sequences_eval = ForkingSequences(
            context_len = config.context_len,
            fcd_samples = -1,
            stride = 1,
        )

        loss_fn = get_loss(config.loss)
        if hasattr(config, "quantiles") and config.quantiles:
            loss_fn.quantiles = list(config.quantiles)
            loss_fn.outputsize_multiplier = len(config.quantiles)
        self.loss_fn = loss_fn

    @abstractmethod
    def forward(self, batch: Dict[str, Tensor]) -> Tensor:
        ...

    def compute_loss(self, batch: Dict[str, Tensor]) -> Tensor:
        """
        pred           : [B, T, H, C]
        outsample_y    : [B, T, H, C]
        outsample_mask : [B, T, H, C]  1=real timestep+channel, 0=padded/missing
        """
        y = batch["outsample_y"]
        preds = batch["preds"]
        B, T, H, C = y.shape

        y = y.reshape(B * T, H, C)
        preds = preds.reshape(B * T, H, C, -1)

        outsample_mask = batch.get("outsample_mask")
        if outsample_mask is not None:
            mask = outsample_mask.reshape(B * T, H, C).float()
        else:
            mask = None

        return self.loss_fn(
            preds=preds, 
            targets=y, 
            mask=mask,
        )

    def setup_training(
        self,
        mcfg,
        train_loader:   DataLoader,
        val_loaders:    Dict[str, DataLoader],
        optimizer:      Optional[torch.optim.Optimizer] = None,
        scheduler       = None,
        loss_fn:        Optional[Callable[[Tensor, Tensor], Tensor]] = None,
        device:         Optional[torch.device] = None,
        seed:           int = 42,
    ) -> "BaseModel":
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
            self.parameters(), lr=mcfg.learning_rate, weight_decay=1e-2
        )

        self.early_stopper = EarlyStopper(
            patience = mcfg.early_stopping_patience,
            mode = "min",
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

    # ── batch preparation ────────────────────────────────────────────────────

    def _to_device(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return {
            k: v.to(self.device) if isinstance(v, Tensor) else v
            for k, v in batch.items()
        }


    def _prepare_batch(self, raw_batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        raw_batch = self._to_device(raw_batch)
        horizon_override = getattr(self.mcfg, "horizon_override", None)
        horizon = int(horizon_override) if horizon_override else int(raw_batch["horizon"][0].item())
        if self.training:
            fcd_samples = self._get_fcd_samples()
            return self._fork_sequences_train(raw_batch, horizon, fcd_samples=fcd_samples)
        return self._fork_sequences_eval(raw_batch, horizon)


    def _get_fcd_samples(self):
        self._assert_training_ready()
        return self.fcd_samples # @ WP TODO: curriculum learning


    def train_step(self, raw_batch: Dict[str, Tensor]) -> float:
        self._assert_training_ready()
        self.train()
        batch = self._prepare_batch(raw_batch)

        self.optimizer.zero_grad(set_to_none=True)
        fwd  = self.__dict__.get('_ddp_model', self)

        batch = self.scaler(batch, norm_type='norm')
        preds = fwd(batch)
        batch["preds"] = preds
        batch = self.scaler(batch, norm_type='denorm')

        loss = self.compute_loss(batch=batch)
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

        batch = self.scaler(batch, norm_type='norm')
        preds = self(batch)
        batch["preds"] = preds
        batch = self.scaler(batch, norm_type='denorm')
    
        return self.compute_loss(batch=batch).item()

    @torch.no_grad()
    def validate(self) -> Dict[str, Dict[str, float]]:
        self._assert_training_ready()
        self.eval()
        results: Dict[str, Dict[str, float]] = {}
        for name, loader in self.val_loaders.items():
            total, n = 0.0, 0
            for raw_batch in loader:
                total += self.val_step(raw_batch)
                n += 1
            results[name] = {"loss": total / n if n > 0 else float("nan")}
        return results

    @torch.no_grad()
    def predict_step(self, raw_batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Single batch inference. Returns pred, targets, outsample_mask for that batch."""
        self.eval()
        raw_batch = self._to_device(raw_batch)
        batch     = self._prepare_batch(raw_batch)

        batch = self.scaler(batch, norm_type='norm')
        preds = self(batch)
        batch["preds"] = preds
        batch = self.scaler(batch, norm_type='denorm')

        preds = batch["preds"].cpu().float()       # [B, n_fcds, H, C_out]
        targets = batch["outsample_y"].cpu()      # [B, n_fcds, H, C]
        outsample_mask = batch.get("outsample_mask")
        if outsample_mask is not None:
            outsample_mask = outsample_mask.cpu()

        return dict(preds=preds, targets=targets, outsample_mask=outsample_mask)

    @torch.no_grad()
    def predict(self, loader, device=None):
        self.eval()
        if device is not None:
            self.to(device)

        results = {}
        for raw_batch in tqdm(loader, desc="Predicting"):
            step           = self.predict_step(raw_batch)
            pred           = step["preds"]
            targets        = step["targets"]
            outsample_mask = step["outsample_mask"]
            dataset_names  = raw_batch.get("dataset_name", ["unknown"] * pred.shape[0])
            channel_ids    = raw_batch.get("channel_ids",  [None]      * pred.shape[0])

            for b in range(pred.shape[0]):
                name   = dataset_names[b]
                n_real = len(channel_ids[b]) if channel_ids[b] is not None else pred.shape[3]
                if name not in results:
                    results[name] = {"channel_ids": channel_ids[b], "preds": [], "targets": [], "outsample_mask": []}
                results[name]["preds"].append(pred[b, :, :, :n_real, :])
                results[name]["targets"].append(targets[b, :, :, :n_real])
                if outsample_mask is not None:
                    results[name]["outsample_mask"].append(outsample_mask[b, :, :, :n_real])

        for name, d in results.items():
            d["preds"]          = torch.cat(d["preds"],          dim=0)
            d["targets"]        = torch.cat(d["targets"],        dim=0)
            d["outsample_mask"] = torch.cat(d["outsample_mask"], dim=0) if d["outsample_mask"] else None

        return results

    def _log_val_metrics(self, val_metrics: Dict[str, Dict[str, float]]):
        for name, metrics in val_metrics.items():
            parts = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            logger.info("  [val/%s] step %d  %s", name, self.global_step, parts)

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

        pbar = tqdm(total=self.mcfg.max_steps, initial=self.global_step, desc="Training")
        while self.global_step < self.mcfg.max_steps:
            train_loss = self.train_step(next(train_iter))
            self.global_step += 1

            if self._world_size > 1:
                t = torch.tensor(train_loss, device=self.device)
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
                train_loss = (t / self._world_size).item()

            self.ckpt_manager.step(
                self.global_step, self,
                optimizer = self.optimizer.state_dict(),
                loss      = train_loss,
            )

            if self.global_step % self.mcfg.val_check_interval == 0:
                val_metrics = self.validate()
                final_metrics = val_metrics
                self._log_val_metrics(val_metrics)

                if primary and primary in val_metrics:
                    monitor_val = val_metrics[primary].get("loss", float("nan"))
                    pbar.set_postfix({
                        "train": f"{train_loss:.4f}",
                        "val":   f"{monitor_val:.4f}",
                    })
                    if self.early_stopper.step(monitor_val):
                        logger.info(
                            "Early stopping at step %d (best=%.4f)",
                            self.global_step, self.early_stopper.best,
                        )
                        break
            else:
                pbar.set_postfix({"train": f"{train_loss:.4f}"})

            pbar.update(1)

        pbar.close()
        return final_metrics

    def save_state(self, path: str | Path):
        self._assert_training_ready()
        torch.save({
            "global_step":   self.global_step,
            "model":         self.state_dict(),
            "optimizer":     self.optimizer.state_dict(),
            "early_stopper": (
                self.early_stopper.state_dict()
                if hasattr(self.early_stopper, "state_dict") else
                vars(self.early_stopper)   # fallback: save __dict__ directly
            ),
            "scheduler":     self.scheduler.state_dict() if self.scheduler else None,
        }, path)
        logger.info("Trainer state saved → %s", path)

    def load_train(self, path: str | Path):
        self._assert_training_ready()
        ckpt = torch.load(path, map_location=self.device)
        self.global_step = ckpt["global_step"]
        self.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("early_stopper"):
            if hasattr(self.early_stopper, "load_state_dict"):
                self.early_stopper.load_state_dict(ckpt["early_stopper"])
            else:
                self.early_stopper.__dict__.update(ckpt["early_stopper"])
        if self.scheduler and ckpt.get("scheduler"):
            self.scheduler.load_state_dict(ckpt["scheduler"])
        logger.info("Trainer state loaded ← %s  (step=%d)", path, self.global_step)

    @staticmethod
    def load_weights(
        path: str | Path,
        model: nn.Module,
        map_location: str = "cpu",
    ) -> nn.Module:
        """Load only model weights — no training state needed."""
        ckpt  = torch.load(path, map_location=map_location)
        state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
        model.load_state_dict(state, strict=True)
        logger.info("Model weights loaded ← %s", path)
        return model

    def setup_inference(
        self,
        mcfg,
        device: Optional[torch.device] = None,
    ) -> "BaseModel":
        self.mcfg   = mcfg
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.to(self.device)
        return self
