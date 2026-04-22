import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Union
import math
import yaml
import torch
import torch.nn as nn

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


class EarlyStopper:
    """
    Counts validation checks without improvement.
    Patience is in units of *checks*, not steps.

    Parameters
    ----------
    patience : int   — number of checks allowed without improvement
    mode     : str   — "min" (lower is better) or "max"
    min_delta: float — minimum change to count as improvement
    """

    def __init__(self, patience: int, mode: str = "min", min_delta: float = 0.0):
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'.")
        self.patience   = patience
        self.mode       = mode
        self.min_delta  = min_delta
        self._best      = math.inf if mode == "min" else -math.inf
        self._counter   = 0

    @property
    def best(self) -> float:
        return self._best

    def step(self, metric: float) -> bool:
        """
        Call after each validation check.
        Returns True if training should stop.
        """
        improved = (
            metric < self._best - self.min_delta
            if self.mode == "min"
            else metric > self._best + self.min_delta
        )
        if improved:
            self._best    = metric
            self._counter = 0
        else:
            self._counter += 1

        return self._counter >= self.patience

    def state_dict(self) -> dict:
        return {"best": self._best, "counter": self._counter}

    def load_state_dict(self, d: dict):
        self._best    = d["best"]
        self._counter = d["counter"]
