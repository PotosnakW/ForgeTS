"""
# Adapted from https://github.com/Nixtla/datasetsforecast/blob/main/datasetsforecast/losses.py
"""

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.loss import _Loss

from .utils import _reduce

import numpy as np


def mae(
    preds:   np.ndarray,               # [B, H, C]
    targets: np.ndarray,               # [B, H, C]
    mask:    np.ndarray | None = None, # [B, H, C]  1=real, 0=padded
) -> float:
    err = np.abs(preds - targets)
    if mask is not None:
        return (err * mask).sum() / max(mask.sum(), 1)
    return err.mean()


def mse(
    preds:   np.ndarray,
    targets: np.ndarray,
    mask:    np.ndarray | None = None,
) -> float:
    err = (preds - targets) ** 2
    if mask is not None:
        return (err * mask).sum() / max(mask.sum(), 1)
    return err.mean()


def rmse(
    preds:   np.ndarray,
    targets: np.ndarray,
    mask:    np.ndarray | None = None,
) -> float:
    return np.sqrt(mse(preds, targets, mask))


def mape(
    preds:   np.ndarray,
    targets: np.ndarray,
    mask:    np.ndarray | None = None,
    eps:     float = 1e-8,
) -> float:
    err = np.abs((targets - preds) / (np.abs(targets) + eps))
    if mask is not None:
        return (err * mask).sum() / max(mask.sum(), 1)
    return err.mean()


def smape(
    preds:   np.ndarray,
    targets: np.ndarray,
    mask:    np.ndarray | None = None,
    eps:     float = 1e-8,
) -> float:
    err = 2 * np.abs(targets - preds) / (np.abs(targets) + np.abs(preds) + eps)
    if mask is not None:
        return (err * mask).sum() / max(mask.sum(), 1)
    return err.mean()


def quantile_loss(
    preds:     np.ndarray,             # [B, H, C * Q]
    targets:   np.ndarray,             # [B, H, C]
    quantiles: list[float],
    mask:      np.ndarray | None = None,
) -> float:
    B, H, n_channels_x_Q = preds.shape
    Q          = len(quantiles)
    assert n_channels_x_Q % Q == 0
    n_channels = n_channels_x_Q // Q

    preds   = preds.reshape(B, H, n_channels, Q)   # [B, H, C, Q]
    targets = targets[..., np.newaxis]              # [B, H, C, 1]
    errors  = targets - preds                       # [B, H, C, Q]

    q    = np.array(quantiles, dtype=preds.dtype)  # [Q]
    loss = np.maximum(q * errors, (q - 1) * errors)  # [B, H, C, Q]

    if mask is not None:
        mask = np.broadcast_to(mask[..., np.newaxis], loss.shape)
        return (loss * mask).sum() / max(mask.sum(), 1)
    return loss.mean()


def coverage(
    preds:     np.ndarray,             # [B, H, C * Q]  — needs at least 2 quantiles
    targets:   np.ndarray,             # [B, H, C]
    quantiles: list[float],
    lo_idx:    int = 0,                # index of lower quantile in quantiles list
    hi_idx:    int = -1,               # index of upper quantile in quantiles list
    mask:      np.ndarray | None = None,
) -> float:
    """Fraction of targets falling within the [lo, hi] quantile interval."""
    B, H, n_channels_x_Q = preds.shape
    Q          = len(quantiles)
    n_channels = n_channels_x_Q // Q
    preds      = preds.reshape(B, H, n_channels, Q)  # [B, H, C, Q]

    lo   = preds[..., lo_idx]   # [B, H, C]
    hi   = preds[..., hi_idx]   # [B, H, C]
    hits = ((targets >= lo) & (targets <= hi)).astype(np.float32)

    if mask is not None:
        return (hits * mask).sum() / max(mask.sum(), 1)
    return hits.mean()
