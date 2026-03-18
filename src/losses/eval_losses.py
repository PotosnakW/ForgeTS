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
    """
    quantile_loss() returns the *mean pinball loss* across quantiles:
    QL = (1 / Q) * Σ_q L_q(y, ŷ_q)

    where L_q is the pinball loss:

        L_q(y, ŷ) = (y - ŷ) * q         if y >= ŷ
                  = (ŷ - y) * (1 - q)    if y <  ŷ
    """

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

def crps(
    preds:     np.ndarray,             # [B, H, C * Q]
    targets:   np.ndarray,             # [B, H, C]
    quantiles: list[float],
    mask:      np.ndarray | None = None,
    scale:     bool = False,           # if True, divide by mean |target| (scaled CRPS)
    eps:       float = 1e-8,
) -> float:
    """
    CRPS approximated via quantile scores.

    Exact CRPS integrates over all quantile levels:

        CRPS(F, y) = ∫₀¹ 2 * L_q(y, F⁻¹(q)) dq

    where L_q is the pinball loss:

        L_q(y, ŷ) = (y - ŷ) * q         if y >= ŷ
                  = (ŷ - y) * (1 - q)    if y <  ŷ

    Approximated over a discrete set of Q quantiles:

        CRPS ≈ (2 / Q) * Σ_q L_q(y, ŷ_q)

    When scale=True, normalises by mean |target|:

        sCRPS = CRPS / E[|y|]

    CRPS is exactly 2 * QL. The factor of 2 is the standard convention that
    makes the discrete approximation consistent with the continuous integral
    definition above — without it, CRPS would be systematically halved
    relative to published benchmarks.

    Use quantile_loss() as a *training objective* (scale doesn't matter there).
    Use CRPS as an *evaluation metric* when you want a proper scoring rule.
    """
    B, H, n_channels_x_Q = preds.shape
    Q          = len(quantiles)
    assert n_channels_x_Q % Q == 0
    n_channels = n_channels_x_Q // Q

    preds   = preds.reshape(B, H, n_channels, Q)   # [B, H, C, Q]
    targets = targets[..., np.newaxis]              # [B, H, C, 1]
    errors  = targets - preds                       # [B, H, C, Q]

    q    = np.array(quantiles, dtype=preds.dtype)  # [Q]
    loss = 2 * np.maximum(q * errors, (q - 1) * errors)  # [B, H, C, Q]

    if mask is not None:
        mask_q = np.broadcast_to(mask[..., np.newaxis], loss.shape)
        score = (loss * mask_q).sum() / max(mask_q.sum(), 1)
    else:
        score = loss.mean()

    if scale:
        t = targets[..., 0]  # [B, H, C]
        if mask is not None:
            denom = (np.abs(t) * mask).sum() / max(mask.sum(), 1)
        else:
            denom = np.abs(t).mean()
        score = score / (denom + eps)

    return score
