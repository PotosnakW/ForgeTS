"""
# Adapted from https://github.com/Nixtla/datasetsforecast/blob/main/datasetsforecast/losses.py
"""

import numpy as np


def mae(
    preds:   np.ndarray,               # [B, T, H, C]
    targets: np.ndarray,               # [B, T, H, C]
    mask:    np.ndarray | None = None, # [B, T, H, C]
) -> float:
    err = np.abs(preds - targets)
    if mask is not None:
        return (err * mask).sum() / max(mask.sum(), 1)
    return err.mean()


def mse(
    preds:   np.ndarray,               # [B, T, H, C]
    targets: np.ndarray,               # [B, T, H, C]
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
    preds:   np.ndarray,               # [B, T, H, C]
    targets: np.ndarray,               # [B, T, H, C]
    mask:    np.ndarray | None = None,
    eps:     float = 1e-8,
) -> float:
    err = np.abs((targets - preds) / (np.abs(targets) + eps))
    if mask is not None:
        return (err * mask).sum() / max(mask.sum(), 1)
    return err.mean()


def smape(
    preds:   np.ndarray,               # [B, T, H, C]
    targets: np.ndarray,               # [B, T, H, C]
    mask:    np.ndarray | None = None,
    eps:     float = 1e-8,
) -> float:
    err = 2 * np.abs(targets - preds) / (np.abs(targets) + np.abs(preds) + eps)
    if mask is not None:
        return (err * mask).sum() / max(mask.sum(), 1)
    return err.mean()


def quantile_loss(
    preds:     np.ndarray,             # [B, T, H, C, Q]
    targets:   np.ndarray,             # [B, T, H, C]
    quantiles: list[float],
    mask:      np.ndarray | None = None,  # [B, T, H, C]
) -> float:
    """
    quantile_loss() returns the *mean pinball loss* across quantiles:
    QL = (1 / Q) * Σ_q L_q(y, ŷ_q)

    where L_q is the pinball loss:

        L_q(y, ŷ) = (y - ŷ) * q         if y >= ŷ
                  = (ŷ - y) * (1 - q)    if y <  ŷ
    """
        
    errors = targets[..., np.newaxis] - preds          # [B, T, H, C, Q]
    q      = np.array(quantiles, dtype=preds.dtype)    # [Q]
    loss   = np.maximum(q * errors, (q - 1) * errors)  # [B, T, H, C, Q]

    if mask is not None:
        mask = np.broadcast_to(mask[..., np.newaxis], loss.shape)
        return (loss * mask).sum() / max(mask.sum(), 1)
    return loss.mean()

def quantile_loss(
    preds, 
    targets, 
    quantiles, 
    mask=None,
    aggregate='mean',
) -> float:
    """
    quantile_loss() returns the *mean pinball loss* across quantiles:
    QL = (1 / Q) * Σ_q L_q(y, ŷ_q)

    where L_q is the pinball loss:

        L_q(y, ŷ) = (y - ŷ) * q         if y >= ŷ
                  = (ŷ - y) * (1 - q)    if y <  ŷ
    """
    errors = targets[..., np.newaxis] - preds
    q      = np.array(quantiles, dtype=preds.dtype)
    loss   = np.maximum(q * errors, (q - 1) * errors)

    if mask is not None:
        mask = np.broadcast_to(mask[..., np.newaxis], loss.shape)
        loss = loss * mask
    
    if aggregate == 'mean':
        return loss.sum() / max(mask.sum(), 1) if mask is not None else loss.mean()
    
    return loss  # aggregate=None, return per-element


def coverage(
    preds:     np.ndarray,             # [B, T, H, C, Q]
    targets:   np.ndarray,             # [B, T, H, C]
    quantiles: list[float],
    lo_idx:    int = 0,
    hi_idx:    int = -1,
    mask:      np.ndarray | None = None,
) -> float:
    lo   = preds[..., lo_idx]                                          # [B, T, H, C]
    hi   = preds[..., hi_idx]                                          # [B, T, H, C]
    hits = ((targets >= lo) & (targets <= hi)).astype(np.float32)

    if mask is not None:
        return (hits * mask).sum() / max(mask.sum(), 1)
    return hits.mean()


def crps(
    preds:     np.ndarray,             # [B, T, H, C, Q]
    targets:   np.ndarray,             # [B, T, H, C]
    quantiles: list[float],
    mask:      np.ndarray | None = None,
    scale:     bool = False,
    eps:       float = 1e-8,
) -> float:
    errors = targets[..., np.newaxis] - preds              # [B, T, H, C, Q]
    q      = np.array(quantiles, dtype=preds.dtype)
    loss   = 2 * np.maximum(q * errors, (q - 1) * errors) # [B, T, H, C, Q]

    if mask is not None:
        mask_q = np.broadcast_to(mask[..., np.newaxis], loss.shape)
        score  = (loss * mask_q).sum() / max(mask_q.sum(), 1)
    else:
        score = loss.mean()

    if scale:
        denom = ((np.abs(targets) * mask).sum() / max(mask.sum(), 1)
                 if mask is not None else np.abs(targets).mean())
        score = score / (denom + eps)

    return score
