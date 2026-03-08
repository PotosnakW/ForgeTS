import torch
import torch.nn as nn


def mae_loss(
    preds:   torch.Tensor,           # [B, H, C]
    targets: torch.Tensor,           # [B, H, C]
    mask:    torch.Tensor | None = None,   # [B, H, C]  1=real, 0=padded
) -> torch.Tensor:
    err = torch.abs(preds - targets)
    if mask is not None:
        return (err * mask).sum() / mask.sum().clamp(min=1)
    return err.mean()

def mse_loss(
    preds:   torch.Tensor,
    targets: torch.Tensor,
    mask:    torch.Tensor | None = None,
) -> torch.Tensor:
    err = (preds - targets) ** 2
    if mask is not None:
        return (err * mask).sum() / mask.sum().clamp(min=1)
    return err.mean()

def quantile_loss(
    preds:     torch.Tensor,         # [B, H, n_channels * Q]
    targets:   torch.Tensor,         # [B, H, n_channels]
    quantiles: list[float],
    mask:      torch.Tensor | None = None,   # [B, H, n_channels]
) -> torch.Tensor:
    B, H, n_channels_x_Q = preds.shape
    Q          = len(quantiles)
    assert n_channels_x_Q % Q == 0
    n_channels = n_channels_x_Q // Q

    preds   = preds.view(B, H, n_channels, Q)    # [B, H, C, Q]
    targets = targets.unsqueeze(-1)               # [B, H, C, 1]
    errors  = targets - preds                     # [B, H, C, Q]

    q    = torch.tensor(quantiles, dtype=preds.dtype, device=preds.device)
    loss = torch.max(q * errors, (q - 1) * errors)   # [B, H, C, Q]

    if mask is not None:
        mask = mask.unsqueeze(-1).expand_as(loss)     # [B, H, C, Q]
        return (loss * mask).sum() / mask.sum().clamp(min=1)
    return loss.mean()