import torch
import torch.nn as nn


def mae_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    MAE Loss.

    preds   : [B, H, C]   B = batch_size in standard mode,
                               batch_size * fcd_samples in forking mode
    targets : [B, H, C]
    """
    return torch.mean(torch.abs(preds - targets))

def mse_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    MSE Loss.

    preds   : [B, H, C]
    targets : [B, H, C]
    """
    return torch.mean((preds - targets) ** 2)

def quantile_loss(
    preds:     torch.Tensor,
    targets:   torch.Tensor,
    quantiles: list[float],
) -> torch.Tensor:
    """
    Pinball (Quantile) Loss for multiple quantiles.

    Args:
        preds:     [B, H, n_channels * Q]  — Q quantile outputs per channel,
                   ordered as [ch0_q0, ch0_q1, ..., ch1_q0, ch1_q1, ...]
        targets:   [B, H, n_channels]      — ground truth (one value per channel)
        quantiles: list of Q quantile values, e.g. [0.1, 0.5, 0.9]

    In forking mode B = batch_size * fcd_samples — the function is unaware of
    and unaffected by the T fold; it treats every row as an independent sample.

    Returns:
        Scalar mean pinball loss.
    """
    B, H, n_channels_x_Q = preds.shape
    Q = len(quantiles)
    assert n_channels_x_Q % Q == 0, (
        f"preds last dim ({n_channels_x_Q}) must be divisible by num quantiles ({Q})"
    )
    n_channels = n_channels_x_Q // Q

    preds   = preds.view(B, H, n_channels, Q)   # [B, H, C, Q]
    targets = targets.unsqueeze(-1)              # [B, H, C, 1]
    errors  = targets - preds                   # [B, H, C, Q]

    q    = torch.tensor(quantiles, dtype=preds.dtype, device=preds.device)  # [Q]
    loss = torch.max(q * errors, (q - 1) * errors)   # [B, H, C, Q]
    return loss.mean()
