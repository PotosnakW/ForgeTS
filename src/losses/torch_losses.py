import torch
import torch.nn as nn


def mae_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    MAE Loss.
    preds:   (B, H, C)
    targets: (B, H, C)
    """
    return torch.mean(torch.abs(preds - targets))

def mse_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    MSE Loss.
    preds:   (B, H, C)
    targets: (B, H, C)
    """
    return torch.mean((preds - targets) ** 2)

def quantile_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    quantiles: list[float],
) -> torch.Tensor:
    """
    Pinball (Quantile) Loss for multiple quantiles.

    Args:
        preds:     (B, H, n_channels * Q)  — Q quantile outputs per channel,
                   assumed ordered as [ch0_q0, ch0_q1, ..., ch1_q0, ch1_q1, ...]
        targets:   (B, H, n_channels)      — ground truth (one value per channel)
        quantiles: list of Q quantile values, e.g. [0.1, 0.5, 0.9]

    Returns:
        Scalar mean loss.
    """
    B, H, n_channels_x_Q = preds.shape
    Q = len(quantiles)
    assert n_channels_x_Q % Q == 0, \
        f"preds last dim ({n_channels_x_Q}) must be divisible by num quantiles ({Q})"
    n_channels = n_channels_x_Q // Q

    # (B, H, n_channels, Q)
    preds = preds.view(B, H, n_channels, Q)

    # (B, H, n_channels, 1)
    targets = targets.unsqueeze(-1)

    # errors: (B, H, n_channels, Q)
    errors = targets - preds

    q = torch.tensor(quantiles, dtype=preds.dtype, device=preds.device)  # (Q,)

    # Pinball loss
    loss = torch.max(q * errors, (q - 1) * errors)  # (B, H, n_channels, Q)

    return loss.mean()
