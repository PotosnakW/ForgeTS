"""
# Adapted from https://github.com/Nixtla/datasetsforecast/blob/main/datasetsforecast/losses.py
"""
import torch

class LossFunction:
    def __init__(self, fn, quantiles=None):
        self.fn = fn
        self.quantiles = quantiles
        self.outputsize_multiplier = len(quantiles) if fn is _quantile_loss else 1

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

def _mae_loss(
    preds:   torch.Tensor,
    targets: torch.Tensor,
    mask:    torch.Tensor | None = None,
) -> torch.Tensor:
    err = torch.abs(preds - targets)
    if mask is not None:
        return (err * mask).sum() / mask.sum().clamp(min=1)
    return err.mean()

def _mse_loss(
    preds:   torch.Tensor,
    targets: torch.Tensor,
    mask:    torch.Tensor | None = None,
) -> torch.Tensor:
    err = (preds - targets) ** 2
    if mask is not None:
        return (err * mask).sum() / mask.sum().clamp(min=1)
    return err.mean()

def _quantile_loss(
    preds:     torch.Tensor,
    targets:   torch.Tensor,
    quantiles: list[float],
    mask:      torch.Tensor | None = None,
) -> torch.Tensor:
    B, H, n_channels_x_Q = preds.shape
    Q          = len(quantiles)
    assert n_channels_x_Q % Q == 0
    n_channels = n_channels_x_Q // Q
    preds   = preds.view(B, H, n_channels, Q)
    targets = targets.unsqueeze(-1)
    errors  = targets - preds
    q    = torch.tensor(quantiles, dtype=preds.dtype, device=preds.device)
    loss = torch.max(q * errors, (q - 1) * errors)
    if mask is not None:
        mask = mask.unsqueeze(-1).expand_as(loss)
        return (loss * mask).sum() / mask.sum().clamp(min=1)
    return loss.mean()

def _huber_loss(
    preds:   torch.Tensor,
    targets: torch.Tensor,
    mask:    torch.Tensor | None = None,
    delta:   float = 1.0,
) -> torch.Tensor:
    err  = torch.abs(preds - targets)
    loss = torch.where(err < delta, 0.5 * err ** 2, delta * (err - 0.5 * delta))
    if mask is not None:
        return (loss * mask).sum() / mask.sum().clamp(min=1)
    return loss.mean()

def get_loss(name: str) -> LossFunction:
    if name not in LOSSES:
        raise ValueError(f"Unknown loss '{name}'. Available: {list(LOSSES.keys())}")
    return LOSSES[name]

LOSSES = {
    "mae":      LossFunction(_mae_loss),
    "mse":      LossFunction(_mse_loss),
    "huber":    LossFunction(_huber_loss),
    "quantile": LossFunction(_quantile_loss, quantiles=[0.1, 0.5, 0.9]),
}
