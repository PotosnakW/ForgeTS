from __future__ import annotations
from typing import Dict, Tuple
import torch
from torch import Tensor


def _gather_block(
    src:          Tensor,   # [B, T_s, C, *extra]
    window_start: Tensor,   # [B]
    block_len:    int,
    T_s:          int,
) -> Tensor:
    """Gather a contiguous block of `block_len` steps from each series."""
    B = src.shape[0]
    extra = src.shape[2:]
    offsets = torch.arange(block_len, device=src.device)
    grid = (window_start.unsqueeze(1) + offsets.unsqueeze(0)).clamp(0, T_s - 1)
    series_blocks = src.gather(1, grid.unsqueeze(-1).unsqueeze(-1).expand(B, block_len, *extra))
    return series_blocks

def _gather_mask(
    mask:         Tensor,   # [B, T_s]
    window_start: Tensor,   # [B]
    block_len:    int,
    T_s:          int,
) -> Tensor:
    """Gather available_mask over the same contiguous block."""
    grid = (window_start.unsqueeze(1) +
            torch.arange(block_len, device=mask.device).unsqueeze(0)
           ).clamp(0, T_s - 1)
    return mask.gather(1, grid)

def _unfold_windows(
    src:  Tensor,
    size: int,
    step: int,
) -> Tensor:
    """
    Unfold time dim into windows, moving the window axis to position 2.

    [B, block_len, C, *extra]  →  [B, T, size, C, *extra]
    [B, block_len]             →  [B, T, size]
    """
    unfolded = src.unfold(dimension=1, size=size, step=step)
    if unfolded.ndim == 3:
        return unfolded.contiguous()
    ndim  = unfolded.ndim
    order = [0, 1, ndim - 1] + list(range(2, ndim - 1))
    return unfolded.permute(*order).contiguous()

def heterogeneous_sampler(
    available_mask: Tensor,
    context_length: int,
    fcd_samples: int,
    horizon: int,
    step_size: int = 1,
) -> Tuple[Tensor, Tensor]:
    """
    Sample one anchor index per series — the last position ever needed
    (end of the last fork's horizon window).

    Returns
    -------
    anchor_idx   : [B]
    window_start : [B]   max(0, anchor_idx - (L + (fcd_samples-1)*lag + H - 1))
    """
    min_needed = context_length + (fcd_samples - 1) * step_size + horizon
    min_anchor = min_needed - 1

    sample_mask = available_mask.clone().float()
    if min_anchor > 0:
        sample_mask[:, :min_anchor] = 0.0

    no_valid = sample_mask.sum(dim=1) == 0
    if no_valid.any():
        rightmost = available_mask.float().cumsum(dim=1).argmax(dim=1)
        sample_mask[no_valid] = 0.0
        sample_mask[no_valid, rightmost[no_valid]] = 1.0

    anchor_idx   = torch.multinomial(sample_mask, num_samples=1).squeeze(1)
    window_start = (anchor_idx - (min_needed - 1)).clamp(min=0)

    return anchor_idx, window_start

def fork_sequences(
    batch:          Dict[str, Tensor],
    context_length: int,
    fcd_samples:    int,
    horizon:        int,
    step_size:      int = 1,
) -> Dict[str, Tensor]:
    """
    Reformat a full-series batch into forking-sequence model inputs.
    Dataloader delivers raw padded series — all windowing happens here.

    Input
    -----
    x_enc          : [B, T_s, C, 1+Vh]
    available_mask : [B, T_s]

    Returns
    -------
    insample_y     : [B, L+T-1, C, 1+Vh]
    outsample_y    : [B, fcd_samples, H, C]
    available_mask : [B, fcd_samples, L]
    """
    x_enc_full = batch["x_enc"]
    available_mask = batch["available_mask"]
    channel_mask = batch.get("channel_mask")
    hist_mask = batch.get("hist_mask")

    B, T_s, C, _ = x_enc_full.shape
    L, H = context_length, horizon

    if fcd_samples != -1:
        block_len = L + (fcd_samples - 1) * step_size + H
        anchor_idx, window_start = heterogeneous_sampler(
            available_mask = available_mask,
            context_length = L,
            fcd_samples = fcd_samples,
            horizon = H,
            step_size = step_size,
        )
        enc_block = _gather_block(x_enc_full,  window_start, block_len, T_s)
        mask_block = _gather_mask(available_mask, window_start, block_len, T_s)
    else:
        enc_block = x_enc_full
        mask_block = available_mask

    # Unfold into fcd_samples windows
    enc_windows = _unfold_windows(enc_block, size=L + H, step=step_size)  # [B, T, L+H, C, 1+Vh]
    enc_size = enc_block.shape[1] - H

    # Split insample / outsample
    out = dict(
        insample_y = enc_block[:, :enc_size],       # [B, L+(T-1)*step_size, C, 1+Vh]
        outsample_y = enc_windows[:, :, L:, :, 0],  # [B, T, H, C]
        available_mask = mask_block[:, :enc_size],      # [B, L+(T-1)*step_size]
    )
    if channel_mask is not None: out["channel_mask"] = channel_mask
    if hist_mask is not None: out["hist_mask"] = hist_mask
    return out