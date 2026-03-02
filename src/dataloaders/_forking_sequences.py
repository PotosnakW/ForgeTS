"""
_forking_sequences.py
─────────────────────
Transforms a full-series dataloader batch into a forking-sequence model batch.

Two modes controlled by fcd_samples
────────────────────────────────────
    fcd_samples == 1  →  standard mode: batch returned unchanged.
                         Dataloader already produced windowed [B, L, C, V].

    fcd_samples  > 0  →  forking mode: sample one contiguous block per item.
                         The FULL block [B, L+T-1, C, V] is fed to the encoder.
                         The model decodes from the last T positions internally.

Output shapes (forking mode)
─────────────────────────────
    x_enc          [B, L+T-1, C, 1+Vh]   full encoder input — model decodes last T tokens
    insample_y     [B, L+T-1, C, Vf]     future covariates aligned to full block
    outsample_y    [B*T, H, C]            H-step targets, T folded into batch
    available_mask [B, L+T-1]             1=real  0=padded/unavailable
    x_stat, channel_mask, hist_mask       unchanged pass-through
    fcd_samples    int                    T, so the model knows how many to decode

Block layout
─────────────
    window_start[b]
        │←────── L + (T-1)*stride raw timesteps ──────→│
        [all tokens encoded together              ]
                                                   │←H→│ outsample fork T-1
                              │←H→│ outsample fork 1
             │←H→│ outsample fork 0   (starts at enc position L-1)

    outsample_y[b, f] = raw series at window_start + L+f .. +L+f+H

Usage in MOMENT.forward()
─────────────────────────
    T = batch.get("fcd_samples", 0)
    x = batch["x_enc"][..., 0]             # [B, L+T-1, C]
    B, seq_len, C = x.shape

    # channel-independent encoding
    x = x.permute(0, 2, 1).reshape(B*C, seq_len)
    emb = self.encoder(x)                  # [B*C, n_patch, d_model]
    emb = emb.reshape(B, C, -1, d_model)   # [B, C, n_patch, d_model]

    if T > 0:
        fork_emb = emb[:, :, -T:, :]      # [B, C, T, d_model]
        pred = self.decoder(fork_emb)      # [B, T, H, C]
    else:
        pred = self.decoder(emb[:, :, -1:, :])  # [B, H, C]
"""

from __future__ import annotations
from typing import Dict
import torch
from torch import Tensor
from fcd_sampler import heterogeneous_sampler


def fork_sequences(
    batch:          Dict[str, Tensor],
    context_length: int,
    fcd_samples:    int,
    stride:         int = 1,
) -> Dict[str, Tensor]:
    """
    Reformat a full-series batch into a forking-sequence model batch.

    Parameters
    ----------
    batch          : from FullSeriesDataset / _full_series_collate_fn
                     Required: x_enc, x_futr, x_stat, available_mask, horizon
    context_length : minimum history before first fork point  L
    fcd_samples    : number of fork points  T.  0 = standard (passthrough).

    Returns (forking mode)
    -------
    x_enc          : [B, L+T-1, C, 1+Vh]   full encoder block
    insample_y     : [B, L+T-1, C, Vf]     future covariates over full block
    outsample_y    : [B*T, H, C]            targets — T folded into batch for loss
    available_mask : [B, L+T-1]
    x_stat         : [B, C, Vs]             unchanged
    channel_mask   : [B, C_max]             pass-through
    hist_mask      : [B, Vh_max]            pass-through
    horizon        : [B]
    anchor_idx     : [B]
    window_start   : [B]
    fcd_samples    : int
    """
    if fcd_samples == 1:
        return batch   # window-sampling mode — dataloader already sliced L

    # fcd_samples=-1: use every valid fork point — T = max over batch
    if fcd_samples == -1:
        series_len = batch.get("series_len")   # [B]
        if series_len is not None:
            T_per_item = ((series_len - L) / stride).long().clamp(min=1)
        else:
            # fallback: derive from safe_mask sum
            T_per_item = ((batch["available_mask"].sum(dim=1) - L) / stride).long().clamp(min=1)
        fcd_samples = int(T_per_item.max().item())


    x_enc_full     = batch["x_enc"]            # [B, T_s, C, 1+Vh]
    x_futr_full    = batch["x_futr"]           # [B, T_s, C, Vf]
    x_stat         = batch["x_stat"]
    available_mask = batch["available_mask"]    # [B, T_s]
    horizon_t      = batch["horizon"]           # [B]
    channel_mask   = batch.get("channel_mask")
    hist_mask      = batch.get("hist_mask")

    B, T_s, C, V = x_enc_full.shape
    Vf = x_futr_full.shape[-1]
    L, T = context_length, fcd_samples
    H = int(horizon_t[0].item())

    # Each of the T fork points shifts by `stride` raw timesteps.
    # enc_size: raw timesteps covering all T context windows.
    # target_size: enc_size plus H steps for the last fork's target.
    enc_size    = L + (T - 1) * stride          # raw encoder input length
    target_size = L + (T - 1) * stride + H      # enc_size + H for targets

    # Zero last H positions so anchor always has H target steps available
    safe_mask = available_mask.clone()
    if H > 0 and T_s > H:
        safe_mask[:, T_s - H:] = 0.0

    # ── 1. Sample anchor ─────────────────────────────────────────────────
    _, _, anchor_idx, window_start = heterogeneous_sampler(
        x              = x_enc_full,
        input_mask     = safe_mask,
        context_length = L,
        fcd_samples    = T,
    )

    # ── 2. Gather helper: extract a contiguous block from window_start ────
    def _gather(src: Tensor, size: int) -> Tensor:
        offsets  = torch.arange(size, device=src.device)
        grid     = (window_start.unsqueeze(1) + offsets.unsqueeze(0)).clamp(0, T_s - 1)
        extra    = src.shape[2:]
        idx      = grid
        for _ in extra:
            idx = idx.unsqueeze(-1)
        return src.gather(1, idx.expand(B, size, *extra))

    # ── 3. Encoder block [B, L+T-1, C, *] ───────────────────────────────
    enc_block  = _gather(x_enc_full,  enc_size)   # [B, L+T-1, C, 1+Vh]
    futr_block = _gather(x_futr_full, enc_size)   # [B, L+T-1, C, Vf]

    mask_grid  = (window_start.unsqueeze(1) +
                  torch.arange(enc_size, device=available_mask.device).unsqueeze(0)
                 ).clamp(0, T_s - 1)
    enc_mask   = available_mask.gather(1, mask_grid)   # [B, L+T-1]

    # ── 4. outsample_y via unfold on target strip ─────────────────────────
    # Only the target strip needs unfold — encoder gets the full block as-is
    target_block = _gather(x_enc_full, target_size)    # [B, L+T+H-1, C, 1+Vh]
    target_strip = target_block[:, L:, :, 0]           # [B, T+H-1, C]
    # step=stride: consecutive fork targets are stride raw timesteps apart
    outsample_y  = (
        target_strip
        .unfold(dimension=1, size=H, step=stride)       # [B, T, C, H]
        .permute(0, 1, 3, 2)                            # [B, T, H, C]
        .contiguous()
        .reshape(B * T, H, C)                           # [B*T, H, C]  — T folded into batch
    )

    # ── 5. Return ─────────────────────────────────────────────────────────
    out = dict(
        x_enc          = enc_block,       # [B, L+T-1, C, 1+Vh]
        insample_y     = futr_block,      # [B, L+T-1, C, Vf]
        outsample_y    = outsample_y,     # [B*T, H, C]
        x_stat         = x_stat,
        available_mask = enc_mask,        # [B, L+T-1]
        horizon        = horizon_t,
        anchor_idx     = anchor_idx,
        window_start   = window_start,
        fcd_samples    = T,
    )
    if channel_mask is not None: out["channel_mask"] = channel_mask
    if hist_mask    is not None: out["hist_mask"]    = hist_mask
    return out
