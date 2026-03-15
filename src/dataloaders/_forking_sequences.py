from typing import Dict, Tuple
import torch
from torch import Tensor


def _gather_block(
    src:          Tensor,   # [B, S, C, *extra]
    window_start: Tensor,   # [B]
    block_len:    int,
    S:            int,
) -> Tensor:
    """Gather a contiguous block of `block_len` steps forward from window_start."""
    B     = src.shape[0]
    extra = src.shape[2:]
    offsets = torch.arange(block_len, device=src.device)
    grid    = (window_start.unsqueeze(1) + offsets.unsqueeze(0)).clamp(0, S - 1)
    return src.gather(
        1,
        grid.unsqueeze(-1).unsqueeze(-1).expand(B, block_len, *extra)
    )

def _gather_mask(
    mask:         Tensor,   # [B, S, C]
    window_start: Tensor,   # [B]
    block_len:    int,
    S:            int,
) -> Tensor:
    B, _, C = mask.shape
    grid = (
        window_start.unsqueeze(1)
        + torch.arange(block_len, device=mask.device).unsqueeze(0)
    ).clamp(0, S - 1)                                     # [B, block_len]
    grid = grid.unsqueeze(-1).expand(B, block_len, C)     # [B, block_len, C]
    return mask.gather(1, grid)                           # [B, block_len, C]

def _unfold_windows(src: Tensor, size: int, step: int) -> Tensor:
    """
    Unfold time dim into sliding windows of `size` spaced `step` apart.

    IMPORTANT: torch.unfold only creates COMPLETE windows.
    If (T - size) % step != 0, the trailing incomplete window is silently
    dropped. This is intentional — we never predict beyond available data.

    Number of windows produced:
        n_fcds = floor((T - size) / step) + 1

    [B, T, C, *extra]  ->  [B, n_fcds, size, C, *extra]
    [B, T]             ->  [B, n_fcds, size]
    """
    unfolded = src.unfold(dimension=1, size=size, step=step)
    if unfolded.ndim == 3:                         # mask: [B, n_fcds, size]
        return unfolded.contiguous()
    ndim  = unfolded.ndim
    order = [0, 1, ndim - 1] + list(range(2, ndim - 1))
    return unfolded.permute(*order).contiguous()   # [B, n_fcds, size, C, *extra]

def n_valid_fcds(T: int, context_length: int, horizon: int, step_size: int) -> int:
    """
    How many complete FCD windows fit in a series of length T.

    A window needs (context_length + horizon) consecutive timesteps.
    torch.unfold enforces completeness automatically; this function makes
    the arithmetic explicit for planning and assertions.

    Example
    -------
    T=10, L=3, H=2, step=2
        window_size = 5
        n_fcds = floor((10 - 5) / 2) + 1 = 3
        Windows cover t=[0..4], [2..6], [4..8]
        t=[6..10] would overflow -> dropped.
    """
    window_size = context_length + horizon
    if T < window_size:
        return 0
    return (T - window_size) // step_size + 1


def heterogeneous_sampler(
    available_mask: Tensor,   # [B, S, C]
    context_length: int,
    fcd_samples:    int,
    horizon:        int,
    step_size:      int = 1,
) -> Tuple[Tensor, int]:
    """
    Sample one window_start per series for a block of fcd_samples FCD windows.
    Only called during training (fcd_samples != -1).
    Val/test passes the full series directly to _unfold_windows.

    Constraints on window_start[b]
    --------------------------------
    Valid positions are those where available_mask == 1 after collapsing
    channels with min — a timestep is valid only if ALL channels have real
    data there. This naturally skips both left-padding AND mid-series gaps.

    Upper bound (scalar, same for all series):
        window_start + block_len - 1 <= S - H - 1
        The last FCD's horizon must not extend into the last H timesteps,
        where targets don't exist in the training data.
        Rearranges to: window_start <= S - block_len - H  (= max_start)

    Sampling is via torch.multinomial, so each series gets an independent
    start index drawn proportionally to its own availability weights.

    T >= L+H is enforced by the dataset so at least one valid position
    always exists per series.

    Returns
    -------
    window_start : [B]   per-series start index sampled from [first_real[b], max_start]
    block_len    : int   total length of the block from window_start
    """
    B, S, C = available_mask.shape

    block_len = context_length + (fcd_samples - 1) * step_size + horizon
    max_start = S - block_len - horizon   # scalar upper bound, inclusive

    # Collapse [B, S, C] -> [B, S] via min:
    # a timestep is only a valid start if ALL channels have real data there.
    time_mask = available_mask.min(dim=2).values                          # [B, S]

    # Zero out positions where the block would overflow into the last H steps.
    # Positions 0..max_start are geometrically valid; beyond that targets don't exist.
    sample_weights = time_mask.clone()
    sample_weights[:, max_start + 1:] = 0.0

    # Multinomial draw: each series gets its own start index sampled proportionally
    # to available_mask — naturally skips left-padding AND mid-series missing values.
    window_start = torch.multinomial(sample_weights, num_samples=1).squeeze(1)  # [B]
    return window_start, block_len


def fork_sequences(
    batch:          Dict[str, Tensor],
    context_length: int,
    fcd_samples:    int,
    horizon:        int,
    step_size:      int = 1,
) -> Dict[str, Tensor]:
    """
    Reformat a full-series batch into forking-sequence model inputs.

    FCD count guarantee
    -------------------
    In both modes the number of FCD windows is:
        n_fcds = floor((enc_block_len - L - H) / step) + 1

    These are always COMPLETE windows — the last FCD horizon never extends
    beyond the available data (see _unfold_windows and heterogeneous_sampler).

    fcd_samples != -1  (training)
        heterogeneous_sampler picks window_start such that the block ends
        at or before S - H (the reserved evaluation tail).
        _unfold_windows produces exactly fcd_samples complete windows.

    fcd_samples == -1  (val / test)
        The full left-padded series is consumed.
        _unfold_windows produces floor((S - L - H) / step) + 1 windows.
        Any incomplete trailing window is dropped — no overflow.

    Inputs  (left-padded, from collate)
    ------------------------------------
    x_enc          : [B, S, C, 1+Vh]
    available_mask : [B, S, C]         0=pad/missing, 1=real

    Outputs
    -------
    insample_y     : [B, enc_size, C, 1+Vh]    enc_size = block_len - H
    outsample_y    : [B, n_fcds,   H,  C]
    available_mask : [B, enc_size, C]
    """
    x_enc_full     = batch["x_enc"]
    available_mask = batch["available_mask"]
    hist_mask      = batch.get("hist_mask")

    B, S, C, _ = x_enc_full.shape
    assert available_mask.shape == (B, S, C), (
        f"available_mask must be [B, S, C] = [{B}, {S}, {C}], "
        f"got {tuple(available_mask.shape)}"
    )
    L, H = context_length, horizon

    if fcd_samples != -1:
        window_start, block_len = heterogeneous_sampler(
            available_mask = available_mask,
            context_length = L,
            fcd_samples    = fcd_samples,
            horizon        = H,
            step_size      = step_size,
        )
        enc_block  = _gather_block(x_enc_full,    window_start, block_len, S)
        mask_block = _gather_mask(available_mask, window_start, block_len, S)
    else:
        # Full series: let _unfold_windows drop incomplete trailing windows.
        enc_block  = x_enc_full
        mask_block = available_mask

    # [B, block_len, C, 1+Vh] -> [B, n_fcds, L+H, C, 1+Vh]
    # torch.unfold drops any trailing incomplete window automatically.
    enc_windows  = _unfold_windows(enc_block,  size=L + H, step=step_size)
    mask_windows = _unfold_windows(mask_block, size=L + H, step=step_size)
    outsample_mask = mask_windows[:, :, L:, :]               # [B, n_fcds, H, C]
    enc_size = enc_block.shape[1] - H                        # block_len - H

    out = dict(
        insample_y     = enc_block[:, :enc_size],             # [B, enc_size, C, 1+Vh]
        outsample_y    = enc_windows[:, :, L:, :, 0],        # [B, n_fcds,   H,  C]
        outsample_mask = outsample_mask,                      # [B, n_fcds,   H,  C]
        available_mask = mask_block[:, :enc_size],            # [B, enc_size, C]
    )
    if hist_mask is not None:
        out["hist_mask"] = hist_mask
    return out
