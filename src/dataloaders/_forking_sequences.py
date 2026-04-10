from typing import Dict, Tuple, Optional
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
    torch.unfold only creates COMPLETE windows (does not exceed available data).

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


def n_valid_fcds(T: int, context_len: int, horizon: int, step_size: int) -> int:
    """
    How many complete FCD windows fit in a series of length T.

    A window needs (context_len + horizon) consecutive timesteps.
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
    window_size = context_len + horizon
    if T < window_size:
        return 0
    return (T - window_size) // step_size + 1


class ForkingSequences:
    """
    Reformat a full-series batch into forking-sequence model inputs.

    Parameters
    ----------
    context_len : int
    step_size      : int
    fcd_sampler    : str    'heterogeneous' (default) | 'homogeneous'

    fcd_samples != -1  (training)
        Sampler picks window_start such that the block fits within [0, S-1].
        The train series is extended by H-1 masked val rows, so the last H-1
        windows within train are now reachable. Predictions landing in the
        masked extension rows have outsample_mask=0 and contribute nothing
        to the loss.

    fcd_samples == -1  (val / test)
        The full series is consumed (ctx_rows + eval rows + H-1 extension).
        _unfold_windows produces floor((S - L - H) / step) + 1 windows.
        ctx_rows and extension rows have available_mask=0, so predictions
        landing there are excluded from the loss. Any incomplete trailing
        window is dropped — no overflow.

    Inputs  (left-padded, from collate)
    ------------------------------------
    x_enc : [B, S, C, 1+Vh]
    available_mask : [B, S, C]  0=pad/missing, 1=real

    Outputs
    -------
    insample_y     : [B, enc_size, C, 1+Vh]    enc_size = block_len - H
    outsample_y    : [B, n_fcds,   H,  C]
    outsample_mask : [B, n_fcds,   H,  C]      0 where loss should be ignored
    available_mask : [B, enc_size, C]
    """

    SAMPLERS = {"heterogeneous", "homogeneous"}

    def __init__(
        self,
        context_len: int,
        step_size:      int = 1,
        fcd_sampler:    str = "heterogeneous",
    ):
        if fcd_sampler not in self.SAMPLERS:
            raise ValueError(
                f"fcd_sampler must be one of {self.SAMPLERS}, got '{fcd_sampler}'"
            )
        self.context_len = context_len
        self.step_size      = step_size
        self.fcd_sampler    = (
            self._heterogeneous_sampler
            if fcd_sampler == "heterogeneous"
            else self._homogeneous_sampler
        )

    def _homogeneous_sampler(
        self,
        available_mask: Tensor,
        fcd_samples:    int,
        horizon:        int,
    ) -> Tuple[Tensor, int]:
        B, S, _ = available_mask.shape
        L, H    = self.context_len, horizon

        block_len = L + (fcd_samples - 1) * self.step_size + H
        max_start = S - block_len
        if max_start < 0:
            raise ValueError(
                f"Series length {S} is too short for block_len {block_len}. "
                f"Reduce fcd_samples, context_len, or horizon."
            )

        # aggregate availability across batch and channels — only sample positions
        # where real data exists for ALL series and ALL channels
        time_mask = available_mask.min(dim=2).values   # [B, S] min over channels
        batch_mask = time_mask.min(dim=0).values       # [S]    min over batch

        sample_weights = batch_mask.float().clone()
        sample_weights[max_start + 1:] = 0.0           # enforce block fits

        if sample_weights.sum() == 0:
            # fallback: no position valid for all series — just respect max_start
            sample_weights = torch.ones(S)
            sample_weights[max_start + 1:] = 0.0

        window_start = torch.multinomial(sample_weights, num_samples=1)  # [1]
        window_start = window_start.repeat(B)                            # [B]
        return window_start, block_len

    def _heterogeneous_sampler(
        self,
        available_mask: Tensor,
        fcd_samples: int,
        horizon: int,
    ) -> Tuple[Tensor, int]:
        """
        Sample one window_start per series, proportional to availability.

        Constraints on window_start[b]
        --------------------------------
        Valid positions are those where available_mask == 1 after collapsing
        channels with min. A timestep is valid only if ALL channels have real
        data there.

        window_start + block_len - 1 <= S - 1
        block_len already includes horizon, so no further subtraction needed.

        The train dataset is extended by H-1 masked rows from the val set, so
        windows whose horizons land in those rows are geometrically valid but
        contribute zero loss (available_mask=0 → outsample_mask=0).

        T >= L+H is enforced by the dataset so at least one valid position
        always exists per series.

        Returns
        -------
        window_start : [B]   per-series index sampled from [first_real[b], max_start]
        block_len    : int   total span of one fcd_samples block
        """
        B, S, C = available_mask.shape
        L, H    = self.context_len, horizon

        block_len = L + (fcd_samples - 1) * self.step_size + H
        max_start = S - block_len

        if max_start < 0:
            raise ValueError(
                f"Series length {S} is too short for context_len={L} + "
                f"fcd_samples={fcd_samples} * step_size={self.step_size} + horizon={H} "
                f"= block_len={block_len}. Reduce context_len or fcd_samples."
            )

        time_mask = available_mask.min(dim=2).values   # [B, S]
        sample_weights = time_mask.clone()
        sample_weights[:, max_start + 1:] = 0.0

        window_start = torch.multinomial(
            sample_weights, num_samples=1
        ).squeeze(1) # [B]
        return window_start, block_len

    def __call__(
        self,
        batch: Dict[str, Tensor],
        horizon: int,
        fcd_samples: int,
    ) -> Dict[str, Tensor]:
        
        x_enc_full = batch["x_enc"]
        available_mask = batch["available_mask"]
        loss_mask = batch["loss_mask"]
        hist_mask = batch.get("hist_mask")

        B, S, C, _ = x_enc_full.shape
        assert available_mask.shape == (B, S, C), (
            f"available_mask must be [B, S, C] = [{B}, {S}, {C}], "
            f"got {tuple(available_mask.shape)}"
        )
        L, H = self.context_len, horizon

        if L != -1 and fcd_samples != -1:
            # fixed context, sampled block
            window_start, block_len = self.fcd_sampler(available_mask, fcd_samples, H)
            enc_block  = _gather_block(src=x_enc_full, window_start=window_start, block_len=block_len, S=S)
            mask_block = _gather_mask( mask=available_mask, window_start=window_start, block_len=block_len, S=S)
            loss_mask_block = _gather_mask( mask=loss_mask, window_start=window_start, block_len=block_len, S=S)
            window_size = L + H
            # fcd_samples already explicit — no change needed

        elif L != -1 and fcd_samples == -1:
            # fixed context, full series
            enc_block   = x_enc_full
            mask_block  = available_mask
            loss_mask_block = loss_mask
            window_size = L + H
            fcd_samples = (S - L - H) // self.step_size + 1

        elif L == -1 and fcd_samples == -1:
            # full context, full series, L defaults to 1
            enc_block   = x_enc_full
            mask_block  = available_mask
            loss_mask_block = loss_mask
            window_size = 1 + H
            fcd_samples = (S - H) // self.step_size

        elif L == -1 and fcd_samples != -1        
            block_len = (fcd_samples - 1) * self.step_size + H
            window_start, _ = self._homogeneous_sampler(available_mask, fcd_samples, H)
            block_end = window_start[0].item() + block_len  # same for all series in batch

            enc_block = x_enc_full[:, :block_end]
            mask_block = available_mask[:, :block_end]
            loss_mask_block = loss_mask[:, :block_end]
            window_size = block_end - (fcd_samples - 1) * self.step_size
            # fcd_samples already explicit — no change needed

        else:
            raise Exception(f"fcd_sample={fcd_samples} and context_len={L} combination not recognized.")

        enc_windows  = _unfold_windows(src=enc_block,  size=window_size, step=self.step_size)
        loss_mask_windows = _unfold_windows(src=loss_mask_block, size=window_size, step=self.step_size)

        eff_L = window_size - H   # 1 when ctx==-1, L otherwise
        outsample_mask = loss_mask_windows[:, :, eff_L:, :]
        enc_size = enc_block.shape[1] - H

        out = dict(
            insample_y     = enc_block[:, :enc_size],
            outsample_y    = enc_windows[:, :, eff_L:, :, 0],
            outsample_mask = outsample_mask,
            available_mask = mask_block[:, :enc_size],
            fcd_samples    = fcd_samples
        )
        if hist_mask is not None:
            out["hist_mask"] = hist_mask
        return out