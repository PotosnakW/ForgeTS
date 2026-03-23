import numpy as np

class Ensembler:
    def __init__(self, ensemble_method='mean', **kwargs):
        self.kwargs = kwargs
        methods = {
            'mean':     self._cumulative_mean,
            'median':   self._cumulative_median,
            'ewm':      self._cumulative_ewm,
            'identity': self._identity,
        }
        if ensemble_method not in methods:
            raise ValueError(f"ensemble_method must be one of {list(methods.keys())}")
        self.ensembler = methods[ensemble_method]

    def ensemble(self, preds, mask=None):
        """
        preds: (B, T, H, C, Q)
        mask:  (B, T, H, C)    int — 1=valid, 0=mask out
        returns: (B, T, H, C, Q)
        """
        B, T, H, C, Q = preds.shape

        # fold Q into C → (B, T, H, C*Q) so all internal logic is unchanged
        preds_flat = preds.reshape(B, T, H, C * Q)

        # expand mask to cover C*Q channels
        mask_flat = (
            np.repeat(mask, Q, axis=-1)   # (B, T, H, C*Q)
            if mask is not None else None
        )

        masked_preds = self._reshape_windows_by_date(preds_flat, mask_flat)  # (B, S, H, C*Q)

        B_, S_, H_, CQ_ = masked_preds.shape
        flat      = masked_preds.reshape(B_ * S_, H_, CQ_)                       # (B*S, H, C*Q)
        ensembled = self.ensembler(flat, **self.kwargs)                           # (B*S, H, C*Q)
        ensembled = ensembled.reshape(B_, S_, H_, CQ_)                           # (B, S, H, C*Q)

        out = self.ensembled_preds_reshape_for_windows(ensembled)                # (B, T, H, C*Q)

        return out.reshape(B, out.shape[1], H, C, Q)                             # (B, T, H, C, Q)

    def _reshape_windows_by_date(self, preds, mask=None):
        """
        Rearranges overlapping forecast windows so that each row contains
        all window predictions for the same target date.
        """

        B, T, H, C = preds.shape
        if mask is not None:
            preds = np.where(mask == 0, np.nan, preds.copy())
        flatten_preds = preds.reshape(B, T * H, C)
        idx          = np.arange(T * H).reshape(T, H)
        flipped_idx  = np.fliplr(idx)
        zs           = np.full((H - 1, H), np.nan)
        padded_idx   = np.concatenate([zs, flipped_idx, zs])
        idx_windows  = np.lib.stride_tricks.sliding_window_view(padded_idx, window_shape=(H, H))
        date_idx     = np.diagonal(idx_windows[:, 0], axis1=1, axis2=2)
        nan_mask     = ~np.isnan(date_idx)
        idx2         = np.where(np.isnan(date_idx), 0, date_idx).astype(int)
        indexed_preds = flatten_preds[:, idx2, :]
        nan_mask_exp  = nan_mask[None, :, :, None]
        return np.where(~nan_mask_exp, np.nan, indexed_preds)                    # (B, S, H, C*Q)

    def ensembled_preds_reshape_for_windows(self, ensembled_preds):
        """unchanged — operates on (B, S, H, C*Q)"""
        B, S, H, C = ensembled_preds.shape
        flipped      = np.flip(ensembled_preds, axis=2)
        windows      = np.lib.stride_tricks.sliding_window_view(flipped, window_shape=(H, H), axis=(1, 2))
        windows_diags = np.diagonal(windows[:, :, 0], axis1=3, axis2=4)         # (B, n_windows, C*Q, H)
        windows_diags = windows_diags.transpose(0, 1, 3, 2)                     # (B, n_windows, H, C*Q)
        return windows_diags

    def _cumulative_mean(self, preds, window_size=None, **kwargs):
        _, H, _ = preds.shape
        if window_size is None:
            cumsum = np.nancumsum(preds, axis=1)
            counts = np.cumsum(~np.isnan(preds), axis=1)
        else:
            cumsum = np.stack([np.nansum(preds[:, max(0, h-window_size+1):h+1, :], axis=1) for h in range(H)], axis=1)
            counts = np.stack([np.sum(~np.isnan(preds[:, max(0, h-window_size+1):h+1, :]), axis=1) for h in range(H)], axis=1)
        ensemble = cumsum / counts
        ensemble[counts == 0] = np.nan
        return ensemble

    def _cumulative_median(self, preds, window_size=None, **kwargs):
        _, H, _ = preds.shape
        return np.stack([
            np.nanmedian(preds[:, max(0, h-window_size+1) if window_size else 0:h+1, :], axis=1)
            for h in range(H)
        ], axis=1)

    def _cumulative_ewm(self, preds, alpha=0.5, window_size=None, **kwargs):
        B, H, C = preds.shape
        beta   = np.log(alpha / (1 - alpha))
        W      = window_size if window_size is not None else H
        h_idx  = np.arange(H)
        i_idx  = np.arange(H)
        starts = np.maximum(0, h_idx - W + 1)
        mask   = (i_idx[None, :] >= starts[:, None]) & (i_idx[None, :] <= h_idx[:, None])
        weight_matrix = np.where(mask, np.exp(beta * (i_idx[None, :] - starts[:, None])), 0.0)
        valid         = ~np.isnan(preds)
        preds_clean   = np.nan_to_num(preds)
        weighted_sum  = np.einsum('hi,bic->bhc', weight_matrix, valid * preds_clean)
        w_sum         = np.einsum('hi,bic->bhc', weight_matrix, valid.astype(float))
        return np.where(w_sum > 0, weighted_sum / w_sum, np.nan)

    def _identity(self, preds, **kwargs):
        return preds
