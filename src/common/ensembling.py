import numpy as np

class Ensembler:
    def __init__(self, ensemble_method='mean', **kwargs):
        self.kwargs = kwargs
        methods = {
            'mean': self._cumulative_mean,
            'median': self._cumulative_median,
            'ewm': self._cumulative_ewm,
        }
        if ensemble_method not in methods:
            raise ValueError(f"ensemble_method must be one of {list(methods.keys())}")
        self.ensembler = methods[ensemble_method]

    def ensemble(self, preds, mask=None):
        """
        preds: (B, T, H, C)
        mask:  (B, T, H, C) int — 1=valid, 0=mask out
        returns: (B, T, H, C)
        """
        B, T, H, C = preds.shape
        masked_preds  = self.preds_reshape_for_ensembling(preds, mask) # (B, S, H, C)

        # reshape to (B*S, H, C) so ensemblers stay unchanged (axis=1 is overlaps)
        B_, test_size, H_, C_ = masked_preds.shape
        flat = masked_preds.reshape(B_ * test_size, H_, C_) # (B*S, H, C)
        ensembled = self.ensembler(flat, **self.kwargs) # (B*S, H, C)
        ensembled_preds = ensembled.reshape(B_, test_size, H_, C_) # (B, test_size, H, C)

        return self.ensembled_preds_reshape_for_windows(ensembled_preds) # (B, T, H, C)

    def preds_reshape_for_ensembling(self, preds, mask=None):
        B, T, H, C = preds.shape

        # Set masked predictions to nan
        if mask is not None:
            preds = np.where(mask == 0, np.nan, preds.copy()) # (B, T, H, C)

        flatten_preds = preds.reshape(B, T * H, C) # (B, T*H, C)

        # index grid — same for all batches
        idx = np.arange(T * H).reshape(T, H)
        flipped_idx = np.fliplr(idx)
        zs = np.full((H - 1, H), np.nan)
        padded_idx = np.concatenate([zs, flipped_idx, zs]) # (T + 2*(H-1), H)

        idx_windows = np.lib.stride_tricks.sliding_window_view(padded_idx, window_shape=(H, H))
        date_idx = np.diagonal(idx_windows[:, 0], axis1=1, axis2=2) # (test_size, H)

        nan_mask = ~np.isnan(date_idx) # (S, H)
        idx2 = np.where(np.isnan(date_idx), 0, date_idx).astype(int) # (S, H)

        indexed_preds = flatten_preds[:, idx2, :] # (B, S, H, C)
        nan_mask_exp = nan_mask[None, :, :, None] # (1, S, H, 1)
        return np.where(~nan_mask_exp, np.nan, indexed_preds) # (B, S, H, C)

    def ensembled_preds_reshape_for_windows(self, ensembled_preds):
        B, S, H, C = ensembled_preds.shape

        # flip H axis so diagonals align — same as before but over batch
        flipped = np.flip(ensembled_preds, axis=2) # (B, S, H, C)

        windows = np.lib.stride_tricks.sliding_window_view(flipped, window_shape=(H, H), axis=(1, 2))
        return np.diagonal(windows[:, :, 0], axis1=2, axis2=3) # (B, n_windows, H, C)

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
        return ensemble # (B, H, C)

    def _cumulative_median(self, preds, window_size=None, **kwargs):
        _, H, _ = preds.shape
        return np.stack([
            np.nanmedian(preds[:, max(0, h-window_size+1) if window_size else 0:h+1, :], axis=1)
            for h in range(H)
        ], axis=1) # (B, H, C)

    def _cumulative_ewm(self, preds, alpha=0.5, window_size=None, **kwargs):
        B, H, C = preds.shape
        beta = np.log(alpha / (1 - alpha))
        W = window_size if window_size is not None else H

        h_idx = np.arange(H)
        i_idx = np.arange(H)
        starts = np.maximum(0, h_idx - W + 1)

        mask = (i_idx[None, :] >= starts[:, None]) & (i_idx[None, :] <= h_idx[:, None])
        weight_matrix = np.where(mask, np.exp(beta * (i_idx[None, :] - starts[:, None])), 0.0)

        valid = ~np.isnan(preds)
        preds_clean = np.nan_to_num(preds)
        weighted_sum = np.einsum('hi,bic->bhc', weight_matrix, valid * preds_clean)
        w_sum = np.einsum('hi,bic->bhc', weight_matrix, valid.astype(float))

        return np.where(w_sum > 0, weighted_sum / w_sum, np.nan) # (B, H, C)
