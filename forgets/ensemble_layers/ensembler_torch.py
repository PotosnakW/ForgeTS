import torch


class Ensembler:
    def __init__(self, config):
        self.stride = config.stride
        methods = {
            'mean':     self._cumulative_mean,
            'median':   self._cumulative_median,
            'ewm':      self._cumulative_ewm,
            'identity': self._identity,
        }
        if config.ensemble_method not in methods:
            raise ValueError(f"ensemble_method must be one of {list(methods.keys())}")
        self.ensembler = methods[config.ensemble_method]
        self.ensemble_window_size = getattr(config, 'ensemble_window_size', config.h)
        self.alpha = getattr(config, 'alpha', 0.9)

    def ensemble(self, preds, mask=None):
        """
        preds: (B, T, H, C, Q) torch.Tensor
        mask:  (B, T, H, C)    torch.Tensor, int/bool — 1=valid, 0=mask out
        returns: (B, T, H, C, Q)
        """
        B, T, H, C, Q = preds.shape
        stride = self.stride

        if stride == H:
            raise ValueError(
                f"stride={stride} equals H={H}: windows are non-overlapping so each "
                "target date has exactly one forecast. Ensembling has no effect — "
                "use ensemble_method='identity' instead."
            )
        elif stride > H:
            raise ValueError(
                f"stride={stride} > H={H}: some target dates will have no forecast coverage. "
                "Please review the selected stride parameter used in your experiment."
            )

        # ... fold Q into C, reshape into overlapping windows, dispatch to
        # self.ensembler, then unfold back to (B, T, H, C, Q) ...

    # --- ensembling strategies (torch versions) ---

    def _cumulative_mean(self, preds, **kwargs):
        cumsum = torch.nan_to_num(preds).cumsum(dim=1)
        counts = (~torch.isnan(preds)).cumsum(dim=1)
        ensemble = cumsum / counts
        ensemble[counts == 0] = float('nan')
        return ensemble

    def _cumulative_median(self, preds, **kwargs):
        _, H, _ = preds.shape
        return torch.stack([
            torch.nanmedian(preds[:, :h+1, :], dim=1).values for h in range(H)
        ], dim=1)

    def _cumulative_ewm(self, preds, **kwargs):
        B, H, C = preds.shape
        alpha = self.alpha
        beta = torch.log(torch.tensor(alpha / (1 - alpha)))
        dtype, device = preds.dtype, preds.device
        W = self.ensemble_window_size

        h_idx = torch.arange(H, device=device)
        i_idx = torch.arange(H, device=device)
        starts = torch.clamp(h_idx - W + 1, min=0)
        mask = (i_idx[None, :] >= starts[:, None]) & (i_idx[None, :] <= h_idx[:, None])

        weight_matrix = torch.where(
            mask,
            torch.exp(beta * (i_idx[None, :] - starts[:, None]).to(dtype)),
            torch.zeros((), dtype=dtype, device=device)
        )
        valid        = ~torch.isnan(preds)
        preds_clean  = torch.nan_to_num(preds)
        weighted_sum = torch.einsum('hi,bic->bhc', weight_matrix, valid.to(dtype) * preds_clean)
        w_sum        = torch.einsum('hi,bic->bhc', weight_matrix, valid.to(dtype))
        return torch.where(w_sum > 0, weighted_sum / w_sum, torch.tensor(float('nan'), dtype=dtype, device=device))

    def _identity(self, preds, **kwargs):
        return preds
