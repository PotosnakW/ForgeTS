import torch
import torch.nn as nn


def _identity(x, stats, stride, norm_type, **kwargs):
    return x


def _standardize(x, stats, stride, norm_type, **kwargs):
    mean  = stats['mean']
    stdev = stats['stdev']
    affine_weight = kwargs['affine_weight']
    affine_bias   = kwargs['affine_bias']
    eps = kwargs['eps']
    causal_fcd_stats = kwargs.get('causal_fcd_stats', None)

    if norm_type == 'norm':
        x = (x - mean) / stdev
        if affine_weight is not None:
            x = x * affine_weight + affine_bias
        return x
    
    elif norm_type == 'denorm':
        if causal_fcd_stats is not None:
            # Already gathered at FCD positions: [B, n_fcds, C, X+1]
            fcd_mean  = causal_fcd_stats['mean'][:, :, :, 0:1].unsqueeze(2)
            fcd_stdev = causal_fcd_stats['stdev'][:, :, :, 0:1].unsqueeze(2)
        else:
            T = x.shape[1]
            fcd_mean  = mean[:, -T*stride::stride, :, 0:1].unsqueeze(2)
            fcd_stdev = stdev[:, -T*stride::stride, :, 0:1].unsqueeze(2)
        if affine_weight is not None:
            x = (x - affine_bias) / (affine_weight + eps ** 2)
        return x * fcd_stdev + fcd_mean

    elif norm_type == 'norm_targets':
        if causal_fcd_stats is not None:
            # Already gathered at FCD positions: [B, n_fcds, C, X+1]
            fcd_mean  = causal_fcd_stats['mean'][:, :, :, 0].unsqueeze(2).expand_as(x)
            fcd_stdev = causal_fcd_stats['stdev'][:, :, :, 0].unsqueeze(2).expand_as(x)
        else:
            T = x.shape[1]
            fcd_mean  = mean[:, -T*stride::stride, :, 0].unsqueeze(2).expand_as(x)
            fcd_stdev = stdev[:, -T*stride::stride, :, 0].unsqueeze(2).expand_as(x)
        x = (x - fcd_mean) / fcd_stdev
        if affine_weight is not None:
            x = x * affine_weight + affine_bias
        return x

    else:
        raise NotImplementedError(f"norm_type must be 'norm', 'denorm', or 'norm_targets', got '{norm_type}'")
    

class Scaler(nn.Module):
    def __init__(self, scaler_type, stride, eps=1e-5, norm_window_size=-1):
        super().__init__()
        self.eps = eps
        self.scaler_type = scaler_type
        self.stride = stride
        self.norm_window_size = norm_window_size
    
        if scaler_type == 'revin':
            self.affine_weight = nn.Parameter(torch.ones(1))
            self.affine_bias = nn.Parameter(torch.zeros(1))
            self.scaler = _standardize
        elif scaler_type == 'standard':
            self.affine_weight = None
            self.affine_bias = None
            self.scaler = _standardize
        elif scaler_type == 'none':
            self.affine_weight = None
            self.affine_bias = None
            self.scaler = _identity
        else:
            raise ValueError(f"Unknown scaler_type '{scaler_type}', must be 'revin', 'standard', or 'none'")
    
    def _get_statistics(self, x, mask):
        """
        Compute statistics based on the norm_window_size.
        x: (B, T, C, X+1) where index 0 is target, rest are exogenous
        mask : (B, T, C)  — 1 = valid, 0 = missing/padded  (optional)
        returns dict with mean, stdev: (B, T, C, X+1) — all slices
        """
        B, T, C, X1 = x.shape

        if self.norm_window_size == -1:
            counts  = torch.cumsum(mask, dim=1).clamp(min=1)
            mean    = torch.cumsum(x * mask, dim=1) / counts
            mean_sq = torch.cumsum((x ** 2) * mask, dim=1) / counts
        else:
            window_size = min(self.norm_window_size, T)
            mask_window = mask.unfold(dimension=1, size=window_size, step=1)
            x_window = x.unfold(dimension=1, size=window_size, step=1)
            counts  = mask_window.sum(dim=2).clamp(min=1)
            mean    = (x_window * mask_window).sum(dim=2) / counts
            mean_sq = ((x_window ** 2) * mask_window).sum(dim=2) / counts

        stdev   = torch.sqrt((mean_sq - mean ** 2).clamp(min=0) + self.eps)
        return {'mean': mean, 'stdev': stdev}

    def forward(self, batch, norm_type):
        causal_stats = batch.get("causal_stats", None)
        causal_fcd_stats = batch.get("causal_fcd_stats", None)

        
        if norm_type == 'norm':
            x = batch["insample_y"].clone() # (B, T, C, X+1)
            mask = batch["available_mask"].unsqueeze(-1).expand_as(x) # (B, T, C, X+1)
            if causal_stats is not None:
                # Use pre-computed causal stats from full series
                self.stats = causal_stats
            elif self.scaler_type != 'none':
                self.stats = self._get_statistics(x, mask)
            else:
                self.stats = {}

        elif norm_type == 'denorm':
            if not hasattr(self, 'stats'):
                raise RuntimeError("denorm called before norm — stats not computed yet")
            x = batch["preds"].clone()                             # (B, T, H, C, Q)

        elif norm_type == 'norm_targets':
            if not hasattr(self, 'stats'):
                raise RuntimeError("norm_targets called before norm — stats not computed yet")
            x = batch["outsample_y"].clone()                       # (B, T, H, C)

        else:
            raise NotImplementedError(f"norm_type must be 'norm', 'denorm', or 'norm_targets', got '{norm_type}'")

        x_scaled = self.scaler(
            x=x,
            stats=self.stats,
            stride=self.stride,
            norm_type=norm_type,
            affine_weight=self.affine_weight,
            affine_bias=self.affine_bias,
            eps=self.eps,
            causal_fcd_stats=causal_fcd_stats if norm_type != 'norm' else None,
        )

        if norm_type == 'norm':
            batch["insample_y"] = x_scaled
        elif norm_type == 'denorm':
            batch["preds"] = x_scaled
        elif norm_type == 'norm_targets':
            batch["outsample_y"] = x_scaled

        return batch
