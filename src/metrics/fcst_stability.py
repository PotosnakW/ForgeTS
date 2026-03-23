import numpy as np
from metrics.eval_losses import quantile_loss


def _reshape_windows_by_date(preds, mask=None):
        """
        Rearranges overlapping forecast windows so that each row contains
        all window predictions for the same target date.
        """
        B, T, H, C = preds.shape
    
        if mask is not None:
            preds = np.where(mask == 0, np.nan, preds.copy())

        flatten_preds = preds.reshape(B, T * H, C)
        idx  = np.arange(T * H).reshape(T, H)
        flipped_idx  = np.fliplr(idx)
        zs = np.full((H - 1, H), np.nan)
        padded_idx = np.concatenate([zs, flipped_idx, zs])
        idx_windows = np.lib.stride_tricks.sliding_window_view(padded_idx, window_shape=(H, H))
        date_idx = np.diagonal(idx_windows[:, 0], axis1=1, axis2=2)
        nan_mask = ~np.isnan(date_idx)
        idx2 = np.where(np.isnan(date_idx), 0, date_idx).astype(int)
        indexed_preds = flatten_preds[:, idx2, :]
        nan_mask_exp = nan_mask[None, :, :, None]
    
        return np.where(~nan_mask_exp, np.nan, indexed_preds)        


def excess_volatility(y, preds, quantiles, scaling=True, mask=None):
    """
    Excess Volatility (EV) — measures harmful forecast instability by comparing
    the cost of a revision against the accuracy improvement it produced.

        EV = QL(ŷ_update_median, ŷ_before)        # revision cost
           - (QL(y, ŷ_before) - QL(y, ŷ_update))  # accuracy improvement

    For each overlapping window pair (ŷ_before, ŷ_update) predicting the same
    target date:
    - revision_cost   = QL(ŷ_update_median, ŷ_before)  how much ŷ_before mispredicts ŷ_update
    - accuracy_before = QL(y, ŷ_before)                 error of older forecast vs truth
    - accuracy_update = QL(y, ŷ_update)                 error of newer forecast vs truth

    Parameters
    ----------
    y : np.ndarray [B, T, H, C]
        Ground truth targets.
    preds : np.ndarray [B, T, H, C, Q]
        Quantile predictions across N forecast windows.
    quantiles : list[float]
        Quantile levels, e.g. [0.1, 0.5, 0.9]. Must contain 0.5 for median extraction.
    scaling : bool
        If True, normalises EV by sum(|y|) to make it scale-independent.
    mask : np.ndarray [B, T, H, C] or None
        1 = real timestep, 0 = padded / missing.

    Returns
    -------
    float
    """
    B, T, H, C, Q = preds.shape

    reshaped_preds = _reshape_windows_by_date(
        preds.reshape(B, T, H, C*Q), 
        mask=np.repeat(mask, Q, axis=-1) if mask is not None else None
    )
    reshaped_y = _reshape_windows_by_date(y)
    reshaped_mask = _reshape_windows_by_date(mask) if mask is not None else None

    y_hat_before = reshaped_preds[:, :, :-1, :] # [B, T, H-1, C*Q]
    y_hat_update = reshaped_preds[:, :,  1:, :] # [B, T, H-1, C*Q]
    reshaped_y = reshaped_y[:, :, :-1, :] # [B, T, H-1, C]
    
    reshaped_mask_before = reshaped_mask[:, :, :-1, :] if mask is not None else None # [B, T, H-1, C] 
    reshaped_mask_update = reshaped_mask[:, :, 1:, :] if mask is not None else None # [B, T, H-1, C] 

    # check if changing to zeros here is good.. it might introduce inf values
    y_hat_before = np.nan_to_num(y_hat_before, nan=0.0).reshape(B, T+H-1, H-1, C, Q)  
    y_hat_update = np.nan_to_num(y_hat_update, nan=0.0).reshape(B, T+H-1, H-1, C, Q)  
    reshaped_y = np.nan_to_num(reshaped_y, nan=0.0)
    reshaped_mask = np.nan_to_num(reshaped_mask, nan=0.0) if mask is not None else None
    reshaped_mask_before = np.nan_to_num(reshaped_mask_before, nan=0.0) if mask is not None else None
    reshaped_mask_update = np.nan_to_num(reshaped_mask_update, nan=0.0) if mask is not None else None
    pair_mask = (
        np.logical_and(
                reshaped_mask[:, :, :-1, :], 
                reshaped_mask[:, :, 1:, :]
            ).astype(float)
        if mask is not None else None
    )
    
    mid = quantiles.index(0.5)
    if len(quantiles)>1: 
        y_hat_update_mid = y_hat_update[..., mid]
    else:
        y_hat_update_mid = y_hat_update.squeeze(-1)

    revision_cost = quantile_loss(
        preds=y_hat_before, 
        targets=y_hat_update_mid, 
        quantiles=quantiles, 
        mask=pair_mask,
    )
    accuracy_before = quantile_loss(
        preds=y_hat_before, 
        targets=reshaped_y, 
        quantiles=quantiles, 
        mask=pair_mask,
    )
    accuracy_update = quantile_loss(
        preds=y_hat_update, 
        targets=reshaped_y, 
        quantiles=quantiles, 
        mask=pair_mask,
    )

    EV = revision_cost - (accuracy_before - accuracy_update)

    if scaling:
        denom = (np.sum(np.abs(reshaped_y) * reshaped_mask_before) + 1e-8
             if reshaped_mask is not None
             else np.sum(np.abs(reshaped_y)) + 1e-8)
        EV /= denom

    return EV


def forecast_percentage_change(preds, scaling=True, mask=None):
    """
    Symmetric Forecast Percentage Change (sFPC) — measures the relative magnitude
    of forecast revisions across consecutive FCDs.

    For each overlapping window pair (ŷ_before, ŷ_update) predicting the same
    target date, computes:

        sQPC = 200 * mean( |ŷ_update - ŷ_before| / (|ŷ_update| + |ŷ_before| + ε) )

    When scaling=False, the denominator is dropped and this reduces to mean absolute
    revision scaled by 200. Higher values indicate greater forecast instability.

    Parameters
    ----------
    preds : np.ndarray [B, T, H, C]
        Predicted values across T forecast windows.
    scaling : bool
        If True, normalises by symmetric denominator (sFPC).
        If False, returns raw mean absolute revision.
    mask : np.ndarray [B, T, H, C] or None
        1 = real timestep, 0 = padded / missing. Masked positions are
        excluded from the mean via nan propagation.

    Returns
    -------
    float
    """
    reshaped_preds = _reshape_windows_by_date(preds=preds, mask=mask)

    y_hat_before = reshaped_preds[:, :, :-1, :]   # [B, S, H-1, C]
    y_hat_update = reshaped_preds[:, :,  1:, :]   # [B, S, H-1, C]

    num = np.abs(y_hat_update - y_hat_before)
    den = np.abs(y_hat_update) + np.abs(y_hat_before) + 1e-8
    val = num / den if scaling else num

    return 200 * np.nanmean(val)