import numpy as np
from metrics.eval_losses import quantile_loss
     

def _reshape_windows_by_date(x, stride, mask=None):
    """
    Rearranges overlapping forecast windows from [B, T, H, C] into
    [B, (T-1)*stride + H, H, C], grouping predictions by their target date.

    In a rolling forecast setup, multiple windows overlap on the same target date — e.g.
    the 1-step-ahead prediction from window t and the 2-step-ahead from window t-1 both
    target date t. This function collects those predictions into a single row, enabling
    direct comparison of forecasts that share the same target.

    The output has (T-1)*stride + H rows (one per unique target date) and H columns
    (one per forecast horizon that could predict that date). Edge dates are partially
    observed and will contain NaNs for horizons that don't reach that date.

    Parameters
    ----------
    x : np.ndarray [B, T, H, C]
    stride : int
        Step size between consecutive forecast windows.
    mask : np.ndarray [B, T, H, C] or None
        If provided, masked positions (mask == 0) are set to NaN before rearranging.

    Returns
    -------
    np.ndarray [B, (T-1)*stride + H, H, C]
    """

    B, T, H, C = x.shape
    S = (T - 1) * stride + H
        
    if mask is not None:
        x = np.where(mask == 0, np.nan, x.copy())

    t_grid, h_grid = np.meshgrid(np.arange(T), np.arange(H), indexing='ij')
    d_grid = t_grid * stride + h_grid  # (T, H): target date for each (t, h) 
    out = np.full((B, S, H, C), np.nan)
    out[:, d_grid, h_grid, :] = x       # scatter (B, T, H, C) → (B, S, H, C)
    return out

def excess_volatility(targets, preds, quantiles, stride=1, scaling=True, mask=None):
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
    targets : np.ndarray [B, T, H, C]
        Ground truth targets.
    preds : np.ndarray [B, T, H, C, Q]
        Quantile predictions across T forecast windows.
    quantiles : list[float]
        Quantile levels, e.g. [0.1, 0.5, 0.9]. Must contain 0.5 for median extraction.
    stride : int
        Step size between consecutive forecast windows.
    scaling : bool
        If True, normalises EV by sum(|y|) to make it scale-independent.
    mask : np.ndarray [B, T, H, C] or None
        1 = real timestep, 0 = padded / missing.

    Returns
    -------
    float
    """

    B, T, H, C, Q = preds.shape
    S = (T-1)*stride + H

    if stride == H:
        raise ValueError(
            f"stride={stride} equals H={H}: windows are non-overlapping so each "
            "target date has exactly one forecast. Cannot measure forecast stability in this case."
        )
    elif stride > H:
        raise ValueError(
            f"stride={stride} > H={H}: some target dates will have no forecast coverage. "
             "Please review the selected stride parameter used in your experiment."
        )

    reshaped_preds = _reshape_windows_by_date(
        x=preds.reshape(B, T, H, C*Q), 
        mask=np.repeat(mask, Q, axis=-1) if mask is not None else None,
        stride=stride,
    )
    reshaped_y = _reshape_windows_by_date(x=targets, stride=stride)
    reshaped_mask = _reshape_windows_by_date(x=mask, stride=stride) if mask is not None else None

    y_hat_before = reshaped_preds[:, :, :-1, :] # [B, S, H-1, C*Q]
    y_hat_update = reshaped_preds[:, :,  1:, :] # [B, S, H-1, C*Q]
    reshaped_y = reshaped_y[:, :, :-1, :] # [B, S, H-1, C]

    y_hat_before = np.nan_to_num(y_hat_before, nan=0.0).reshape(B, S, H-1, C, Q)  
    y_hat_update = np.nan_to_num(y_hat_update, nan=0.0).reshape(B, S, H-1, C, Q)  
    reshaped_y = np.nan_to_num(reshaped_y, nan=0.0)
    pair_mask = (
        np.logical_and(
            np.nan_to_num(reshaped_mask[:, :, :-1, :], nan=0.0),
            np.nan_to_num(reshaped_mask[:, :,  1:, :], nan=0.0)
        ).astype(float)
        if mask is not None else None
    )
    
    mid = quantiles.index(0.5)
    y_hat_update_mid = y_hat_update[..., mid] if len(quantiles) > 1 else y_hat_update.squeeze(-1)

    # aggregate=None: return per-element losses so EV is computed before any aggregation
    revision_cost = quantile_loss(
        preds=y_hat_before, 
        targets=y_hat_update_mid, 
        quantiles=quantiles, 
        mask=pair_mask,
        aggregate=None,
    )
    accuracy_before = quantile_loss(
        preds=y_hat_before, 
        targets=reshaped_y, 
        quantiles=quantiles, 
        mask=pair_mask,
        aggregate=None,
    )
    accuracy_update = quantile_loss(
        preds=y_hat_update, 
        targets=reshaped_y, 
        quantiles=quantiles, 
        mask=pair_mask,
        aggregate=None,
    )

    EV = (revision_cost - (accuracy_before - accuracy_update)).sum()
    denom = np.sum(np.abs(reshaped_y) * pair_mask) + 1e-8 if mask is not None else np.sum(np.abs(reshaped_y)) + 1e-8

    return EV / denom if scaling else EV


def forecast_percentage_change(preds, stride=1, scaling=True, mask=None):
    """
    Symmetric Forecast Percentage Change (sFPC) — measures the relative magnitude
    of forecast revisions across consecutive Forecast Creation Dates (FCDs).

    For each overlapping window pair (ŷ_before, ŷ_update) predicting the same
    target date, computes:

        sFPC = 200 * mean( |ŷ_update - ŷ_before| / (|ŷ_update| + |ŷ_before| + ε) )

    When scaling=False, the denominator is dropped and this reduces to mean absolute
    revision scaled by 200. Higher values indicate greater forecast instability.

    Parameters
    ----------
    preds : np.ndarray [B, T, H, C]
        Point predictions across T forecast windows.
    stride : int
        Step size between consecutive forecast windows.
    scaling : bool
        If True, normalises by symmetric denominator (sFPC).
        If False, returns raw mean absolute revision.
        Both cases are scaled by 200.
    mask : np.ndarray [B, T, H, C] or None
        1 = real timestep, 0 = padded / missing. Masked positions are
        excluded from the mean via nan propagation.

    Returns
    -------
    float
    """

    _, _, H, _ = preds.shape

    if stride == H:
        raise ValueError(
            f"stride={stride} equals H={H}: windows are non-overlapping so each "
            "target date has exactly one forecast. Cannot measure forecast stability in this case."
        )
    elif stride > H:
        raise ValueError(
            f"stride={stride} > H={H}: some target dates will have no forecast coverage. "
             "Please review the selected stride parameter used in your experiment."
        )
    
    reshaped_preds = _reshape_windows_by_date(
        x=preds, 
        mask=mask, 
        stride=stride
    )

    y_hat_before = reshaped_preds[:, :, :-1, :]   # [B, S, H-1, C]
    y_hat_update = reshaped_preds[:, :,  1:, :]   # [B, S, H-1, C]

    num = np.abs(y_hat_update - y_hat_before)
    den = np.abs(y_hat_update) + np.abs(y_hat_before) + 1e-8
    val = num / den if scaling else num

    return 200 * np.nanmean(val)
