import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch

from statsmodels.tsa.arima.model import ARIMA
from chronos import Chronos2Pipeline
import timesfm


def run_experiment_batched(series, W, H, predict_batch_fn):
    """
    predict_batch_fn(ctx_matrix: (n, W), h: int) -> (n, H) forecasts.
    Returns None if the series is too short to produce any windows.
    """
    starts_idx = np.arange(W, len(series) - H)
    n = len(starts_idx)
    if n == 0:
        return None
 
    ctx_mat = np.stack([series[idx - W: idx] for idx in starts_idx])  # (n, W)
    fut_mat = np.stack([series[idx: idx + H] for idx in starts_idx])  # (n, H)
 
    # lag-aware ("leaky") norm: stats from the context window itself
    mu_lag = ctx_mat.mean(axis=1)
    sig_lag = np.clip(ctx_mat.std(axis=1), 1e-6, None)
 
    # causal norm: stats from everything up to idx, computed via cumsum (O(n))
    cumsum = np.cumsum(series)
    cumsq = np.cumsum(series ** 2)
    counts = starts_idx.astype(np.float64)          # len(series[:idx]) == idx
    sum_prefix = cumsum[starts_idx - 1]
    sq_prefix = cumsq[starts_idx - 1]
    mu_caus = sum_prefix / counts
    var_caus = np.clip(sq_prefix / counts - mu_caus ** 2, 0, None)
    sig_caus = np.clip(np.sqrt(var_caus), 1e-6, None)
 
    ctx_norm_lag = (ctx_mat - mu_lag[:, None]) / sig_lag[:, None]
    ctx_norm_caus = (ctx_mat - mu_caus[:, None]) / sig_caus[:, None]
 
    preds_lag = predict_batch_fn(ctx_norm_lag, H)
    preds_caus = predict_batch_fn(ctx_norm_caus, H)
 
    fut_norm_lag = (fut_mat - mu_lag[:, None]) / sig_lag[:, None]
    fut_norm_caus = (fut_mat - mu_caus[:, None]) / sig_caus[:, None]
 
    mae_lag = np.abs(preds_lag - fut_norm_lag).mean(axis=1)
    mae_caus = np.abs(preds_caus - fut_norm_caus).mean(axis=1)
 
    return {
        "origin_idx": starts_idx,
        "mu_lag": mu_lag, "sig_lag": sig_lag,
        "mu_caus": mu_caus, "sig_caus": sig_caus,
        "pred_lag_norm": preds_lag,      # (n_windows, H)
        "pred_caus_norm": preds_caus,    # (n_windows, H)
        "future_raw": fut_mat,           # (n_windows, H), original scale
        "mae_lag": mae_lag,
        "mae_caus": mae_caus,
    }


# -------------------------------------------------------
# ------------------------ ARIMA ------------------------
# -------------------------------------------------------

def arima_forecast(ctx_norm, h, order=(1, 1, 1)):
    fit = ARIMA(ctx_norm, order=order).fit()
    return fit.forecast(steps=h)

def arima_forecast_fn(ctx_norm, h):
    return arima_forecast(ctx_norm, h)


class IdentityInstanceNorm(torch.nn.Module):
    """Returns input unchanged; reports loc=0, scale=1 so de-norm is a no-op."""
    def forward(self, x, loc_scale=None):
        loc   = torch.zeros(*x.shape[:-1], 1, device=x.device, dtype=x.dtype)
        scale = torch.ones_like(loc)
        return x, (loc, scale)

    def inverse(self, x, loc_scale):
        return x  # identity: no de-norm needed
    

# -------------------------------------------------------
# ----------------------- Chronos-2 ---------------------
# -------------------------------------------------------
    
def chronos_forecast(pipeline, ctx_norm, h):
    # Chronos-2 needs (n_series=1, n_variates=1, history_length)
    ctx_tensor = torch.tensor(ctx_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, W)
    with torch.no_grad():
        forecast = pipeline.predict(ctx_tensor, prediction_length=h)
        print(type(forecast), forecast.shape if hasattr(forecast, 'shape') else [f.shape for f in forecast])
    return forecast[0].median(dim=0).values.cpu().numpy()


def init_chronos():
    chronos_pipeline = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2",
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        )
    chronos_pipeline.model.instance_norm = IdentityInstanceNorm()

def chronos_forecast_fn(model, ctx_norm, h):
    return chronos_forecast(pipeline=model, ctx_norm=ctx_norm, h=h)


# -------------------------------------------------------
# ----------------------- TimesFM -----------------------
# -------------------------------------------------------

def timesfm_forecast(model, ctx_norm, h):
    point_forecast, _ = model.forecast(
        inputs=[ctx_norm],
        freq=[0],
        normalize=False,  # ← disable internal normalization
    )
    return point_forecast[0][:h]


def init_timesfm():
    model = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend='gpu',
                per_core_batch_size=32,
                horizon_len=128,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id='google/timesfm-1.0-200m-pytorch'
            ),
        )
    return model

def forecast_fn(model, ctx_norm, h):
    return timesfm_forecast(model=model, ctx_norm=ctx_norm, h=h)


# ===========================================================================
# Dataset loaders -> dict[unique_id] -> 1D np.float64 array, sorted by time
# ===========================================================================
def load_m4(dataset_name: str, max_series: int | None = None) -> dict[str, np.ndarray]:
    df = gluonts_to_long_dataframe(dataset_name=dataset_name, split="train_test")
    df = df.sort_values(["unique_id", "ds"])
    series = {}
    for uid, g in df.groupby("unique_id"):
        series[uid] = g["y"].to_numpy(dtype=np.float64)
        if max_series is not None and len(series) >= max_series:
            break
    return series
 
 
def load_favorita(
    csv_path: str,
    max_series: int | None = None,
    min_length: int = 60,
) -> dict[str, np.ndarray]:
    """
    ASSUMPTION: standard Kaggle Favorita `train.csv` layout with columns
    ['date', 'store_nbr', 'item_nbr', 'unit_sales']. Each (store_nbr, item_nbr)
    pair is treated as one series, daily-resampled, missing days filled with 0.
    If your file/columns differ, adjust this function accordingly.
    """
    usecols = ["date", "store_nbr", "item_nbr", "unit_sales"]
    df = pd.read_csv(csv_path, usecols=usecols, parse_dates=["date"])
    df["unique_id"] = df["store_nbr"].astype(str) + "_" + df["item_nbr"].astype(str)
 
    series = {}
    for uid, g in df.groupby("unique_id"):
        g = g.set_index("date").sort_index()
        full_idx = pd.date_range(g.index.min(), g.index.max(), freq="D")
        y = g["unit_sales"].reindex(full_idx, fill_value=0.0).to_numpy(dtype=np.float64)
        if len(y) < min_length:
            continue
        series[uid] = y
        if max_series is not None and len(series) >= max_series:
            break
    return series
 
