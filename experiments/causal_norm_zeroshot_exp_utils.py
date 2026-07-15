"""
Lag-aware vs. causal-normalization forecasting experiment.
 
Contains all model wrappers, dataset loaders, and the CLI entrypoint in one file.
 
Usage
-----
    python run_experiment.py --dataset m4_yearly --W 24 --H 6 --methods arima
    python run_experiment.py --dataset favorita --favorita-csv /path/to/train.csv \
        --W 90 --H 14 --max-series 500 --step 14 --methods chronos timesfm
"""
 
import sys
import argparse
import functools
import pickle
import traceback
from pathlib import Path
 
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from statsmodels.tsa.arima.model import ARIMA


# ---- repo-local import (adjust if your checkout lives elsewhere) ----------
_REPO_ROOT = Path("/home/wpotosna/forgets")
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from preprocessing.gluonts_preprocessor import gluonts_to_long_dataframe
 
 
# ===========================================================================
# Core expanding-window experiment
# ===========================================================================
def run_experiment_batched(series, W, H, predict_batch_fn, step=1,
                           test_only=False):
    """
    predict_batch_fn(ctx_matrix: (n, W), h: int) -> (n, H) forecasts.
    Returns None if the series is too short to produce any windows.
 
    test_only : bool
        Restrict origins to the forking-sequences test partition
        Ytest = Y[-2H+1 :].  Origins range from max(W, T-2H+1) .. T-H,
        giving at most H windows (with step=1).
    """
    T = len(series)
    if T < 2 * H - 1:
        return None  # not enough data even for test partition

    # left-pad if series too short for full context windows
    pad_len = max(0, W + 2 * H - 1 - T)
    if pad_len > 0:
        series = np.concatenate([np.zeros(pad_len), series])
        T = len(series)  # update T after padding

    min_origin = T - 2 * H + 1
    starts_idx = np.arange(min_origin, T - H + 1, step=1)
    n = len(starts_idx)
    if n == 0:
        return None
 
    ctx_mat = np.stack([series[idx - W: idx] for idx in starts_idx])
    fut_mat = np.stack([series[idx: idx + H] for idx in starts_idx])
 
    # lag norm: stats from the context window itself
    mu_lag = ctx_mat.mean(axis=1)
    sig_lag = np.clip(ctx_mat.std(axis=1), 1e-6, None)
 
    # causal norm: stats from everything up to idx, computed via cumsum
    cumsum = np.cumsum(series)
    cumsq = np.cumsum(series ** 2)
    counts = starts_idx.astype(np.float64)
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
        "pred_lag_norm": preds_lag,
        "pred_caus_norm": preds_caus,
        "future_raw": fut_mat,
        "mae_lag": mae_lag,
        "mae_caus": mae_caus,
    }
 
 
# ===========================================================================
# Batching helpers
# ===========================================================================
def make_gpu_batched_fn(raw_batch_fn, batch_size):
    """Chunk large (n, W) inputs into GPU-friendly slices."""
    def batched(ctx_matrix, h):
        n = ctx_matrix.shape[0]
        if n <= batch_size:
            return raw_batch_fn(ctx_matrix, h)
        parts = []
        for start in range(0, n, batch_size):
            parts.append(raw_batch_fn(ctx_matrix[start:start + batch_size], h))
        return np.concatenate(parts, axis=0)
    return batched
 
 
def make_arima_batched_fn(n_jobs=-1, order=(1, 1, 1)):
    """Fit one ARIMA per window row, parallelised with joblib."""
    def _fit_one(ctx_row, h):
        try:
            fit = ARIMA(ctx_row, order=order).fit()
            return fit.forecast(steps=h)
        except Exception:
            return np.full(h, np.nan)
 
    def batched(ctx_matrix, h):
        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_fit_one)(ctx_matrix[i], h) for i in range(ctx_matrix.shape[0])
        )
        return np.stack(results)
    return batched
 
 
# ===========================================================================
# Chronos-2
# ===========================================================================
class IdentityInstanceNorm(torch.nn.Module):
    """Returns input unchanged so we control normalisation externally."""
    def forward(self, x, loc_scale=None):
        loc   = torch.zeros(*x.shape[:-1], 1, device=x.device, dtype=x.dtype)
        scale = torch.ones_like(loc)
        return x, (loc, scale)
 
    def inverse(self, x, loc_scale):
        return x
 
 
def init_chronos(device="cuda:0"):
    from chronos import Chronos2Pipeline
    pipeline = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2",
        device_map=device,
        dtype=torch.bfloat16,
    )
    pipeline.model.instance_norm = IdentityInstanceNorm()
    return pipeline
 
 
def chronos_forecast_batch(pipeline, ctx_matrix, h, device="cuda:0"):
    ctx_tensor = (
        torch.tensor(ctx_matrix, dtype=torch.float32)
        .unsqueeze(1)
    )
    with torch.no_grad():
        forecast = pipeline.predict(ctx_tensor, prediction_length=h)
    if isinstance(forecast, list):
        forecast = torch.stack(forecast)
    arr = forecast.squeeze(1).cpu().numpy()
    return np.median(arr, axis=1)
 
 
# ===========================================================================
# TimesFM
# ===========================================================================
def init_timesfm(H=128):
    import timesfm
    model = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="gpu",
            per_core_batch_size=32,
            horizon_len=H,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
        ),
    )
    return model
 
 
def timesfm_forecast_batch(model, ctx_matrix, h):
    inputs = [ctx_matrix[i] for i in range(ctx_matrix.shape[0])]
    freqs = [0] * len(inputs)
    point_forecasts, _ = model.forecast(
        inputs=inputs,
        freq=freqs,
        normalize=False,
    )
    return np.array(point_forecasts)[:, :h]


# ===========================================================================
# Toto 2.0
# ===========================================================================

class IdentityScaler(torch.nn.Module):
    """No-op replacement for Toto2's PatchedCausalStdScaler (loc=0, scale=1).
 
    Toto 2.0 normally re-normalizes the context internally using an expanding
    (causal) mean/std before applying arcsinh, then de-scales the output
    quantiles with `sinh() * scale + loc`. That fights with the external
    lag-norm / causal-norm stats already applied to ctx_matrix in
    run_experiment_batched, so we swap it for a pass-through and let the
    caller's own normalization be the only one in effect.
    """
    def forward(self, data, mask=None):
        return data, torch.zeros_like(data), torch.ones_like(data)
 
 
# Quantile levels returned by Toto2Model.forecast(): [0.1, ..., 0.9]
TOTO_MEDIAN_IDX = 4
TOTO_PATCH_SIZE = 32
 
 
def init_toto(checkpoint="Datadog/Toto-2.0-22m", device="cpu"):
    from toto2 import Toto2Model
 
    model = Toto2Model.from_pretrained(checkpoint)
    model = model.to(device).eval()
    model.scaler = IdentityScaler()  # disable internal causal norm
    return model
 
 
def _pad_to_patch(ctx_matrix):
    """Left-pad rows so length is a multiple of TOTO_PATCH_SIZE (>= 32).
 
    Returns (padded_matrix, mask_matrix) both of shape (n, padded_len).
    """
    n, W = ctx_matrix.shape
    min_len = max(TOTO_PATCH_SIZE,
                  TOTO_PATCH_SIZE * ((W + TOTO_PATCH_SIZE - 1) // TOTO_PATCH_SIZE))
    if W >= min_len:
        return ctx_matrix, np.ones_like(ctx_matrix, dtype=bool)
 
    pad_len = min_len - W
    padded = np.concatenate([np.zeros((n, pad_len)), ctx_matrix], axis=1)
    mask = np.concatenate([np.zeros((n, pad_len), dtype=bool),
                           np.ones((n, W), dtype=bool)], axis=1)
    return padded, mask
 
 
def toto_forecast_batch(model, ctx_matrix, h, device="cpu"):
    """Batch forecast: (n, W) -> (n, H)."""
    ctx_padded, mask_np = _pad_to_patch(ctx_matrix)
    has_padding = not mask_np.all()
 
    target      = torch.tensor(ctx_padded, dtype=torch.float32, device=device).unsqueeze(1)  # (n, 1, W')
    target_mask = torch.tensor(mask_np, device=device).unsqueeze(1)                          # (n, 1, W')
    series_ids  = torch.zeros(target.shape[0], 1, dtype=torch.long, device=device)
 
    with torch.no_grad():
        quantiles = model.forecast(
            {"target": target, "target_mask": target_mask, "series_ids": series_ids},
            horizon=h,
            decode_block_size=768,
            has_missing_values=has_padding,
        )
    # quantiles: (9, batch, n_variates, horizon) -> median, drop n_variates dim
    return quantiles[TOTO_MEDIAN_IDX, :, 0, :].cpu().numpy()
 

 
# ===========================================================================
# Dataset loaders
# ===========================================================================
def load_m4(dataset_name: str) -> dict[str, np.ndarray]:
    df = gluonts_to_long_dataframe(dataset_name=dataset_name, split="train_test")
    df = df.sort_values(["unique_id", "ds"])
    series = {}
    for uid, g in df.groupby("unique_id"):
        series[uid] = g["y"].to_numpy(dtype=np.float64)
    return series
 
 
def load_favorita(
    csv_path: str,
    min_length: int = 60,
) -> dict[str, np.ndarray]:
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
    return series
 
