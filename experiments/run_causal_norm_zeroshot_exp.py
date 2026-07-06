"""
CLI entrypoint: run the lag-aware vs. causal-normalization forecasting experiment
across every series in a dataset (M4 or Favorita), for ARIMA, Chronos-2, and TimesFM.

Usage
-----
    python run_all_series.py --dataset m4_yearly --W 24 --H 6
    python run_all_series.py --dataset favorita --favorita-csv /path/to/train.csv --W 90 --H 14 --max-series 500

Output
------
    results_<dataset>.pkl
        {"dataset", "W", "H", "summary": [{"unique_id", "method", "origin_idx",
         "mae_lag", "mae_caus"}, ...]}  — one dict entry per (unique_id, method)
    predictions/<dataset>/<method>/<unique_id>.pkl
        dict with the full (n_windows, H) prediction arrays (both norm variants),
        the actual future values, and the norm stats used.
"""

import argparse
import pickle
import traceback
from pathlib import Path

from lib import (
    run_experiment_batched,
    make_arima_batched_fn,
    init_chronos,
    chronos_forecast_batch,
    init_timesfm,
    timesfm_forecast_batch,
    make_gpu_batched_fn,
    load_m4,
    load_favorita,
)
import functools


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                   help="e.g. 'm4_yearly', 'm4_quarterly', 'm4_monthly', or 'favorita'")
    p.add_argument("--favorita-csv", default="/home/claude/favorita_train.csv")
    p.add_argument("--W", type=int, default=24)
    p.add_argument("--H", type=int, default=6)
    p.add_argument("--max-series", type=int, default=None)
    p.add_argument("--methods", nargs="+", default=["arima", "chronos", "timesfm"],
                   choices=["arima", "chronos", "timesfm"])
    p.add_argument("--batch-size", type=int, default=64,
                   help="GPU batch size for Chronos-2 / TimesFM windows per call")
    p.add_argument("--arima-n-jobs", type=int, default=-1,
                   help="CPU processes for ARIMA (joblib); -1 = all cores")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-dir", default="/mnt/user-data/outputs")
    p.add_argument("--save-predictions", action="store_true", default=True)
    p.add_argument("--no-save-predictions", dest="save_predictions", action="store_false")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / "predictions" / args.dataset

    # ---- load data ----------------------------------------------------
    if args.dataset == "favorita":
        all_series = load_favorita(args.favorita_csv, max_series=args.max_series)
    else:
        all_series = load_m4(args.dataset, max_series=args.max_series)
    print(f"Loaded {len(all_series)} series from '{args.dataset}'")

    # ---- build batched forecast_fn registry (heavy models loaded once) -
    forecast_fns = {}
    if "arima" in args.methods:
        forecast_fns["arima"] = make_arima_batched_fn(n_jobs=args.arima_n_jobs)
    if "chronos" in args.methods:
        chronos_pipeline = init_chronos(device=args.device)
        forecast_fns["chronos"] = make_gpu_batched_fn(
            functools.partial(chronos_forecast_batch, chronos_pipeline), args.batch_size
        )
    if "timesfm" in args.methods:
        timesfm_model = init_timesfm()
        forecast_fns["timesfm"] = make_gpu_batched_fn(
            functools.partial(timesfm_forecast_batch, timesfm_model), args.batch_size
        )

    for method in forecast_fns:
        (pred_dir / method).mkdir(parents=True, exist_ok=True)

    # ---- run experiment for every series x method ----------------------
    summary = []  # list of dicts, one per (unique_id, method)
    n = len(all_series)
    for s_i, (uid, series) in enumerate(all_series.items(), start=1):
        if len(series) <= args.W + args.H:
            print(f"[{s_i}/{n}] skip {uid}: too short ({len(series)} points)")
            continue

        for method, fn in forecast_fns.items():
            try:
                result = run_experiment_batched(series, args.W, args.H, fn)
            except Exception as e:
                print(f"[{s_i}/{n}] {uid} / {method} FAILED: {e}")
                traceback.print_exc()
                continue

            if result is None:
                continue

            if args.save_predictions:
                safe_uid = str(uid).replace("/", "_")
                with open(pred_dir / method / f"{safe_uid}.pkl", "wb") as f:
                    pickle.dump({"unique_id": uid, "method": method,
                                 "W": args.W, "H": args.H, **result}, f)

            summary.append({
                "unique_id": uid,
                "method": method,
                "origin_idx": result["origin_idx"],   # (n_windows,)
                "mae_lag": result["mae_lag"],          # (n_windows,)
                "mae_caus": result["mae_caus"],        # (n_windows,)
            })

        if s_i % 10 == 0 or s_i == n:
            print(f"[{s_i}/{n}] series processed")

    results_path = out_dir / f"results_{args.dataset}.pkl"
    with open(results_path, "wb") as f:
        pickle.dump({"dataset": args.dataset, "W": args.W, "H": args.H, "summary": summary}, f)
    print(f"Saved raw results -> {results_path}")
    if args.save_predictions:
        print(f"Saved per-series prediction pickles -> {pred_dir}")


if __name__ == "__main__":
    main()