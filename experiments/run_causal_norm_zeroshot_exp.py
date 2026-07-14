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

from causal_norm_zeroshot_exp_utils import (
    run_experiment_batched,
    make_arima_batched_fn,
    init_chronos,
    chronos_forecast_batch,
    init_timesfm,
    timesfm_forecast_batch,
    init_toto,
    toto_forecast_batch,
    make_gpu_batched_fn,
    load_m4,
    load_favorita,
)
import functools


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True,
                   help="'m3_yearly', 'm3_quarterly', 'm3_monthly', 'm3_other', "
                        "'m4_yearly', 'm4_quarterly', 'm4_monthly', or 'favorita'")
    p.add_argument("--favorita-csv", default="/home/wpotosna/favorita_train.csv")
    p.add_argument("--W", type=int, default=24)
    p.add_argument("--H", type=int, default=6)
    p.add_argument("--step", type=int, default=None,
                   help="Stride between forecast origins (default: H)")
    p.add_argument("--test-only", action="store_true", default=False,
                   help="Evaluate only on forking-sequences test partition "
                        "Y[-2H+1:] (at most H origins per series)")
    p.add_argument("--max-series", type=int, default=None)
    p.add_argument("--methods", nargs="+", default=["arima", "chronos", "timesfm", "toto"],
                   choices=["arima", "chronos", "timesfm", "toto"])
    p.add_argument("--batch-size", type=int, default=64,
                   help="GPU batch size for Chronos-2 / TimesFM")
    p.add_argument("--arima-n-jobs", type=int, default=-1,
                   help="CPU processes for ARIMA (-1 = all cores)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-dir", default="/home/wpotosna/forgets/results")
    p.add_argument("--save-predictions", action="store_true", default=True)
    p.add_argument("--no-save-predictions", dest="save_predictions", action="store_false")
    args = p.parse_args()
 
    step = args.step or args.H
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / "predictions" / args.dataset
 
    # ---- load data ----
    if args.dataset == "favorita":
        all_series = load_favorita(args.favorita_csv, max_series=args.max_series)
    else:
        all_series = load_m4(args.dataset, max_series=args.max_series)
    print(f"Loaded {len(all_series)} series from '{args.dataset}'")
 
    # ---- build forecast functions (heavy models loaded once) ----
    forecast_fns = {}
    if "arima" in args.methods:
        forecast_fns["arima"] = make_arima_batched_fn(n_jobs=args.arima_n_jobs)
    if "chronos" in args.methods:
        pipeline = init_chronos(device=args.device)
        forecast_fns["chronos"] = make_gpu_batched_fn(
            functools.partial(chronos_forecast_batch, pipeline, device=args.device),
            args.batch_size,
        )
    if "timesfm" in args.methods:
        tfm = init_timesfm(H=args.H)
        forecast_fns["timesfm"] = make_gpu_batched_fn(
            functools.partial(timesfm_forecast_batch, tfm), args.batch_size
        )
    if "toto" in args.methods:
        tfm = init_toto()
        forecast_fns["toto"] = make_gpu_batched_fn(
            functools.partial(toto_forecast_batch, tfm), args.batch_size
        )
 
    for method in forecast_fns:
        (pred_dir / method).mkdir(parents=True, exist_ok=True)
 
    # ---- run ----
    summary = []
    n = len(all_series)
    for s_i, (uid, series) in enumerate(all_series.items(), start=1):
        if len(series) <= args.W + args.H:
            print(f"[{s_i}/{n}] skip {uid}: too short ({len(series)} pts)")
            continue
 
        for method, fn in forecast_fns.items():
            try:
                result = run_experiment_batched(
                    series, args.W, args.H, fn, step=step,
                    test_only=args.test_only)
            except Exception as e:
                print(f"[{s_i}/{n}] {uid}/{method} FAILED: {e}")
                traceback.print_exc()
                continue
            if result is None:
                continue
 
            if args.save_predictions:
                safe_uid = str(uid).replace("/", "_")
                with open(pred_dir / method / f"{safe_uid}.pkl", "wb") as f:
                    pickle.dump({"unique_id": uid, "method": method,
                                 "W": args.W, "H": args.H, "step": step,
                                 **result}, f)
 
            summary.append({
                "unique_id": uid, "method": method,
                "origin_idx": result["origin_idx"],
                "mae_lag": result["mae_lag"],
                "mae_caus": result["mae_caus"],
            })
 
        if s_i % 10 == 0 or s_i == n:
            print(f"[{s_i}/{n}] series processed")
 
    methods_tag = "-".join(args.methods)
    results_path = out_dir / f"results_{args.dataset}_{methods_tag}.pkl"
    with open(results_path, "wb") as f:
        pickle.dump({"dataset": args.dataset, "W": args.W, "H": args.H,
                      "step": step, "test_only": args.test_only,
                      "summary": summary}, f)
    print(f"Saved -> {results_path}")
 
 
if __name__ == "__main__":
    main()
 
