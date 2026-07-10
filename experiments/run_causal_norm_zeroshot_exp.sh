#!/usr/bin/env bash
set -euo pipefail

PY="python run_causal_norm_zeroshot_exp.py"
OUTDIR="/home/wpotosna/forgets/results"
FAVORITA_CSV="/home/wpotosna/favorita_train.csv"
DEVICE="cuda:0"
BS=64                          # GPU batch size
FILTER="${1:-all}"

should_run() {
    # returns 0 (true) if any argument matches $FILTER
    for tag in "$@"; do [[ "$FILTER" == "$tag" ]] && return 0; done
    [[ "$FILTER" == "all" ]] && return 0
    return 1
}

# ===========================  M1  ==========================================
# Yearly   : 181 series, len 15–58,   H=6,  W=8
# Quarterly: 203 series, len 18–114,  H=8,  W=8
# Monthly  : 617 series, len 48–150,  H=18, W=24

for m in arima; do #chronos timesfm; do
should_run m1 "$m" && \
$PY --dataset m1_yearly    --W 8  --H 6  --step 1 --methods "$m" \
    --out-dir "$OUTDIR" --device "$DEVICE" --batch-size "$BS"
done

# for m in arima chronos timesfm; do
# should_run m1 "$m" && \
# $PY --dataset m1_quarterly --W 8  --H 8  --step 1 --methods "$m" \
#     --out-dir "$OUTDIR" --device "$DEVICE" --batch-size "$BS"
# done

# for m in arima chronos timesfm; do
# should_run m1 "$m" && \
# $PY --dataset m1_monthly   --W 24 --H 18 --step 1 --methods "$m" \
#     --out-dir "$OUTDIR" --device "$DEVICE" --batch-size "$BS"
# done

# # ===========================  M3  ==========================================
# # Yearly   : 645 series,  len 20–47,   H=6,  W=12
# # Quarterly: 756 series,  len 24–72,   H=8,  W=12
# # Monthly  : 1428 series, len 66–144,  H=18, W=36
# # Other    : 174 series,  len 71–104,  H=8,  W=16

# for m in arima chronos timesfm; do
# should_run m3 "$m" && \
# $PY --dataset m3_yearly    --W 12 --H 6  --step 1 --methods "$m" \
#     --out-dir "$OUTDIR" --device "$DEVICE" --batch-size "$BS"
# done

# for m in arima chronos timesfm; do
# should_run m3 "$m" && \
# $PY --dataset m3_quarterly --W 12 --H 8  --step 1 --methods "$m" \
#     --out-dir "$OUTDIR" --device "$DEVICE" --batch-size "$BS"
# done

# for m in arima chronos timesfm; do
# should_run m3 "$m" && \
# $PY --dataset m3_monthly   --W 36 --H 18 --step 1 --methods "$m" \
#     --out-dir "$OUTDIR" --device "$DEVICE" --batch-size "$BS"
# done

# for m in arima chronos timesfm; do
# should_run m3 "$m" && \
# $PY --dataset m3_other     --W 16 --H 8  --step 1 --methods "$m" \
#     --out-dir "$OUTDIR" --device "$DEVICE" --batch-size "$BS"
# done

# # ===========================  M4  ==========================================
# # Yearly   : 23000 series, len 19–841,   H=6,  W=12
# # Quarterly: 24000 series, len 24–874,   H=8,  W=12
# # Monthly  : 48000 series, len 60–2812,  H=18, W=36
# # Weekly   : 359 series,   len 93–2610,  H=13, W=26
# # Daily    : 4227 series,  len 107–9933, H=14, W=28
# # Hourly   : 414 series,   len 748–1008, H=48, W=168

# for m in arima chronos timesfm; do
# should_run m4 "$m" && \
# $PY --dataset m4_yearly    --W 12  --H 6  --step 1 --methods "$m" \
#     --out-dir "$OUTDIR" --device "$DEVICE" --batch-size "$BS"
# done

# for m in arima chronos timesfm; do
# should_run m4 "$m" && \
# $PY --dataset m4_quarterly --W 12  --H 8  --step 1 --methods "$m" \
#     --out-dir "$OUTDIR" --device "$DEVICE" --batch-size "$BS"
# done

# for m in arima chronos timesfm; do
# should_run m4 "$m" && \
# $PY --dataset m4_monthly   --W 36  --H 18 --step 1 --methods "$m" \
#     --out-dir "$OUTDIR" --device "$DEVICE" --batch-size "$BS"
# done

# for m in arima chronos timesfm; do
# should_run m4 "$m" && \
# $PY --dataset m4_weekly    --W 26  --H 13 --step 1 --methods "$m" \
#     --out-dir "$OUTDIR" --device "$DEVICE" --batch-size "$BS"
# done

# for m in arima chronos timesfm; do
# should_run m4 "$m" && \
# $PY --dataset m4_daily     --W 28  --H 14 --step 1 --methods "$m" \
#     --out-dir "$OUTDIR" --device "$DEVICE" --batch-size "$BS"
# done

# for m in arima chronos timesfm; do
# should_run m4 "$m" && \
# $PY --dataset m4_hourly    --W 168 --H 48 --step 1 --methods "$m" \
#     --out-dir "$OUTDIR" --device "$DEVICE" --batch-size "$BS"
# done

# # ===========================  Tourism  =====================================
# # Yearly   : 518 series, len 11–47,   H=6,  W=4
# # Quarterly: 427 series, len 30–130,  H=8,  W=16
# # Monthly  : 366 series, len 91–333,  H=18, W=36

# for m in arima chronos timesfm; do
# should_run tourism "$m" && \
# $PY --dataset tourism_yearly    --W 4  --H 6  --step 1 --methods "$m" \
#     --out-dir "$OUTDIR" --device "$DEVICE" --batch-size "$BS"
# done

# for m in arima chronos timesfm; do
# should_run tourism "$m" && \
# $PY --dataset tourism_quarterly --W 16 --H 8  --step 1 --methods "$m" \
#     --out-dir "$OUTDIR" --device "$DEVICE" --batch-size "$BS"
# done

# for m in arima chronos timesfm; do
# should_run tourism "$m" && \
# $PY --dataset tourism_monthly   --W 36 --H 18 --step 1 --methods "$m" \
#     --out-dir "$OUTDIR" --device "$DEVICE" --batch-size "$BS"
# done

# # ===========================  Favorita  ====================================
# # Daily retail, H=16 (Kaggle competition horizon)
# # --test-only restricts to Y[-2H+1:] = last 31 days, step=1

# for m in arima chronos timesfm; do
# should_run favorita "$m" && \
# $PY --dataset favorita --favorita-csv "$FAVORITA_CSV" \
#     --W 90 --H 16 --step 1 --test-only --max-series 500 --methods "$m" \
#     --out-dir "$OUTDIR" --device "$DEVICE" --batch-size "$BS"
# done

# echo "Done."