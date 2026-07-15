#!/usr/bin/env bash
set -euo pipefail

SCRIPT="run_causal_norm_zeroshot_exp.py"
OUTDIR="/home/wpotosna/forgets/results"
FAVORITA_CSV="/home/wpotosna/favorita_train.csv"
DEVICE="cuda:0"
BS=64                          # GPU batch size
FILTER="${1:-all}"

should_run() {
    for tag in "$@"; do [[ "$FILTER" == "$tag" ]] && return 0; done
    [[ "$FILTER" == "all" ]] && return 0
    return 1
}

# ---- per-method env dispatch ----------------------------------------------
env_for() {
    case "$1" in
        toto) echo "toto_env" ;;
        timesfm) echo "timesfm_env" ;;
        arima|chronos) echo "tsfm_base" ;;
        *) echo "Unknown method for env dispatch: $1" >&2; exit 1 ;;
    esac
}

run_py() {
    local m="$1"; shift
    local env; env="$(env_for "$m")"
    conda run -n "$env" python "$SCRIPT" --methods "$m" "$@"
}

METHODS="toto" #"arima chronos timesfm toto"

# ===========================  M1  ==========================================
for m in $METHODS; do
should_run m1 "$m" && \
run_py "$m" --dataset m1_yearly    --W 12 --H 6  --step 1 \
    --out-dir "$OUTDIR" --device "$DEVICE" --batch-size "$BS"
done

for m in $METHODS; do
should_run m1 "$m" && \
run_py "$m" --dataset m1_quarterly --W 16 --H 8  --step 1 \
    --out-dir "$OUTDIR" --device "$DEVICE" --batch-size "$BS"
done

for m in $METHODS; do
should_run m1 "$m" && \
run_py "$m" --dataset m1_monthly   --W 36 --H 18 --step 1 \
    --out-dir "$OUTDIR" --device "$DEVICE" --batch-size "$BS"
done

# ===========================  M3  ==========================================
for m in $METHODS; do
should_run m3 "$m" && \
run_py "$m" --dataset m3_yearly    --W 12 --H 6  --step 1 \
    --out-dir "$OUTDIR" --device "$DEVICE" --batch-size "$BS"
done

for m in $METHODS; do
should_run m3 "$m" && \
run_py "$m" --dataset m3_quarterly --W 16 --H 8  --step 1 \
    --out-dir "$OUTDIR" --device "$DEVICE" --batch-size "$BS"
done

for m in $METHODS; do
should_run m3 "$m" && \
run_py "$m" --dataset m3_monthly   --W 36 --H 18 --step 1 \
    --out-dir "$OUTDIR" --device "$DEVICE" --batch-size "$BS"
done

for m in $METHODS; do
should_run m3 "$m" && \
run_py "$m" --dataset m3_other     --W 16 --H 8  --step 1 \
    --out-dir "$OUTDIR" --device "$DEVICE" --batch-size "$BS"
done

# ===========================  M4  ==========================================
for m in $METHODS; do
should_run m4 "$m" && \
run_py "$m" --dataset m4_yearly    --W 12 --H 6  --step 1 \
    --out-dir "$OUTDIR" --device "$DEVICE" --batch-size "$BS"
done

for m in $METHODS; do
should_run m4 "$m" && \
run_py "$m" --dataset m4_quarterly --W 16 --H 8  --step 1 \
    --out-dir "$OUTDIR" --device "$DEVICE" --batch-size "$BS"
done

for m in $METHODS; do
should_run m4 "$m" && \
run_py "$m" --dataset m4_monthly   --W 36 --H 18 --step 1 \
    --out-dir "$OUTDIR" --device "$DEVICE" --batch-size "$BS"
done

for m in $METHODS; do
should_run m4 "$m" && \
run_py "$m" --dataset m4_weekly    --W 26 --H 13 --step 1 \
    --out-dir "$OUTDIR" --device "$DEVICE" --batch-size "$BS"
done

for m in $METHODS; do
should_run m4 "$m" && \
run_py "$m" --dataset m4_daily     --W 28 --H 14 --step 1 \
    --out-dir "$OUTDIR" --device "$DEVICE" --batch-size "$BS"
done

for m in $METHODS; do
should_run m4 "$m" && \
run_py "$m" --dataset m4_hourly    --W 96 --H 48 --step 1 \
    --out-dir "$OUTDIR" --device "$DEVICE" --batch-size "$BS"
done

# ===========================  Tourism  =====================================
for m in $METHODS; do
should_run tourism "$m" && \
run_py "$m" --dataset tourism_yearly    --W 12 --H 6  --step 1 \
    --out-dir "$OUTDIR" --device "$DEVICE" --batch-size "$BS"
done

for m in $METHODS; do
should_run tourism "$m" && \
run_py "$m" --dataset tourism_quarterly --W 16 --H 8  --step 1 \
    --out-dir "$OUTDIR" --device "$DEVICE" --batch-size "$BS"
done

for m in $METHODS; do
should_run tourism "$m" && \
run_py "$m" --dataset tourism_monthly   --W 36 --H 18 --step 1 \
    --out-dir "$OUTDIR" --device "$DEVICE" --batch-size "$BS"
done

# ===========================  Favorita  ====================================
for m in $METHODS; do
should_run favorita "$m" && \
run_py "$m" --dataset favorita --favorita-csv "$FAVORITA_CSV" \
    --W 32 --H 16 --step 1 --test-only \
    --out-dir "$OUTDIR" --device "$DEVICE" --batch-size "$BS"
done

echo "Done."
