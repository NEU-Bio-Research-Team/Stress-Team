#!/usr/bin/env bash
# Step 4 lock test for cfgC_dim350_dsp020 with explicit floor censoring.

set -uo pipefail

DRY_RUN=0
for arg in "$@"; do
  [[ "$arg" == "--dry-run" ]] && DRY_RUN=1
done

SEED_START=200
SEED_END=239
RUNS_PER_SHARD=5
PARALLELISM="${PARALLELISM:-20}"

CFG_NAME="cfgC_dim350_dsp020"
DIM=3.5
DSP=0.20
MIN_PRICE_FRACTION=0.60
RESILIENCE_MIN_DAMP=0.10
GUARD_WINDOW=5

OUT_ROOT="data/processed/tardis/phase2_outputs/phase2_censored_causal_candidate_cfgC_dim350_dsp020_v1_200runs_gw5t"
OUT_DIR="$OUT_ROOT/raw"
ANALYSIS_DIR="$OUT_ROOT/analysis"
PANEL_FILE="$OUT_ROOT/censored_panel.csv"
LOG_DIR="/tmp/cfgC_step4_lock_test_logs"

mkdir -p "$OUT_DIR" "$ANALYSIS_DIR" "$LOG_DIR"

echo "============================================================"
echo " Step 4 lock test: $CFG_NAME"
echo " seeds=$SEED_START..$SEED_END  runs/shard=$RUNS_PER_SHARD  parallel=$PARALLELISM"
echo " min_price_fraction=$MIN_PRICE_FRACTION  guard_window=$GUARD_WINDOW"
echo " out=$OUT_ROOT"
[[ $DRY_RUN -eq 1 ]] && echo " *** DRY RUN - no jobs submitted ***"
echo "============================================================"
echo ""

if [[ "$PARALLELISM" -gt 20 ]]; then
  echo "[warn] Hardware audit capped parallelism at 20 logical cores; PARALLELISM=$PARALLELISM will oversubscribe CPU."
fi

if [[ $DRY_RUN -eq 0 ]]; then
  seq "$SEED_START" "$SEED_END" | \
    nice -n 10 xargs -P"$PARALLELISM" -I{} bash -c '
      seed="$1"
      cfg="$2"
      dim="$3"
      dsp="$4"
      rdamp="$5"
      nruns="$6"
      outdir="$7"
      logdir="$8"
      mpf="$9"

      name="${cfg}_s${seed}"
      json="${outdir}/lob_pilot_${name}.json"
      log="${logdir}/lob_pilot_${name}.log"

      if [[ -f "$json" ]]; then
        echo "[skip] $name"
        exit 0
      fi

      echo "[start] $name"

      conda run --no-capture-output -n comosa_phase1 python \
        scripts/stage2_economics/18_lob_mini_runner.py \
        --scenario llm \
        --n-runs "$nruns" \
        --seed "$seed" \
        --tick-ms 100 \
        --impact-scale 2.0 \
        --intensity-scale 1.2289 \
        --crash-window-ticks 10 \
        --crash-threshold-pct 1.93 \
        --base-order-size 0.10 \
        --drop-sell-pressure "$dsp" \
        --drop-impact-mult "$dim" \
        --min-price-fraction "$mpf" \
        --resilience-floor-fraction 0.85 \
        --resilience-min-damp "$rdamp" \
        --mm-vol-threshold-mult 1.4 \
        --mm-withdrawal-strength 1.8 \
        --max-drop-ticks 5000 \
        --max-recovery-ticks 3000 \
        --max-post-ticks 2000 \
        --max-pre-ticks 2000 \
        --output-csv "${outdir}/lob_pilot_${name}.csv" \
        --summary-json "$json" \
        --log-every-runs 1 \
        --log-every-ticks 200 \
        >"$log" 2>&1

      ec=$?
      if [[ $ec -eq 0 ]]; then
        echo "[done] $name"
      else
        echo "[FAIL] $name exit=$ec -> $log"
      fi
    ' _ {} "$CFG_NAME" "$DIM" "$DSP" "$RESILIENCE_MIN_DAMP" \
          "$RUNS_PER_SHARD" "$OUT_DIR" "$LOG_DIR" "$MIN_PRICE_FRACTION"
fi

if [[ $DRY_RUN -eq 1 ]]; then
  echo ""
  echo "--- Step 4 censor/analyze ---"
  echo "[dry-run] would run 25_step4_censor_panel.py on $OUT_DIR"
  echo ""
  echo "============================================================"
  echo " Step 4 dry run complete."
  echo " Raw shards : $OUT_DIR"
  echo " Analysis   : $ANALYSIS_DIR"
  echo " Panel      : $PANEL_FILE"
  echo " Logs       : $LOG_DIR"
  echo "============================================================"
  exit 0
fi

echo ""
echo "--- Step 4 censor/analyze ---"
conda run --no-capture-output -n comosa_phase1 python \
  scripts/stage2_economics/25_step4_censor_panel.py \
  --input-dir "$OUT_DIR" \
  --output-dir "$ANALYSIS_DIR" \
  --panel-file "$PANEL_FILE" \
  --cfg-label "$CFG_NAME" \
  --guard-window "$GUARD_WINDOW" \
  --crash-window 10 \
  --min-price-fraction "$MIN_PRICE_FRACTION"

echo ""
echo "============================================================"
echo " Step 4 complete."
echo " Raw shards : $OUT_DIR"
echo " Analysis   : $ANALYSIS_DIR"
echo " Panel      : $PANEL_FILE"
echo " Logs       : $LOG_DIR"
echo "============================================================"