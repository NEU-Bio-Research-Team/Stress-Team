#!/usr/bin/env bash
# run_cfgC_9screen.sh  —  Blocked 9-config emergent-causal screen (Step 2 in teacher plan)
#
# Fixed params (definition layer – do NOT tune):
#   tick_ms=100, crash_window_ticks=10, crash_threshold_pct=1.93
#   impact_scale=2.0, intensity_scale=1.2289
#
# Fixed screening params:
#   min_price_fraction=0.60      (safety rail, NOT a mechanism)
#   resilience_floor_fraction=0.85
#   resilience_min_damp=0.10     (fixed for Stage 1A; rescue sweep in Stage 1B)
#
# Grid (9 configs):
#   drop_impact_mult   ∈ {2.5, 3.0, 3.5}
#   drop_sell_pressure ∈ {0.12, 0.16, 0.20}
#
# Blocked design:
#   Seeds 100-119 shared by ALL configs → same event-mix for fair comparison.
#   5 runs/shard × 20 shards = 100 runs per config  → 900 total runs.
#   20 parallel workers per config.
#
# Gates checked per config after all shards:
#   Screening:  crash_rate ∈ [0.05, 0.20]   avg_frac_at_min < 0.20   floor_hit_rate < 0.20
#   Causal-lock (flag only, not enforced here):
#               crash_rate ∈ [0.08, 0.12]   avg_frac_at_min < 0.05   floor_hit_rate < 0.05
#
# Usage:
#   bash scripts/stage2_economics/run_cfgC_9screen.sh [--dry-run]

set -uo pipefail

DRY_RUN=0
for arg in "$@"; do
  [[ "$arg" == "--dry-run" ]] && DRY_RUN=1
done

SEED_START=100
SEED_END=119
RUNS_PER_SHARD=5
PARALLELISM=20

OUT_DIR="data/processed/tardis/phase2_outputs/cfgC_screen"
LOG_DIR="/tmp/cfgC_screen_logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"

DIMS=(2.5 3.0 3.5)
DSPS=(0.12 0.16 0.20)
RESILIENCE_MIN_DAMP=0.10
MIN_PRICE_FRACTION=0.60

echo "============================================================"
echo " cfgC 9-config blocked screen  (teacher plan Step 2)"
echo " seeds=$SEED_START..$SEED_END  runs/shard=$RUNS_PER_SHARD  parallel=$PARALLELISM"
echo " min_price_fraction=$MIN_PRICE_FRACTION  resilience_min_damp=$RESILIENCE_MIN_DAMP"
echo " out=$OUT_DIR  logs=$LOG_DIR"
[[ $DRY_RUN -eq 1 ]] && echo " *** DRY RUN – no jobs submitted ***"
echo "============================================================"
echo ""

for DIM in "${DIMS[@]}"; do
  for DSP in "${DSPS[@]}"; do
    # Sanitise floats for filenames: 2.5 → 250, 0.12 → 012
    DIM_TAG=$(printf "%.0f" "$(echo "$DIM * 100" | bc)")
    DSP_TAG=$(printf "%03.0f" "$(echo "$DSP * 100" | bc)")
    CFG_NAME="cfgC_dim${DIM_TAG}_dsp${DSP_TAG}"

    echo "------------------------------------------------------------"
    echo "Config: $CFG_NAME  (dim=$DIM  dsp=$DSP  rdamp=$RESILIENCE_MIN_DAMP  floor=$MIN_PRICE_FRACTION)"
    echo "------------------------------------------------------------"

    if [[ $DRY_RUN -eq 1 ]]; then
      echo "[dry-run] would launch seeds $SEED_START..$SEED_END"
      continue
    fi

    # Launch 20 parallel shards for this config
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

        echo "[start] $name  seed=$seed  dim=$dim  dsp=$dsp  rdamp=$rdamp  floor=$mpf"

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
          echo "[FAIL] $name exit=$ec  → $log"
        fi
      ' _ {} "$CFG_NAME" "$DIM" "$DSP" "$RESILIENCE_MIN_DAMP" \
            "$RUNS_PER_SHARD" "$OUT_DIR" "$LOG_DIR" "$MIN_PRICE_FRACTION"

    # Per-config summary with floor artifact metrics
    echo ""
    echo "--- Summary: $CFG_NAME ---"
    conda run --no-capture-output -n comosa_phase1 python3 - \
        "$CFG_NAME" "$OUT_DIR" "$MIN_PRICE_FRACTION" <<'PYSUM'
import glob, json, sys, math
import pandas as pd
import numpy as np

cfg, out_dir, floor_frac_str = sys.argv[1], sys.argv[2], sys.argv[3]
floor_frac = float(floor_frac_str)

files = sorted(glob.glob(f"{out_dir}/lob_pilot_{cfg}_s*.json"))
if not files:
    print("  No JSON files found.")
    sys.exit(0)

crash_rates, dd_means, frac_at_mins, floor_hits = [], [], [], []
time_to_floors = []

def safe_nanmean(values):
  arr = np.asarray(values, dtype=float)
  if arr.size == 0 or np.all(np.isnan(arr)):
    return float("nan")
  return float(np.nanmean(arr))

def safe_nanmedian(values):
  arr = np.asarray(values, dtype=float)
  if arr.size == 0 or np.all(np.isnan(arr)):
    return float("nan")
  return float(np.nanmedian(arr))

for p in files:
    with open(p) as f:
        d = json.load(f)
    crash_rates.append(float(d.get("flash_crash_rate", 0.0)))
    dd_means.append(float(d.get("run_max_drawdown_pct", d.get("mean_drawdown_pct", float("nan")))))

    # Floor artifact metrics from CSV
    csv_path = p.replace(".json", ".csv")
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        frac_at_mins.append(float("nan"))
        floor_hits.append(float("nan"))
        time_to_floors.append(float("nan"))
        continue

    price_col = next((c for c in ["close", "mid", "price"] if c in df.columns), None)
    if price_col is None or "phase" not in df.columns:
        frac_at_mins.append(float("nan"))
        floor_hits.append(float("nan"))
        time_to_floors.append(float("nan"))
        continue

    run_fracs, run_floor_hits, run_ttf = [], [], []
    run_col = "run_id" if "run_id" in df.columns else None
    groups = df.groupby(run_col) if run_col else [(None, df)]

    for _, run_df in groups:
        pre = run_df[run_df["phase"] == "pre"]
        drop = run_df[run_df["phase"] == "drop"]
        if len(pre) == 0 or len(drop) == 0:
            continue
        init_price = pre[price_col].iloc[-1]
        floor = init_price * floor_frac

        at_floor = drop[price_col] <= floor * 1.001
        run_fracs.append(at_floor.mean())
        run_floor_hits.append(1.0 if at_floor.any() else 0.0)

        # time_to_floor: first tick index in drop where price hits floor
        first_hit_idx = at_floor.values.argmax() if at_floor.any() else None
        run_ttf.append(first_hit_idx if first_hit_idx is not None else float("nan"))

    frac_at_mins.append(safe_nanmean(run_fracs))
    floor_hits.append(safe_nanmean(run_floor_hits))
    time_to_floors.append(safe_nanmedian([t for t in run_ttf if not math.isnan(t)]))

def fmt(v):
    return f"{v:.4f}" if not math.isnan(v) else "  nan "

cr_mean = safe_nanmean(crash_rates)
fam_mean = safe_nanmean(frac_at_mins)
fhr_mean = safe_nanmean(floor_hits)
ttf_med  = safe_nanmedian(time_to_floors)

print(f"  cfg                = {cfg}")
print(f"  shards             = {len(files)}")
print(f"  crash_rate  mean   = {fmt(cr_mean)}  target [0.05-0.20]  strict [0.08-0.12]")
print(f"  avg_frac_at_min    = {fmt(fam_mean)}  gate <0.20 screen / <0.05 causal-lock")
print(f"  floor_hit_rate     = {fmt(fhr_mean)}  gate <0.20 screen / <0.05 causal-lock")
print(f"  median_ttf_ticks   = {fmt(ttf_med)}  (ticks into drop phase when floor first hit)")
print(f"  mean_drawdown_pct  = {fmt(safe_nanmean(dd_means))}")

# Gate evaluation
screen_pass = (
    0.05 <= cr_mean <= 0.20
    and (math.isnan(fam_mean) or fam_mean < 0.20)
    and (math.isnan(fhr_mean) or fhr_mean < 0.20)
)
causal_lock = (
    0.08 <= cr_mean <= 0.12
    and (math.isnan(fam_mean) or fam_mean < 0.05)
    and (math.isnan(fhr_mean) or fhr_mean < 0.05)
)
print(f"  SCREEN gate pass   = {screen_pass}")
print(f"  CAUSAL-LOCK flag   = {causal_lock}")
PYSUM
    echo ""

  done
done

echo "============================================================"
echo " All 9 configs done. Check $LOG_DIR for per-shard logs."
echo " Run 19_analyze_dim_sweep.py or a custom aggregator"
echo " to rank by composite score."
echo "============================================================"
