#!/usr/bin/env bash
# run_pilot_20parallel.sh
# Usage:
#   bash scripts/stage2_economics/run_pilot_20parallel.sh \
#       <cfg_name> <drop_impact_mult> <drop_sell_pressure> <resilience_min_damp> \
#       <seed_start> <seed_end> <runs_per_shard> [min_price_fraction]
#
# Resume-safe: skips any shard whose .json output already exists.
# Logs per shard: /tmp/pilot20p_logs/lob_pilot_<cfg>_s<seed>.log
#
# Scientific basis for tunable params:
#   drop_impact_mult     : price-impact amplifier during drop phase
#                          (Bouchaud et al. 2018 – nonlinear impact in LOB)
#   drop_sell_pressure   : extra sell-side order-flow bias in drop phase
#                          (calibrated from BTC empirical OFI distribution in
#                           data/processed/tardis/confounder_outputs/Event_Dynamics_100ms.csv)
#   resilience_min_damp  : minimum damping near the resilience zone start (0=hard wall, 1=no effect)
#                          0.30 follows Obizhaeva & Wang (2013) – gradual LOB replenishment
#   min_price_fraction   : hard absolute floor (safety net, default 0.85)
#                          Setting < resilience_floor_fraction (0.85) creates a soft-damping zone
#                          between hard floor and resilience start – motivated by Bouchaud (2018)
#                          LOB thinning zone before complete market breakdown

set -uo pipefail

CFG_NAME=${1:?usage: run_pilot_20parallel.sh <cfg_name> <drop_impact_mult> <drop_sell_pressure> <resilience_min_damp> <seed_start> <seed_end> <runs_per_shard> [min_price_fraction]}
DROP_IMPACT_MULT=${2:?missing drop_impact_mult}
DROP_SELL_PRESSURE=${3:?missing drop_sell_pressure}
RESILIENCE_MIN_DAMP=${4:?missing resilience_min_damp}
SEED_START=${5:?missing seed_start}
SEED_END=${6:?missing seed_end}
RUNS_PER_SHARD=${7:?missing runs_per_shard}
MIN_PRICE_FRACTION=${8:-0.85}

OUT_DIR="data/processed/tardis/phase2_outputs/pilot_20p"
LOG_DIR="/tmp/pilot20p_logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"

echo "=== PILOT 20-PARALLEL ==="
echo "cfg=$CFG_NAME  dim=$DROP_IMPACT_MULT  dsp=$DROP_SELL_PRESSURE  rdamp=$RESILIENCE_MIN_DAMP  floor=$MIN_PRICE_FRACTION"
echo "seeds=$SEED_START..$SEED_END  runs/shard=$RUNS_PER_SHARD"
echo "out=$OUT_DIR  logs=$LOG_DIR"
echo ""

# xargs-safe: pass all config as positional args to avoid export-f portability issues
seq "$SEED_START" "$SEED_END" | \
  nice -n 10 xargs -P20 -I{} bash -c '
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

    # Resume: skip if output JSON already exists
    if [[ -f "$json" ]]; then
      echo "[skip] $name (already done)"
      exit 0
    fi

    echo "[start] $name seed=$seed"
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
      --log-every-ticks 500 \
      >"$log" 2>&1
    ec=$?
    if [[ $ec -eq 0 ]]; then
      echo "[done] $name"
    else
      echo "[FAIL] $name exit=$ec  see $log"
    fi
  ' _ {} "$CFG_NAME" "$DROP_IMPACT_MULT" "$DROP_SELL_PRESSURE" "$RESILIENCE_MIN_DAMP" \
        "$RUNS_PER_SHARD" "$OUT_DIR" "$LOG_DIR" "$MIN_PRICE_FRACTION"

echo ""
echo "=== RESULTS SUMMARY ==="
# Pass cfg and out_dir as argv to avoid bash expanding Python f-string syntax
# in the heredoc (unquoted <<PY would mis-expand {var} under set -u).
python3 - "$CFG_NAME" "$OUT_DIR" <<'PY'
import glob, json, statistics, sys
cfg, out_dir = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(f"{out_dir}/lob_pilot_{cfg}_s*.json"))
rates = []
for p in files:
    with open(p) as f:
        j = json.load(f)
    rates.append(float(j.get("flash_crash_rate", 0.0)))
if rates:
    print(f"cfg               = {cfg}")
    print(f"shards_done       = {len(files)}")
    print(f"crash_rate  mean  = {sum(rates)/len(rates):.4f}")
    print(f"crash_rate  min   = {min(rates):.4f}")
    print(f"crash_rate  max   = {max(rates):.4f}")
    if len(rates) > 1:
        print(f"crash_rate  stdev = {statistics.stdev(rates):.4f}")
    n_crash = sum(1 for r in rates if r > 0)
    print(f"shards_with_crash = {n_crash}/{len(rates)}")
    target_ok = 0.05 <= sum(rates)/len(rates) <= 0.40
    print(f"gate_pass [5-40%] = {target_ok}")
else:
    print("No JSON output files found.")
PY
