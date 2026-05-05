#!/usr/bin/env bash
# Narrow rescue sweep around cfgC_dim350_dsp020, varying resilience_min_damp only.

set -uo pipefail

DRY_RUN=0
for arg in "$@"; do
  [[ "$arg" == "--dry-run" ]] && DRY_RUN=1
done

SEED_START="${SEED_START:-240}"
SEED_END="${SEED_END:-243}"
RUNS_PER_SHARD="${RUNS_PER_SHARD:-5}"
PARALLELISM="${PARALLELISM:-20}"

BASE_CFG="cfgC_dim350_dsp020"
DIM=3.5
DSP=0.20
MIN_PRICE_FRACTION=0.60
GUARD_WINDOW=5
RDAMPS_STR="${RDAMPS_STR:-0.10 0.12 0.14 0.16 0.18}"
read -r -a RDAMPS <<< "$RDAMPS_STR"

OUT_ROOT="data/processed/tardis/phase2_outputs/rescue_sweep_cfgC_dim350_dsp020_rdamp_v1"
SUMMARY_CSV="$OUT_ROOT/rescue_sweep_summary.csv"
LOG_DIR="/tmp/cfgC_rescue_rdamp_logs"

mkdir -p "$OUT_ROOT" "$LOG_DIR"

rdamp_tag() {
  echo "$1" | tr -d '.'
}

cfg_name_for_rdamp() {
  local tag
  tag="$(rdamp_tag "$1")"
  printf '%s_rdamp%s' "$BASE_CFG" "$tag"
}

echo "============================================================"
echo " cfgC rescue rdamp sweep"
echo " base_cfg=$BASE_CFG  dim=$DIM  dsp=$DSP  floor=$MIN_PRICE_FRACTION"
echo " seeds=$SEED_START..$SEED_END  runs/shard=$RUNS_PER_SHARD  parallel=$PARALLELISM"
echo " rdamps=${RDAMPS[*]}"
echo " out=$OUT_ROOT"
[[ $DRY_RUN -eq 1 ]] && echo " *** DRY RUN - no jobs submitted ***"
echo "============================================================"
echo ""

if [[ "$PARALLELISM" -gt 20 ]]; then
  echo "[warn] Hardware audit capped parallelism at 20 logical cores; PARALLELISM=$PARALLELISM will oversubscribe CPU."
fi

for rdamp in "${RDAMPS[@]}"; do
  cfg="$(cfg_name_for_rdamp "$rdamp")"
  mkdir -p "$OUT_ROOT/$cfg/raw" "$OUT_ROOT/$cfg/analysis/floor" "$OUT_ROOT/$cfg/analysis/censored"
done

task_stream() {
  local rdamp cfg seed
  for rdamp in "${RDAMPS[@]}"; do
    cfg="$(cfg_name_for_rdamp "$rdamp")"
    for seed in $(seq "$SEED_START" "$SEED_END"); do
      printf '%s %s %s %s\n' "$seed" "$cfg" "$rdamp" "$OUT_ROOT/$cfg/raw"
    done
  done
}

export DIM DSP RUNS_PER_SHARD MIN_PRICE_FRACTION LOG_DIR

if [[ $DRY_RUN -eq 0 ]]; then
  task_stream | xargs -n4 -P"$PARALLELISM" bash -c '
    seed="$1"
    cfg="$2"
    rdamp="$3"
    outdir="$4"

    name="${cfg}_s${seed}"
    json="${outdir}/lob_pilot_${name}.json"
    log="${LOG_DIR}/lob_pilot_${name}.log"

    if [[ -f "$json" ]]; then
      echo "[skip] $name"
      exit 0
    fi

    echo "[start] $name  rdamp=$rdamp"

    conda run --no-capture-output -n comosa_phase1 python \
      scripts/stage2_economics/18_lob_mini_runner.py \
      --scenario llm \
      --n-runs "$RUNS_PER_SHARD" \
      --seed "$seed" \
      --tick-ms 100 \
      --impact-scale 2.0 \
      --intensity-scale 1.2289 \
      --crash-window-ticks 10 \
      --crash-threshold-pct 1.93 \
      --base-order-size 0.10 \
      --drop-sell-pressure "$DSP" \
      --drop-impact-mult "$DIM" \
      --min-price-fraction "$MIN_PRICE_FRACTION" \
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
  ' _
fi

for rdamp in "${RDAMPS[@]}"; do
  cfg="$(cfg_name_for_rdamp "$rdamp")"
  raw_dir="$OUT_ROOT/$cfg/raw"
  floor_dir="$OUT_ROOT/$cfg/analysis/floor"
  censored_dir="$OUT_ROOT/$cfg/analysis/censored"
  panel_file="$OUT_ROOT/$cfg/censored_panel.csv"

  echo ""
  echo "--- Analyze: $cfg (rdamp=$rdamp) ---"

  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[dry-run] would analyze $raw_dir"
    continue
  fi

  conda run --no-capture-output -n comosa_phase1 python \
    scripts/stage2_economics/22_analyze_cfgC_floor_artifacts.py \
    --input-dir "$raw_dir" \
    --output-dir "$floor_dir" \
    --min-price-fraction "$MIN_PRICE_FRACTION" \
    --guard-window "$GUARD_WINDOW"

  conda run --no-capture-output -n comosa_phase1 python \
    scripts/stage2_economics/25_step4_censor_panel.py \
    --input-dir "$raw_dir" \
    --output-dir "$censored_dir" \
    --panel-file "$panel_file" \
    --cfg-label "$cfg" \
    --guard-window "$GUARD_WINDOW" \
    --crash-window 10 \
    --min-price-fraction "$MIN_PRICE_FRACTION"
done

if [[ $DRY_RUN -eq 1 ]]; then
  echo ""
  echo "Dry run complete."
  echo "Summary would be written to: $SUMMARY_CSV"
  exit 0
fi

python3 - "$OUT_ROOT" "$SUMMARY_CSV" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

out_root = Path(sys.argv[1])
summary_csv = Path(sys.argv[2])
rows = []

for cfg_dir in sorted(out_root.glob("cfgC_dim350_dsp020_rdamp*")):
    cfg = cfg_dir.name
    tag = cfg.rsplit("rdamp", 1)[1]
    rdamp = float("0." + tag[1:])

    floor_path = cfg_dir / "analysis" / "floor" / "per_config_floor_timing.csv"
    censored_path = cfg_dir / "analysis" / "censored" / "step4_lock_test_summary.json"
    if not floor_path.exists() or not censored_path.exists():
        continue

    floor_df = pd.read_csv(floor_path)
    if floor_df.empty:
        continue
    floor = floor_df.iloc[0]

    with censored_path.open("r", encoding="utf-8") as f:
        censored = json.load(f)

    crash_rate = float(censored["crash_rate_censored"])
    rows.append(
        {
            "cfg": cfg,
            "rdamp": rdamp,
            "n_runs": int(censored["n_runs"]),
            "crash_rate_raw": float(censored["crash_rate_raw"]),
            "crash_rate_censored": crash_rate,
            "pre_floor_crash_rate": float(floor["pre_floor_crash_rate"]),
            "floor_hit_rate": float(floor["floor_hit_rate"]),
            "avg_frac_at_min": float(floor["avg_frac_at_min"]),
            "usable_pre_floor_fraction": float(floor["usable_pre_floor_fraction"]),
            "floor_exposure": float(censored["floor_exposure"]),
            "avg_frac_at_min_censored": float(censored["avg_frac_at_min_censored"]),
            "usable_fraction_total": float(censored["usable_fraction_total_mean"]),
            "usable_fraction_drop": float(censored["usable_fraction_drop_mean"]),
            "ofi_crash_window_mean": float(censored["ofi_crash_window"]["mean"]),
            "spread_crash_window_mean": float(censored["spread_crash_window"]["mean"]),
            "depth_crash_window_mean": float(censored["depth_crash_window"]["mean"]),
            "target_dist": abs(crash_rate - 0.10),
            "gate_target_8_12": int(0.08 <= crash_rate <= 0.12),
            "gate_zero_floor": int(float(censored["floor_exposure"]) == 0.0),
        }
    )

summary_df = pd.DataFrame(rows)
if summary_df.empty:
    raise SystemExit("No completed rescue sweep outputs found.")

summary_df = summary_df.sort_values(
    ["gate_target_8_12", "gate_zero_floor", "target_dist", "floor_hit_rate", "avg_frac_at_min"],
    ascending=[False, False, True, True, True],
).reset_index(drop=True)
summary_df.to_csv(summary_csv, index=False)

cols = [
    "rdamp",
    "crash_rate_censored",
    "pre_floor_crash_rate",
    "floor_hit_rate",
    "avg_frac_at_min",
    "usable_fraction_drop",
    "floor_exposure",
    "target_dist",
]
print("=== RESCUE SWEEP SUMMARY ===")
print(summary_df[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"Saved summary: {summary_csv}")
PY

echo ""
echo "============================================================"
echo " Rescue sweep complete."
echo " Out root : $OUT_ROOT"
echo " Summary  : $SUMMARY_CSV"
echo " Logs     : $LOG_DIR"
echo "============================================================"