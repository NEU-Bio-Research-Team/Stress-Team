#!/usr/bin/env bash

set -euo pipefail

CFG_PATH=${1:?"Usage: $0 <config_json> <cfg_label> [total_runs] [runs_per_shard] [parallel_jobs] [seed_start]"}
CFG_LABEL=${2:?"Usage: $0 <config_json> <cfg_label> [total_runs] [runs_per_shard] [parallel_jobs] [seed_start]"}
TOTAL_RUNS=${3:-500}
RUNS_PER_SHARD=${4:-25}
PARALLEL_JOBS=${5:-20}
SEED_START=${6:-500}

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

OUT_ROOT="data/processed/tardis/phase2_outputs/${CFG_LABEL}_${TOTAL_RUNS}runs"
RAW_DIR="${OUT_ROOT}/raw"
LOG_DIR="/tmp/${CFG_LABEL}_logs"

mkdir -p "$RAW_DIR" "$LOG_DIR"

if [[ ! -f "$CFG_PATH" ]]; then
  echo "Missing config: $CFG_PATH"
  exit 1
fi

if (( TOTAL_RUNS <= 0 || RUNS_PER_SHARD <= 0 || PARALLEL_JOBS <= 0 )); then
  echo "TOTAL_RUNS, RUNS_PER_SHARD, and PARALLEL_JOBS must be positive integers."
  exit 1
fi

SHARD_COUNT=$(( (TOTAL_RUNS + RUNS_PER_SHARD - 1) / RUNS_PER_SHARD ))

echo "=== PHASE 2 ABLATION PARALLEL RUN ==="
echo "config=$CFG_PATH"
echo "cfg_label=$CFG_LABEL"
echo "total_runs=$TOTAL_RUNS runs_per_shard=$RUNS_PER_SHARD shard_count=$SHARD_COUNT parallel_jobs=$PARALLEL_JOBS seed_start=$SEED_START"
echo "out_root=$OUT_ROOT"
echo ""

seq 0 $((SHARD_COUNT - 1)) | \
  xargs -P"$PARALLEL_JOBS" -I{} bash -c '
    shard_idx="$1"
    total_runs="$2"
    runs_per_shard="$3"
    seed_start="$4"
    cfg_path="$5"
    cfg_label="$6"
    raw_dir="$7"
    log_dir="$8"

    seed=$((seed_start + shard_idx))
    remaining=$((total_runs - shard_idx * runs_per_shard))
    shard_runs="$runs_per_shard"
    if (( remaining < runs_per_shard )); then
      shard_runs="$remaining"
    fi

    if (( shard_runs <= 0 )); then
      exit 0
    fi

    name="${cfg_label}_s${seed}"
    csv_path="${raw_dir}/${name}.csv"
    json_path="${raw_dir}/${name}.json"
    log_path="${log_dir}/${name}.log"

    if [[ -f "$json_path" && -f "$csv_path" ]]; then
      echo "[skip] $name"
      exit 0
    fi

    echo "[start] $name runs=$shard_runs seed=$seed"
    conda run --no-capture-output -n comosa_phase1 python \
      scripts/stage2_economics/18_lob_mini_runner.py \
      --config-json "$cfg_path" \
      --n-runs "$shard_runs" \
      --seed "$seed" \
      --output-csv "$csv_path" \
      --summary-json "$json_path" \
      >"$log_path" 2>&1
    ec=$?
    if [[ $ec -eq 0 ]]; then
      echo "[done] $name"
    else
      echo "[FAIL] $name exit=$ec see $log_path"
      exit $ec
    fi
  ' _ {} "$TOTAL_RUNS" "$RUNS_PER_SHARD" "$SEED_START" "$CFG_PATH" "$CFG_LABEL" "$RAW_DIR" "$LOG_DIR"

python3 - "$RAW_DIR" "$OUT_ROOT" "$TOTAL_RUNS" "$CFG_PATH" "$CFG_LABEL" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

raw_dir = Path(sys.argv[1])
out_root = Path(sys.argv[2])
target_runs = int(sys.argv[3])
cfg_path = sys.argv[4]
cfg_label = sys.argv[5]

csv_paths = sorted(raw_dir.glob("*.csv"))
json_paths = sorted(raw_dir.glob("*.json"))

if not csv_paths or not json_paths:
    raise SystemExit("No shard outputs found to merge.")

frames = []
run_id_offset = 0
observed_runs = 0
crash_count = 0
flash_crash_rates = []

for csv_path in csv_paths:
    frame = pd.read_csv(csv_path)
    if "run_id" not in frame.columns:
        raise SystemExit(f"Missing run_id column in {csv_path}")

    frame["run_id"] = frame["run_id"].astype(int) + run_id_offset
    unique_runs = int(frame["run_id"].nunique())
    observed_runs += unique_runs
    crash_count += int(frame.groupby("run_id")["flash_crash_flag"].max().sum())
    run_id_offset += unique_runs
    frames.append(frame)

for json_path in json_paths:
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    flash_crash_rates.append(float(payload.get("flash_crash_rate", 0.0)))

merged = pd.concat(frames, ignore_index=True)
merged_csv = out_root / f"lob_full_simulation_{cfg_label}.csv"
summary_json = out_root / f"lob_full_summary_{cfg_label}.json"
merged.to_csv(merged_csv, index=False)

summary = {
    "config_json": cfg_path,
    "target_runs": target_runs,
    "observed_runs": observed_runs,
    "shard_count": len(csv_paths),
    "flash_crash_rate_mean_of_shards": sum(flash_crash_rates) / len(flash_crash_rates),
    "flash_crash_rate_from_merged_panel": crash_count / observed_runs if observed_runs else 0.0,
    "crash_event_count": crash_count,
    "top_up_recommended": observed_runs >= 500 and crash_count < 20,
    "top_up_guidance": "Add 200-300 more runs, do not jump directly to 1000." if observed_runs >= 500 and crash_count < 20 else "No top-up required under the current crash-count rule.",
}

with summary_json.open("w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)

print("=== MERGE COMPLETE ===")
print(f"merged_csv={merged_csv}")
print(f"summary_json={summary_json}")
print(f"observed_runs={observed_runs}")
print(f"crash_event_count={crash_count}")
print(f"flash_crash_rate={summary['flash_crash_rate_from_merged_panel']:.4f}")
print(f"top_up_recommended={summary['top_up_recommended']}")
PY
