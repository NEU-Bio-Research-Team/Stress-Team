#!/usr/bin/env bash

set -euo pipefail

LABEL=${1:?usage: run_phase2_experiment_parallel.sh <label> <total_runs> <runs_per_shard> <parallel_jobs> <seed_start> [runner overrides ...]}
TOTAL_RUNS=${2:?usage: run_phase2_experiment_parallel.sh <label> <total_runs> <runs_per_shard> <parallel_jobs> <seed_start> [runner overrides ...]}
RUNS_PER_SHARD=${3:?usage: run_phase2_experiment_parallel.sh <label> <total_runs> <runs_per_shard> <parallel_jobs> <seed_start> [runner overrides ...]}
PARALLEL_JOBS=${4:?usage: run_phase2_experiment_parallel.sh <label> <total_runs> <runs_per_shard> <parallel_jobs> <seed_start> [runner overrides ...]}
SEED_START=${5:?usage: run_phase2_experiment_parallel.sh <label> <total_runs> <runs_per_shard> <parallel_jobs> <seed_start> [runner overrides ...]}
shift 5

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG_JSON="config/phase2_canonical_config.json"
OUT_ROOT="data/processed/tardis/phase2_outputs/${LABEL}_${TOTAL_RUNS}runs"
RAW_DIR="${OUT_ROOT}/raw"
LOG_DIR="/tmp/${LABEL}_logs"
EXTRA_ARGS_FILE="${OUT_ROOT}/runner_extra_args.txt"

mkdir -p "$RAW_DIR" "$LOG_DIR"

if [[ ! -f "$CONFIG_JSON" ]]; then
  echo "Missing config: $CONFIG_JSON"
  exit 1
fi

for arg in "$@"; do
  if [[ "$arg" == "--config-json" || "$arg" == "--output-csv" || "$arg" == "--summary-json" ]]; then
    echo "Do not override $arg; this wrapper manages config and output paths."
    exit 1
  fi
done

printf '%s\n' "$@" > "$EXTRA_ARGS_FILE"

SHARD_COUNT=$(( (TOTAL_RUNS + RUNS_PER_SHARD - 1) / RUNS_PER_SHARD ))

echo "=== PHASE 2 EXPERIMENT PARALLEL RUN ==="
echo "label         : $LABEL"
echo "config        : $CONFIG_JSON"
echo "total_runs    : $TOTAL_RUNS"
echo "runs_per_shard: $RUNS_PER_SHARD"
echo "parallel_jobs : $PARALLEL_JOBS"
echo "seed_start    : $SEED_START"
echo "shard_count   : $SHARD_COUNT"
echo "out_root      : $OUT_ROOT"
echo "extra_args    : $*"
echo ""

seq 0 $((SHARD_COUNT - 1)) | \
  xargs -P "$PARALLEL_JOBS" -I{} bash -c '
    shard_idx="$1"
    label="$2"
    total_runs="$3"
    runs_per_shard="$4"
    seed_start="$5"
    config_json="$6"
    raw_dir="$7"
    log_dir="$8"
    extra_args_file="$9"

    seed=$((seed_start + shard_idx))
    run_start=$((shard_idx * runs_per_shard))
    remaining=$((total_runs - run_start))
    if [[ $remaining -le 0 ]]; then
      exit 0
    fi

    shard_runs=$runs_per_shard
    if [[ $remaining -lt $runs_per_shard ]]; then
      shard_runs=$remaining
    fi

    tag="${label}_shard$(printf "%03d" "$shard_idx")_s${seed}"
    csv_path="${raw_dir}/${tag}.csv"
    json_path="${raw_dir}/${tag}.json"
    log_path="${log_dir}/${tag}.log"

    if [[ -f "$json_path" && -f "$csv_path" ]]; then
      echo "[skip] $tag"
      exit 0
    fi

    mapfile -t extra_args < "$extra_args_file"

    echo "[start] $tag runs=$shard_runs"
    conda run --no-capture-output -n comosa_phase1 python \
      scripts/stage2_economics/18_lob_mini_runner.py \
      --config-json "$config_json" \
      --n-runs "$shard_runs" \
      --seed "$seed" \
      --output-csv "$csv_path" \
      --summary-json "$json_path" \
      "${extra_args[@]}" \
      >"$log_path" 2>&1
    echo "[done] $tag"
  ' _ {} "$LABEL" "$TOTAL_RUNS" "$RUNS_PER_SHARD" "$SEED_START" "$CONFIG_JSON" "$RAW_DIR" "$LOG_DIR" "$EXTRA_ARGS_FILE"

python3 - "$RAW_DIR" "$OUT_ROOT" "$LABEL" "$TOTAL_RUNS" "$CONFIG_JSON" "$EXTRA_ARGS_FILE" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

raw_dir = Path(sys.argv[1])
out_root = Path(sys.argv[2])
label = sys.argv[3]
target_runs = int(sys.argv[4])
config_json = sys.argv[5]
extra_args_file = Path(sys.argv[6])

csv_paths = sorted(raw_dir.glob("*.csv"))
json_paths = sorted(raw_dir.glob("*.json"))
if not csv_paths or not json_paths:
    raise SystemExit("No shard outputs were produced.")

frames = []
run_offset = 0
for csv_path in csv_paths:
    frame = pd.read_csv(csv_path)
    if frame.empty:
        continue
    if "run_id" not in frame.columns:
        raise SystemExit(f"Missing run_id column in {csv_path}")
    frame["run_id"] = pd.to_numeric(frame["run_id"], errors="coerce").fillna(0).astype(int) + run_offset
    run_offset = int(frame["run_id"].max()) + 1
    frames.append(frame)

if not frames:
    raise SystemExit("All shard CSVs were empty.")

merged = pd.concat(frames, ignore_index=True)
merged_csv = out_root / f"lob_full_simulation_{label}.csv"
summary_json = out_root / f"lob_full_summary_{label}.json"
merged.to_csv(merged_csv, index=False)

crash_by_run = merged.groupby("run_id")["flash_crash_flag"].max()
extra_args = [line.strip() for line in extra_args_file.read_text(encoding="utf-8").splitlines() if line.strip()]

payload = {
    "label": label,
    "config_json": config_json,
    "extra_runner_args": extra_args,
    "target_total_runs": target_runs,
    "observed_total_runs": int(crash_by_run.shape[0]),
    "crash_count": int(crash_by_run.sum()),
    "flash_crash_rate": float(crash_by_run.mean()),
    "merged_csv": str(merged_csv),
    "source_shards": len(csv_paths),
}

summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

print(f"observed_total_runs = {payload['observed_total_runs']}")
print(f"crash_count         = {payload['crash_count']}")
print(f"flash_crash_rate    = {payload['flash_crash_rate']:.4f}")
print(f"merged_csv          = {payload['merged_csv']}")
print(f"merged_summary      = {summary_json}")
PY