# Normal Baseline Runbook (Phase 1 + Phase 2 Bridge)

## Scope

This runbook adds normal-market baseline calibration into the existing COMOSA pipeline.

Updated components:

- `config/settings.py`
- `scripts/stage2_economics/05b_download_normal_week.py`
- `scripts/stage2_economics/06_micro_feature_engineering.py` (`--mode normal`)
- `scripts/stage2_economics/11_compute_prior_anchors.py` (normal merge)
- `scripts/stage2_economics/18_lob_mini_runner.py` (`--calibration-phase`)
- `scripts/stage2_economics/phase1_llm_elicitation/common.py`
- `scripts/stage2_economics/phase1_llm_elicitation/13_write_prompts.py` (`--include-normal`)

## Environment

```bash
conda activate comosa_phase1
pip install -r requirements.txt
```

## Default Normal Windows

Configured in `config/settings.py`:

- `normal_bull`: `2021-10-04` to `2021-10-08`
- `normal_bear`: `2023-02-06` to `2023-02-10`

Outputs are written under `data/processed/tardis/normal_baseline/`.

## Step 1 - Smoke Download (Light)

Run only 1-2 days first.

```bash
python scripts/stage2_economics/05b_download_normal_week.py --max-days 2
```

Expected artifacts:

- `data/processed/tardis/normal_baseline/baseline_download_manifest.csv`
- `data/processed/tardis/normal_baseline/baseline_prior_stats.json`
- at least one folder like `data/processed/tardis/normal_baseline/normal_bull_2021-10-04/aggtrades.parquet`

## Step 2 - Smoke Micro Features on Normal Data

```bash
python scripts/stage2_economics/06_micro_feature_engineering.py \
  --mode normal \
  --normal-dir data/processed/tardis/normal_baseline \
  --resolution 100
```

Expected artifacts:

- `data/processed/tardis/normal_baseline/micro/res_100ms/*.parquet`
- `data/processed/tardis/normal_baseline/micro/micro_features_summary.csv`

## Step 3 - Merge Anchors

```bash
python scripts/stage2_economics/11_compute_prior_anchors.py
```

Expected in `data/processed/tardis/confounder_outputs/prior_anchors.json`:

- `metadata.has_normal_baseline = true`
- `metadata.phases` includes `normal_bull` and `normal_bear`
- flat keys present:
  - `ofi_percentiles_per_phase.normal_bull`
  - `trade_intensity_per_phase.normal_bear`
  - `noise_trader_lambda.normal_bull`
  - `order_size_pareto_alpha.normal_bear`

## Step 4 - Smoke Simulation with Calibration Phase

```bash
python scripts/stage2_economics/18_lob_mini_runner.py \
  --scenario llm \
  --calibration-phase normal_bull \
  --n-runs 3 \
  --output-csv data/processed/tardis/phase2_outputs/lob_mini_simulation_llm_normalbull_smoke.csv \
  --summary-json data/processed/tardis/phase2_outputs/lob_mini_summary_llm_normalbull_smoke.json
```

Repeat with `--calibration-phase pre` and `--calibration-phase normal_bear` for A/B/C comparisons.

## Step 5 - Prompt Generation (Optional)

If your input CSV contains `normal_bull`/`normal_bear` in `phase` column, include them:

```bash
python scripts/stage2_economics/phase1_llm_elicitation/13_write_prompts.py --include-normal
```

If normal phases are missing in CSV, script 13 will warn and skip them.

## Full Sequence (After Smoke)

```bash
python scripts/stage2_economics/05b_download_normal_week.py
python scripts/stage2_economics/06_micro_feature_engineering.py --mode normal --normal-dir data/processed/tardis/normal_baseline
python scripts/stage2_economics/11_compute_prior_anchors.py
python scripts/stage2_economics/18_lob_mini_runner.py --scenario llm --calibration-phase pre --n-runs 50
python scripts/stage2_economics/18_lob_mini_runner.py --scenario uniform --calibration-phase normal_bull --n-runs 50
python scripts/stage2_economics/18_lob_mini_runner.py --scenario literature --calibration-phase normal_bear --n-runs 50
```
