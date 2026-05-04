# FULL E2E RUNBOOK (All Major Scripts)

## Scope

This runbook provides one consolidated full-run command sequence across the project:

- Phase 1 data engineering
- Stage 2 event-driven economics pipeline
- Normal baseline calibration branch
- Phase 1 LLM elicitation pipeline
- Phase 2 simulation and causal analysis

## Existing Runbooks

More focused runbooks still exist:

- scripts/stage2_economics/NORMAL_BASELINE_RUNBOOK.md
- scripts/stage2_economics/PHASE2_RUNBOOK.md
- scripts/stage2_economics/phase1_llm_elicitation/RUNBOOK.md

This file is the single consolidated command path.

## Environment

```bash
conda activate comosa_phase1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Optional (for full vLLM inference):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Full End-to-End Commands

Run from repo root.

```bash
set -euo pipefail

# ------------------------------------------------------------------
# A) Phase 1 Data Engineering
# ------------------------------------------------------------------
python scripts/phase1_data_engineering/00_fetch_tardis.py --mode klines
python scripts/phase1_data_engineering/03_audit_tardis.py
python scripts/phase1_data_engineering/06_preprocess_tardis.py
python scripts/phase1_data_engineering/07_extract_features.py
python scripts/phase1_data_engineering/09_stylized_facts.py

# ------------------------------------------------------------------
# B) Core Event-Driven Economics Pipeline
# ------------------------------------------------------------------
python scripts/phase1_data_engineering/00_fetch_tardis.py --mode aggtrades
python scripts/stage2_economics/00_reindex_ticks.py
python scripts/stage2_economics/01_build_multiresolution_bars.py
python scripts/stage2_economics/02_hft_feature_engineering.py
python scripts/stage2_economics/04_detect_flash_crashes.py
python scripts/stage2_economics/05_download_event_ticks.py
python scripts/stage2_economics/06_micro_feature_engineering.py
python scripts/stage2_economics/07_dag_validation.py
python scripts/stage2_economics/08_refine_event_timestamps.py
python scripts/stage2_economics/09_produce_confounder_outputs.py
python scripts/stage2_economics/10_augment_dynamics_features.py
python scripts/stage2_economics/12_bbo_proxies.py
python scripts/stage2_economics/11_compute_prior_anchors.py

# ------------------------------------------------------------------
# C) Normal Baseline Branch
# ------------------------------------------------------------------
python scripts/stage2_economics/05b_download_normal_week.py
python scripts/stage2_economics/06_micro_feature_engineering.py --mode normal --normal-dir data/processed/tardis/normal_baseline
python scripts/stage2_economics/11_compute_prior_anchors.py

# ------------------------------------------------------------------
# D) Phase 1 LLM Elicitation
# ------------------------------------------------------------------
python scripts/stage2_economics/phase1_llm_elicitation/13_write_prompts.py --include-normal --overwrite

# Option 1: Full inference with vLLM
python scripts/stage2_economics/phase1_llm_elicitation/14_run_inference.py \
  --backend vllm \
  --model-path models/mistral-7b-instruct \
  --batch-size 32 \
  --max-retries 3 \
  --temperature 0.8 \
  --overwrite

# Option 2: If no GPU/vLLM model is available, use mock backend
# python scripts/stage2_economics/phase1_llm_elicitation/14_run_inference.py --backend mock --overwrite

python scripts/stage2_economics/phase1_llm_elicitation/15_extract_parameters.py --overwrite
python scripts/stage2_economics/phase1_llm_elicitation/16_fit_distributions.py --overwrite
python scripts/stage2_economics/phase1_llm_elicitation/17_pre_simulation_sanity_check.py

# ------------------------------------------------------------------
# E) Phase 2 Simulation + Validation + Causal
# ------------------------------------------------------------------
python scripts/stage2_economics/18_lob_mini_runner.py --scenario llm --calibration-phase pre --n-runs 50 --output-csv data/processed/tardis/phase2_outputs/lob_mini_simulation_llm.csv --summary-json data/processed/tardis/phase2_outputs/lob_mini_summary_llm.json
python scripts/stage2_economics/18_lob_mini_runner.py --scenario uniform --calibration-phase normal_bull --n-runs 50 --output-csv data/processed/tardis/phase2_outputs/lob_mini_simulation_uniform.csv --summary-json data/processed/tardis/phase2_outputs/lob_mini_summary_uniform.json
python scripts/stage2_economics/18_lob_mini_runner.py --scenario literature --calibration-phase normal_bear --n-runs 50 --output-csv data/processed/tardis/phase2_outputs/lob_mini_simulation_literature.csv --summary-json data/processed/tardis/phase2_outputs/lob_mini_summary_literature.json

python scripts/stage2_economics/19_stylised_facts_validation.py \
  --sim-llm data/processed/tardis/phase2_outputs/lob_mini_simulation_llm.csv \
  --sim-uniform data/processed/tardis/phase2_outputs/lob_mini_simulation_uniform.csv \
  --sim-literature data/processed/tardis/phase2_outputs/lob_mini_simulation_literature.csv \
  --report-json reports/validation/phase2_stylised_facts_validation.json \
  --report-md reports/validation/phase2_stylised_facts_validation.md

python scripts/stage2_economics/18_lob_mini_runner.py --scenario llm --calibration-phase pre --n-runs 1000 --output-csv data/processed/tardis/phase2_outputs/lob_full_simulation_llm.csv --summary-json data/processed/tardis/phase2_outputs/lob_full_summary_llm.json

python scripts/stage2_economics/20_causal_discovery.py \
  --sim-panel data/processed/tardis/phase2_outputs/lob_full_simulation_llm.csv \
  --edges-csv data/processed/tardis/phase2_outputs/causal_discovery_edges.csv \
  --report-json reports/validation/phase2_causal_discovery.json \
  --report-md reports/validation/phase2_causal_discovery.md

python scripts/stage2_economics/21_intervention_analysis.py \
  --sim-panel data/processed/tardis/phase2_outputs/lob_full_simulation_llm.csv \
  --report-json reports/validation/phase2_intervention_analysis.json \
  --report-md reports/validation/phase2_intervention_analysis.md
```

## Final Sanity Checks

```bash
python - <<'PY'
import json
import pandas as pd
from pathlib import Path

checks = [
    Path('data/processed/tardis/confounder_outputs/prior_anchors.json'),
    Path('data/processed/tardis/phase1_outputs/behavioral_priors.json'),
    Path('data/processed/tardis/phase2_outputs/lob_mini_summary_llm.json'),
    Path('reports/validation/phase2_stylised_facts_validation.json'),
    Path('reports/validation/phase2_causal_discovery.json'),
    Path('reports/validation/phase2_intervention_analysis.json'),
]

print('=== Artifact existence ===')
for p in checks:
    print(p, 'OK' if p.exists() else 'MISSING')

sim = Path('data/processed/tardis/phase2_outputs/lob_mini_simulation_llm.csv')
if sim.exists():
    df = pd.read_csv(sim)
    print('\n=== Simulation quick stats ===')
    print('rows:', len(df), 'runs:', df.run_id.nunique())
    print('crash_rate:', df.groupby('run_id')['flash_crash_flag'].max().mean())
PY
```

## Runtime Notes

- This full pipeline is long-running and can take many hours to days depending on hardware and network.
- Script 14 is the heaviest component. Use mock backend if model/GPU stack is not ready.
- Script 05 and 05b depend on Binance Vision availability.
- Script 20 may use fallback methods if NOTEARS/LiNGAM stack is unavailable.
