# Phase 2 Simulation and Causal Runbook

## Scope

This runbook covers the Phase 2 pipeline implemented in this directory:

- `18_lob_mini_runner.py`
- `19_stylised_facts_validation.py`
- `20_causal_discovery.py`
- `21_intervention_analysis.py`

The execution workflow is intentionally aligned with Phase 1:

1. Start with a mini run and inspect outputs.
2. Validate gate metrics.
3. Tune parameters if gates fail.
4. Scale up to full run.
5. Run causal and intervention analyses.

## Inputs and Artifacts

Required inputs:

- `data/processed/tardis/phase1_outputs/behavioral_priors.json`
- `data/processed/tardis/confounder_outputs/prior_anchors.json`
- `data/processed/tardis/confounder_outputs/Flash_Crash_Events_Labeled.csv`

Primary Phase 2 outputs:

- `data/processed/tardis/phase2_outputs/lob_mini_simulation_llm.csv`
- `data/processed/tardis/phase2_outputs/lob_mini_summary_llm.json`
- `reports/validation/phase2_stylised_facts_validation.json`
- `reports/validation/phase2_stylised_facts_validation.md`
- `data/processed/tardis/phase2_outputs/causal_discovery_edges.csv`
- `reports/validation/phase2_causal_discovery.json`
- `reports/validation/phase2_causal_discovery.md`
- `reports/validation/phase2_intervention_analysis.json`
- `reports/validation/phase2_intervention_analysis.md`

## Environment

Use the same environment strategy as Phase 1.

```bash
conda activate comosa_phase1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Optional causal packages (recommended for Script 20):

```bash
pip install causalnex lingam
```

If NOTEARS and LiNGAM packages are missing, Script 20 will still run with a correlation-based fallback DAG.

## Smoke Test (Quick)

Run a 10-run smoke test first.

```bash
python scripts/stage2_economics/18_lob_mini_runner.py \
  --scenario llm \
  --n-runs 10 \
  --output-csv data/processed/tardis/phase2_outputs/lob_mini_simulation_llm_smoke.csv \
  --summary-json data/processed/tardis/phase2_outputs/lob_mini_summary_llm_smoke.json

python scripts/stage2_economics/19_stylised_facts_validation.py \
  --sim-llm data/processed/tardis/phase2_outputs/lob_mini_simulation_llm_smoke.csv \
  --report-json reports/validation/phase2_stylised_facts_validation_smoke.json \
  --report-md reports/validation/phase2_stylised_facts_validation_smoke.md
```

Expected smoke-test checks:

- Output panel exists and includes all contract columns.
- Validation report is generated without schema errors.
- Crash rate is finite and non-trivial.

## Full Mini Run (Phase 2 Day 1)

Run 50–100 LLM-prior simulations.

```bash
python scripts/stage2_economics/18_lob_mini_runner.py \
  --scenario llm \
  --n-runs 50 \
  --output-csv data/processed/tardis/phase2_outputs/lob_mini_simulation_llm.csv \
  --summary-json data/processed/tardis/phase2_outputs/lob_mini_summary_llm.json
```

## Ablation Runs for H1 (Phase 2 Day 2)

Generate baseline panels under uniform and literature priors.

```bash
python scripts/stage2_economics/18_lob_mini_runner.py \
  --scenario uniform \
  --n-runs 50 \
  --output-csv data/processed/tardis/phase2_outputs/lob_mini_simulation_uniform.csv \
  --summary-json data/processed/tardis/phase2_outputs/lob_mini_summary_uniform.json

python scripts/stage2_economics/18_lob_mini_runner.py \
  --scenario literature \
  --n-runs 50 \
  --output-csv data/processed/tardis/phase2_outputs/lob_mini_simulation_literature.csv \
  --summary-json data/processed/tardis/phase2_outputs/lob_mini_summary_literature.json
```

Then run validation with all three panels.

```bash
python scripts/stage2_economics/19_stylised_facts_validation.py \
  --sim-llm data/processed/tardis/phase2_outputs/lob_mini_simulation_llm.csv \
  --sim-uniform data/processed/tardis/phase2_outputs/lob_mini_simulation_uniform.csv \
  --sim-literature data/processed/tardis/phase2_outputs/lob_mini_simulation_literature.csv \
  --report-json reports/validation/phase2_stylised_facts_validation.json \
  --report-md reports/validation/phase2_stylised_facts_validation.md
```

## Gate Criteria

Script 19 checks 4 primary gates:

- `kurtosis_excess > 3.0`
- `acf_vol_lag1 > 0.10`
- `ofi_drop_mean < ofi_pre_mean`
- `crash_rate_sim in [0.05, 0.40]`

Ablation H1 check:

- `kurtosis_llm_prior > kurtosis_uniform`

Pass interpretation:

- Full Phase 2 stylized-facts pass does not require pixel-perfect price trajectory matching.
- Success means simulated dynamics reproduce targeted stylized properties.

## Tuning Guide if Gates Fail

Suggested adjustments:

- `crash_rate_sim > 0.40`
- Increase `--mm-vol-threshold-mult`
- Decrease `--mm-withdrawal-strength`

- `crash_rate_sim < 0.05`
- Increase `--impact-scale`
- Increase `--intensity-scale`

- `acf_vol_lag1 < 0.10`
- Increase `--impact-scale`
- Increase persistence by reusing higher `kyle_lambda` phases

- `ofi_drop_mean >= ofi_pre_mean`
- Check phase-conditioned side behavior in Script 18
- Increase sell bias for momentum/noise in drop phase

## Scale Up (Phase 2 Day 3)

Once mini-run gates pass, scale to 500–1000 runs.

```bash
python scripts/stage2_economics/18_lob_mini_runner.py \
  --scenario llm \
  --n-runs 1000 \
  --output-csv data/processed/tardis/phase2_outputs/lob_full_simulation_llm.csv \
  --summary-json data/processed/tardis/phase2_outputs/lob_full_summary_llm.json
```

Validate the full panel with Script 19 by swapping `--sim-llm` to the full output path.

## Causal Discovery (Phase 2 Day 4)

```bash
python scripts/stage2_economics/20_causal_discovery.py \
  --sim-panel data/processed/tardis/phase2_outputs/lob_mini_simulation_llm.csv \
  --edges-csv data/processed/tardis/phase2_outputs/causal_discovery_edges.csv \
  --report-json reports/validation/phase2_causal_discovery.json \
  --report-md reports/validation/phase2_causal_discovery.md
```

Expected theoretical mechanism for comparison:

- `OFI -> spread_bps`
- `spread_bps -> depth_imbalance`
- `depth_imbalance -> flash_crash_flag`
- `OFI -> flash_crash_flag`
- `leverage_proxy -> flash_crash_flag`

## Intervention Analysis (Phase 2 Day 5)

```bash
python scripts/stage2_economics/21_intervention_analysis.py \
  --sim-panel data/processed/tardis/phase2_outputs/lob_mini_simulation_llm.csv \
  --report-json reports/validation/phase2_intervention_analysis.json \
  --report-md reports/validation/phase2_intervention_analysis.md
```

Primary intervention targets:

- `do(OFI=0)` should reduce crash propensity materially.
- `do(leverage=0)` should reduce amplification and severity.

H3 decision rule in script:

- Pass if `do(OFI=0)` implies predicted crash-rate reduction >= 30%.

## Recommended Execution Order

Use this exact sequence:

```bash
# Day 1
python scripts/stage2_economics/18_lob_mini_runner.py --scenario llm --n-runs 50

# Day 2
python scripts/stage2_economics/18_lob_mini_runner.py --scenario uniform --n-runs 50 --output-csv data/processed/tardis/phase2_outputs/lob_mini_simulation_uniform.csv --summary-json data/processed/tardis/phase2_outputs/lob_mini_summary_uniform.json
python scripts/stage2_economics/18_lob_mini_runner.py --scenario literature --n-runs 50 --output-csv data/processed/tardis/phase2_outputs/lob_mini_simulation_literature.csv --summary-json data/processed/tardis/phase2_outputs/lob_mini_summary_literature.json
python scripts/stage2_economics/19_stylised_facts_validation.py

# Day 3
python scripts/stage2_economics/18_lob_mini_runner.py --scenario llm --n-runs 1000 --output-csv data/processed/tardis/phase2_outputs/lob_full_simulation_llm.csv --summary-json data/processed/tardis/phase2_outputs/lob_full_summary_llm.json

# Day 4
python scripts/stage2_economics/20_causal_discovery.py --sim-panel data/processed/tardis/phase2_outputs/lob_full_simulation_llm.csv

# Day 5
python scripts/stage2_economics/21_intervention_analysis.py --sim-panel data/processed/tardis/phase2_outputs/lob_full_simulation_llm.csv
```

## Quick Sanity Check Commands

```bash
python - <<'PY'
import pandas as pd
p = "data/processed/tardis/phase2_outputs/lob_mini_simulation_llm.csv"
df = pd.read_csv(p)
print("rows", len(df), "runs", df.run_id.nunique())
print("columns", df.columns.tolist())
print("phase counts")
print(df.phase.value_counts().to_string())
print("crash rate", df.groupby("run_id")["flash_crash_flag"].max().mean())
PY
```
