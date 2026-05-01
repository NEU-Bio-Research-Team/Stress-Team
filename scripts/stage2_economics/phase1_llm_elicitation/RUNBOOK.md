# Phase 1 LLM Elicitation Runbook

## Scope

This runbook covers the Phase 1 pipeline implemented in this directory:

- `13_write_prompts.py`
- `14_run_inference.py`
- `15_extract_parameters.py`
- `16_fit_distributions.py`

The pipeline is resume-safe by default. Re-run without `--overwrite` to continue from the last completed checkpoint.

## Environment

Use Python 3.11 in a fresh conda environment. Do not rely on the system `python` binary unless you have verified it resolves to the new environment.

```bash
conda create -n comosa_phase1 python=3.11 -y
conda activate comosa_phase1

python -m pip install --upgrade pip setuptools wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Notes:

- `torch` is installed separately so you can match the correct CUDA wheel.
- If your shell still resolves `python` to something unexpected, use `python3` explicitly.
- `14_run_inference.py` can run a `mock` backend for smoke tests even when `vllm` or the model is not installed yet.

## Model Download

Download the local Mistral checkpoint once:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    local_dir="models/mistral-7b-instruct",
    ignore_patterns=["*.pt", "original/"],
)
PY
```

If you are using an AWQ-quantized variant, place it under `models/mistral-7b-instruct` or pass the final path with `--model-path`.

## Smoke Test

Run the full chain on 3 samples first:

```bash
python scripts/stage2_economics/phase1_llm_elicitation/13_write_prompts.py --dry-run --overwrite
python scripts/stage2_economics/phase1_llm_elicitation/14_run_inference.py --dry-run --backend mock --overwrite
python scripts/stage2_economics/phase1_llm_elicitation/15_extract_parameters.py --dry-run --overwrite
python scripts/stage2_economics/phase1_llm_elicitation/16_fit_distributions.py --dry-run --overwrite
```

Expected dry-run outputs:

- `data/processed/tardis/phase1_outputs/phase1_prompts.dry_run.json`
- `data/processed/tardis/phase1_outputs/raw_elicited.dry_run.json`
- `data/processed/tardis/phase1_outputs/raw_elicited.dry_run.csv`
- `data/processed/tardis/phase1_outputs/behavioral_priors.dry_run.json`

## Full Run

When the smoke test looks correct, run the full sequence:

```bash
python scripts/stage2_economics/phase1_llm_elicitation/13_write_prompts.py --overwrite
python scripts/stage2_economics/phase1_llm_elicitation/14_run_inference.py \
  --backend vllm \
  --model-path models/mistral-7b-instruct \
  --batch-size 32 \
  --max-retries 3 \
  --temperature 0.8 \
  --overwrite
python scripts/stage2_economics/phase1_llm_elicitation/15_extract_parameters.py --overwrite
python scripts/stage2_economics/phase1_llm_elicitation/16_fit_distributions.py --overwrite
```

If a long run is interrupted, restart the same command without `--overwrite`.

## Output Contract

The final outputs land here:

- `data/processed/tardis/phase1_outputs/phase1_prompts.json`
- `data/processed/tardis/phase1_outputs/raw_elicited.json`
- `data/processed/tardis/phase1_outputs/raw_elicited.csv`
- `data/processed/tardis/phase1_outputs/behavioral_priors.json`

Important runtime note for the current workspace snapshot:

- The checked-in `Event_Dynamics_100ms*.csv` files currently only contain `pre` and `drop` phase rows.
- `13_write_prompts.py` therefore marks missing `recovery` and `post` phase samples as `anchor_imputed_phase_sample` and records the donor phase in `sample_source_phase`.
- If you regenerate the full event dynamics table later, the script will automatically switch back to observed phase sampling whenever those rows exist.

## Quick Checks

After the full run, verify these conditions:

```bash
python - <<'PY'
import json
import pandas as pd
from pathlib import Path

base = Path("data/processed/tardis/phase1_outputs")

prompts = json.loads((base / "phase1_prompts.json").read_text())
raw = json.loads((base / "raw_elicited.json").read_text())
parsed = pd.read_csv(base / "raw_elicited.csv")
priors = json.loads((base / "behavioral_priors.json").read_text())

print("phase1_prompts", len(prompts))
print("raw_elicited", len(raw))
print("parsed_total", len(parsed))
print("parsed_success", int((parsed["parse_status"] == "parsed").sum()))
print("agent_phase_entries", sum(len(priors[agent]) for agent in priors if agent != "metadata"))
PY
```

Target checks:

- `phase1_prompts == 512`
- `raw_elicited == 512`
- parsed success rate `>= 85%`
- `behavioral_priors.json` contains 16 agent-phase entries plus metadata

## Verify Distribution of canceled order
```
cd /home/mluser/BRT-FDA/MinhQuang/Stress-Team && /home/mluser/.conda/envs/comosa_phase1/bin/python - <<'PY'
import pandas as pd
from pathlib import Path
p=Path('data/processed/tardis/phase1_outputs/raw_elicited.csv')
df=pd.read_csv(p)
parsed=df[df['parse_status']=='parsed'].copy()
for col in ['aggressiveness','cancel_probability','order_size_multiplier','inventory_sensitivity']:
    parsed[col]=pd.to_numeric(parsed[col], errors='coerce')
print('parsed_rows', len(parsed))
print('cancel_probability_unique', sorted(parsed['cancel_probability'].dropna().unique().tolist())[:20])
print('cancel_probability_mean_by_agent')
print(parsed.groupby('agent_type')['cancel_probability'].agg(['mean','std','min','max','nunique']).round(3).to_string())
print('\norder_type by agent')
print(pd.crosstab(parsed['agent_type'], parsed['order_type'], normalize='index').round(3).to_string())
print('\nside by agent')
print(pd.crosstab(parsed['agent_type'], parsed['side'], normalize='index').round(3).to_string())
PY
```