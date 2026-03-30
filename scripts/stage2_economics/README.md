# Stage 2: Market Simulator & Agent-Based Modeling

> **Status:** NOT STARTED  
> **Depends on:** Stage 1 (Bio Stage) — CLOSED

## Inputs from Bio Stage

- OU stress dynamics: `dσ = θ(μ - σ)dt + η dW` (universal mean-reversion, 15/15 subjects)
- σ(t) extraction pipeline: `src/preprocessing/` + `src/features/`
- Calibration targets: 5/5 Cont stylized facts (`data/processed/tardis/stylized_facts.json`)
- Flash crash scenarios: 7 events (`data/processed/tardis/flash_crashes.csv`)
- Agent heterogeneity: θ ~ N(0.074, 0.024²) at 5s scale — RECALIBRATE at ABM Δt

## Key constraints from Bio Stage

1. θ must be calibrated at ABM's time-step (NOT fixed at 0.074 — window artifact)
2. Standard OU sufficient — no fractional extension needed
3. ~13% extreme responders with 2× stress amplitude
4. Stress signal is in cardiac TIMING (HRV), not EEG/ECG morphology
5. Lab stress (TSST) ≠ trading stress — coupling function needs explicit design

## Scripts

### HFT Reindex Pipeline (Scripts 00–03)

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `00_reindex_ticks.py` | Parse raw .csv.gz → ms-resolution parquet | `data/raw/tardis/trades/*.csv.gz` | `data/processed/tardis/ticks/<date>.parquet` |
| `01_build_multiresolution_bars.py` | Aggregate ticks → time/volume/dollar/tick bars | tick parquets | `data/processed/tardis/bars/<type>/<date>.parquet` |
| `02_hft_feature_engineering.py` | Compute HFT microstructure features | bar parquets | `data/processed/tardis/hft_features/<res>/<date>.parquet` |
| `03_bio_market_alignment.py` | Build bio σ(t) ↔ market coupling layer | WESAD features + HFT features | `data/processed/tardis/hft_features/bio_market_alignment/` |

Run order: `00 → 01 → 02 → 03`

```bash
python scripts/stage2_economics/00_reindex_ticks.py
python scripts/stage2_economics/01_build_multiresolution_bars.py
python scripts/stage2_economics/02_hft_feature_engineering.py
python scripts/stage2_economics/03_bio_market_alignment.py
```

### Future scripts (26+)

ABM simulator scripts will be numbered 26+ continuing from Bio Stage.
