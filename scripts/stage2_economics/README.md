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

## Planned scripts

Scripts will be numbered 26+ continuing from Bio Stage.
