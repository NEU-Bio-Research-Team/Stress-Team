# Phase 2 Crash Calibration Decision (2026-05-02)

## Decision Locked

- Detector window: `10` ticks (with `tick_ms=100`, equivalent to 1 second).
- Detector threshold: `crash_threshold_pct = 1.93`.
- Calibration target: `flash_crash_rate` around `10%` as the balanced target for Phase 3 positive-example coverage.

## Scientific Framing Used

- Separation between:
  - Flash-crash definition threshold (empirical labeling choice).
  - Simulator crash-rate target (calibration and statistical power choice).
- Existing repo validation gate remains: `crash_rate_sim in [0.05, 0.40]`.

## Empirical Inputs Used (Workspace Data)

Source file:
- `data/processed/tardis/confounder_outputs/Event_Dynamics_100ms.csv`

Computed on `phase == "drop"`:
- Row-level quantiles of `drop_from_local_pct`:
  - q90 = `0.166395%`
  - q95 = `0.250243%`
  - q99 = `0.526910%`
- Event-level quantiles of `max(drop_from_local_pct)`:
  - q90 = `2.153755%`
  - q95 = `2.693401%`
  - q99 = `3.646393%`
- Detector-consistent (rolling 10 ticks on close) event-level mapping:
  - target 20% positives -> threshold `1.468675%`
  - target 10% positives -> threshold `1.931845%`
  - target 5% positives -> threshold `2.362234%`

Chosen value `1.93%` aligns with the detector-consistent mapping for ~10% event positives.

## Code Changes Applied

File updated:
- `scripts/stage2_economics/18_lob_mini_runner.py`

### 1) Fix dead `drop_sell_pressure`

- Refactored `infer_side_probability(...)` to compute base `p_buy` for all 4 archetypes first.
- Applied uniform drop-phase sell tilt afterward for all archetypes:
  - `if phase == "drop": p_buy = p_buy - drop_sell_pressure`
- Returned `clip01(p_buy)`.

### 2) Update default threshold

- Changed parser default:
  - from `--crash-threshold-pct=1.0`
  - to `--crash-threshold-pct=1.93`

### 3) Add run-level max drawdown logger

- Added helper `max_drawdown_pct_rolling(close_series, window_ticks)` using same drawdown logic as crash detector.
- Added per-run log line fields:
  - `max_dd_{window}t=...%`
  - `crash={0|1}`
- Added summary JSON fields:
  - `crash_window_ticks`
  - `crash_threshold_pct`
  - `run_max_drawdown_pct` with `mean/p50/p90/p95/p99/max`
- Added final stdout summary line for run max drawdown stats.

## Post-patch Verification

Environment used:
- `conda` env: `comosa_phase1`

### A) Default run check (n_runs=3, seed=123)

Command:
- `python scripts/stage2_economics/18_lob_mini_runner.py --scenario llm --n-runs 3 --seed 123 ...`

Observed:
- `Crash rate      : 0.0000`
- `Run max DD (%) : mean=0.0362 p95=0.0748 max=0.0806`
- Summary JSON contains:
  - `crash_threshold_pct: 1.93`
  - `crash_window_ticks: 10`

### B) `drop_sell_pressure` sensitivity check (dead-parameter fix verification)

A/B setup:
- A: `--drop-sell-pressure 0.12`
- B: `--drop-sell-pressure 0.90`
- Same `n_runs=3`, same seed `123`.

Observed (post-patch):
- `summary_equal_a_b = False`
- `panel_equal_a_b = False`
- `mean_ofi_drop` changed:
  - A: `-0.133472...`
  - B: `-0.472204...`
- `run_max_drawdown_pct.mean` changed:
  - A: `0.036175...`
  - B: `0.073995...`

Conclusion:
- `drop_sell_pressure` is no longer dead; it now has measurable effect on simulator outputs.

## Notes

- This patch intentionally does not modify impact calibration (`impact_scale`, leverage feedback loop) beyond selected scope.
- Next calibration steps should be done against the locked target (`~10%`) and validated against Phase 2 stylized-fact gates.

## Blocker Quantification (Pre-Advanced Tasks)

Using the manual formula agreed in review:

`impact_scale = (D/100 * P0) / (W * kyle_lambda_drop * |mean_net_flow_drop|)`

With:
- `D = 1.93` (%)
- `W = 10` ticks
- `kyle_lambda_drop = 0.77641797` (from `prior_anchors.json`)
- `P0 = 36747.9774` (mean `tick_start_price` from `Flash_Crash_Events_Labeled.csv`)
- `mean_net_flow_drop = -0.133472...` (from patched default run output)

Computed:
- `impact_scale_required ≈ 684.39`

Sensitivity (different observed drop OFI means):
- if `|mean_net_flow_drop| = 0.472204...` -> `impact_scale ≈ 193.45`
- if `|mean_net_flow_drop| = 0.055020...` -> `impact_scale ≈ 1660.25`

Interpretation:
- With current dynamics and default `impact_scale=1.0`, crash detector threshold `1.93%` is unlikely to trigger frequently.
- A dedicated impact calibration step is required before expecting `flash_crash_rate ~ 10%`.

## Pending Manual Decisions (Before Advanced Tasks)

1. Contrarian immunity policy under systemic sell-pressure:
  - keep uniform shift for all archetypes (current implementation), or
  - exempt contrarian in drop phase.
2. Impact calibration strategy for target crash-rate:
  - choose whether to calibrate via `impact_scale` first, or
  - jointly calibrate `impact_scale` and drop-phase OFI amplification.
