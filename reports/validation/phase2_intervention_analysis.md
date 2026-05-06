# Phase 2 Intervention Analysis

## Data

- sim_panel: data/processed/tardis/phase2_outputs/phase2_canonical_tuned_legacy_500runs/lob_full_simulation_llm_tuned_legacy.csv
- runs_total: 500
- crash_runs_total: 88
- runs_used: 500
- crash_runs_used: 9
- window_pre_ticks: 50
- min_window_rows: 25
- floor_policy: censor_rows
- floor_touched_run_rate: 1.0
- floor_touched_used_rate: 1.0
- mean_window_rows: 50.0

## Model

- n_runs: 500
- logistic_auc: 0.9798596967639738
- positive_rate_raw: 0.018

## Observational

- crash_rate_raw: 0.018
- crash_rate_pred: 0.018019967194536576
- severity_max_drawdown_1s_pct_pred: 0.979502801073535
- severity_max_drawdown_1s_pct_raw: 0.9795028010735349

## do(OFI=0)

- crash_rate_pred: 5.530476097423166e-08
- relative_reduction_pct: 99.9996930917777
- h3_target_30pct_pass: True

## do(leverage=0)

- crash_rate_pred: 8.686486735755835e-17
- relative_reduction_pct: 99.99999999999952
- severity_max_drawdown_1s_pct_pred: 0.0
- severity_reduction_pct: 100.0
