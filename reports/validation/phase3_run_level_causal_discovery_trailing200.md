# Phase 3 Run-Level Causal Discovery

## Data

- sim_panel: data/processed/tardis/phase2_outputs/phase2_canonical_tuned_legacy_500runs/lob_full_simulation_llm_tuned_legacy.csv
- run_panel_csv: data/processed/tardis/phase2_outputs/phase2_canonical_tuned_legacy_500runs/run_level_causal_panel_trailing200.csv
- runs_total: 500
- crash_runs_total: 88
- pre_gap_ticks: 20
- trailing_pre_rows: 200
- mean_pre_rows: 200.0
- min_pre_rows: 200
- max_pre_rows: 200
- crash_rate: 0.176
- kept_nodes: ofi_pre_mean, depth_imb_pre_mean, leverage_pre_max, mean_wealth_pre_mean, wealth_concentration_pre_mean, crashed, crash_severity_pct
- dropped_nodes: spread_pre_mean, kyle_lambda_pre_mean, vpin_pre_max, pct_insolvent_pre_max
- dagma_status: ok
- direct_lingam_status: ok
- pcmci_status: not_run: run-level panel is cross-sectional rather than an ordered time series

## Variance

- ofi_pre_mean: 0.0202651
- spread_pre_mean: 0
- depth_imb_pre_mean: 3.22602e-05
- leverage_pre_max: 0.0118012
- kyle_lambda_pre_mean: 1.2326e-32
- vpin_pre_max: 1.34632e-24
- mean_wealth_pre_mean: 2.86228e+06
- pct_insolvent_pre_max: 0
- wealth_concentration_pre_mean: 0.000223858
- crashed: 0.145024
- crash_severity_pct: 3.69943e-05

## Window Strategies

- pre_phase_trailing_window: 500

## Method Summaries

- dagma: edges=11
  outcome_edges=['leverage_pre_max->crashed (0.063)']
  top_edges=['depth_imb_pre_mean->ofi_pre_mean (0.980)', 'mean_wealth_pre_mean->wealth_concentration_pre_mean (-0.977)', 'leverage_pre_max->wealth_concentration_pre_mean (-0.817)', 'wealth_concentration_pre_mean->ofi_pre_mean (-0.813)', 'crashed->wealth_concentration_pre_mean (-0.426)', 'crashed->mean_wealth_pre_mean (0.389)', 'crash_severity_pct->mean_wealth_pre_mean (0.276)', 'crash_severity_pct->leverage_pre_max (0.111)', 'leverage_pre_max->ofi_pre_mean (0.076)', 'crash_severity_pct->depth_imb_pre_mean (0.069)']
- direct_lingam: edges=9
  outcome_edges=['mean_wealth_pre_mean->crashed (0.834)']
  top_edges=['depth_imb_pre_mean->ofi_pre_mean (0.999)', 'mean_wealth_pre_mean->wealth_concentration_pre_mean (-0.907)', 'mean_wealth_pre_mean->crashed (0.834)', 'mean_wealth_pre_mean->depth_imb_pre_mean (0.826)', 'mean_wealth_pre_mean->leverage_pre_max (0.776)', 'crash_severity_pct->mean_wealth_pre_mean (0.461)', 'depth_imb_pre_mean->leverage_pre_max (0.120)', 'crashed->leverage_pre_max (0.089)', 'leverage_pre_max->wealth_concentration_pre_mean (-0.078)']
