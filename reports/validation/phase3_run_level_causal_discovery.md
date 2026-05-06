# Phase 3 Run-Level Causal Discovery

## Data

- sim_panel: /home/mluser/BRT-FDA/MinhQuang/Stress-Team/data/processed/tardis/phase2_outputs/phase2_canonical_tuned_legacy_500runs/lob_full_simulation_llm_tuned_legacy.csv
- run_panel_csv: /home/mluser/BRT-FDA/MinhQuang/Stress-Team/data/processed/tardis/phase2_outputs/phase2_canonical_tuned_legacy_500runs/run_level_causal_panel_llm_tuned_legacy.csv
- runs_total: 500
- crash_runs_total: 88
- pre_gap_ticks: 20
- mean_pre_rows: 9945.22
- min_pre_rows: 1832
- max_pre_rows: 11750
- crash_rate: 0.176
- kept_nodes: ofi_pre_mean, spread_pre_mean, depth_imb_pre_mean, leverage_pre_max, kyle_lambda_pre_mean, mean_wealth_pre_mean, pct_insolvent_pre_max, wealth_concentration_pre_mean, crashed, crash_severity_pct
- dropped_nodes: vpin_pre_max
- dagma_status: ok
- direct_lingam_status: ok
- pcmci_status: not_run: run-level panel is cross-sectional rather than an ordered time series

## Variance

- ofi_pre_mean: 0.450021
- spread_pre_mean: 0.473322
- depth_imb_pre_mean: 0.000695629
- leverage_pre_max: 0.100253
- kyle_lambda_pre_mean: 0.000165614
- vpin_pre_max: 1.2951e-26
- mean_wealth_pre_mean: 1.40125e+11
- pct_insolvent_pre_max: 0.00412051
- wealth_concentration_pre_mean: 0.00279861
- crashed: 0.145024
- crash_severity_pct: 3.69943e-05

## Window Strategies

- before_crash_minus_gap: 88
- full_run_no_crash: 412

## Method Summaries

- dagma: edges=24
  outcome_edges=['kyle_lambda_pre_mean->crashed (-1.773)', 'spread_pre_mean->crashed (0.954)', 'ofi_pre_mean->crashed (0.899)']
  top_edges=['kyle_lambda_pre_mean->crashed (-1.773)', 'kyle_lambda_pre_mean->spread_pre_mean (1.707)', 'mean_wealth_pre_mean->depth_imb_pre_mean (1.523)', 'mean_wealth_pre_mean->spread_pre_mean (-1.306)', 'wealth_concentration_pre_mean->ofi_pre_mean (1.107)', 'depth_imb_pre_mean->ofi_pre_mean (0.980)', 'spread_pre_mean->crashed (0.954)', 'ofi_pre_mean->crashed (0.899)', 'mean_wealth_pre_mean->kyle_lambda_pre_mean (0.869)', 'leverage_pre_max->mean_wealth_pre_mean (-0.764)']
- direct_lingam: edges=27
  outcome_edges=['crash_severity_pct->crashed (0.356)']
  top_edges=['depth_imb_pre_mean->pct_insolvent_pre_max (36.920)', 'ofi_pre_mean->pct_insolvent_pre_max (-35.844)', 'spread_pre_mean->pct_insolvent_pre_max (-4.387)', 'crashed->pct_insolvent_pre_max (3.294)', 'spread_pre_mean->mean_wealth_pre_mean (-2.614)', 'spread_pre_mean->depth_imb_pre_mean (2.163)', 'crashed->mean_wealth_pre_mean (1.940)', 'crashed->depth_imb_pre_mean (-1.857)', 'kyle_lambda_pre_mean->pct_insolvent_pre_max (1.475)', 'depth_imb_pre_mean->mean_wealth_pre_mean (1.404)']
