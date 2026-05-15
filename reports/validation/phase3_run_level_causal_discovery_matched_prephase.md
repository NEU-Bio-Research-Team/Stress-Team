# Phase 3 Run-Level Causal Discovery

## Data

- sim_panel: /home/mluser/BRT-FDA/MinhQuang/Stress-Team/data/processed/tardis/phase2_outputs/phase2_canonical_tuned_legacy_500runs/lob_full_simulation_llm_tuned_legacy.csv
- run_panel_csv: /home/mluser/BRT-FDA/MinhQuang/Stress-Team/data/processed/tardis/phase2_outputs/phase2_canonical_tuned_legacy_500runs/run_level_causal_panel_matched_prephase.csv
- runs_total: 500
- crash_runs_total: 88
- pre_gap_ticks: 20
- mean_pre_rows: 1730.0
- min_pre_rows: 1730
- max_pre_rows: 1730
- crash_rate: 0.176
- kept_nodes: ofi_pre_mean, depth_imb_pre_mean, leverage_pre_max, mean_wealth_pre_mean, wealth_concentration_pre_mean, crashed, crash_severity_pct
- dropped_nodes: spread_pre_mean, kyle_lambda_pre_mean, vpin_pre_max, pct_insolvent_pre_max
- dagma_status: ok
- direct_lingam_status: ok
- pcmci_status: not_run: run-level panel is cross-sectional rather than an ordered time series

## Variance

- ofi_pre_mean: 0.0307863
- spread_pre_mean: 0
- depth_imb_pre_mean: 4.88345e-05
- leverage_pre_max: 0.0432039
- kyle_lambda_pre_mean: 0
- vpin_pre_max: 2.98628e-24
- mean_wealth_pre_mean: 777671
- pct_insolvent_pre_max: 0
- wealth_concentration_pre_mean: 5.98639e-05
- crashed: 0.145024
- crash_severity_pct: 3.69943e-05

## Window Strategies

- pre_phase_minus_gap: 500

## Method Summaries

- dagma: edges=10
  outcome_edges=['leverage_pre_max->crash_severity_pct (0.438)']
  top_edges=['depth_imb_pre_mean->ofi_pre_mean (0.980)', 'mean_wealth_pre_mean->wealth_concentration_pre_mean (-0.975)', 'ofi_pre_mean->leverage_pre_max (0.930)', 'wealth_concentration_pre_mean->leverage_pre_max (-0.517)', 'leverage_pre_max->crash_severity_pct (0.438)', 'crashed->mean_wealth_pre_mean (0.378)', 'crashed->wealth_concentration_pre_mean (-0.360)', 'wealth_concentration_pre_mean->ofi_pre_mean (-0.301)', 'wealth_concentration_pre_mean->depth_imb_pre_mean (-0.154)', 'crashed->leverage_pre_max (0.081)']
- direct_lingam: edges=8
  outcome_edges=['wealth_concentration_pre_mean->crashed (-1.191)', 'depth_imb_pre_mean->crashed (-0.368)']
  top_edges=['wealth_concentration_pre_mean->crashed (-1.191)', 'wealth_concentration_pre_mean->mean_wealth_pre_mean (-0.997)', 'depth_imb_pre_mean->ofi_pre_mean (0.997)', 'wealth_concentration_pre_mean->depth_imb_pre_mean (-0.967)', 'wealth_concentration_pre_mean->leverage_pre_max (-0.777)', 'crash_severity_pct->wealth_concentration_pre_mean (-0.457)', 'depth_imb_pre_mean->crashed (-0.368)', 'depth_imb_pre_mean->leverage_pre_max (0.200)']
