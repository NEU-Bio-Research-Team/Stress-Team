# Phase 2 Causal Discovery

## Data

- sim_panel: data/processed/tardis/phase2_outputs/phase2_canonical_tuned_legacy_500runs/lob_full_simulation_llm_tuned_legacy.csv
- rows_raw: 5810200
- rows_after_floor_policy: 1463170
- rows_after_slice: 620
- rows_used: 620
- runs_total: 500
- crash_runs_total: 88
- runs_used: 9
- nodes: ofi, spread_bps, depth_imbalance, leverage_proxy, kyle_lambda, vpin, flash_crash_flag, mean_wealth_t, pct_insolvent, wealth_concentration
- phase_filter: drop
- crash_window_pre_ticks: 50
- crash_window_post_ticks: 20
- floor_policy: censor_rows
- floor_touched_run_rate: 1.0
- floor_touched_used_rate: 1.0
- notears_status: ok
- var_lingam_status: ok (sequences_used=9, skipped_short=0, failed=0)

## Expected Edges

- depth_imbalance -> flash_crash_flag
- leverage_proxy -> flash_crash_flag
- mean_wealth_t -> ofi
- ofi -> flash_crash_flag
- ofi -> spread_bps
- pct_insolvent -> spread_bps
- spread_bps -> depth_imbalance
- wealth_concentration -> leverage_proxy

## Method Scores

- notears: precision=0.000, recall=0.000, f1=0.000, shd=14, edges=8, matched=[]
  strongest=['ofi->depth_imbalance@lag0 (0.899)', 'leverage_proxy->depth_imbalance@lag0 (-0.885)', 'wealth_concentration->pct_insolvent@lag0 (0.746)', 'pct_insolvent->vpin@lag0 (-0.627)', 'vpin->depth_imbalance@lag0 (-0.335)', 'mean_wealth_t->vpin@lag0 (-0.302)', 'leverage_proxy->wealth_concentration@lag0 (0.224)', 'flash_crash_flag->leverage_proxy@lag0 (0.120)']
- varlingam: precision=0.080, recall=0.250, f1=0.121, shd=28, edges=25, matched=['mean_wealth_t->ofi', 'wealth_concentration->leverage_proxy']
  strongest=['wealth_concentration->leverage_proxy@lag0 (55.788)', 'wealth_concentration->depth_imbalance@lag1 (52.205)', 'mean_wealth_t->depth_imbalance@lag1 (-27.052)', 'mean_wealth_t->ofi@lag1 (-22.263)', 'mean_wealth_t->leverage_proxy@lag1 (-21.089)', 'wealth_concentration->vpin@lag0 (8.467)', 'wealth_concentration->mean_wealth_t@lag1 (0.889)', 'mean_wealth_t->vpin@lag1 (0.560)', 'ofi->leverage_proxy@lag0 (-0.539)', 'leverage_proxy->ofi@lag0 (-0.446)']
