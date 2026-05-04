# Phase 2 Causal Discovery

## Data

- rows_used: 120000
- nodes: ofi, spread_bps, depth_imbalance, leverage_proxy, kyle_lambda, vpin, flash_crash_flag, mean_wealth_t, pct_insolvent, wealth_concentration
- notears_status: notears_unavailable: No module named 'notears'
- lingam_status: ok

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

- lingam: precision=0.000, recall=0.000, edges=12, matched=[]
