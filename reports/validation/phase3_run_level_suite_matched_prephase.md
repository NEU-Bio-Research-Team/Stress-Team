# Phase 3 Run-Level Suite

## Data

- sim_panel: /home/mluser/BRT-FDA/MinhQuang/Stress-Team/data/processed/tardis/phase2_outputs/phase2_canonical_tuned_legacy_500runs/lob_full_simulation_llm_tuned_legacy.csv
- floor_touch_rate_060: 1.0
- crashed=0,floor_touched=1: 412
- crashed=1,floor_touched=1: 88

## Main Causal

- pre_gap_ticks: 20
- node_preset: crash_only
- kept_nodes: ofi_pre_mean, depth_imb_pre_mean, leverage_pre_max, mean_wealth_pre_mean, wealth_concentration_pre_mean, crashed
- dagma_status: ok
- direct_lingam_status: ok
- dagma_outcome_edges: []
- direct_lingam_outcome_edges: []

## Main Intervention

- pre_gap_ticks: 20
- feature_preset: market_core
- kept_features: ofi_pre_mean, depth_imb_pre_mean, leverage_pre_max
- observational_rate_raw: 0.176
- observational_rate_pred: 0.1759944189103194
- auc_in_sample: 0.9802791262135921
- cv_auc_mean: 0.9807720600239312
- cv_auc_std: 0.006404268111693443
- ofi_reduction_pct: 61.10733671390077
- leverage_reduction_pct: 99.3805009516132
- severity_reduction_pct: 0.02238205097475459

Group means by crashed:
- crashed=0: {'crashed': 0, 'ofi_pre_mean': 0.12482316515037527, 'depth_imb_pre_mean': 0.004978849958610323, 'leverage_pre_max': 1.1620869536434986, 'crash_severity_pct': 50.001755430271864}
- crashed=1: {'crashed': 1, 'ofi_pre_mean': 0.48630058339025006, 'depth_imb_pre_mean': 0.019370603043855874, 'leverage_pre_max': 1.6072402479912293, 'crash_severity_pct': 50.00743339144453}

## Causal Ablations

- pre_gap=0, preset=crash_only: dagma_outcome_edges=[], direct_lingam_outcome_edges=[]
- pre_gap=20, preset=crash_only: dagma_outcome_edges=[], direct_lingam_outcome_edges=[]
- pre_gap=50, preset=crash_only: dagma_outcome_edges=[], direct_lingam_outcome_edges=[]

## Intervention Ablations

- pre_gap=0, preset=market_core: cv_auc_mean=0.9807783021132833, ofi_reduction_pct=61.38446451184273, leverage_reduction_pct=99.33770855002949, severity_reduction_pct=0.021049959157782436
- pre_gap=0, preset=market_plus_wealth: cv_auc_mean=0.9804636047777912, ofi_reduction_pct=26.590289084787944, leverage_reduction_pct=78.31174575399467, severity_reduction_pct=0.02351715281171294
- pre_gap=20, preset=market_core: cv_auc_mean=0.9807720600239312, ofi_reduction_pct=61.10733671390077, leverage_reduction_pct=99.3805009516132, severity_reduction_pct=0.02238205097475459
- pre_gap=20, preset=market_plus_wealth: cv_auc_mean=0.9800383744446941, ofi_reduction_pct=25.66293866121902, leverage_reduction_pct=78.4042390888416, severity_reduction_pct=0.024887517572468525
- pre_gap=50, preset=market_core: cv_auc_mean=0.9807783021132833, ofi_reduction_pct=61.10337224314117, leverage_reduction_pct=99.41157934366795, severity_reduction_pct=0.021921940754716235
- pre_gap=50, preset=market_plus_wealth: cv_auc_mean=0.9797673717346671, ofi_reduction_pct=25.298973218573913, leverage_reduction_pct=78.62163012546978, severity_reduction_pct=0.02436987367026676
