# Phase 3 Run-Level Suite

## Data

- sim_panel: /home/mluser/BRT-FDA/MinhQuang/Stress-Team/data/processed/tardis/phase2_outputs/phase2_canonical_tuned_legacy_500runs/lob_full_simulation_llm_tuned_legacy.csv
- floor_touch_rate_060: 1.0
- crashed=0,floor_touched=1: 412
- crashed=1,floor_touched=1: 88

## Main Causal

- pre_gap_ticks: 20
- node_preset: crash_only
- kept_nodes: ofi_pre_mean, spread_pre_mean, depth_imb_pre_mean, leverage_pre_max, kyle_lambda_pre_mean, mean_wealth_pre_mean, pct_insolvent_pre_max, wealth_concentration_pre_mean, crashed
- dagma_status: ok
- direct_lingam_status: ok
- dagma_outcome_edges: ['kyle_lambda_pre_mean->crashed (-1.773)', 'spread_pre_mean->crashed (0.954)', 'ofi_pre_mean->crashed (0.899)']
- direct_lingam_outcome_edges: []

## Main Intervention

- pre_gap_ticks: 20
- feature_preset: market_core
- kept_features: ofi_pre_mean, spread_pre_mean, depth_imb_pre_mean, leverage_pre_max, kyle_lambda_pre_mean
- observational_rate_raw: 0.176
- observational_rate_pred: 0.17600773952816112
- auc_in_sample: 1.0
- cv_auc_mean: 1.0
- cv_auc_std: 0.0
- ofi_reduction_pct: -29.67539256653518
- leverage_reduction_pct: 33.67424345719204
- severity_reduction_pct: 0.022575841274191846

Group means by crashed:
- crashed=0: {'crashed': 0, 'ofi_pre_mean': -2.256432170425704, 'spread_pre_mean': 3.544473934574763, 'depth_imb_pre_mean': -0.08843319953076245, 'leverage_pre_max': 1.7147296838216917, 'kyle_lambda_pre_mean': 0.6292769011213891, 'crash_severity_pct': 50.001755430271864}
- crashed=1: {'crashed': 1, 'ofi_pre_mean': -0.6376235162629237, 'spread_pre_mean': 5.3045015974165945, 'depth_imb_pre_mean': -0.024857489719546252, 'leverage_pre_max': 2.235666911029008, 'kyle_lambda_pre_mean': 0.6045834433776122, 'crash_severity_pct': 50.00743339144453}

## Causal Ablations

- pre_gap=0, preset=crash_only: dagma_outcome_edges=['kyle_lambda_pre_mean->crashed (-1.397)', 'leverage_pre_max->crashed (1.056)', 'spread_pre_mean->crashed (0.954)', 'ofi_pre_mean->crashed (0.895)'], direct_lingam_outcome_edges=[]
- pre_gap=20, preset=crash_only: dagma_outcome_edges=['kyle_lambda_pre_mean->crashed (-1.773)', 'spread_pre_mean->crashed (0.954)', 'ofi_pre_mean->crashed (0.899)'], direct_lingam_outcome_edges=[]
- pre_gap=50, preset=crash_only: dagma_outcome_edges=['kyle_lambda_pre_mean->crashed (-1.732)', 'spread_pre_mean->crashed (0.955)', 'wealth_concentration_pre_mean->crashed (0.716)'], direct_lingam_outcome_edges=[]
- pre_gap=20, preset=crash_plus_severity: dagma_outcome_edges=['kyle_lambda_pre_mean->crashed (-1.773)', 'spread_pre_mean->crashed (0.954)', 'ofi_pre_mean->crashed (0.899)'], direct_lingam_outcome_edges=['crash_severity_pct->crashed (0.356)']

## Intervention Ablations

- pre_gap=0, preset=market_core: cv_auc_mean=1.0, ofi_reduction_pct=-29.267614465550896, leverage_reduction_pct=40.385276079166665, severity_reduction_pct=0.017840430860072212
- pre_gap=0, preset=market_plus_wealth: cv_auc_mean=1.0, ofi_reduction_pct=-21.92175318153997, leverage_reduction_pct=17.223196815441668, severity_reduction_pct=0.012212528476781646
- pre_gap=20, preset=market_core: cv_auc_mean=1.0, ofi_reduction_pct=-29.67539256653518, leverage_reduction_pct=33.67424345719204, severity_reduction_pct=0.022575841274191846
- pre_gap=20, preset=market_plus_wealth: cv_auc_mean=1.0, ofi_reduction_pct=-19.130674118109646, leverage_reduction_pct=11.325009346642222, severity_reduction_pct=0.018696161050459237
- pre_gap=50, preset=market_core: cv_auc_mean=1.0, ofi_reduction_pct=-28.72388579403349, leverage_reduction_pct=28.258315650522224, severity_reduction_pct=0.026594966790770513
- pre_gap=50, preset=market_plus_wealth: cv_auc_mean=1.0, ofi_reduction_pct=-13.362476779102797, leverage_reduction_pct=8.024982759663153, severity_reduction_pct=0.020768302089815945
