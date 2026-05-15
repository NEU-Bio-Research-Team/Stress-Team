# Phase 3 Run-Level Suite

## Data

- sim_panel: data/processed/tardis/phase2_outputs/phase2_canonical_tuned_legacy_500runs/lob_full_simulation_llm_tuned_legacy.csv
- floor_touch_rate_060: 1.0
- crashed=0,floor_touched=1: 412
- crashed=1,floor_touched=1: 88

## Main Causal

- pre_gap_ticks: 20
- trailing_pre_rows: 200
- node_preset: crash_only
- kept_nodes: ofi_pre_mean, depth_imb_pre_mean, leverage_pre_max, mean_wealth_pre_mean, wealth_concentration_pre_mean, crashed
- dagma_status: ok
- direct_lingam_status: ok
- dagma_outcome_edges: ['leverage_pre_max->crashed (0.059)']
- direct_lingam_outcome_edges: []

## Main Intervention

- pre_gap_ticks: 20
- trailing_pre_rows: 200
- feature_preset: market_plus_wealth
- intervention_mode: local_sd_clip_to_stable_median
- kept_features: ofi_pre_mean, depth_imb_pre_mean, leverage_pre_max, mean_wealth_pre_mean, wealth_concentration_pre_mean
- observational_rate_raw: 0.176
- observational_rate_pred: 0.1759840838664039
- auc_in_sample: 0.9835061782877317
- cv_auc_mean: 0.981676874883561
- cv_auc_std: 0.007240472247651819
- ofi_reduction_pct: 0.9824333553332552
- ofi_counterfactual: {'target_floor': 0.0021672802131483325, 'shift_sd': 0.14235549043461485}
- leverage_reduction_pct: 22.825192543883293
- leverage_counterfactual: {'target_floor': 1.0979017331799055, 'shift_sd': 0.108633159276476}
- severity_reduction_pct: 0.0005330001727144604

Group means by crashed:
- crashed=0: {'crashed': 0, 'ofi_pre_mean': 0.011218257696348176, 'depth_imb_pre_mean': 0.00044771300049254424, 'leverage_pre_max': 1.1114836847457867, 'mean_wealth_pre_mean': 175485.8196332096, 'wealth_concentration_pre_mean': 0.9455666732270648, 'crash_severity_pct': 50.001755430271864}
- crashed=1: {'crashed': 1, 'ofi_pre_mean': 0.26457862407636323, 'depth_imb_pre_mean': 0.010553035234810547, 'leverage_pre_max': 1.3446429431126312, 'mean_wealth_pre_mean': 179189.33237813384, 'wealth_concentration_pre_mean': 0.9128102216647279, 'crash_severity_pct': 50.00743339144453}

## Causal Ablations

- pre_gap=0, trailing=200, preset=crash_only: dagma_outcome_edges=[], direct_lingam_outcome_edges=[]
- pre_gap=20, trailing=200, preset=crash_only: dagma_outcome_edges=['leverage_pre_max->crashed (0.059)'], direct_lingam_outcome_edges=[]
- pre_gap=50, trailing=200, preset=crash_only: dagma_outcome_edges=['leverage_pre_max->crashed (0.064)'], direct_lingam_outcome_edges=[]

## Intervention Ablations

- pre_gap=0, trailing=200, preset=market_core: cv_auc_mean=0.9788295218175428, ofi_reduction_pct=6.83341982031775, leverage_reduction_pct=43.6360273183781, severity_reduction_pct=0.0010924165392374834
- pre_gap=0, trailing=200, preset=market_plus_wealth: cv_auc_mean=0.9804011838842698, ofi_reduction_pct=1.0629911559317242, leverage_reduction_pct=20.64239028245509, severity_reduction_pct=0.00043709791589739553
- pre_gap=20, trailing=200, preset=market_core: cv_auc_mean=0.9802282299931434, ofi_reduction_pct=5.720426678748163, leverage_reduction_pct=43.36669845248876, severity_reduction_pct=0.0011863004122898448
- pre_gap=20, trailing=200, preset=market_plus_wealth: cv_auc_mean=0.981676874883561, ofi_reduction_pct=0.9824333553332552, leverage_reduction_pct=22.825192543883293, severity_reduction_pct=0.0005330001727144604
- pre_gap=50, trailing=200, preset=market_core: cv_auc_mean=0.9807639933238453, ofi_reduction_pct=6.702993343178261, leverage_reduction_pct=42.069048502131814, severity_reduction_pct=0.0009721032330231739
- pre_gap=50, trailing=200, preset=market_plus_wealth: cv_auc_mean=0.9809743997510847, ofi_reduction_pct=2.495322458651806, leverage_reduction_pct=22.53822382330262, severity_reduction_pct=0.000195326638047927
