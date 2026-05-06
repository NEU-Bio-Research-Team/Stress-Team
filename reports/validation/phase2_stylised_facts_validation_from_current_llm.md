# Phase 2 Stylised Facts Validation

## LLM Metrics

- rows: 5810200
- runs: 500
- kurtosis_excess: 29.08417701780441
- acf_vol_lag1: 1.0
- ofi_pre_mean: 0.18690711279890576
- ofi_drop_mean: -5.039080724277167
- crash_rate_sim: 0.176

## Ablation

- kurtosis_uniform: missing input
- kurtosis_literature: missing input
- h1_pass_kurtosis_llm_gt_uniform: None

## Gate Results

- kurtosis_excess: pass=True observed=29.08417701780441 target=> 3.0
- acf_vol_lag1: pass=True observed=1.0 target=> 0.10
- ofi_drop_less_than_pre: pass=True observed={'ofi_pre_mean': 0.18690711279890576, 'ofi_drop_mean': -5.039080724277167} target=ofi_drop_mean < ofi_pre_mean
- crash_rate_sim: pass=True observed=0.176 target=in [0.05, 0.40]

## Recommendations

- All stylized-fact gates pass for current thresholds
