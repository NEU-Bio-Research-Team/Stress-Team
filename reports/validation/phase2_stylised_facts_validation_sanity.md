# Phase 2 Stylised Facts Validation

## LLM Metrics

- rows: 35250
- runs: 3
- kurtosis_excess: 4.590799528341342
- acf_vol_lag1: 1.0
- ofi_pre_mean: 0.013040299623220635
- ofi_drop_mean: -0.030599514828818078
- crash_rate_sim: 0.0

## Ablation

- kurtosis_uniform: missing input
- kurtosis_literature: missing input
- h1_pass_kurtosis_llm_gt_uniform: None

## Gate Results

- kurtosis_excess: pass=True observed=4.590799528341342 target=> 3.0
- acf_vol_lag1: pass=True observed=1.0 target=> 0.10
- ofi_drop_less_than_pre: pass=True observed={'ofi_pre_mean': 0.013040299623220635, 'ofi_drop_mean': -0.030599514828818078} target=ofi_drop_mean < ofi_pre_mean
- crash_rate_sim: pass=False observed=0.0 target=in [0.05, 0.40]

## Recommendations

- crash_rate_sim < 0.05: increase sell pressure in drop phase or raise leverage amplification
