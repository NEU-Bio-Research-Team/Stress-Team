# Phase 2 Stylised Facts Validation

## LLM Metrics

- rows: 578100
- runs: 50
- kurtosis_excess: 15.999627942018972
- acf_vol_lag1: 1.0
- ofi_pre_mean: 0.020444175203658002
- ofi_drop_mean: -0.05502005463185628
- crash_rate_sim: 0.0

## Ablation

- kurtosis_uniform: 11.617639079178907
- kurtosis_literature: 20.88545318157489
- h1_pass_kurtosis_llm_gt_uniform: True

## Gate Results

- kurtosis_excess: pass=True observed=15.999627942018972 target=> 3.0
- acf_vol_lag1: pass=True observed=1.0 target=> 0.10
- ofi_drop_less_than_pre: pass=True observed={'ofi_pre_mean': 0.020444175203658002, 'ofi_drop_mean': -0.05502005463185628} target=ofi_drop_mean < ofi_pre_mean
- crash_rate_sim: pass=False observed=0.0 target=in [0.05, 0.40]

## Recommendations

- crash_rate_sim < 0.05: increase sell pressure in drop phase or raise leverage amplification
