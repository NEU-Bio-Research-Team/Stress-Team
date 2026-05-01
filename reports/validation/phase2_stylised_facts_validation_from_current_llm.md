# Phase 2 Stylised Facts Validation

## LLM Metrics

- rows: 19316282
- runs: 50
- kurtosis_excess: 19.94777997224679
- acf_vol_lag1: 1.0
- ofi_pre_mean: 0.0040684732322267755
- ofi_drop_mean: 0.031419050500512684
- crash_rate_sim: 0.0

## Ablation

- kurtosis_uniform: missing input
- kurtosis_literature: missing input
- h1_pass_kurtosis_llm_gt_uniform: None

## Gate Results

- kurtosis_excess: pass=True observed=19.94777997224679 target=> 3.0
- acf_vol_lag1: pass=True observed=1.0 target=> 0.10
- ofi_drop_less_than_pre: pass=False observed={'ofi_pre_mean': 0.0040684732322267755, 'ofi_drop_mean': 0.031419050500512684} target=ofi_drop_mean < ofi_pre_mean
- crash_rate_sim: pass=False observed=0.0 target=in [0.05, 0.40]

## Recommendations

- crash_rate_sim < 0.05: increase sell pressure in drop phase or raise leverage amplification
- OFI phase ordering invalid: verify phase-conditioned side probabilities and Poisson rates
