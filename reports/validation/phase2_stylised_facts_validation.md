# Phase 2 Stylised Facts Validation

## LLM Metrics

- rows: 581300
- runs: 50
- kurtosis_excess: 12.36941759054868
- acf_vol_lag1: 1.0
- ofi_pre_mean: 0.0830266272663147
- ofi_drop_mean: -0.5298376095794275
- crash_rate_sim: 0.0

## Ablation

- kurtosis_uniform: 12.174634228083361
- kurtosis_literature: 11.962762196626354
- h1_pass_kurtosis_llm_gt_uniform: True

## Gate Results

- kurtosis_excess: pass=True observed=12.36941759054868 target=> 3.0
- acf_vol_lag1: pass=True observed=1.0 target=> 0.10
- ofi_drop_less_than_pre: pass=True observed={'ofi_pre_mean': 0.0830266272663147, 'ofi_drop_mean': -0.5298376095794275} target=ofi_drop_mean < ofi_pre_mean
- crash_rate_sim: pass=False observed=0.0 target=in [0.05, 0.40]

## Recommendations

- crash_rate_sim < 0.05: increase sell pressure in drop phase or raise leverage amplification
