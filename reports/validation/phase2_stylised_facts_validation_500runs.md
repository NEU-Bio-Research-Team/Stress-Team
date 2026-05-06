# Phase 2 Stylised Facts Validation

## Scenario Comparison

| scenario | rows | runs | kurtosis_excess | acf_vol_lag1 | ofi_pre_mean | ofi_drop_mean | crash_rate_sim |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LLM | 5810200 | 500 | 29.084177 | 1.000000 | 0.186907 | -5.039081 | 0.176000 |
| Uniform | 5804400 | 500 | 26.492039 | 1.000000 | 0.115426 | -2.911987 | 0.008000 |
| Literature | 5810600 | 500 | 77.195527 | 1.000000 | 0.184757 | -2.881011 | 0.152000 |

## Ablation

- h1_pass_kurtosis_llm_gt_uniform: True
- kurtosis_ratio_llm_over_uniform: 1.097846
- kurtosis_ratio_llm_over_literature: 0.376760
- kurtosis_gap_llm_minus_uniform: 2.592138
- kurtosis_gap_llm_minus_literature: -48.111350
- crash_rate_gap_llm_minus_uniform: 0.168000
- crash_rate_gap_llm_minus_literature: 0.024000

## Gate Results

- kurtosis_excess: pass=True observed=29.08417701780441 target=> 3.0
- acf_vol_lag1: pass=True observed=1.0 target=> 0.10
- ofi_drop_less_than_pre: pass=True observed={'ofi_pre_mean': 0.18690711279890576, 'ofi_drop_mean': -5.039080724277167} target=ofi_drop_mean < ofi_pre_mean
- crash_rate_sim: pass=True observed=0.176 target=in [0.05, 0.40]

## Recommendations

- All stylized-fact gates pass for current thresholds
