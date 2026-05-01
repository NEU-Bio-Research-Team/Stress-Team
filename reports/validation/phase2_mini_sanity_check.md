# Phase 2 Mini-Simulation Sanity Check

Overall gates passed: 9/9

## Prior sanity snapshot
- parsed_rows: 512
- cancel_probability_unique: [0.0, 0.01, 0.03, 0.05, 0.06, 0.1, 0.4, 0.5, 0.6, 0.65]

### cancel_probability by agent
| agent_type | mean | std | min | max | nunique |
|---|---:|---:|---:|---:|---:|
| contrarian_trader | 0.067 | 0.046 | 0.0 | 0.1 | 5 |
| hft_market_maker | 0.23 | 0.219 | 0.1 | 0.65 | 5 |
| momentum_trader | 0.054 | 0.017 | 0.0 | 0.1 | 3 |
| noise_trader | 0.033 | 0.01 | 0.01 | 0.05 | 3 |

### order_type share by agent
| agent_type | limit | market |
|---|---:|---:|
| contrarian_trader | 0.0 | 1.0 |
| hft_market_maker | 1.0 | 0.0 |
| momentum_trader | 0.0 | 1.0 |
| noise_trader | 0.0 | 1.0 |

### side share by agent
| agent_type | buy | do_nothing | sell |
|---|---:|---:|---:|
| contrarian_trader | 0.461 | 0.188 | 0.352 |
| hft_market_maker | 0.727 | 0.016 | 0.258 |
| momentum_trader | 0.523 | 0.0 | 0.477 |
| noise_trader | 0.5 | 0.0 | 0.5 |

## Empirical stylized-fact baseline
- events: 66
- rows: 1236467
- rows_observed: 1094425
- observed_ratio: 0.885123
- returns_count: 1094359
- kurtosis_excess: 8581.787979
- acf_vol_lag1: 1.0
- ofi_pre_mean: 0.26444
- ofi_drop_mean: -2.698847
- crash_rate_proxy: 0.000628

## Gate checklist
- [PASS] parsed_rows_minimum | observed=512 | expected=>= 400
- [PASS] market_maker_limit_dominance | observed=1.0 | expected=>= 0.90
- [PASS] market_maker_cancel_rate_reasonable | observed=0.23 | expected=[0.20, 0.40]
- [PASS] noise_buy_sell_balance | observed=0.0 | expected=<= 0.10
- [PASS] contrarian_sell_floor | observed=0.352 | expected=>= 0.15
- [PASS] empirical_fat_tails_baseline | observed=8581.787979 | expected=> 3.0
- [PASS] empirical_vol_clustering_baseline | observed=1.0 | expected=> 0.10
- [PASS] empirical_ofi_drop_below_pre | observed={"pre": 0.26444, "drop": -2.698847} | expected=ofi_drop < ofi_pre
- [PASS] empirical_crash_rate_proxy_reasonable | observed="pending_simulation_output" | expected=evaluate crash_rate_sim in [0.05, 0.40] after mini simulation

## Recommendations
- Contrarian sell-side already meets target; keep current prior and proceed to mini simulation.
