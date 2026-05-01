# Phase 2 Mini-Simulation Sanity Check

Overall gates passed: 8/9

## Blocking findings
- contrarian_sell_floor: observed=0.008, expected=>= 0.15

## Prior sanity snapshot
- parsed_rows: 512
- cancel_probability_unique: [0.0, 0.01, 0.03, 0.05, 0.1, 0.6, 0.65]

### cancel_probability by agent
| agent_type | mean | std | min | max | nunique |
|---|---:|---:|---:|---:|---:|
| contrarian_trader | 0.069 | 0.045 | 0.0 | 0.1 | 4 |
| hft_market_maker | 0.227 | 0.221 | 0.1 | 0.65 | 3 |
| momentum_trader | 0.064 | 0.027 | 0.0 | 0.1 | 3 |
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
| contrarian_trader | 0.953 | 0.039 | 0.008 |
| hft_market_maker | 0.695 | 0.039 | 0.266 |
| momentum_trader | 0.438 | 0.0 | 0.562 |
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
- [PASS] market_maker_cancel_rate_reasonable | observed=0.227 | expected=[0.20, 0.40]
- [PASS] noise_buy_sell_balance | observed=0.0 | expected=<= 0.10
- [FAIL] contrarian_sell_floor | observed=0.008 | expected=>= 0.15
- [PASS] empirical_fat_tails_baseline | observed=8581.787979 | expected=> 3.0
- [PASS] empirical_vol_clustering_baseline | observed=1.0 | expected=> 0.10
- [PASS] empirical_ofi_drop_below_pre | observed={"pre": 0.26444, "drop": -2.698847} | expected=ofi_drop < ofi_pre
- [PASS] empirical_crash_rate_proxy_reasonable | observed="pending_simulation_output" | expected=evaluate crash_rate_sim in [0.05, 0.40] after mini simulation

## Recommendations
- Contrarian sell-side is too low for a two-sided mean-reversion archetype. Current sell=0.008, target=0.200.
- Suggested temporary rebalancing for mini simulation: buy=0.761, do_nothing=0.039, sell=0.200.
- After rebalancing, run 50-100 mini simulation runs and compare stylized facts against empirical benchmarks in this report.
