# Phase 2 — Task 1 Validation Report: LOB Mini Simulation (floor=0.85)

**Date**: 2026-05-03  
**Scenario**: `llm`, calibration phase `pre`  
**Parameter set**: `floor085` — `min-price-fraction=0.85`  
**Runs**: 3 per shard × 10 shards = **30 total runs**  
**Pass threshold**: flash-crash rate **< 20%** per group

---

## Simulation Parameters (common)

| Parameter | Value |
|---|---|
| `--tick-ms` | 100 |
| `--impact-scale` | 2.0 |
| `--intensity-scale` | 1.2289 |
| `--crash-window-ticks` | 10 |
| `--crash-threshold-pct` | 1.93 % |
| `--base-order-size` | 0.10 |
| `--drop-sell-pressure` | 0.18 |
| `--min-price-fraction` | **0.85** |
| `--max-drop-ticks` | 5000 |
| `--max-recovery-ticks` | 3000 |
| `--max-post-ticks` | 2000 |
| `--max-pre-ticks` | 2000 |
| `--mm-vol-threshold-mult` | 1.4 |
| `--mm-withdrawal-strength` | 1.8 |

---

## Per-Shard Results

| Shard | Group | Seed | Crash rate | DD mean % | DD p95 % | DD max % | Pct insolvent | Wealth conc drop |
|---|---|---|---|---|---|---|---|---|
| dim300_r3_s111 | 3.00× | 111 | 0.0000 | 0.3206 | 0.4526 | 0.4574 | 0.050 | 0.9646 |
| dim300_r3_s112 | 3.00× | 112 | 0.0000 | 0.4052 | 0.8603 | 0.9365 | 0.050 | 0.9702 |
| dim300_r3_s113 | 3.00× | 113 | 0.0000 | 0.9198 | 1.5038 | 1.5926 | 0.050 | 1.0434 |
| dim300_r3_s114 | 3.00× | 114 | 0.0000 | 0.4830 | 0.9013 | 0.9662 | 0.030 | 0.9439 |
| dim300_r3_s115 | 3.00× | 115 | 0.0000 | 0.3577 | 0.7467 | 0.8121 | 0.030 | 0.9573 |
| dim600_r3_s211 | 6.00× | 211 | 0.0000 | 0.6931 | 1.0943 | 1.1595 | 0.020 | 1.0370 |
| dim600_r3_s212 | 6.00× | 212 | 0.0000 | 0.6212 | 0.7714 | 0.7851 | 0.020 | 1.0329 |
| dim600_r3_s213 | 6.00× | 213 | 0.0000 | 0.3336 | 0.3901 | 0.3981 | 0.000 | 0.9877 |
| dim600_r3_s214 | 6.00× | 214 | 0.0000 | 0.5027 | 0.6149 | 0.6316 | 0.010 | 1.0006 |
| **dim600_r3_s215** | **6.00×** | **215** | **0.3333** | **1.6051** | **2.7947** | **2.9245** | **0.090** | **1.0925** |

---

## Group-Level Summary

### dim300 — `drop-impact-mult = 3.00` (seeds 111–115)

| Metric | Value |
|---|---|
| Crash rate (shard avg) | **0.0000** |
| Shards with any crash | 0 / 5 |
| DD mean (avg across shards) | 0.4972 % |
| DD max (worst shard) | 1.5926 % |
| Pct insolvent peak (avg) | 4.20 % |
| Wealth conc drop (avg) | 0.9759 |
| **VERDICT** | ✅ **PASS** (crash rate 0.00% < 20%) |

### dim600 — `drop-impact-mult = 6.00` (seeds 211–215)

| Metric | Value |
|---|---|
| Crash rate (shard avg) | **0.0667** |
| Shards with any crash | 1 / 5 |
| DD mean (avg across shards) | 0.7511 % |
| DD max (worst shard) | 2.9245 % (seed 215) |
| Pct insolvent peak (avg) | 2.80 % |
| Wealth conc drop (avg) | 1.0302 |
| **VERDICT** | ✅ **PASS** (crash rate 6.67% < 20%) |

### Overall

| Metric | Value |
|---|---|
| Overall crash rate | **0.0333** (1/30 runs) |
| Shards with crash | 1 / 10 |
| **OVERALL VERDICT** | ✅ **PASS** |

---

## Notes on the Single Crash (seed 215, run 1)

- **Event**: `event_id=57` — a very-low starting-price BTC snapshot (~\$12,400 pre-crash).
- **Mechanism**: With `drop-impact-mult=6.00`, the 6× amplified sell impact drove price immediately to the `min-price-fraction=0.85` floor (~\$10,409) within the first 250 ticks of the drop phase.
- **Floor binding**: The floor held throughout; price never breached 0.85 × initial.
- **Insolvent fraction peaked at 9.0%** — the highest across all 30 runs, consistent with the very low absolute price level creating leveraged-agent insolvency faster.
- **DD**: 2.9245% over a 10-tick window — the only run that exceeded the 1.93% crash threshold.
- **Interpretation**: This crash is a tail event driven by the combination of an extreme low-price starting point (event 57 is an outlier in the historical dataset) and the strongest drop multiplier (6.00×). It does **not** indicate a systemic instability in the calibration.

---

## Floor-Binding Verification

The `min-price-fraction=0.85` floor was observed to be binding in multiple runs across both groups:

| Shard | Event | Initial price | Floor price | Observed floor |
|---|---|---|---|---|
| s111 run3 | 12 | ~98,376 | ~83,620 | 83,619.60 ✅ |
| s112 run1 | 12 | ~98,376 | ~83,620 | 83,619.60 ✅ |
| s113 run1 | 19 | ~22,651 | ~19,253 | 19,239.15 ✅ |
| s113 run2 | 1  | ~11,972 | ~10,176 | 10,176.20 ✅ |
| s114 run1 | 10 | ~43,171 | ~36,695 | 36,695.18 ✅ |
| s215 run1 | 57 | ~12,247 | ~10,410 | 10,409.51 ✅ |

Floor enforcement is operating correctly.

---

## Output Files

All outputs written to `data/processed/tardis/phase2_outputs/`:

| File type | Pattern | Count |
|---|---|---|
| Tick-level panel CSV | `lob_mini_simulation_llm_floor085_task1res_dim{300,600}_r3_s{111–115,211–215}.csv` | 10 |
| Summary JSON | `lob_mini_summary_llm_floor085_task1res_dim{300,600}_r3_s{111–115,211–215}.json` | 10 |
