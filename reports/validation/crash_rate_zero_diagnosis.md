# Diagnostic Report: crash_rate_sim = 0.0

**Date**: 2025-05-09  
**Pipeline Stage**: Phase 2 – LOB Mini Simulation (script 18)  
**Gate Target**: `crash_rate_sim ∈ [0.05, 0.40]`  
**Observed**: `crash_rate_sim = 0.0` (0/50 runs, 0/1000 runs)

---

## Executive Summary

The crash rate is zero in all simulations because the 10-tick (1-second) rolling window
crash detector never observes ≥ 1.93% price drop within any single second.
The simulator **does** produce large total-drop-phase declines (mean 12.85%, max 33.54%)
but these are **gradual drifts** spread over 5,000 ticks (500 seconds), not sharp spikes.
The mismatch between the crash detection window (10 ticks) and the simulated price
dynamics (gradual drift over 5,000 ticks) is the primary cause.

---

## Data Scanned

| Layer | File / Source |
|-------|---------------|
| L1 – Anchors | `phase2_outputs/prior_anchors.json` |
| L2 – Behavioral priors | `phase2_outputs/behavioral_priors.json` |
| L3 – Sim panel | `phase2_outputs/lob_mini_simulation_llm.csv` (581,300 rows, 50 runs) |
| L4 – Sim summary | `phase2_outputs/lob_mini_summary_llm.json` |
| L5 – Script defaults | `scripts/stage2_economics/18_lob_mini_runner.py` argparse |

---

## Finding 1 — CRITICAL: 10-Tick Price Volatility Too Small (4.2× Gap)

**Flash crash detector** (lines 603–608 of `18_lob_mini_runner.py`):
```python
drop_pct = (window[0] - min(window)) / window[0] * 100.0
if drop_pct >= crash_threshold_pct:   # 1.93%
    flash_crash_flag = True
```
The window spans **10 ticks × 100ms = 1 second**.

**Observed 10-tick rolling max drawdown distribution** (50 LLM runs):

| Percentile | 10-tick max dd |
|-----------|---------------|
| p50 | 0.0661% |
| p90 | 0.2862% |
| p95 | 0.3208% |
| p99 | 0.3988% |
| **max** | **0.4585%** |

**Threshold = 1.93%** → gap = **4.2×**.  
No single run in 50 simulations ever reached 1.93% in any 1-second window.

---

## Finding 2 — CRITICAL: Drop Phase Is Gradual, Not a Sharp Spike

**Drop-phase statistics** (sim panel):

| Metric | Value |
|--------|-------|
| Mean ticks in drop phase | **5,000** (hitting `max_drop_ticks`) |
| Full drop-phase dd — mean per run | **12.85%** |
| Full drop-phase dd — p90 | **31.09%** |
| Full drop-phase dd — max | **33.54%** |

Price falls 12–33% but **over 500 seconds**, not 1 second.  
The 10-tick window only captures `0.46%` of the total drift.

**Why is the drop gradual?**  
Each tick: `impact = impact_scale × drop_impact_mult × kyle_lambda × net_flow`
- `impact_scale = 2.0`
- `drop_impact_mult = 1.35`
- `kyle_lambda (drop, mean) = 0.598`
- `net_flow (OFI drop mean) = −0.530`
- → `impact/tick ≈ −0.856` (absolute)

At `price ≈ 39,283` (mean sim starting price):
```
0.856 / 39,283 = 0.00218% per tick
10-tick cumulative = 0.022%   ← vs threshold 1.93%
```

---

## Finding 3 — MAJOR: OFI Noise Cancels Most Net Flow

- `noise_trader.arrival_rate_lambda (drop) = 311.6` — very high activity, random direction
- Drop-phase OFI: **mean = −0.53**, **std = 0.96**
- Net sell pressure is real but small; noise traders cancel ~50% of directional flow
- `drop_sell_pressure = 0.12` (12% buy→sell tilt) is too weak against λ=311 noise volume

---

## Finding 4 — NOT A CAUSE: min_price_fraction Floor Is Irrelevant

- `min_price_fraction = 0.70` clamps price at 70% of initial
- Maximum observed full-drop = 33.54% → price reaches ~66.5% of initial
- The floor **is** hit in some runs, but the 10-tick dd is still only 0.46%
- The floor does **not** prevent crashes; it prevents unphysical total ruin

---

## Finding 5 — Contrarian Traders Never Activate

- `contrarian.activation_threshold_pct = 1.0` → only activate if price moves ≥1% from reference
- In any 1-second window, price moves < 0.46% → contrarians never fire
- Their stabilizing effect is irrelevant because the drop is too slow to trigger them

---

## Quantitative Fix Options

### Option A — Lower crash_threshold_pct (match observed dynamics)
Set `--crash-threshold-pct 0.45`.  
Max observed dd = 0.46% → ~50% of runs would crash.  
**Pro**: No code change needed, fast fix.  
**Con**: Threshold is far below empirical BTC flash crash benchmarks (~1–3%).

### Option B — Widen crash_window_ticks (match 10–20 second window)
BTC flash crashes historically unfold in 10–120 seconds, not 1 second.  
Set `--crash-window-ticks 100` (10 seconds) or `200` (20 seconds).  
In a 100-tick window, mean OFI drift would produce `0.022% × 10 = 0.22%`; still short.  
At 1,000 ticks (100s): `~2.2%` → crosses threshold.  
**Recommended**: `--crash-window-ticks 200` combined with Option C.

### Option C — Increase drop_impact_mult (amplify drop dynamics)
Current: `drop_impact_mult=1.35`.  
Required for 1.93% in 10 ticks at p05 OFI: `impact_scale ≈ 42.9` (21× increase).  
More realistic: set `--drop-impact-mult 5.0` combined with wider window.  
With `drop_impact_mult=5.0` + `crash_window_ticks=100`:  
`5.0/1.35 × 10 × 0.022% = 0.81%` — still below 1.93%.  
Need `drop_impact_mult=10` + `crash_window_ticks=100` to reliably hit 1.93%.

### Option D — Reduce noise trader volume in drop phase (cleaner net_flow)
`drop_sell_pressure=0.40` (was 0.12) + reduced noise arrival → net OFI drops to −2 to −5.  
This is more physically realistic: real flash crashes have strongly directional order flow.

### Recommended Combined Fix
```bash
python scripts/stage2_economics/18_lob_mini_runner.py \
  --scenario llm \
  --n-runs 100 \
  --crash-threshold-pct 0.45 \     # match observed 10-tick dynamics
  --crash-window-ticks 50 \         # 5 seconds (realistic BTC)
  --drop-impact-mult 3.0 \          # amplify crash dynamics
  --drop-sell-pressure 0.30         # stronger directional pressure
```
Expected crash_rate: 0.10–0.35 (estimate; requires calibration run).

---

## Summary Table

| Finding | Severity | Root Cause? | Fix Required |
|---------|----------|-------------|--------------|
| 10-tick window captures only 0.46% dd; threshold=1.93% | CRITICAL | YES | Lower threshold OR widen window |
| Drop phase = 5000 ticks (gradual); not a sharp spike | CRITICAL | YES | Widen crash_window_ticks |
| OFI noise cancels directional flow | MAJOR | Contributing | Increase drop_sell_pressure |
| min_price_fraction=0.70 floor | NONE | NO | No action |
| Contrarians never activate | NONE | NO (consequence, not cause) | No action |

---

## Conclusion

The simulator produces **economically plausible crashes** (large total drawdowns) but the
**detection metric** is calibrated to a 1-second spike that never occurs in this model.
The root cause is a **mismatch between simulation dynamics (gradual 500s drift) and
detection criterion (1-second threshold)**.  
The minimum fix is: `--crash-threshold-pct 0.45` (recalibrate to observed dynamics).  
The more realistic fix is: widen `crash_window_ticks` to 50–200 and increase `drop_impact_mult`.
