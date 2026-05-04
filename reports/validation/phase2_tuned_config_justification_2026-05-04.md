# Phase 2 Tuned Configuration Justification (2026-05-04)

## Scope

This note records the provenance of the currently discussed Phase 2 tuned configuration and separates:

- empirically anchored parameters,
- calibration-derived parameters, and
- operational safeguards.

It is intended as the audit-trail companion for the current tuned 50-run benchmark.

## Canonical Tuned Command Under Review

```bash
python scripts/stage2_economics/18_lob_mini_runner.py \
  --scenario llm \
  --calibration-phase pre \
  --n-runs 50 \
  --seed 42 \
  --tick-ms 100 \
  --impact-scale 2.0 \
  --intensity-scale 1.2289 \
  --crash-window-ticks 10 \
  --crash-threshold-pct 1.93 \
  --base-order-size 0.10 \
  --drop-impact-mult 2.20 \
  --drop-sell-pressure 0.18 \
  --min-price-fraction 0.50 \
  --max-drop-ticks 5000 \
  --max-recovery-ticks 3000 \
  --max-post-ticks 2000 \
  --max-pre-ticks 2000 \
  --mm-vol-threshold-mult 1.4 \
  --mm-withdrawal-strength 1.8
```

## Summary Verdict

The tuned configuration is mostly auditable and scientifically defensible on five of the six main knobs.

The single unresolved parameter is `min_price_fraction=0.50`.

This parameter is not merely undocumented: it is actively binding in the current tuned 50-run result and therefore materially affects dynamics. It should not be described as empirically calibrated from BTC event data in its current form.

## Parameter Audit Table

| Parameter | Value | Status | Source / derivation | Notes |
|---|---:|---|---|---|
| `impact-scale` | 2.0 | Calibrated | Step-2 sweep in `phase2_crash_calibration_decision_2026-05-02.md` | Selected after bounded sweep with target crash-rate around 10% |
| `intensity-scale` | 1.2289 | Calibrated | Step-1 OFI p50 calibration in `phase2_crash_calibration_decision_2026-05-02.md` | Achieved OFI drop p50 within 7.7% of empirical target |
| `crash-window-ticks` | 10 | Empirical / design-locked | Locked detector definition | 10 ticks at 100 ms = 1 second |
| `crash-threshold-pct` | 1.93 | Empirical | Detector-consistent event mapping from `Event_Dynamics_100ms.csv` | Chosen to align with about 10% empirical event positives |
| `base-order-size` | 0.10 | Operational | Internal control under wealth cap | In practice constrained by `MAX_WEALTH_FRACTION = 0.05`; affects solvency path more than raw impact |
| `drop-impact-mult` | 2.20 | Empirically anchored | `realized_vol_drop / realized_vol_pre` from `prior_anchors.json` | Empirical ratio is about 2.44 to 2.48; 2.20 is a conservative approximation |
| `drop-sell-pressure` | 0.18 | Empirically anchored | OFI asymmetry from `prior_anchors.json` | `ofi_p50_pre = +0.003`, `ofi_p50_drop = -0.117` support stronger sell tilt in drop phase |
| `min-price-fraction` | 0.50 | Pending / unresolved | No direct empirical anchor currently documented | Active lower bound in current tuned simulation; see dedicated section below |

## Verified Empirical Anchors

### 1) `drop-impact-mult = 2.20`

From `data/processed/tardis/confounder_outputs/prior_anchors.json`:

- `realized_vol_per_phase.pre.mean = 0.00101183`
- `realized_vol_per_phase.drop.mean = 0.00250743`
- mean ratio = `0.00250743 / 0.00101183 = 2.478`

- `realized_vol_per_phase.pre.median = 0.00056551`
- `realized_vol_per_phase.drop.median = 0.0013788`
- median ratio = `0.0013788 / 0.00056551 = 2.438`

Therefore `drop-impact-mult = 2.20` sits about 10% below the empirical volatility ratio and can be described as a conservative stress-impact multiplier.

### 2) `drop-sell-pressure = 0.18`

From `data/processed/tardis/confounder_outputs/prior_anchors.json`:

- `ofi_percentiles_per_phase.pre.p50 = +0.003`
- `ofi_percentiles_per_phase.drop.p50 = -0.117`

This confirms that drop-phase order flow is empirically sell-skewed relative to pre-phase conditions.

### 3) `base-order-size = 0.10`

From `scripts/stage2_economics/18_lob_mini_runner.py`:

- `MAX_WEALTH_FRACTION = 0.05`
- `AGENT_INIT_WEALTH['noise_trader'] = 50_000`

At price around 58,000 USDT, the wealth cap is:

`(50,000 * 0.05) / 58,000 = 0.0431 BTC`

This means both `base_order_size=0.25` and `base_order_size=0.10` are typically clipped by the wealth cap for noise traders, so the main practical effect is on solvency and persistence rather than direct raw order-size impact.

## Unresolved Parameter: `min-price-fraction = 0.50`

### What real event data says

From `data/processed/tardis/confounder_outputs/Flash_Crash_Events_Labeled.csv`:

- `min(tick_bottom_price / tick_start_price) = 0.855502`
- `p01(bottom / start) = 0.867793`
- `p05(bottom / start) = 0.890684`
- `p10(bottom / start) = 0.904778`
- `max event drop = 14.4498%`

Across all 66 labeled events, the worst observed event bottom is still 85.55% of start price.

So `min-price-fraction = 0.50` is not an empirical lower bound. It is far below the observed event range.

### What current tuned simulation says

From `data/processed/tardis/phase2_outputs/lob_mini_simulation_llm_tuned_parallel_50.csv`, using per-run:

- `sim_min_close_frac = min(close) / first(close)`
- minimum across runs = `0.499835`
- p01 across runs = `0.499863`
- p05 across runs = `0.499904`
- all 50 runs fall below `0.70`
- 40 of 50 runs fall below `0.50` by tiny floating-point margin after normalization to first observed close

Interpretation:

- the tuned run is pressing directly against the 0.50 floor,
- therefore this parameter is active and materially shapes the simulation,
- it cannot be described as a harmless nonbinding safeguard.

## Practical Conclusion

The current tuned configuration is not fully locked for a main production run yet.

What is already solid:

- `impact-scale = 2.0`
- `intensity-scale = 1.2289`
- `crash-threshold-pct = 1.93`
- `crash-window-ticks = 10`
- `drop-impact-mult = 2.20`
- `drop-sell-pressure = 0.18`

What still needs an explicit decision:

- whether `min-price-fraction = 0.50` is an intentional stress-test guardrail,
- or whether it should be reset closer to the empirical event-bottom range and the crash-rate re-calibrated.

## Recommendation Before Main Run

Do not describe `min-price-fraction = 0.50` as empirically calibrated.

Use one of these two framings instead:

1. Conservative numerical safeguard:
   `min-price-fraction` is an operational floor added to prevent pathological collapse-to-zero, acknowledged as outside the empirical BTC event range.

2. Empirical floor option:
   Replace `0.50` with a floor tied to labeled event data (for example near `0.85` or a lower empirical quantile such as `0.89`) and rerun calibration.

Until that choice is made, the only outstanding scientific issue in the tuned Phase 2 configuration is the floor parameter.