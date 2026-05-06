# Phase 2 Status Report and Phase 3 Plan (2026-05-06)

## Executive Summary

Phase 2 is now in a usable state for downstream analysis.

The critical blocker from the previous round was the false zero-crash result from the canonical 500-run batch. That issue has been resolved by restoring the intended tuned-legacy behavior on the current runner. The corrected canonical 500-run panel now produces a non-degenerate crash regime, passes the core stylised-facts gates, and is the panel that should be treated as the authoritative Phase 2 output for subsequent work.

At the same time, Task 3 is only partially successful. The synthetic panel is now valid enough to run causal discovery and intervention scripts, but the current Task 3 outputs are not yet strong enough to support a clean causal claim in the paper. In particular, causal edge recovery is weak and the OFI intervention behaves in the wrong direction under the current modeling setup.

## What Was Fixed Before Finalizing Phase 2

### Root cause of the earlier zero-crash failure

The earlier canonical batch that returned `crash_event_count = 0` was not a real scientific result. It came from a behavior mismatch between:

- the historical tuned-legacy benchmark path from 2026-05-04, and
- the current runner implementation after resilience-damping logic was added.

The tuned-legacy benchmark under review predates the later resilience-damping mechanism in `18_lob_mini_runner.py`. On the current runner, leaving that damping active inside the canonical config suppressed drop-phase impact too strongly and collapsed the crash regime.

### Final canonical config state

The canonical Phase 2 config remains:

- `config/phase2_canonical_config.json`

To preserve tuned-legacy behavior on the current runner, it now explicitly fixes:

- `resilience_min_damp = 1.0`

This does not remove the known floor problem. It only ensures that the canonical config still reproduces the intended tuned-legacy regime instead of the broken zero-crash regime.

The known limitation remains documented in:

- `KNOWN_LIMITATIONS.md`

## Final Phase 2 Results

### Canonical 500-run output

Authoritative output path:

- `data/processed/tardis/phase2_outputs/phase2_canonical_tuned_legacy_500runs/lob_full_simulation_llm_tuned_legacy.csv`
- `data/processed/tardis/phase2_outputs/phase2_canonical_tuned_legacy_500runs/lob_full_summary_llm_tuned_legacy.json`

Final batch summary:

- target runs: `500`
- observed runs: `500`
- shard count: `20`
- crash event count: `88`
- flash crash rate: `0.176`
- top-up required: `False`

This means Task 4's `500-first` policy succeeded. Under the current crash-count rule, Phase 2 does not need a top-up to 750 or 1000 runs.

### Stylised-facts validation on the corrected full panel

Validation artifact:

- `reports/validation/phase2_stylised_facts_validation_from_current_llm.json`
- `reports/validation/phase2_stylised_facts_validation_from_current_llm.md`

Observed LLM metrics on the corrected 500-run panel:

- rows: `5,810,200`
- runs: `500`
- kurtosis excess: `29.0842`
- volatility ACF lag 1: `1.0`
- OFI pre mean: `0.1869`
- OFI drop mean: `-5.0391`
- crash rate: `0.176`

Gate outcome:

- all core gates pass: `True`

Interpretation:

- The corrected canonical panel is no longer scientifically blocked at the stylised-facts stage.
- It is strong enough to be used as the main synthetic panel for the next analysis steps.

### Important limitation that still remains

The Phase 2 canonical panel is usable, but it is not limitation-free.

The key remaining limitation is still the binding price floor:

- `min_price_fraction = 0.50`

This must continue to be treated as a structural regime boundary, not as an empirically calibrated BTC lower bound. Any analysis that claims mechanism or policy relevance from this panel should still distinguish floor-touched and floor-clean runs.

## What I Already Tried For Task 3

Task 3 in the approved plan is the Phase 3-style causal package built on top of the Phase 2 synthetic panel:

- causal discovery on the synthetic panel
- intervention analysis on the synthetic panel

I have already run both downstream scripts on the corrected 500-run canonical panel, not on the earlier broken zero-crash panel.

### 1. Causal discovery

Artifacts produced:

- `data/processed/tardis/phase2_outputs/causal_discovery_edges.csv`
- `reports/validation/phase2_causal_discovery.json`
- `reports/validation/phase2_causal_discovery.md`

What was tried:

- ran NOTEARS on the corrected panel
- ran LiNGAM on the corrected panel
- used the expected theoretical edge set already encoded in the script for comparison

What happened:

- both NOTEARS and LiNGAM were available and ran successfully
- rows used for discovery: `120000`
- NOTEARS status: `ok`
- LiNGAM status: `ok`

Current score quality:

- LiNGAM: precision `0.040`, recall `0.125`, matched edge only `spread_bps -> depth_imbalance`
- NOTEARS: precision `0.000`, recall `0.000`

Interpretation:

- the pipeline executes correctly,
- but the current causal recovery is weak,
- so Task 3 is computationally working but not yet scientifically persuasive.

### 2. Intervention analysis

Artifacts produced:

- `reports/validation/phase2_intervention_analysis.json`
- `reports/validation/phase2_intervention_analysis.md`

What was tried:

- built the current intervention model on the corrected 500-run panel
- evaluated `do(OFI = 0)`
- evaluated `do(leverage_proxy = 0)`

Observed outputs:

- samples used: `150000`
- logistic AUC: `0.9978`
- raw positive rate: `0.000533`

Counterfactual results:

- `do(OFI = 0)` changed predicted crash rate in the wrong direction
- reported relative reduction for `do(OFI = 0)`: `-51.69%`
- H3 target pass: `False`
- `do(leverage = 0)` nearly eliminated predicted crash risk
- reported relative reduction for `do(leverage = 0)`: `99.997%`

Interpretation:

- the current intervention script runs end-to-end,
- but the OFI intervention result is not usable as a paper claim,
- and the leverage intervention is likely too extreme to accept uncritically without further diagnostic work.

### 3. What Task 3 result means right now

Task 3 is not blocked by infrastructure anymore. It is blocked by model interpretation quality.

The current state is:

- Phase 2 synthetic data generation is now valid enough to feed Task 3.
- Task 3 scripts run successfully on the corrected panel.
- The present Task 3 outputs should be treated as exploratory diagnostics, not final claims.

## Phase 3 Plan From Here

The next Phase 3 plan should be pragmatic and ordered around what is already unblocked by the corrected 500-run panel.

### Phase 3 Priority 1 - Freeze the corrected canonical panel as the main baseline

Use the corrected 500-run LLM panel as the main reference panel for all immediate downstream work.

This means:

- do not use the earlier zero-crash canonical batch for any conclusion
- do not mix old broken outputs into new validation tables
- treat the current corrected 500-run panel as the authoritative Phase 2 baseline

### Phase 3 Priority 2 - Repair the Task 3 scientific interpretation layer

The next concrete job is not to rerun the same Task 3 scripts blindly. The next job is to make their outputs interpretable.

Recommended next steps:

1. Rebuild causal discovery on a more targeted panel slice.
2. Separate at least `pre`, `drop`, and `recovery` regimes instead of learning one DAG from the fully mixed panel.
3. Add floor-touch stratification before causal interpretation.
4. Re-check whether the expected edge set is recoverable on crash-proximate windows only.

Reason:

- the current full-panel mix likely dilutes the very crash mechanics that the DAG is supposed to recover.

### Phase 3 Priority 3 - Rework intervention modeling before using H3

The current intervention model is useful as a warning sign, not as final evidence.

Recommended next steps:

1. Move the intervention target from rare tick-level `flash_crash_flag` alone to a run-level or event-window outcome.
2. Add floor-touch controls so the intervention does not confound mechanism with hard-floor saturation.
3. Diagnose why `do(OFI = 0)` increases predicted crash risk under the current specification.
4. Re-estimate intervention effects after restricting the modeling window to crash-relevant regions.

Current conclusion for H3:

- H3 is not supported yet.

### Phase 3 Priority 4 - Lock the main Task 2 comparison table on the corrected baseline

The strongest next paper-facing deliverable is still the main prior comparison table:

- LLM prior
- Uniform prior
- Literature prior

This remains the cleanest route to support the main claim that LLM-elicited heterogeneous priors improve fidelity relative to simpler baselines.

This comparison is now complete on the corrected framework. The 500-run validation report shows:

- LLM kurtosis excess = 29.08
- Uniform kurtosis excess = 26.49
- Literature kurtosis excess = 77.20
- LLM crash rate = 0.176
- Uniform crash rate = 0.008
- Literature crash rate = 0.152

Interpretation:

- the LLM prior clearly dominates the Uniform prior on tail behavior while remaining in the target crash-rate band;
- the Literature prior produces much heavier tails than the corrected LLM baseline and is therefore not the preferred fidelity match under the current stylised-facts targets;
- the Uniform prior is now a weak baseline because it materially under-produces crashes.

Recommended execution order from here:

1. keep the corrected 500-run LLM baseline and the completed Uniform/Literature 500-run panels as the locked Task 2 comparison set
2. use the 500-run stylised-facts comparison as the main paper-facing evidence for the prior-ablation claim
3. only then extend causal/intervention comparison if the baseline comparison remains the stable narrative

### Phase 3 Priority 5 - Add robustness framing around the floor

Before strong causal claims are written, Phase 3 should explicitly report whether the discovered mechanisms are:

- present in all runs,
- concentrated in floor-touched runs only, or
- stable in floor-clean runs.

This is the most important robustness protection against reviewer criticism on the canonical tuned legacy panel.

## Practical Recommendation

At this point, the correct working interpretation is:

- Phase 2 is operationally complete for the corrected canonical LLM panel.
- The corrected panel is good enough to support Phase 3 experimentation.
- Task 3 has already been attempted and is running technically, but its current outputs are exploratory and need refinement before they can support the main paper claims.

## Immediate Next Actions

Recommended next sequence:

1. Keep the corrected 500-run canonical LLM panel as the locked Phase 2 baseline.
2. Keep the completed Uniform and Literature 500-run panels as the locked ablation baselines for Task 2.
3. Use the 500-run stylised-facts comparison as the main paper-facing comparison table before expanding the causal story.
4. Rework Task 3 on regime-sliced and floor-stratified data rather than the fully pooled panel.
5. Only promote causal/intervention results into paper claims after OFI intervention behavior is corrected or convincingly explained.