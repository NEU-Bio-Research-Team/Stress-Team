# COMOSA Master Report: Phase 1 To Phase 3

Updated: 2026-05-07

This document is the current end-to-end report for the market-only COMOSA workspace. It is written for readers who have not worked in this repository before and need one place to understand:

- what was done in each phase,
- how the pipeline works,
- which inputs and parameters were used,
- which outputs were produced,
- what the main experiments and ablations showed,
- and how reliable each result currently is.

For repository layout, file-level workflow commands, and the broader project map, see [README.md](../README.md).

When numbers in older notes disagree, prefer this report plus the linked validation artifacts over older benchmark notes.

## 1. Executive Summary

The workspace implements a three-phase research pipeline for BTC flash-crash simulation with LLM-informed heterogeneous agents.

| Phase | Main purpose | Current status | Reliability | Bottom line |
| --- | --- | --- | --- | --- |
| Phase 1 | Build behavioral priors from LLM elicitation anchored to empirical BTC event data | Completed | Moderate | The elicitation pipeline ran successfully and produced usable priors for all 4 archetypes across 4 phases. |
| Phase 2 | Run the LOB simulator and compare LLM priors against baseline prior families | Completed | Moderate to strong | The corrected canonical 500-run panel is the strongest current result in the workspace. |
| Phase 3 | Learn causal structure and estimate interventions from the synthetic panel | Partially successful | Exploratory to moderate | The original tick-level approach was weak; the run-level rewrite is usable for cross-run risk structure, but not yet for strong within-event causal claims. |

The project-level conclusion is:

1. Phase 1 is real, not hypothetical. The repository contains checked-in outputs from a full local Mistral elicitation run.
2. Phase 2 is the main evidence layer. The LLM prior clearly beats the uniform baseline on crash generation and tail behavior while staying inside the target crash-rate band.
3. Phase 3 is now operational in run-level form, but it should be framed as cross-run crash-risk structure rather than definitive within-event causal mechanism recovery.

## 2. End-To-End Workflow

The research logic is:

1. Build empirical flash-crash features and prior anchors from real BTC event data.
2. Use those anchors to prompt a local LLM for agent behavior.
3. Fit parametric priors from the elicited responses.
4. Feed those priors into a heterogeneous-agent LOB simulator.
5. Validate whether the simulated panel reproduces stylized facts.
6. Compare the LLM prior against simpler baselines.
7. Use the simulated panel for downstream causal and intervention analysis.

In concrete repository terms:

1. Empirical event outputs and anchors are produced under [data/processed/tardis/confounder_outputs](../data/processed/tardis/confounder_outputs).
2. Phase 1 elicitation outputs are produced under [data/processed/tardis/phase1_outputs](../data/processed/tardis/phase1_outputs).
3. Phase 2 simulation outputs are produced under [data/processed/tardis/phase2_outputs](../data/processed/tardis/phase2_outputs).
4. Validation and interpretation reports are written under [reports/validation](../reports/validation).

## 3. Project Inputs And Shared Assumptions

The whole workflow depends on the following global design choices.

| Item | Value |
| --- | --- |
| Market | BTCUSDT perpetual futures |
| Main time resolution | 100 ms |
| Empirical event set | 66 BTC flash-crash events |
| Agent archetypes | momentum trader, contrarian trader, HFT market maker, noise trader |
| Main synthetic baseline | corrected canonical tuned-legacy 500-run LLM panel |
| Main known structural limitation | binding price floor at `min_price_fraction = 0.50` in the canonical Phase 2 config |

The main code and artifact entry points are:

- [README.md](../README.md)
- [reports/DATA_PIPELINE.md](DATA_PIPELINE.md)
- [scripts/stage2_economics/phase1_llm_elicitation/RUNBOOK.md](../scripts/stage2_economics/phase1_llm_elicitation/RUNBOOK.md)
- [scripts/stage2_economics/PHASE2_RUNBOOK.md](../scripts/stage2_economics/PHASE2_RUNBOOK.md)

## 4. Phase 1: LLM Elicitation Of Behavioral Priors

### 4.1 Goal

Phase 1 converts empirical event-level BTC flash-crash features into agent-behavior priors by prompting a local instruction model and fitting parametric distributions to the returned parameters.

### 4.2 What Was Done

The implemented Phase 1 pipeline is documented in [scripts/stage2_economics/phase1_llm_elicitation/RUNBOOK.md](../scripts/stage2_economics/phase1_llm_elicitation/RUNBOOK.md) and consists of four steps:

1. Generate structured prompts from real event-level anchors.
2. Run local model inference over those prompts.
3. Parse the returned JSON responses into tabular parameters.
4. Fit distributions that become the behavioral priors for Phase 2.

The relevant scripts are:

- [13_write_prompts.py](../scripts/stage2_economics/phase1_llm_elicitation/13_write_prompts.py)
- [14_run_inference.py](../scripts/stage2_economics/phase1_llm_elicitation/14_run_inference.py)
- [15_extract_parameters.py](../scripts/stage2_economics/phase1_llm_elicitation/15_extract_parameters.py)
- [16_fit_distributions.py](../scripts/stage2_economics/phase1_llm_elicitation/16_fit_distributions.py)

### 4.3 Main Inputs

| Input | Role |
| --- | --- |
| [prior_anchors.json](../data/processed/tardis/confounder_outputs/prior_anchors.json) | empirical anchor statistics per phase |
| [Event_Dynamics_100ms_gridded.csv](../data/processed/tardis/confounder_outputs/Event_Dynamics_100ms_gridded.csv) | event-level sampled market states |
| [prompt_details](../prompt_details) | archetype prompts and specifications |
| [models/mistral-7b-instruct](../models/mistral-7b-instruct) | local model checkpoint |

### 4.4 Key Runtime Parameters

The full-run configuration in the runbook uses:

| Parameter | Value |
| --- | --- |
| Environment | `comosa_phase1` |
| Python | 3.11 |
| Model | `mistral-7b-instruct` |
| Backend | `vllm` |
| Batch size | 32 |
| Max retries | 3 |
| Temperature | 0.8 |
| Prompt count | 512 |

### 4.5 Main Outputs

The final output contract is:

- [phase1_prompts.json](../data/processed/tardis/phase1_outputs/phase1_prompts.json)
- [raw_elicited.json](../data/processed/tardis/phase1_outputs/raw_elicited.json)
- [raw_elicited.csv](../data/processed/tardis/phase1_outputs/raw_elicited.csv)
- [behavioral_priors.json](../data/processed/tardis/phase1_outputs/behavioral_priors.json)

### 4.6 Quantitative Results

The checked-in Phase 1 outputs show:

| Metric | Value |
| --- | --- |
| Prompt records | 512 |
| Raw elicited records | 512 |
| Parsed rows in fitted priors | 512 |
| Agent archetypes | 4 |
| Market phases | 4 |
| Agent-phase groups | 16 |
| Fit backend | scipy |

The fitted prior metadata is recorded in [behavioral_priors.json](../data/processed/tardis/phase1_outputs/behavioral_priors.json).

An example fitted group is momentum trader, pre phase:

| Parameter | Distribution | Fitted value |
| --- | --- | --- |
| aggressiveness | Beta | alpha = 139.133792, beta = 61.508124 |
| cancel_probability | Beta | alpha = 18.861634, beta = 325.808561 |
| order_size_multiplier | LogNormal | shape = 0.174157, scale = 2.790633 |
| inventory_sensitivity_vg | Gamma | shape = 100.0, scale = 0.005 |
| order_type_market_fraction | Scalar | 1.0 |

The sanity gate summary in [phase2_mini_sanity_check.md](validation/phase2_mini_sanity_check.md) reports 9/9 gates passed. Representative checks include:

| Check | Result |
| --- | --- |
| parsed rows minimum | PASS |
| market-maker limit dominance | PASS |
| market-maker cancel rate reasonable | PASS |
| noise buy/sell balance | PASS |
| contrarian sell floor | PASS |

### 4.7 Reliability And Limitations

Phase 1 is usable, but not perfect.

Strengths:

1. The run is real. The repository contains full checked-in outputs, not only a plan.
2. The prompts are anchored to empirical phase statistics from real BTC event data.
3. The fitted priors cover all 4 archetypes across all 4 phases.

Current limitations:

1. Sample size is still moderate at roughly 32 records per agent-phase group.
2. The checked-in event-dynamics table currently underrepresents observed recovery and post rows, so some Phase 1 prompt sampling falls back to anchor-imputed phase samples.
3. The local LLM has not yet been stress-tested with adversarial elicitation or cross-model replication.

### 4.8 Phase 1 Conclusion

Phase 1 is completed and operational. Its outputs are strong enough to serve as Phase 2 priors. The correct description of the current workspace is that Phase 1 elicitation was implemented and run successfully.

## 5. Phase 2: LOB Simulation And Prior Ablation

### 5.1 Goal

Phase 2 uses the Phase 1 priors inside a heterogeneous-agent LOB simulator and tests whether the resulting synthetic panel reproduces empirical flash-crash stylized facts better than simpler baselines.

### 5.2 What Was Done

The main simulation engine is [18_lob_mini_runner.py](../scripts/stage2_economics/18_lob_mini_runner.py). The corrected authoritative result is the canonical tuned-legacy 500-run LLM panel discussed in [phase2_status_and_phase3_plan_2026-05-06.md](validation/phase2_status_and_phase3_plan_2026-05-06.md).

The critical correction before freezing Phase 2 was restoring the intended tuned-legacy regime by neutralizing later resilience damping in the canonical config.

### 5.3 Main Inputs

| Input | Role |
| --- | --- |
| [behavioral_priors.json](../data/processed/tardis/phase1_outputs/behavioral_priors.json) | LLM-derived behavioral priors |
| [prior_anchors.json](../data/processed/tardis/confounder_outputs/prior_anchors.json) | empirical calibration anchors |
| [Flash_Crash_Events_Labeled.csv](../data/processed/tardis/confounder_outputs/Flash_Crash_Events_Labeled.csv) | event-level initialization reference |
| [phase2_canonical_config.json](../config/phase2_canonical_config.json) | canonical tuned-legacy shared runner knobs |

### 5.4 Canonical Phase 2 Parameters

The authoritative canonical configuration is [phase2_canonical_config.json](../config/phase2_canonical_config.json).

| Parameter | Value |
| --- | --- |
| scenario | llm |
| calibration_phase | pre |
| seed | 42 |
| tick_ms | 100 |
| impact_scale | 2.0 |
| intensity_scale | 1.2289 |
| base_order_size | 0.1 |
| mm_vol_threshold_mult | 1.4 |
| mm_withdrawal_strength | 1.8 |
| crash_window_ticks | 10 |
| crash_threshold_pct | 1.93 |
| max_drop_ticks | 5000 |
| max_recovery_ticks | 3000 |
| max_post_ticks | 2000 |
| max_pre_ticks | 2000 |
| drop_impact_mult | 2.2 |
| drop_sell_pressure | 0.18 |
| min_price_fraction | 0.50 |
| resilience_floor_fraction | 0.85 |
| resilience_min_damp | 1.0 |

The known limitation for this config is explicit: the 0.50 minimum price fraction is a structural floor, not an empirical BTC lower bound.

### 5.5 Main Outputs

The authoritative Phase 2 canonical outputs are:

- [lob_full_simulation_llm_tuned_legacy.csv](../data/processed/tardis/phase2_outputs/phase2_canonical_tuned_legacy_500runs/lob_full_simulation_llm_tuned_legacy.csv)
- [lob_full_summary_llm_tuned_legacy.json](../data/processed/tardis/phase2_outputs/phase2_canonical_tuned_legacy_500runs/lob_full_summary_llm_tuned_legacy.json)
- [phase2_stylised_facts_validation_500runs.md](validation/phase2_stylised_facts_validation_500runs.md)

### 5.6 Main Phase 2 Results

The corrected canonical 500-run panel produced:

| Metric | Value |
| --- | --- |
| Runs | 500 |
| Rows | 5,810,200 |
| Crash count | 88 |
| Crash rate | 0.176 |
| Kurtosis excess | 29.084177 |
| Volatility ACF lag 1 | 1.0 |
| OFI pre mean | 0.186907 |
| OFI drop mean | -5.039081 |
| Core stylized-fact gates | all pass |

This makes the corrected canonical panel the main synthetic baseline for the rest of the workspace.

### 5.7 Main Ablation: LLM vs Uniform vs Literature Priors

The main prior-ablation report is [phase2_stylised_facts_validation_500runs.md](validation/phase2_stylised_facts_validation_500runs.md).

| Scenario | Rows | Runs | Kurtosis excess | ACF vol lag 1 | OFI pre mean | OFI drop mean | Crash rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LLM | 5,810,200 | 500 | 29.084177 | 1.0 | 0.186907 | -5.039081 | 0.176 |
| Uniform | 5,804,400 | 500 | 26.492039 | 1.0 | 0.115426 | -2.911987 | 0.008 |
| Literature | 5,810,600 | 500 | 77.195527 | 1.0 | 0.184757 | -2.881011 | 0.152 |

Key comparisons:

| Comparison | Result |
| --- | --- |
| H1 kurtosis test (`LLM > Uniform`) | PASS |
| LLM vs Uniform crash-rate gap | +0.168 |
| LLM vs Literature crash-rate gap | +0.024 |
| LLM vs Literature kurtosis gap | LLM lower by 48.111350 |

Interpretation:

1. The LLM prior clearly outperforms the uniform prior for crash generation and tail behavior.
2. The literature prior also generates crashes, but with much heavier tails than the corrected LLM baseline.
3. The LLM prior is the preferred balanced baseline under the current stylized-facts targets, but it is not best on every possible metric.

### 5.8 Reliability And Limitations

Strengths:

1. The corrected canonical 500-run panel is non-degenerate and passes the core stylized-fact gates.
2. The main LLM vs uniform ablation now supports the central prior-quality claim.
3. The outputs are large enough to support downstream analysis.

Current limitations:

1. The canonical panel still uses a binding structural price floor.
2. Mechanism claims should always mention floor sensitivity.
3. The literature prior remains competitive on crash rate and exceeds the LLM panel on tail heaviness.

### 5.9 Phase 2 Conclusion

Phase 2 is the strongest current contribution in the workspace. The main paper-facing result is that the LLM prior materially improves fidelity relative to the uniform baseline while preserving a realistic crash regime.

## 6. Phase 3: Causal Discovery And Intervention

### 6.1 Goal

Phase 3 tries to learn structure from the synthetic panel and test counterfactual interventions that relate crash risk to pre-crash market state.

### 6.2 What Was Tried First

The first Phase 3 attempt used a crash-window tick-level panel with NOTEARS and VAR-LiNGAM. That pipeline ran, but the slice was degenerate because key variables became constant or nearly constant inside the crash window. This made the original tick-level causal package computationally runnable but scientifically weak.

### 6.3 Phase 3 Fix

The workspace now uses a run-level aggregation strategy built from the locked 500-run canonical panel:

1. aggregate one row per simulation run,
2. summarize pre-crash features,
3. variance-filter unusable variables,
4. run cross-sectional causal discovery and intervention analysis.

The key scripts are:

- [26_run_level_causal_discovery.py](../scripts/stage2_economics/26_run_level_causal_discovery.py)
- [27_phase3_run_level_suite.py](../scripts/stage2_economics/27_phase3_run_level_suite.py)

### 6.4 Main Inputs

| Input | Role |
| --- | --- |
| [lob_full_simulation_llm_tuned_legacy.csv](../data/processed/tardis/phase2_outputs/phase2_canonical_tuned_legacy_500runs/lob_full_simulation_llm_tuned_legacy.csv) | full 500-run synthetic panel |
| pre-gap aggregation rule | define pre-crash summary window |
| variance filter | remove near-constant variables |

### 6.5 Main Phase 3 Parameters

The main suite report is [phase3_run_level_suite.md](validation/phase3_run_level_suite.md).

Main causal setting:

| Parameter | Value |
| --- | --- |
| pre_gap_ticks | 20 |
| node preset | crash_only |
| min_pre_rows | 25 |
| min_variance | 1e-6 |
| weight_threshold | 0.05 |
| causal methods | DAGMA, DirectLiNGAM |

Main intervention setting:

| Parameter | Value |
| --- | --- |
| pre_gap_ticks | 20 |
| feature preset | market_core |
| kept features | ofi_pre_mean, spread_pre_mean, depth_imb_pre_mean, leverage_pre_max, kyle_lambda_pre_mean |
| target | crashed |
| classifier | logistic regression with standard scaling |
| cross-validation | 5-fold stratified ROC AUC |

### 6.6 Main Outputs

The main run-level Phase 3 outputs are:

- [run_level_causal_panel_llm_tuned_legacy.csv](../data/processed/tardis/phase2_outputs/phase2_canonical_tuned_legacy_500runs/run_level_causal_panel_llm_tuned_legacy.csv)
- [phase3_run_level_causal_discovery.md](validation/phase3_run_level_causal_discovery.md)
- [phase3_run_level_suite.md](validation/phase3_run_level_suite.md)
- [phase3_run_level_ablation_summary.csv](../data/processed/tardis/phase2_outputs/phase2_canonical_tuned_legacy_500runs/phase3_run_level_ablation_summary.csv)

### 6.7 Main Phase 3 Results

The run-level panel has:

| Metric | Value |
| --- | --- |
| Runs | 500 |
| Crash runs | 88 |
| Crash rate | 0.176 |
| Mean pre-window rows | 9945.22 |
| Floor-touch rate at 0.60 rule | 1.0 |
| Dropped run-level variable | `vpin_pre_max` |

Main causal result:

| Method | Outcome-facing result |
| --- | --- |
| DAGMA | `kyle_lambda_pre_mean -> crashed (-1.773)`, `spread_pre_mean -> crashed (0.954)`, `ofi_pre_mean -> crashed (0.899)` |
| DirectLiNGAM | no usable direct outcome edge in the main `crash_only` preset |

Main intervention result:

| Metric | Value |
| --- | --- |
| Observed crash rate (raw) | 0.176 |
| Predicted crash rate (observational) | 0.176008 |
| CV ROC AUC | 1.0 |
| `do(OFI = 0)` relative reduction | -29.675393% |
| `do(leverage = 0)` relative reduction | 33.674243% |
| Severity reduction under `do(leverage = 0)` | 0.022576% |

Interpretation:

1. The causal layer is now operational in a run-level form.
2. The DAGMA result gives stable cross-run crash-risk edges.
3. The leverage intervention moves in the expected direction and is moderately sized.
4. The OFI intervention still moves in the wrong direction under the current modeling setup.
5. Crash severity is still heavily compressed, so severity-based intervention conclusions are weak.

### 6.8 Main Ablations

The Phase 3 suite ran both causal and intervention ablations.

#### Causal ablations

| Ablation | Result |
| --- | --- |
| `pre_gap = 0`, `crash_only` | DAGMA recovered `leverage_pre_max -> crashed` in addition to `kyle_lambda`, `spread`, and `ofi` |
| `pre_gap = 20`, `crash_only` | DAGMA recovered `kyle_lambda`, `spread`, `ofi` into `crashed` |
| `pre_gap = 50`, `crash_only` | DAGMA recovered `kyle_lambda`, `spread`, and `wealth_concentration` into `crashed` |
| `pre_gap = 20`, `crash_plus_severity` | DAGMA unchanged; DirectLiNGAM added `crash_severity_pct -> crashed`, which should be treated cautiously |

The main practical reading is that DAGMA is reasonably stable across pre-gap choices, while DirectLiNGAM is not the preferred summary method here.

#### Intervention ablations

| Setting | CV AUC | `do(OFI = 0)` | `do(leverage = 0)` |
| --- | ---: | ---: | ---: |
| `pre_gap = 0`, `market_core` | 1.0 | -29.267614% | 40.385276% |
| `pre_gap = 0`, `market_plus_wealth` | 1.0 | -21.921753% | 17.223197% |
| `pre_gap = 20`, `market_core` | 1.0 | -29.675393% | 33.674243% |
| `pre_gap = 20`, `market_plus_wealth` | 1.0 | -19.130674% | 11.325009% |
| `pre_gap = 50`, `market_core` | 1.0 | -28.723886% | 28.258316% |
| `pre_gap = 50`, `market_plus_wealth` | 1.0 | -13.362477% | 8.024983% |

The robust pattern is:

1. leverage reduction stays positive across all tested intervention settings,
2. OFI intervention stays negative across all tested intervention settings,
3. adding wealth features weakens the leverage effect magnitude.

### 6.9 Reliability And Limitations

Strengths:

1. Phase 3 now uses the full 500-run panel rather than a tiny degenerate crash-window slice.
2. The run-level panel restores variance for most variables that were unusable in the earlier tick-level slice.
3. DAGMA produces stable outcome-facing edges across several pre-gap settings.

Current limitations:

1. All 500 runs are floor-touched under the 0.60 floor-touch rule, so floor-stratified ablation is not available on this panel.
2. The run-level panel answers a cross-run risk-structure question, not a within-event tick-order question.
3. The perfect CV AUC should be interpreted as a sign of strong synthetic separability, not automatically as real-world generalization.
4. `vpin_pre_max` remains effectively constant and is filtered out.
5. The OFI intervention does not support the intended H3 direction.

### 6.10 Phase 3 Conclusion

Phase 3 is now useful, but it should be framed carefully. The run-level package supports a cross-run crash-risk story with stable DAGMA edges and a meaningful leverage intervention signal. It does not yet support a clean OFI intervention claim or a definitive within-event causal mechanism claim.

## 7. Cross-Phase Conclusions

The combined project-level conclusions are:

1. Phase 1 successfully produced LLM-derived behavioral priors from empirical BTC crash anchors.
2. Phase 2 produced a corrected canonical 500-run synthetic panel that passes the core stylized-facts gates and supports the main prior-ablation result.
3. Phase 2 shows the clearest paper-facing contribution: the LLM prior materially outperforms the uniform baseline while remaining in a realistic crash-rate regime.
4. Phase 3 is best treated as a supplementary but meaningful analysis layer. The strongest takeaways are the stable run-level DAGMA crash edges and the positive leverage intervention effect.
5. The current weakest claim in the workspace remains the OFI intervention story. It is not yet supported in the intended direction.

## 8. Recommended Paper Framing

If this workspace is written up as a paper in its current state, the most defensible framing is:

1. main contribution: LLM-derived heterogeneous priors materially change synthetic flash-crash behavior relative to simple baselines,
2. supporting contribution: the corrected 500-run canonical panel reproduces the target stylized-facts gates,
3. exploratory contribution: run-level causal analysis identifies stable cross-run crash-risk structure, especially around spread, OFI, and leverage-related quantities,
4. explicit caution: the current Phase 3 intervention layer does not yet validate the intended OFI intervention claim.

## 9. Key Files For Readers

If you want to continue past this master report, the best next files are:

1. [README.md](../README.md) for repository structure and workflow commands.
2. [reports/DATA_PIPELINE.md](DATA_PIPELINE.md) for empirical data lineage.
3. [reports/COMOSA_Tong_Hop_Phase1_Phase2.md](COMOSA_Tong_Hop_Phase1_Phase2.md) for the older detailed Phase 1 and Phase 2 narrative.
4. [reports/validation/phase2_status_and_phase3_plan_2026-05-06.md](validation/phase2_status_and_phase3_plan_2026-05-06.md) for the corrected canonical 500-run Phase 2 interpretation.
5. [reports/validation/phase3_run_level_suite.md](validation/phase3_run_level_suite.md) for the latest Phase 3 main results and ablations.