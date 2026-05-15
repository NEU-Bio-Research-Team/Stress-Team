## 4. Experimental Setup

### 4.1 Dataset, Locked Design, and Comparison Criterion

We evaluate the simulator on a locked three-condition design built from the curated BTCUSDT flash-crash panel. The primary condition uses behavioral priors elicited offline and frozen before simulation. The first baseline replaces these priors with broad randomized parameter draws, while the second uses hand-crafted values adapted from prior simulation studies. All three conditions are run on the same event panel, use the same 100 ms simulation clock, and follow the same seed discipline. Cross-condition differences can therefore be attributed to the prior specification rather than to differences in the market environment.

The quantitative comparison in this paper is fixed to the authoritative 500-run artifacts stored in the repository. This freeze matters because the same locked panel supports both the distributional validation and the downstream mechanism analysis. We regard the LLM condition as successful if it dominates the broad-prior baseline and remains non-inferior to the literature baseline on crash incidence and mean short-horizon drawdown, while preserving the correct sign ordering of directional flow between the pre-crash and drop phases. Tail thickness is treated as a gate-based realism criterion rather than as a point-calibration target, because the empirical reference is computed from a crash-conditioned event panel rather than from an unconditional BTC return series.

To operationalize that comparison, we evaluate each condition along four dimensions: the heaviness of return tails, the persistence of short-run volatility, the directional shift in order-flow pressure from the pre-crash phase to the drop phase, and the incidence and sharpness of simulated crashes. These dimensions are intentionally complementary. Tail shape and volatility persistence test whether the simulator preserves familiar high-frequency regularities. The phase shift in directional pressure tests whether the model reproduces the selling asymmetry of crash windows. Crash incidence and short-horizon drawdown test whether the simulator produces economically meaningful stress episodes rather than merely noisy volatility.

| Evaluation goal | Evidence used in the paper | Why it matters |
| --- | --- | --- |
| Distributional realism | Tail thickness and short-run volatility persistence | Verifies gate-based realism for high-frequency return behavior |
| Phase fidelity | Shift in directional pressure from the pre-crash phase to the drop phase | Verifies that simulated crashes are driven by the correct sign and timing of flow pressure |
| Crash realism | Crash incidence and mean short-horizon drawdown | Distinguishes economically meaningful crashes from weak disturbances |
| Mechanism interpretability | Pre-crash run summaries, descriptive separation tests, and counterfactual crash-rate changes | Tests whether the simulated panel contains usable pre-crash structure |

### 4.2 Run-Level Mechanism Design

Mechanism analysis is conducted only on the locked LLM panel, because the goal is not to compare graph recovery across all baselines but to determine whether the best-performing simulation condition contains interpretable pre-crash signals. A run is labeled as a crash if the price falls by at least 1.93% within a 10-step window, which corresponds to one second on the 100 ms grid. For crash runs, the summary window ends two seconds before the first detected crash so that the predictors remain genuinely pre-crash. For non-crash runs, the full trajectory is retained. This design yields 500 run-level observations, of which 88 are crash runs and 412 are non-crash runs.

The run-level panel includes summary measures for directional pressure, quoted trading cost, book imbalance, leverage amplification, and price impact. We then use the panel in two complementary ways. First, we apply two directed-dependence recovery procedures and treat agreement between them as stronger evidence than a result obtained by only one procedure. Second, we report descriptive separation tests and single-feature discrimination scores for the main pre-crash summaries. These tests are useful for quantifying signal strength, but they are not treated as standalone evidence for edge orientation. Because the panel is simulated rather than observed, the analysis is explicitly framed as mechanism recovery within the simulator, not as identification of causal effects in the real BTC market.

### 4.3 Counterfactual Intervention Protocol

The final experimental block asks whether the simulator supports meaningful intervention analysis once the run-level panel is fixed. We estimate the baseline predicted crash rate from the run summaries and then apply two targeted counterfactual changes: removing directional pressure and removing leverage amplification. Both interventions are evaluated against the same pre-crash feature set, with all untargeted summaries held unchanged. The pre-registered success criterion from the runbook is a reduction of at least 30% in the predicted crash rate. This threshold converts the intervention analysis from a descriptive exercise into a falsifiable test.

All experiments are fully reproducible from repository artifacts. The simulator panels, validation reports, seed discipline, and cached LLM outputs are all frozen in the current workspace. Exact implementation labels, artifact paths, and internal variable names are listed in Appendix Table A1. The broad-prior ranges, crash detector definition, and key structural constraints are listed in Appendix Table A2 so that the main text can stay focused on economic interpretation rather than software details.

Figure 2 about here. Source file: reports/figures/paper/figure_02_temporal_anatomy_llm.png

## 5. Results

### 5.1 Reproducing Distributional and Crash-Level Targets

Table 3 shows the locked three-way comparison. The main result is narrower, and more defensible, than raw point matching. The LLM-based prior clearly dominates the broad-prior baseline on crash incidence and short-horizon drawdown, and it remains competitive with the literature baseline on those same crash-level metrics. At the same time, it does not reproduce the full empirical tail magnitude of the crash-conditioned BTC reference panel, and it overshoots the empirical directional-flow magnitude in the drop phase. The correct reading is therefore improved fidelity under empirical constraints, not exact replication of every empirical moment.

| Criterion | Empirical crash-conditioned reference | LLM-based priors | Gap vs. empirical reference (LLM) | Broad priors | Literature priors | Interpretation |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| Tail thickness of returns | 8581.7880 | 29.0842 | 99.66% below empirical, about 295x smaller | 26.4920 | 77.1955 | The LLM condition improves on the broad baseline but does not match the raw extremity of the crash-conditioned tail reference |
| Short-run volatility persistence | 1.0000 | 1.0000 | matched | 1.0000 | 1.0000 | All three conditions clear the persistence gate |
| Directional flow, pre-crash phase | 0.2644 | 0.1869 | 29.32% below empirical | 0.1154 | 0.1848 | The LLM and literature conditions begin from similar pre-crash levels |
| Directional flow, drop phase | -2.6988 | -5.0391 | 86.71% more negative than empirical | -2.9120 | -2.8810 | The LLM condition captures the correct sign and timing but overshoots the empirical magnitude |
| Crash incidence | 0.05 to 0.40 target band | 0.1760 | n/a | 0.0080 | 0.1520 | Only the LLM and literature conditions remain inside the target band |
| Mean short-horizon drawdown (%) | n/a | 1.3280 | n/a | 0.6122 | 1.1276 | The LLM condition is materially sharper than the broad baseline and non-inferior to the literature baseline |

Two clarifications are important for interpreting Table 3. First, the empirical tail reference of 8581.7880 is computed from a crash-conditioned 100 ms event panel. It should therefore be read as an upper-envelope benchmark for extremal flash-crash tails, not as a point target that the simulator is expected to match exactly. The current simulator also contains structural features that compress raw tail magnitude, including a hard price floor and bounded impact dynamics. For that reason, the LLM condition remains about 295 times smaller than the empirical reference, and we do not claim raw tail-magnitude matching.

Second, the directional-flow comparison shows correct sign and timing but imperfect amplitude calibration. The LLM drop-phase mean of -5.0391 is 86.71% more negative than the empirical crash-conditioned reference of -2.6988. This overshoot is consistent with the deliberate drop-phase sell tilt and impact amplification used to bring crash incidence into the target band. The result is still informative, but it should be framed as a sign-and-timing match rather than as a magnitude match.

The broad baseline is therefore too weak for the intended use case. Its crash incidence of 0.008 falls far below the target band, and its mean short-horizon drawdown is less than half that of the LLM condition. The literature baseline is stronger, but it does not dominate the LLM condition uniformly. It produces a realistic crash incidence of 0.152 and a high drawdown profile, yet its return tails remain substantially heavier than those of the LLM condition. Taken together, this evidence is consistent with the view that empirically disciplined LLM priors can improve simulation fidelity relative to an undisciplined broad baseline in this BTCUSDT flash-crash setting. The literature baseline remains a serious comparator, so the evidence supports a mixed but defensible advantage rather than blanket dominance.

Appendix Figure A1 provides an optional compact visual summary of the locked three-condition comparison. Source file: reports/figures/paper/appendix_figure_a1_locked_comparison.png

### 5.2 Temporal Anatomy of Simulated Crashes

The locked LLM panel does not generate crash flags through isolated price jumps. Figure 2 shows a coordinated microstructure sequence once runs are aligned at the first crash onset. Directional pressure turns sharply negative before and during the drop, quoted trading cost rises at the same time that book balance deteriorates, and trading activity accelerates into the event window. Price then falls in a concentrated burst and only partially normalizes during the early recovery period. This co-movement matters because it shows that the simulator reproduces a crash anatomy rather than a threshold artifact.

The phase statistics in Table 3 reinforce this picture, but they also qualify it. In the LLM condition, mean directional pressure shifts from 0.1869 in the pre-crash phase to -5.0391 in the drop phase. The corresponding shift is materially weaker under both baselines, which supports the claim that the LLM condition captures directional asymmetry more clearly than the alternatives. At the same time, the LLM condition overshoots the empirical drop-phase magnitude. The temporal anatomy should therefore be read as evidence of correct sequencing and co-movement, not as perfect amplitude calibration.

### 5.3 Run-Level Mechanism Evidence

The run-level analysis provides partial, rather than complete, recovery of the hypothesized mechanism. Across the two complementary structure-recovery procedures, two signals stand out most clearly: wider quoted trading cost before the event and stronger directional pressure before the event. The book-thinning channel is present only intermittently, and leverage remains unstable in the structure-recovery stage even though crash runs display visibly larger leverage peaks than non-crash runs.

| Proposed mechanism | Evidence across structure recovery | Interpretation |
| --- | --- | --- |
| Directional pressure raises crash risk | Recovered in one procedure, absent in the second | Partial support |
| Wider trading cost precedes crashes | Recovered in one procedure and consistent with large pre-crash group separation | Strongest support |
| Wider trading cost co-moves with thinner book | Recovered in one procedure only | Partial support |
| Thinner book raises crash risk | Not stably recovered | Limited support |
| Leverage raises crash risk | Unstable in structure recovery, but larger in crash runs and confirmed by intervention | Mixed but substantive support |

To quantify the descriptive side of this evidence, Table 4 reports pre-crash group means, rank-based significance tests, and single-feature discrimination scores. The strongest and most stable precursor is quoted trading cost, which rises from 3.5445 in non-crash runs to 5.3045 in crash runs. Leverage also increases materially, from 1.7147 to 2.2357. Directional pressure and book imbalance show equally strong separation, but their interpretation is more delicate because the current window-construction rule makes the run-level panel highly separable by design.

| Pre-crash summary | Non-crash mean | Crash mean | Mann-Whitney p-value | Best single-feature AUC | Reading |
| --- | ---: | ---: | ---: | ---: | --- |
| Directional pressure | -2.2564 | -0.6376 | 3.90e-49 | 1.000 | Strong separator, but the sign should be interpreted cautiously under the current window construction |
| Quoted trading cost | 3.5445 | 5.3045 | 1.13e-86 | 1.000 | Strongest and most stable precursor |
| Book imbalance | -0.0884 | -0.0249 | 3.90e-49 | 1.000 | Signal is present, but edge orientation remains unstable |
| Leverage peak | 1.7147 | 2.2357 | 1.23e-38 | 0.941 | Consistent with the successful intervention result |

Welch unequal-variance tests point in the same direction for all four summaries. However, these statistics quantify association within the constructed run-level panel rather than independent support for edge orientation. The near-perfect AUC values for several summaries indicate that the current pre-crash window design makes crash and non-crash runs highly separable. For that reason, we use these tests as evidence that the panel contains meaningful pre-crash signal, while reserving the more cautious phrase partial mechanism recovery for the causal conclusion itself.

### 5.4 Counterfactual Interventions

The intervention results sharpen the previous section by asking which pre-crash channel behaves like an actionable amplifier rather than a correlated warning signal. Table 5 shows a clear asymmetry. Removing leverage amplification lowers the predicted crash rate from 0.1760 to 0.1167, a reduction of 33.67% that clears the pre-registered threshold. By contrast, removing directional pressure increases the predicted crash rate to 0.2282, so this intervention fails the pre-registered test.

| Counterfactual change | Baseline predicted crash rate | Post-change rate | Change | Pre-registered threshold | Verdict |
| --- | ---: | ---: | ---: | --- | --- |
| Remove directional pressure | 0.1760 | 0.2282 | +29.68% | At least 30% reduction | Fail |
| Remove leverage amplification | 0.1760 | 0.1167 | -33.67% | At least 30% reduction | Pass |

This result matters because it distinguishes correlation from leverage within the simulated system. Directional pressure is clearly associated with crashes, but naively forcing that summary to zero does not protect the system once the rest of the pre-crash state is held fixed. Leverage behaves differently. When leverage is removed, the predicted crash rate falls by roughly one third. Within the simulator, leverage therefore emerges as the cleaner intervention target, while directional pressure remains better understood as part of a broader pre-crash configuration rather than as a single control knob.

Figure 3 about here. Source file: reports/figures/paper/figure_03_precrash_and_interventions.png

## Appendix Table A1. Exact Reproducibility Labels Used in Sections 4 and 5

| Paper term in the main text | Exact implementation label or frozen artifact |
| --- | --- |
| Locked three-condition comparison | reports/validation/phase2_stylised_facts_validation_500runs.md and reports/validation/phase2_stylised_facts_validation_500runs.json |
| Locked run-level mechanism and intervention suite | reports/validation/phase3_run_level_suite.md and reports/validation/phase3_run_level_suite.json |
| Sparse acyclic recovery procedure | dagma |
| Linear non-Gaussian recovery procedure | direct_lingam |
| Tail thickness of returns | kurtosis_excess |
| Short-run volatility persistence | acf_vol_lag1 |
| Directional flow before and during the crash window | ofi_pre_mean and ofi_drop_mean |
| Pre-crash quoted trading cost | spread_pre_mean |
| Pre-crash book balance measure | depth_imb_pre_mean |
| Pre-crash leverage peak | leverage_pre_max |

## Appendix Table A2. Baseline and Detector Details Omitted from the Main Prose

| Implementation detail | Frozen value used in the current comparison |
| --- | --- |
| Broad-prior baseline | aggressiveness Uniform(0.0, 1.0); cancel probability Uniform(0.0, 0.8); order-size multiplier Uniform(0.5, 1.5); inventory sensitivity Uniform(0.0, 1.0); market-order fraction Uniform(0.3, 0.9) |
| Crash detector | a run is labeled as crashed when drawdown is at least 1.93% within 10 ticks on the 100 ms grid, equivalent to one second |
| Drop-phase amplification | sell tilt = 0.18 for all archetypes except the contrarian type; impact multiplier = 2.2 in the drop phase |
| Hard price floor | min price fraction = 0.50 of the initial price |
| Known limitation | any crash-rate claim must be read together with the binding floor context |
