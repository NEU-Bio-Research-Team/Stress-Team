# COMOSA Research Plan (Rewritten, Stress-Free)

> **LLM-Elicited Behavioral Priors for Heterogeneous Agent Simulation of BTC Flash Crash Dynamics with Causal Learning.**
> This document consolidates the 4 tabs of the original Google Doc into a single, self-contained specification. All WESAD / bio / stress coupling material has been removed — this plan uses market microstructure only. The "stress_proxy" column that appears in the current `Event_Dynamics_100ms.csv` is a **market-side composite of OFI / VPIN / Amihud / realized vol** (see §4.2) and is *not* a physiological variable; it is renamed `order_flow_toxicity` below and is optional.

---

## 1. Motivation and Research Questions

The research builds an agent-based financial-market simulator in which an LLM is used **offline, once**, to generate behavioral priors for four trader archetypes. The entire online simulation is then pure mathematical LOB dynamics. Two contributions are targeted:

1. **LLM as a behavioral calibration instrument** (not an online decision-maker), producing parametric distributions for agent heterogeneity.
2. **Causal discovery on simulation output** to recover the causal mechanism — not just the predictive correlate — of BTC flash crashes.

### Research Questions

- **RQ1 (primary).** Can LLM-elicited behavioral priors produce heterogeneous agents realistic enough to reproduce BTC microstructure stylised facts (flash-crash anatomy, spread dynamics, OFI)?
- **RQ2 (secondary).** Which combinations of behavioral parameters (`aggressiveness`, `cancel_prob`, `order_type`) predict flash-crash occurrence in simulation?
- **RQ3 (causal).** What causal structure among microstructure variables (OFI, spread, depth, leverage, inventory) produces flash crashes, and can this DAG be learned from simulation output?

### Hypotheses

- **H1 (Calibration).** LLM-elicited priors yield more realistic heterogeneous populations than uniform random initialization, measured by stylised-fact reproduction (fat tails, volatility clustering, spread patterns).
- **H2 (Trigger).** Strongly negative OFI **combined with** high leverage **and** Market-Maker withdrawal are necessary conditions for flash crash; leverage acts as an interaction/amplification term in the causal DAG.
- **H3 (Intervention).** Do-calculus interventions `do(OFI < τ)` or `do(Leverage < L_max)` reduce flash-crash probability by >70% in simulation, demonstrating policy-actionability.

---

## 2. The 3-Phase Architecture

A clear picture of what each phase consumes and produces is essential for deciding what data we still need to finish preparing.

```
Phase 1  : LLM (offline, ~500 prompts)
           → 4 sets of parametric distributions {Beta, Gamma, Poisson, Pareto}
                │
                ▼
Phase 2  : LOB Simulation Engine (math model, 100 ms ticks, ~1000 runs)
           → Synthetic panel data {OFI, spread, depth, leverage, inventory, crash}
                │
                ▼
Phase 3  : Causal Discovery (NOTEARS / LiNGAM / optionally ARCADIA)
           → DAG + do-calculus intervention validation
```

### 2.1 Phase 1 — LLM-Elicited Behavioral Priors (offline, once)

**Goal.** Replace hard-coded agent parameters with parametric distributions *measured* from an LLM acting as a behavioral respondent, anchored in **empirical BTC microstructure statistics** from our 66-event dataset.

**Pipeline.**

1. Build prompt templates for each archetype: Momentum, Noise, Market Maker, Contrarian.
2. Anchor each prompt with **real percentiles** extracted from `Event_Dynamics_100ms.csv` per phase (`pre` / `drop` / `recovery` / `post`). Example:
   *"You are a Market Maker. Current OFI = −2.3 (90th percentile of OFI during `drop` across 66 BTC flash crashes). Realized vol (50-bin) = 0.0042. Amihud illiquidity is in the top decile. Describe your order submission."*
3. For each response, extract `{order_type, aggressiveness ∈ [0,1], cancel_prob ∈ [0,1], inventory_sensitivity, order_size}`.
4. Repeat 100–500 times per archetype with varied scenario anchors.
5. Fit parametric families (Beta, Gamma, Normal, Pareto for size) with MLE + bootstrap CIs.
6. Freeze distributions → these are the **only** LLM-derived artifacts consumed by Phase 2.

**Example output.**

```
Momentum:     aggressiveness ~ Beta(α=2.1, β=0.8)     cancel_prob ~ Beta(0.5, 3.2)
Market Maker: aggressiveness ~ Beta(α=0.6, β=2.4)     cancel_prob ~ Beta(2.8, 0.9)
Noise:        arrival_rate   ~ Poisson(λ=3.2 / 100ms)
Contrarian:   inventory_sens ~ Gamma(α=1.8, β=0.4)
```

**Key design choice: FinGPT + tool-calling.** To address the "LLM is just stereotyping finance blog text" objection (Problem 1 in §7), the prompting backbone uses a finance-tuned LLM (FinGPT or equivalent) with a retrieval / tool-calling layer that lets the model query the actual empirical distributions it is being asked about. The LLM is a **measurement instrument** calibrated against real statistics, not an oracle.

### 2.2 Phase 2 — LOB Simulation Engine

**The LOB object.** A two-sided queue of resting limit orders per price level. Market orders consume the opposite side; limit orders join the queue at the specified level; cancels remove them.

```
Price   | Bid queue (buy)    | Ask queue (sell)
--------|--------------------|------------------
50,010  |                    | [MM:2 BTC][MM:1]   ← best ask
50,005  |                    | [MM:3]
50,000  | [CT:1]             |                     ← mid
49,995  | [MM:2]             |                     ← best bid
49,990  | [MT:5]             |
```

**Four archetypes, their rules.**

| Archetype      | Decision rule                                                                 | Parameters sampled from Phase 1 prior                     |
|----------------|--------------------------------------------------------------------------------|------------------------------------------------------------|
| Momentum       | Extrapolate trend from moving average; aggressive market order in trend dir.   | `aggressiveness`, `size` ~ Pareto(wealth)                  |
| Noise          | Poisson arrival; random buy/sell; random size.                                 | `λ`, `size_mean`                                           |
| Market Maker   | Passive two-sided quoting; withdraw all quotes when realized vol > τ_vol.       | `spread_target`, `inventory_sens`, `vol_threshold`         |
| Contrarian     | Buy on sustained drawdown (overreaction correction).                            | `drawdown_threshold`, `size`                               |

**Per-tick loop (Δt = 100 ms).**

1. Each agent samples its *static* parameters once at instantiation from the Phase-1 distributions (see the "static prior" justification in §7, Problem 4).
2. Each agent observes LOB state `{best_bid, best_ask, spread, depth_imbalance, recent_OFI, realized_vol}`.
3. Each agent emits an `Action ∈ {submit_market, submit_limit(level,size), cancel(id), do_nothing}` according to its rule.
4. The matching engine clears, updates the book, and emits tick-level state.
5. Detector: `flash_crash_indicator = 1` if `close.pct_change(last_N_ticks) ≤ −3%`.

**Why static priors are *correct*, not a limitation.** Panic is an **emergent** property of structural interaction — MM withdrawal at vol-threshold, momentum amplification, spread→depth feedback — so it does not need to be encoded inside an individual agent. Moreover, NOTEARS/LiNGAM assume a stationary data-generating process; a dynamic prior would conflate structural composition with real-time behavioral adaptation and make the Phase-3 causal graph un-interpretable.

### 2.3 Phase 3 — Causal Discovery on Simulation Output

**Panel built in Phase 2.**

| run_id | tick | OFI  | spread | depth | leverage_proxy | vpin | flash_crash |
|--------|------|------|--------|-------|-----------------|------|-------------|
| 001    | 100  | −0.3 | 8 bps  | 450   | 0.12           | 0.31 | 0           |
| 001    | 347  | −4.1 | 45 bps | 20    | 0.91           | 0.88 | 1           |

**Methods.**

- **NOTEARS** (Zheng et al. 2018): continuous acyclicity-constrained optimization, \[\min_W \tfrac{1}{2n}\lVert X - XW\rVert_F^2 + \lambda\lVert W\rVert_1\ \text{s.t.}\ h(W)=\mathrm{tr}(e^{W\circ W})-d=0\].
- **LiNGAM** (Shimizu et al. 2006): linear non-Gaussian identification.
- **ARCADIA** (NeurIPS 2025, optional): LLM-guided iterative DAG proposal + validation.

**Validation.**

1. Compare learned DAG to a **BTC-native** benchmark (e.g., published Kirilenko-style structure **re-derived on a different BTC flash-crash sample** — not the E-mini 2010 DAG, see §7 Problem 3).
2. Run `do(leverage=0)` and `do(OFI > τ)` interventions in the simulator; measure crash-rate reduction.
3. Phase-1/2/3 ablations (see §5).

**Note on tautology risk (§7 Problem 2).** Since Phase 2 rules are coded by us, Phase 3 is framed as **confirmatory** — verifying that causal discovery recovers the *structural* mechanism embedded in the simulator — not as blind discovery. This is a controlled methodological benchmark: the simulator provides ground truth, and the contribution is showing that standard causal-discovery methods (NOTEARS/LiNGAM) *can* recover microstructure DAGs under realistic, heterogeneous agent populations.

---

## 3. Dataset: Current State

The empirical dataset is produced by the BTC / Tardis pipeline in [`NEU-Bio-Research-Team/Stress-Team`](https://github.com/NEU-Bio-Research-Team/Stress-Team), specifically the **`scripts/stage2_economics/`** folder. (The `phase1_data_engineering` scripts 01/02/04/05 are bio-related and are ignored here.)

### 3.1 Repo pipeline that matters (stress-free path)

```
scripts/phase1_data_engineering/
  00_fetch_tardis.py          # download BTCUSDT futures (Binance Vision free or Tardis paid)
  03_audit_tardis.py          # T1–T15 data quality checks
  06_preprocess_tardis.py     # trades → 1-min OHLCV + orderbook features + crash events
  07_extract_features.py      # (only extract_market_features() is relevant)
  09_stylized_facts.py        # SF-1…SF-5 on BTC returns

scripts/stage2_economics/                 ← the 66-event tick-level pipeline
  00_reindex_ticks.py                     # raw csv.gz → ms-resolution parquet
  01_build_multiresolution_bars.py        # time / volume / dollar / tick bars
  02_hft_feature_engineering.py           # HFT features on daily bars
  04_detect_flash_crashes.py              # Tier 1 (macro): klines → event_catalog.csv
  05_download_event_ticks.py              # aggTrades + bookTicker around each event
  06_micro_feature_engineering.py         # Tier 2 (micro): event ticks → 10/100/1000 ms features
  08_refine_event_timestamps.py           # tick-anchored event timestamps
  09_produce_confounder_outputs.py        # Event_Dynamics_100ms.csv (+ news labels, phase)
```

The decisive file for Phases 1–3 calibration is `Event_Dynamics_100ms.csv`, produced by `09_produce_confounder_outputs.py` (lines ~300–400 define the exact columns).

### 3.2 Columns currently in `Event_Dynamics_100ms.csv`

Extracted verbatim from `keep_cols` in script 09:

| Column                              | Meaning                                                                 |
|-------------------------------------|-------------------------------------------------------------------------|
| `event_id`                          | One of 66 flash-crash events (integer)                                  |
| `date`                              | Event date                                                              |
| `timestamp_ms`, `timestamp_utc`     | 100-ms grid timestamp                                                   |
| `time_from_drop_start_ms`           | Offset relative to drop-start epoch                                     |
| `phase`                             | `pre` / `drop` / `recovery` / `post` (see §3.3)                         |
| `close`                             | Last trade price in bin                                                 |
| `ofi`                               | Order flow imbalance (signed volume, Lee–Ready)                         |
| `trade_intensity`                   | Trades per second (Poisson λ proxy)                                     |
| `amihud_illiq`                      | Amihud 2002 illiquidity ratio                                           |
| `kyle_lambda`                       | Kyle 1985 price impact (rolling)                                        |
| `vpin`                              | Easley–López de Prado–O'Hara 2012, time-clock approximation             |
| `realized_vol_50`                   | Realized volatility over 50 bins                                        |
| `stress_proxy`                      | **Market-side composite**, 0.40·z(−OFI)+0.25·z(VPIN)+0.20·z(Amihud)+0.15·z(RV). *Not a bio signal.* Rename `order_flow_toxicity`. |
| `velocity_pct_per_100ms`            | Signed return velocity                                                  |
| `drop_velocity_pct_per_100ms`       | `max(−velocity, 0)`                                                     |
| `panic_acceleration_pct_per_100ms2` | 2nd derivative of price                                                 |
| `drop_1s_pct`, `drop_from_local_pct`| Short-horizon drawdowns                                                 |
| `delta_from_news_ms`                | Distance from nearest curated news timestamp                            |

### 3.3 Phase labels (script 09 logic)

```
phase = "pre"       if timestamp_ms < start_epoch
phase = "drop"      if start_epoch  ≤ timestamp_ms ≤ bottom_epoch
phase = "recovery"  if bottom_epoch < timestamp_ms ≤ end_epoch
phase = "post"      otherwise
end_epoch = first timestamp where close ≥ bottom + 0.50·(start − bottom)   (50% retrace)
```

### 3.4 Scale

- 66 events × ~9,144 rows/event ≈ **603,500 rows at 100 ms**.
- Sufficient for NOTEARS / LiNGAM (stable from ~5–10k rows for 5–8 variables).

---

## 4. Dataset: Gaps and Required Additions

The plan's causal DAG uses 6 variables; the dataset currently exposes **5**. Fixing this is the precondition to finishing data prep.

### 4.1 Variable ↔ column gap matrix

| DAG variable            | Column in `Event_Dynamics_100ms.csv`           | Status                                     |
|-------------------------|------------------------------------------------|--------------------------------------------|
| X1 = OFI                | `ofi`                                          | ✅ available                                |
| X2 = Bid–Ask Spread     | —                                              | ❌ **missing**, must be added (P0)          |
| X3 = LOB Depth          | `amihud_illiq` (proxy)                         | ⚠️ proxy only; add `depth_imbalance` (P0)  |
| X4 = Leverage           | —                                              | ❌ **missing**, operationalization needed (P0) |
| X5 = Inventory pressure | `vpin`                                         | ✅ proxy, acceptable                        |
| Y  = Flash Crash        | `phase == "drop"` (or `drop_velocity` threshold)| ✅ available                                |

### 4.2 Features to add in a new `12_augment_dynamics_features.py`

All of these come from either the raw `bookTicker.parquet` (already present in the repo's event download) or from algebraic transforms of existing columns. No fabrication.

#### A. From `bookTicker.parquet` (present but dropped in `keep_cols`)

- **`spread_bps` = (ask − bid) / mid × 10000**   — Roll 1984; O'Hara 1995. Script `06_micro_feature_engineering.py` already computes it; simply re-add it to `keep_cols` in script 09.
- **`depth_imbalance` = (Q_bid − Q_ask) / (Q_bid + Q_ask) ∈ [−1, 1]**   — Cao–Chen–Griffin 2005. Best direct proxy for X3.
- **`mid_price`, `touch_depth`**   — book-state features useful for agent observation vectors.

#### B. Leverage operationalization (choose one; I recommend A + B in parallel)

- **Option A — Funding rate** (empirically cleanest): download Binance USDT-M **funding rate** history (8-hourly) for every event date; merge as `funding_abs`. Literature anchor: Liu & Tsyvinski 2021. Cost: ~10 MB total, one CSV per year.
- **Option B — Velocity × acceleration composite** (uses only existing columns): `leverage_proxy = |drop_velocity| · |panic_acceleration| / realized_vol_50`. Anchored in Brunnermeier & Pedersen 2009 (margin-call cascade). Fully computable now.
- **Option C — Redefine X4** as `inventory_pressure_composite` = z(VPIN) + z(Amihud) + z(drop_velocity), and drop the word "leverage". Scientifically defensible and avoids the funding-rate dependency. Recommend *only* if Option A is blocked.

#### C. Regular 100 ms grid per event (required by NOTEARS)

NOTEARS and LiNGAM both assume regularly-spaced observations. The current binning can have gaps when no trade lands in a 100 ms bin. Re-index:

```python
grid = np.arange(t_start, t_end + 100, 100)
df = df.set_index("timestamp_ms").reindex(grid)

# Book state is persistent → forward-fill
df[["spread_bps", "depth_imbalance", "mid_price"]] = df[["spread_bps","depth_imbalance","mid_price"]].ffill()

# Trade flow is additive → zero-fill
df[["ofi", "trade_intensity", "volume"]] = df[["ofi","trade_intensity","volume"]].fillna(0)
```

#### D. Normal-market reference (for Phase-2 baseline calibration)

The current data is event-centric (±30 min around each crash). The LOB engine needs a **non-crash baseline** for Market Maker / Noise Trader behavior in the pre-crash regime. Two options:

- Use the `pre` phase aggregated across 66 events (cheap, slightly biased because it's still just-before-crash).
- Download 1 week of `BTCUSDT-aggTrades` from Binance Vision as a true normal-market reference (~100 MB; free).

### 4.3 Priority-ordered action checklist

| Priority | Action                                                                                                             |
|---------|---------------------------------------------------------------------------------------------------------------------|
| P0       | Re-emit `spread_bps` and `depth_imbalance` from `bookTicker.parquet` into `Event_Dynamics_100ms.csv`.               |
| P0       | Decide leverage operationalization (Option A, B, or A+B). Patch `09_produce_confounder_outputs.py` accordingly.     |
| P0       | Rename `stress_proxy` → `order_flow_toxicity` to remove the (misleading) bio connotation.                           |
| P1       | Add the regular-grid re-indexer as `12_augment_dynamics_features.py` (per-event, 100 ms step).                      |
| P1       | Pull ~1 week of normal-market aggTrades for non-event baseline distributions.                                       |
| P2       | Compute per-phase `OFI` percentiles + Poisson λ + Pareto α: this is the **Phase-1 prompt anchor file**.             |

### 4.4 Phase-1 anchor statistics (`10_compute_prior_anchors.py`)

Aggregates to emit into a JSON the prompting code will consume:

| Statistic                       | Formula / code                                                                              | Purpose                                |
|--------------------------------|----------------------------------------------------------------------------------------------|----------------------------------------|
| OFI percentiles per phase       | `df.groupby("phase")["ofi"].quantile([.05,.25,.50,.75,.95])`                                | Anchor OFI scenario in LLM prompt       |
| Noise-trader λ                  | `λ = mean(trade_intensity)` (Poisson MLE)                                                  | Calibrate arrival rate                  |
| Order-size Pareto α             | `α̂ = n / Σ log(x_i / x_min)` (Clauset–Shalizi–Newman 2009)                                 | Calibrate Pareto size distribution      |
| Kyle λ per regime               | `df.groupby("phase")["kyle_lambda"].mean()`                                                  | MM impact threshold                     |
| Realized vol per phase          | `df.groupby("phase")["realized_vol_50"].mean()`                                              | MM activation threshold                 |

### 4.5 Microstructure formulas (audit trail)

All formulas already in script 06; kept here as a single reference so reviewers can cite primary sources:

- Spread (Roll 1984; O'Hara 1995): \( \mathrm{spread\_bps} = (P_{ask}-P_{bid})/P_{mid}\cdot 10{,}000 \)
- Depth imbalance (Cao–Chen–Griffin 2005): \( (Q_{bid}-Q_{ask})/(Q_{bid}+Q_{ask}) \)
- Kyle λ (Kyle 1985): \( \lambda_t = \mathrm{Cov}(\Delta P,\ \mathrm{OFI})/\mathrm{Var}(\mathrm{OFI}) \), rolling
- OFI (Lee & Ready 1991): \( \mathrm{OFI}_t = \sum_{i\in\mathrm{bin}_t}\mathrm{sign}_i\cdot q_i \)
- Amihud (2002): \( \mathrm{ILLIQ}_t = |r_t|/DV_t \)
- VPIN (Easley–López de Prado–O'Hara 2012): \( \big|\sum V^B - \sum V^S\big|/\sum V^{tot} \) over rolling bucket
- Funding-rate leverage proxy (Liu & Tsyvinski 2021): \( \mathrm{leverage\_proxy}_t = |r_f(t)| \)
- Velocity-acceleration leverage proxy (Brunnermeier–Pedersen 2009): \( |\dot P|\cdot|\ddot P|/\sigma \)

---

## 5. Validation, Ablation, Timeline

### 5.1 Stylised-facts validation

Compare simulator output to real BTC (`Event_Dynamics_100ms.csv`):
(a) return fat tails (excess kurtosis > 3); (b) volatility clustering (ACF of squared returns); (c) spread widening before `drop`; (d) OFI regime shift across phases.

### 5.2 Causal-DAG validation

- BTC-native benchmark: withhold *k* events, learn DAG on the rest, compare edge set and signs.
- Do-calculus: `do(leverage=0)` vs baseline → crash-rate delta; H3 target ≥ 70% reduction.

### 5.3 Ablation (Phase 1)

| Condition | Parameters source            | Expected outcome                              |
|-----------|------------------------------|-----------------------------------------------|
| A         | Uniform random               | Poor stylised-fact match, weak DAG            |
| B         | LLM-elicited (ours)          | Best stylised-fact match, interpretable DAG   |
| C         | Hand-calibrated from papers  | Middle performance, low scalability           |

### 5.4 Timeline (8 weeks)

- W1–2: Finish data gaps (§4), run `10_compute_prior_anchors.py`, LLM elicitation.
- W3–4: LOB engine, 1000 simulation runs.
- W5–6: NOTEARS / LiNGAM, do-calculus.
- W7–8: Validation, ablation, write-up.

---

## 6. Tools

LLM: FinGPT (+ tool-calling) as primary; GPT-4o / Claude-3.5 as fallback (500–1000 prompts, ~$5–15). Causal: DoWhy + NOTEARS + LiNGAM. LOB: custom Python (or Mesa). Viz: NetworkX + matplotlib. Data: repo `stage2_economics` pipeline + additional funding-rate + 1-week aggTrades.

---

## 7. Addressing the 4 Objections (from the Problem-Solving tab)

### Problem 1 — "Is the LLM hallucinating stereotypes?"

**Defense.** (a) The LLM is framed as a **measurement instrument**, not an oracle, per Calibrating Behavioral Parameters with LLMs (arXiv 2025). (b) Every prompt is **anchored on real empirical percentiles** extracted from the 66-event dataset, turning LLM outputs into conditional responses rather than free generation. (c) Use FinGPT + tool-calling so the LLM can query actual empirical distributions, making its outputs contestable and auditable. (d) The ablation in §5.3 directly quantifies the LLM-prior's marginal contribution over uniform random and hand-calibrated baselines.

### Problem 2 — "Isn't causal discovery on simulator output just reverse-engineering your own code?"

**Defense.** The framing is explicitly **confirmatory causal validation**, not blind discovery: we know the ground truth because we coded Phase 2, and the research claim is that *standard causal-discovery algorithms can recover microstructure DAGs in a controlled, heterogeneous-agent setting where ground truth is known*. This is a methodological benchmark contribution, comparable to synthetic-data evaluation in the NOTEARS and LiNGAM original papers.

### Problem 3 — "Kirilenko's E-mini 2010 DAG is not a BTC ground truth."

**Defense.** Replace the E-mini benchmark with a **BTC-native, different-sample benchmark**: hold out a subset of the 66 events (temporal split), learn a reference DAG on held-out data, and compare to the simulator-derived DAG. Kirilenko (2017) is kept only as a qualitative reference for the *type* of mechanism (MM withdrawal + OFI + leverage), not as numerical ground truth.

### Problem 4 — "Static priors cannot model panic."

**Defense.** Panic is an **emergent property** of LOB structural interaction (MM withdrawal at `realized_vol > τ`, momentum amplification, spread→depth positive feedback). Static agent-level priors are furthermore a **prerequisite** for identifiable causal discovery: dynamic priors would conflate structural composition with real-time behavioral adaptation and break NOTEARS/LiNGAM stationarity. The causal claim `structural interaction → flash crash` is in fact *stronger* than `individual panic → flash crash` and closer to Glosten–Milgrom / Kyle microstructure theory.

---

## 8. Data-Prep Stop Criterion

Data preparation is complete when, for every one of the 66 events, `Event_Dynamics_100ms.csv` contains on a regular 100 ms grid:

`event_id, timestamp_ms, phase, close, ofi, spread_bps, depth_imbalance, kyle_lambda, vpin, amihud_illiq, trade_intensity, realized_vol_50, leverage_proxy, velocity_pct_per_100ms, drop_velocity_pct_per_100ms, panic_acceleration_pct_per_100ms2, order_flow_toxicity (renamed stress_proxy), delta_from_news_ms`

plus a companion `prior_anchors.json` with per-phase percentiles of OFI, Poisson λ, Pareto α, Kyle λ, realized vol. Once this contract is met, Phase 1 LLM elicitation can start and Phase 2 simulation can be calibrated against real statistics.
