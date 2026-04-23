# COMOSA / Algorithmic Panic - BTC Flash Crash Microstructure Research

> LLM-Elicited Behavioral Priors for Heterogeneous Agent Simulation of BTC Flash Crash Dynamics with Causal Learning.

This workspace is the market-only branch of the original project. The runnable code now focuses on BTCUSDT futures microstructure, flash crash event extraction, event-level feature engineering, and causal-analysis inputs for COMOSA. The old bio, WESAD, DREAMER, and PyTorch-heavy pipeline is no longer part of the active code path.

## Feature Audit At A Glance

The documentation currently targets a richer event-level contract than the checked-in `Event_Dynamics_100ms.csv` always contains. The table below separates:

- features required by the current COMOSA market-only docs,
- whether the feature is already produced by code today,
- whether the variable is literature-grounded, a reasonable proxy, or a heuristic composite,
- and whether it depends on BBO / book state.

### Event-Level Feature Contract

| Feature | Needed by docs | Current code status | Scientific status | How it is computed or sourced | Needs BBO? |
|------|------|------|------|------|------|
| `event_id` | Yes | Computed | Metadata | Event identifier from event catalog and refined timestamp pipeline | No |
| `date` | Yes | Computed | Metadata | Event date copied from catalog | No |
| `timestamp_ms` | Yes | Computed | Metadata | 100ms bin timestamp from event micro bars, then regularized by script 10 | No |
| `timestamp_utc` | Yes | Computed | Metadata | UTC conversion of `timestamp_ms` | No |
| `time_from_drop_start_ms` | Yes | Computed | Event-study alignment | `timestamp_ms - drop_start_ms` | No |
| `phase` | Yes | Computed | Event-study heuristic | `pre / drop / recovery / post` split around start, bottom, and 50% retrace recovery marker | No |
| `close` | Yes | Computed | Market microstructure standard | Last trade price in each fixed-width bin from `aggTrades` | No |
| `ofi` | Yes | Computed | Proxy | Sum of signed trade volume in bin using Binance `is_buyer_maker` side inference; useful but not full queue-based OFI | No |
| `trade_intensity` | Yes | Computed | Market microstructure standard | `trade_count / bin_seconds` | No |
| `amihud_illiq` | Yes | Computed | Literature-grounded | `abs(log_return) / dollar_volume` | No |
| `kyle_lambda` | Yes | Computed | Literature-grounded proxy | Rolling slope-like estimator `Cov(dp, signed_volume) / Var(signed_volume)` | No |
| `vpin` | Yes | Computed | Proxy | Rolling imbalance ratio using buy vs sell trade volume on time bars; not canonical volume-bucket VPIN | No |
| `realized_vol_50` | Yes | Computed | Literature-grounded | Rolling `sqrt(sum(log_return^2))` over 50 bins | No |
| `leverage_proxy` | Yes | Computed | Heuristic proxy | `abs(drop_velocity) * abs(panic_acceleration) / realized_vol_50`; intended as margin-call stress proxy, not direct leverage measurement | No |
| `order_flow_toxicity` | Yes | Computed | Heuristic composite | Equal-weight z-score composite of sell pressure from OFI, VPIN, Amihud, and realized volatility | No |
| `velocity_pct_per_100ms` | Yes | Computed | Kinematic transform | Per-bin percent return rescaled to 100ms units | No |
| `drop_velocity_pct_per_100ms` | Yes | Computed | Kinematic transform | Negative part of velocity, expressed as positive downward speed | No |
| `panic_acceleration_pct_per_100ms2` | Yes | Computed | Kinematic transform | First difference of velocity, rescaled to 100ms squared units | No |
| `drop_1s_pct` | Yes | Computed | Heuristic crash summary | Rolling 1-second sum of `drop_velocity_pct_per_100ms` | No |
| `drop_from_local_pct` | Yes | Computed | Heuristic local dislocation proxy | Drawdown from rolling local reference price over recent bins | No |
| `delta_from_news_ms` | Yes | Partially computed | Event-study standard | Offset to matched news timestamp when a timestamp-level news catalog exists; currently often empty because the catalog is missing | No |
| `mid_price` | Yes | Proxy computed in gridded output | Market microstructure standard when real, heuristic fallback today | Real target is `(best_bid_price + best_ask_price) / 2`; current fallback from script 12 is `close` | Yes |
| `spread_bps` | Yes | Proxy computed in gridded output | Market microstructure standard when real, proxy fallback today | Real target is `(best_ask - best_bid) / mid_price * 10000`; current fallback from script 12 rescales `amihud_illiq * close * 10000` | Yes |
| `touch_depth` | Yes | Proxy computed in gridded output | Market microstructure standard when real, proxy fallback today | Real target is `best_bid_qty + best_ask_qty`; current fallback from script 12 is winsorized inverse absolute `kyle_lambda` | Yes |
| `depth_imbalance` | Yes | Proxy computed in gridded output | Literature-grounded when real, proxy fallback today | Real target is `(best_bid_qty - best_ask_qty) / (best_bid_qty + best_ask_qty)`; current fallback from script 12 is clipped `ofi / (trade_intensity * 100)` | Yes |

### Current Reading Of The Pipeline

At the moment, the implemented event pipeline is strongest on trade-side features and crash dynamics. In practical terms:

- the current outputs already support a useful trade-only event study,
- the gridded CSV now includes transparent fallback proxies for the missing BBO fields,
- true BBO data is still preferable and should overwrite those proxies when available,
- and the weakest current variables are still the exploratory composites like `order_flow_toxicity` and `leverage_proxy`.

### What Is Missing Beyond BBO

The repo is not blocked only by missing BBO. These gaps also matter:

| Gap | Current status | Why it matters |
|------|------|------|
| Timestamp-level news catalog | Missing | `delta_from_news_ms` stays empty without `data/processed/tardis/news/news_events_utc.csv` |
| Direct leverage measurement | Missing | `leverage_proxy` is only a heuristic; direct measures would come from funding, open interest, or liquidation data |
| Canonical order-book OFI | Missing | Current `ofi` is signed trade imbalance, not queue-dynamics OFI from order-book updates |
| Canonical VPIN | Missing | Current implementation is a useful time-bar proxy, not full volume-bucket VPIN |
| Export of extra trade-side features | Fixed in current `Event_Dynamics_100ms.csv` | Script 09 now exports `volume`, `dollar_volume`, `buy_ratio`, `vwap`, `log_return`, `realized_vol_10`, `realized_vol_200`, and `ofi_autocorr_20` |
| Robust Phase-1 anchor logic | Partially fixed | Script 11 now excludes exact zero-filled OFI bins when building phase quantiles |

### How Missing BBO Can Be Obtained

The missing BBO features are in scope of the existing design and do not require inventing new formulas. They can be recovered in either of these ways:

| Source | How to use it | Expected outputs |
|------|------|------|
| Binance Vision `bookTicker` daily archives | Use [scripts/stage2_economics/05_download_event_ticks.py](scripts/stage2_economics/05_download_event_ticks.py) to download event windows with `aggTrades + bookTicker`, then rerun scripts 06, 09, 10, 11 | `mid_price`, `spread_bps`, `touch_depth`, `depth_imbalance`, plus any BBO-derived cross-features |
| Tardis order-book feeds (`incremental_book_L2`) | Rebuild best bid / ask snapshots from order-book updates when Binance `bookTicker` coverage is unavailable or incomplete | Same BBO fields as above, potentially with richer queue-depth features |
| Trade-only fallback proxy path | Use [scripts/stage2_economics/12_bbo_proxies.py](scripts/stage2_economics/12_bbo_proxies.py) after script 10 to append proxy BBO columns to the gridded CSV | Proxy `mid_price`, `spread_bps`, `touch_depth`, `depth_imbalance` for Phase 1 and Phase 3 unblocking |

If `bookTicker` is unavailable for a given period, the recommended fallback is not to fabricate BBO values from trades. The correct fallback is to reconstruct the touch from an order-book feed or explicitly run the analysis as a trade-only proxy pipeline.

## What This Repo Contains

The repository is organized around three linked goals:

| Phase | Goal | Status in this repo |
|------|------|---------------------|
| Phase 1 | Prepare empirical anchors for offline LLM prompting | Script support is present via scripts 10 and 11; the prompt-execution layer is not implemented here |
| Phase 2 | Build BTC flash crash event data from raw market feeds | Implemented |
| Phase 3 | Validate causal structure and export DAG-oriented artifacts | Implemented |

In practical terms, the codebase currently does four things well:

- fetches BTC futures data from Binance Vision or Tardis,
- audits and preprocesses raw market data,
- constructs event-level flash crash datasets at 10ms / 100ms / 1s resolution,
- exports artifacts used for DAG validation and Phase-1 prior calibration.

## Source Of Truth

If you are new to the workspace, read these files first:

1. `Comosa-Plan-Rewritten.md` - high-level research specification for the market-only COMOSA path.
2. `reports/DATA_PIPELINE.md` - lineage from raw BTC data to event-level outputs.
3. `config/settings.py` - central paths, thresholds, and pipeline constants.
4. `DAG/` - conceptual DAG notes and per-agent diagrams.

Important note: some files under `DAG/` still use older bio-technical language. The executable source code and `Comosa-Plan-Rewritten.md` are the source of truth for the current scope.

## Environment Setup

Create a clean Python environment and install the minimal dependencies:

```powershell
conda create -n stress python=3.10 -y
conda activate stress
pip install -r requirements.txt
```

Optional environment variable for paid Tardis downloads:

```powershell
$env:TARDIS_API_KEY = "your_key_here"
```

If `TARDIS_API_KEY` is not set, the fetch pipeline defaults to Binance Vision.

## Repository Layout

```text
config/
  settings.py                    Central configuration for paths and thresholds

scripts/
  phase1_data_engineering/
    00_fetch_tardis.py           Download BTC futures data
    03_audit_tardis.py           Run T1-T15 audit checks
    06_preprocess_tardis.py      Daily preprocessing pipeline
    07_extract_features.py       Aggregate processed parquet into market_features.csv
    09_stylized_facts.py         Validate stylized facts on BTC returns

  stage2_economics/
    00_reindex_ticks.py          Raw trades -> ms parquet
    01_build_multiresolution_bars.py
    02_hft_feature_engineering.py
    04_detect_flash_crashes.py   Detect flash crash events
    05_download_event_ticks.py   Download event windows (aggTrades + bookTicker)
    06_micro_feature_engineering.py
    07_dag_validation.py         Structural break, Granger, IRF analyses
    08_refine_event_timestamps.py
    09_produce_confounder_outputs.py
    10_augment_dynamics_features.py
    12_bbo_proxies.py
    11_compute_prior_anchors.py

src/
  analysis/ audit/ data/ features/ preprocessing/ utils/

data/
  raw/tardis/                    Raw downloads
  processed/tardis/              Main generated artifacts

reports/
  DATA_PIPELINE.md               Pipeline lineage documentation
  audit/                         Audit outputs
  validation/                    Currently empty in this branch

DAG/
  DAG.md
  Explanation_DAG.md
  DAG_for_4_agents/
```

## Recommended Workflows

There are three distinct workflows in the repo. Use the one that matches your goal.

### 1. Daily Market Pipeline

Use this path if you want a lightweight market-data build, audit, and stylized-facts check.

```powershell
conda activate stress

python scripts/phase1_data_engineering/00_fetch_tardis.py --mode klines
python scripts/phase1_data_engineering/03_audit_tardis.py
python scripts/phase1_data_engineering/06_preprocess_tardis.py
python scripts/phase1_data_engineering/07_extract_features.py
python scripts/phase1_data_engineering/09_stylized_facts.py
```

Typical outputs from this path:

- `data/processed/tardis/btc_features_1min.parquet`
- `data/processed/market_features.csv`
- `data/processed/tardis/stylized_facts.json`
- `reports/audit/tardis_audit.csv`

### 2. Core Event-Driven Flash Crash Pipeline

Use this path if you want the main COMOSA dataset for event-level analysis.

Important: `00_fetch_tardis.py` defaults to `--mode klines`. For the event pipeline you typically want raw trade data.

```powershell
conda activate stress

python scripts/phase1_data_engineering/00_fetch_tardis.py --mode aggtrades
python scripts/stage2_economics/00_reindex_ticks.py
python scripts/stage2_economics/04_detect_flash_crashes.py
python scripts/stage2_economics/05_download_event_ticks.py
python scripts/stage2_economics/06_micro_feature_engineering.py
python scripts/stage2_economics/08_refine_event_timestamps.py
python scripts/stage2_economics/09_produce_confounder_outputs.py
python scripts/stage2_economics/10_augment_dynamics_features.py
python scripts/stage2_economics/12_bbo_proxies.py
python scripts/stage2_economics/11_compute_prior_anchors.py
```

Key outputs from this path:

- `data/processed/tardis/event_catalog.csv`
- `data/processed/tardis/event_catalog_tick_refined.csv`
- `data/processed/tardis/confounder_outputs/Event_Dynamics_100ms.csv`
- `data/processed/tardis/confounder_outputs/Event_Dynamics_100ms_gridded.csv`
- `data/processed/tardis/confounder_outputs/Flash_Crash_Events_Labeled.csv`
- `data/processed/tardis/confounder_outputs/Empirical_Benchmarks.json`
- `data/processed/tardis/confounder_outputs/prior_anchors.json`

### 3. Optional HFT Bar And Diagnostics Pipeline

Use this path if you want bar-based microstructure features or DAG-oriented diagnostics beyond the event CSV.

```powershell
conda activate stress

python scripts/stage2_economics/01_build_multiresolution_bars.py
python scripts/stage2_economics/02_hft_feature_engineering.py
python scripts/stage2_economics/07_dag_validation.py
```

Outputs from this path include:

- `data/processed/tardis/bars/`
- `data/processed/tardis/hft_features/`
- `data/processed/tardis/dag_validation/granger_results.csv`
- `data/processed/tardis/dag_validation/impulse_responses.csv`
- `data/processed/tardis/dag_validation/crash_anatomy.csv`
- `data/processed/tardis/dag_validation/dag_validation_summary.json`

## Current Checked-In Artifacts

The repository already contains generated outputs under `data/processed/tardis/`, including:

- `btc_features_1min.parquet`
- `event_catalog.csv`
- `event_catalog_tick_refined.csv`
- `flash_crashes.csv`
- `stylized_facts.json`
- `confounder_outputs/Event_Dynamics_100ms.csv`
- `confounder_outputs/Flash_Crash_Events_Labeled.csv`
- `confounder_outputs/Empirical_Benchmarks.json`
- `dag_validation/granger_results.csv`
- `dag_validation/impulse_responses.csv`
- `dag_validation/crash_anatomy.csv`
- `dag_validation/regime_statistics.csv`

Some outputs are generated on demand and may not be committed in every branch, especially:

- `Event_Dynamics_100ms_gridded.csv`
- `prior_anchors.json`
- `market_features.csv`

## Important Script Notes

- `scripts/phase1_data_engineering/00_fetch_tardis.py` supports both Binance Vision and Tardis. Binance Vision is the free default.
- `scripts/stage2_economics/00_reindex_ticks.py` expects raw trade files under `data/raw/tardis/trades/`.
- `scripts/stage2_economics/05_download_event_ticks.py` downloads targeted event windows from Binance Vision, not from the earlier daily processed artifacts.
- `scripts/stage2_economics/10_augment_dynamics_features.py` regularizes the 100ms event data for NOTEARS / LiNGAM style downstream use.
- `scripts/stage2_economics/11_compute_prior_anchors.py` converts event-level empirical distributions into Phase-1 anchor statistics for offline LLM prompting.

## Research Positioning

This branch stops at empirical calibration and causal-analysis preparation. The following are planned conceptually but are not implemented as runnable modules in the current repo:

- the offline LLM prompt execution layer for archetype priors,
- the heterogeneous-agent LOB simulator,
- policy-intervention experiments on the simulator output.

In other words, the current repository is strongest on data engineering, market microstructure feature construction, event labeling, and causal-analysis inputs.

## Quick Start For New Readers

If you only want to understand the project without running everything:

1. Read `Comosa-Plan-Rewritten.md`.
2. Read `reports/DATA_PIPELINE.md`.
3. Inspect `config/settings.py`.
4. Open `data/processed/tardis/confounder_outputs/Event_Dynamics_100ms.csv`.
5. Review `data/processed/tardis/dag_validation/dag_validation_summary.json`.

## Last Updated

2026-04-23