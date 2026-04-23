# Data Pipeline Documentation

> End-to-end lineage from raw Binance Futures data to NOTEARS-ready gridded CSV.

---

## Pipeline Overview

```
Binance Vision / Tardis API
    ↓  (00_fetch_tardis)
data/raw/tardis/  *.csv.gz
    ↓  (03_audit_tardis)           → reports/audit/tardis_audit.csv
    ↓  (06_preprocess_tardis)      → data/processed/tardis/ 1-min OHLCV
    ↓  (stage2: 00_reindex_ticks)  → ms-resolution parquet
    ↓  (stage2: 04_detect_flash_crashes)
data/processed/tardis/event_catalog.csv   (66 events)
    ↓  (stage2: 05_download_event_ticks)
data/processed/tardis/events/event_NNN/   aggTrades + bookTicker
    ↓  (stage2: 06_micro_feature_engineering)
data/processed/tardis/event_micro/event_NNN.parquet  (10/100/1000ms)
    ↓  (stage2: 08_refine_event_timestamps)
    ↓  (stage2: 09_produce_confounder_outputs)
data/processed/tardis/confounder_outputs/
    Event_Dynamics_100ms.csv              ← primary output
    Flash_Crash_Events_Labeled.csv
    Empirical_Benchmarks.json
    ↓  (stage2: 10_augment_dynamics_features)
    Event_Dynamics_100ms_gridded.csv      ← regular grid for NOTEARS
    ↓  (stage2: 11_compute_prior_anchors)
    prior_anchors.json                    ← Phase-1 LLM input
```

---

## Flash Crash Detection Criteria

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `DROP_PCT` | 3.0% | Kirilenko et al. (2017): 5-min 5% for E-mini; scaled to crypto 24/7 vol |
| `WINDOW_MIN` | 5 min | SEC (2010) flash crash definition: rapid decline within minutes |
| `RECOVERY_PCT` | 50% | Cont (2010): V-shaped recovery threshold; SEC (2015) circuit breaker calibration |
| `MIN_SEPARATION_HR` | 4 h | Independence assumption; prevents double-counting same event |

Result: **66 flash crash events** spanning 2020-06 to 2024-12.

---

## Event_Dynamics_100ms.csv Schema

| Column | Type | Source Script | Description |
|--------|------|--------------|-------------|
| `event_id` | int | 09 | Event identifier (0-65) |
| `timestamp` | datetime | 06 | UTC timestamp at 100ms resolution |
| `phase` | str | 09 | pre_crash / crash / recovery |
| `close` | float | 06 | Mid-price at bin close |
| `ofi` | float | 06 | Signed order flow imbalance |
| `spread_bps` | float | 06 | Bid-ask spread in basis points |
| `depth_imbalance` | float | 06 | (Q_bid − Q_ask) / (Q_bid + Q_ask) |
| `mid_price` | float | 06 | (best_bid + best_ask) / 2 |
| `touch_depth` | float | 06 | bid_qty + ask_qty at BBO |
| `kyle_lambda` | float | 06 | Rolling price impact coefficient |
| `vpin` | float | 06 | Volume-weighted PIN |
| `amihud_illiq` | float | 06 | Amihud illiquidity ratio |
| `realized_vol_50` | float | 06 | 50-bin realized volatility |
| `trade_intensity` | float | 06 | Trades per second |
| `leverage_proxy` | float | 09 | \|velocity\| × \|acceleration\| / σ |
| `order_flow_toxicity` | float | 09 | Z-score composite of OFI, VPIN, Kyle λ |

---

## Gridded CSV (10_augment)

The NOTEARS and LiNGAM algorithms require strictly regular time series. Script 10 re-indexes each event onto an exact 100ms grid:

- **Book-state columns** (spread, depth, mid_price, close): forward-filled
- **Trade-flow columns** (ofi, trade_intensity): zero-filled (no trades = 0 flow)
- **Derived columns** (kyle_lambda, vpin, amihud_illiq, realized_vol_50): linearly interpolated

Output: `Event_Dynamics_100ms_gridded.csv` with identical schema.

---

## Prior Anchors JSON (11_compute)

Per-phase empirical statistics for anchoring LLM prompts in Phase 1:

```json
{
  "pre_crash": {
    "ofi_p25": ..., "ofi_p50": ..., "ofi_p75": ...,
    "trade_intensity_mean": ...,        // → Poisson λ
    "trade_intensity_std": ...,
    "pareto_alpha": ...,                // Clauset-Shalizi-Newman MLE
    "kyle_lambda_median": ...,
    "realized_vol_50_median": ...,
    "spread_bps_median": ...,
    "depth_imbalance_median": ...
  },
  "crash": { ... },
  "recovery": { ... }
}
```

---

## Audit Report

Script `03_audit_tardis.py` produces `reports/audit/tardis_audit.csv` with 15 checks (T1–T15):

- T1: Data completeness (daily gaps)
- T2: Timestamp monotonicity
- T3: Price outlier detection (z > 10)
- T5: Kurtosis check (> 3.0 for fat tails)
- T7: Missing data percentage (< 1%)
- T10-T15: Orderbook integrity, spread validity, trade–quote alignment

---

## Stylized Facts Validation

Script `09_stylized_facts.py` (phase1) validates SF-1 through SF-5:

| ID | Fact | Test |
|----|------|------|
| SF-1 | Fat tails in returns | Kurtosis > 3, Jarque-Bera p < 0.05 |
| SF-2 | Volatility clustering | ARCH-LM test, squared returns ACF |
| SF-3 | Volume-volatility correlation | Pearson ρ between volume and |return| |
| SF-4 | Leverage effect | Asymmetric volatility response to negative returns |
| SF-5 | Long memory in |returns| | Hurst exponent > 0.5 via R/S analysis |
