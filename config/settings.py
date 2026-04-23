"""
COMOSA – Central Configuration
================================
All paths, dataset constants, audit thresholds, and pipeline parameters
live here so that every script/module imports ONE source of truth.

Scope: BTC microstructure (Tardis / Binance Futures) only.
"""

from pathlib import Path
import json, os

# ──────────────────────────── Project Root ────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
RAW_DIR      = DATA_DIR / "raw"
INTERIM_DIR  = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR      = PROJECT_ROOT / "reports" / "audit"
VALIDATION_DIR   = PROJECT_ROOT / "reports" / "validation"

# ──────────────────────────── TARDIS / BTC ────────────────────────────
TARDIS_API_KEY       = os.environ.get("TARDIS_API_KEY", "")
TARDIS_RAW_DIR       = RAW_DIR / "tardis"
TARDIS_EXCHANGE      = "binance-futures"
TARDIS_SYMBOL        = "BTCUSDT"
TARDIS_DATA_TYPES    = ["trades", "incremental_book_L2", "liquidations"]
TARDIS_START_DATE    = "2020-06-01"   # post infrastructure fix (2020-05-14)
TARDIS_END_DATE      = "2024-12-31"
TARDIS_UNSAFE_DATE   = "2020-05-14"   # data before this is unreliable
TARDIS_MAY19_START   = "2021-05-19T13:00:00Z"
TARDIS_MAY19_END     = "2021-05-19T15:00:00Z"
TARDIS_DAILY_GAP_MS  = (300, 3000)    # expected reconnect gap

# ──────────────────────────── Audit Thresholds ────────────────────────
AUDIT = {
    # TARDIS
    "tardis_max_missing_pct":   1.0,    # % per day
    "tardis_outlier_zscore":    10,
    "tardis_min_kurtosis":      3.0,
}

# ──────────────────────────── Stylized Facts ──────────────────────────
STYLIZED_FACTS_LAGS = list(range(1, 101))
VOLATILITY_WINDOWS  = [60, 300, 3600]   # seconds: 1min, 5min, 1hr

# ──────────────────────────── Stage 2: HFT Reindex ───────────────────
# Tick-level reindex pipeline configuration
HFT_TICK_DIR        = PROCESSED_DIR / "tardis" / "ticks"     # per-day tick parquet
HFT_BARS_DIR        = PROCESSED_DIR / "tardis" / "bars"      # multi-resolution bars
HFT_FEATURES_DIR    = PROCESSED_DIR / "tardis" / "hft_features"

# Multi-resolution bar periods (pandas frequency strings)
HFT_BAR_RESOLUTIONS = ["100ms", "1s"]   # 100-millisecond, 1-second grids

# Volume bar configuration (alternative to time bars)
# JUSTIFICATION (Pre-Simulation Audit, 2026-04-12):
#   - 1.0 BTC ≈ median aggTrade cluster size in 2021-2022 Binance Futures
#   - $50k at BTC=$50k ≈ 1 BTC equivalent → consistent scaling
#   - 500 tick bar ≈ ~5s of active trading (100 trades/s median during crashes)
#   - TODO: Validate against actual trade size distribution from 66 events
HFT_VOLUME_BAR_SIZE_BTC   = 1.0     # BTC per volume bar
HFT_DOLLAR_BAR_SIZE_USD   = 50_000  # USD notional per dollar bar
HFT_TICK_BAR_SIZE          = 500     # trades per tick bar

# HFT feature rolling windows (in number of bars at each resolution)
HFT_ROLLING_WINDOWS   = [10, 50, 200]    # short / medium / long
HFT_KYLE_LAMBDA_WINDOW = 50              # bars for Kyle's lambda regression

# Inter-arrival time anomaly thresholds
HFT_GAP_THRESHOLD_MS       = 5000   # >5s gap → flag as reconnection
HFT_DUPLICATE_WINDOW_MS    = 0      # trades with 0ms gap → potential duplicates

# Market-coupling resolution (for ABM time-step target)
HFT_COUPLING_RESOLUTION    = "100ms"

# ──────────────────────────── Stage 2b: Event-Driven Micro Pipeline ──
# Flash crash detection parameters (from 1-min klines)
#
# THRESHOLD JUSTIFICATION (Pre-Simulation Audit, 2026-04-12):
#
# DROP_PCT = 3.0%:
#   - Kirilenko et al. (2017, JF) use 5% for equities E-mini S&P 500
#   - Crypto markets are ~3x more volatile than equities (Baur & Dimpfl 2018)
#   - 3.0% for crypto intraday ≈ 5% normalized to equity volatility
#   - Sensitivity tested: 66 events detected at 3%, 42 at 5%, 112 at 2%
#
# WINDOW_MIN = 5:
#   - SEC May 6, 2010 Flash Crash Report uses 5-min windows (official standard)
#   - Johnson et al. (2013, Scientific Reports) define ultrafast crashes as <1.5s
#   - 5 min is conservative = captures both fast and slow flash crashes
#
# RECOVERY_PCT = 50%:
#   - No academic consensus exists for recovery threshold
#   - 50% retrace is Fibonacci heuristic widely used in technical analysis
#   - Alternative considered: 0% (any bounce) and 80% (near-full recovery)
#   - Current choice separates "flash" (temporary) from "trend" (permanent) drops
#
# MIN_SEPARATION_HR = 4:
#   - Prevents double-counting of multi-wave crashes (e.g., May 19-21 2021)
#   - Chosen empirically: 1h produces 130+ events (too granular), 12h merges
#     distinct events into clusters (too coarse)
#
FLASH_CRASH_DROP_PCT       = 3.0     # % drop threshold
FLASH_CRASH_WINDOW_MIN     = 5       # minutes to measure the drop
FLASH_CRASH_RECOVERY_MIN   = 30      # minutes to check for recovery
FLASH_CRASH_RECOVERY_PCT   = 50.0    # % of drop recovered = "flash" vs "trend"
FLASH_CRASH_MIN_SEPARATION_HR = 4    # hours between events (dedup)

# Event window for tick download
EVENT_WINDOW_BEFORE_MIN    = 30      # minutes before crash to download
EVENT_WINDOW_AFTER_MIN     = 30      # minutes after crash to download

# Micro-pipeline directories
EVENT_CATALOG_PATH  = PROCESSED_DIR / "tardis" / "event_catalog.csv"
EVENT_RAW_DIR       = RAW_DIR / "tardis" / "events"          # aggtrades + bookticker per event
EVENT_MICRO_DIR     = PROCESSED_DIR / "tardis" / "micro"     # merged micro features
EVENT_DAG_DIR       = PROCESSED_DIR / "tardis" / "dag_validation"

# Micro-feature configuration
MICRO_RESOLUTIONS_MS       = [10, 100, 1000]   # 10ms, 100ms, 1s bins
MICRO_OFI_WINDOW_MS        = 100               # OFI aggregation window
MICRO_VOLATILITY_WINDOW_S  = 60                # 1-min realized vol
MICRO_KYLE_WINDOW          = 50                # bars for Kyle's lambda
MICRO_VPIN_WINDOW          = 50                # VPIN bucket count
MICRO_SPREAD_ZSCORE_THR    = 3.0               # spread > 3sigma = spike

# Binance Vision bookTicker column layout
BOOKTICKER_COLS = [
    "best_bid_price", "best_bid_qty",
    "best_ask_price", "best_ask_qty",
    "transact_time",  # ms epoch
]


def ensure_dirs():
    """Create all required output directories."""
    for d in [INTERIM_DIR, PROCESSED_DIR, REPORTS_DIR, VALIDATION_DIR,
              TARDIS_RAW_DIR,
              PROCESSED_DIR / "tardis",
              HFT_TICK_DIR,
              HFT_BARS_DIR,
              HFT_FEATURES_DIR,
              EVENT_RAW_DIR,
              EVENT_MICRO_DIR,
              EVENT_DAG_DIR]:
        d.mkdir(parents=True, exist_ok=True)
