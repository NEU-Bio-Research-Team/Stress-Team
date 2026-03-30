"""
Algorithmic Panic – Central Configuration
==========================================
All paths, dataset constants, audit thresholds, and pipeline parameters
live here so that every script/module imports ONE source of truth.
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

# ──────────────────────────── WESAD ────────────────────────────────────
WESAD_RAW_DIR       = RAW_DIR / "wesad" / "WESAD_extracted" / "WESAD"
WESAD_EXPECTED_SUBJECTS = [f"S{i}" for i in range(2, 18) if i != 12]  # S2-S17 minus S12
WESAD_CHEST_SR      = 700       # Hz – RespiBAN chest device
WESAD_WRIST_BVP_SR  = 64        # Hz – Empatica E4 BVP
WESAD_WRIST_EDA_SR  = 4         # Hz – Empatica E4 EDA
WESAD_WRIST_TEMP_SR = 4         # Hz – Empatica E4 TEMP
WESAD_WRIST_ACC_SR  = 32        # Hz – Empatica E4 ACC
WESAD_LABELS = {0: "undefined", 1: "baseline", 2: "stress",
                3: "amusement", 4: "meditation", 5: "undefined_2",
                6: "undefined_3", 7: "undefined_4"}
WESAD_WINDOW_SEC    = 5         # seconds – non-overlapping window
WESAD_ECG_BANDPASS  = (0.5, 40) # Hz
WESAD_IBI_RANGE     = (250, 2000)  # ms – physiological range for R-R

# ──────────────────────────── DREAMER ─────────────────────────────────
DREAMER_MAT_PATH    = RAW_DIR / "dreamer" / "DREAMER.mat"
DREAMER_N_SUBJECTS  = 23
DREAMER_N_TRIALS    = 18
DREAMER_EEG_SR      = 128       # Hz – Emotiv EPOC
DREAMER_EEG_CHANNELS = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"
]
DREAMER_EEG_BANDPASS  = (0.1, 40)    # Hz
DREAMER_NOTCH_FREQ    = 50           # Hz (powerline 50 Hz)
DREAMER_NOTCH_BW      = 2            # Hz (48-52 Hz)
DREAMER_BASELINE_SEC  = 61           # seconds per trial
DREAMER_STRESS_AROUSAL_THR = 3       # >=3 → high arousal
DREAMER_STRESS_VALENCE_THR = 3       # <=3 → low valence
# Stress proxy: low valence AND high arousal
DREAMER_ICA_N_COMPONENTS = 13        # recommended for 14 channels

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
    # WESAD
    "wesad_min_subjects":       15,
    "wesad_max_missing_pct":    5.0,    # %
    "wesad_sync_tol_pct":       2.0,    # % – max allowable deviation in wrist sample count vs. expected
    "wesad_snr_min_db":         10,
    "wesad_kappa_min":          0.6,
    # WESAD stress fraction ~10-13% of total recording (1 stress block / ~80 min protocol)
    "wesad_stress_mean_range":  (0.08, 0.18),
    "wesad_stress_std_range":   (0.05, 0.15),
    # DREAMER
    "dreamer_min_subjects":     23,
    "dreamer_n_channels":       14,
    "dreamer_artifact_std_thr": 3.0,
    "dreamer_brain_comp_pct":   70,
    # TARDIS
    "tardis_max_missing_pct":   1.0,    # % per day
    "tardis_outlier_zscore":    10,
    "tardis_min_kurtosis":      3.0,
}

# ──────────────────────────── Phase 2 Validation ──────────────────────
VALIDATION = {
    # Learnability thresholds
    "strong_signal_threshold":  0.70,   # balanced_accuracy for "strong signal"
    "weak_signal_threshold":    0.55,   # below this → "no signal"
    # Subject probe
    "subject_probe_folds":      5,
    "encoding_ratio_high":      5.0,    # probe_acc / chance
    "encoding_ratio_moderate":  2.0,
    # Permutation test
    "n_permutations":           100,
    "p_value_threshold":        0.05,
    # Adversarial
    "grl_lambda":               1.0,    # gradient reversal strength
    "grl_epochs":               100,
    # Feature stability
    "kendall_tau_stable":       0.7,
    "kendall_tau_moderate":     0.4,
    # DREAMER ICA
    "ica_flagged_subjects":     ["S10", "S17", "S21", "S23"],
    "ica_skip_threshold":       0.01,   # <1% difference → skip ICA
    "ica_required_threshold":   0.03,   # >3% → ICA required
    # Ablation
    "feature_critical_delta":   0.03,   # drop > 3% → critical feature
}

# ──────────────────────────── Coupling / Cross-dataset ────────────────
STRESS_TIME_RESOLUTION_SEC = 5   # σ(t) output resolution
STRESS_VALUE_RANGE = (0.0, 1.0)  # calibrated probability
STRESS_EXPECTED_MEAN = (0.2, 0.4)
STRESS_EXPECTED_STD  = (0.05, 0.15)

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
HFT_VOLUME_BAR_SIZE_BTC   = 1.0     # BTC per volume bar
HFT_DOLLAR_BAR_SIZE_USD   = 50_000  # USD notional per dollar bar
HFT_TICK_BAR_SIZE          = 500     # trades per tick bar

# HFT feature rolling windows (in number of bars at each resolution)
HFT_ROLLING_WINDOWS   = [10, 50, 200]    # short / medium / long
HFT_KYLE_LAMBDA_WINDOW = 50              # bars for Kyle's lambda regression

# Inter-arrival time anomaly thresholds
HFT_GAP_THRESHOLD_MS       = 5000   # >5s gap → flag as reconnection
HFT_DUPLICATE_WINDOW_MS    = 0      # trades with 0ms gap → potential duplicates

# Bio-market coupling resolution
HFT_COUPLING_RESOLUTION    = "100ms"     # ABM time-step target
HFT_BIO_IBI_PRECISION_MS   = 1           # R-R interval precision

# ──────────────────────────── Stage 2b: Event-Driven Micro Pipeline ──
# Flash crash detection parameters (from 1-min klines)
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
              PROCESSED_DIR / "wesad",
              PROCESSED_DIR / "dreamer",
              PROCESSED_DIR / "tardis",
              HFT_TICK_DIR,
              HFT_BARS_DIR,
              HFT_FEATURES_DIR,
              EVENT_RAW_DIR,
              EVENT_MICRO_DIR,
              EVENT_DAG_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    """Create all required output directories."""
    for d in [INTERIM_DIR, PROCESSED_DIR, REPORTS_DIR, VALIDATION_DIR,
              TARDIS_RAW_DIR,
              PROCESSED_DIR / "wesad",
              PROCESSED_DIR / "dreamer",
              PROCESSED_DIR / "tardis",
              HFT_TICK_DIR,
              HFT_BARS_DIR,
              HFT_FEATURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)
