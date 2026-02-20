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


def ensure_dirs():
    """Create all required output directories."""
    for d in [INTERIM_DIR, PROCESSED_DIR, REPORTS_DIR, VALIDATION_DIR,
              TARDIS_RAW_DIR,
              PROCESSED_DIR / "wesad",
              PROCESSED_DIR / "dreamer",
              PROCESSED_DIR / "tardis"]:
        d.mkdir(parents=True, exist_ok=True)
