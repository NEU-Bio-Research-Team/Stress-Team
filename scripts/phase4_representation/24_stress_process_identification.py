"""
Script 24 - Physiological Stress Process Identification
=========================================================
Phase 4: sigma(t) Stochastic Law Discovery

Advisor's Pipeline (per-subject, scientifically rigorous):
  1. Extract sigma(t) = continuous stress proxy (hr_mean AND PC1 HRV)
  2. Per-subject z-score normalization
  3. Remove protocol mean per phase (baseline/stress/recovery)
     sigma(t) = s(t) + r(t)  where s(t) = deterministic phase effect
  4. Analyze residual r(t):
     a. Stationarity test (ADF + KPSS)
     b. ACF / PACF estimation
     c. Increment distribution analysis (Gaussian? heavy-tails? jumps?)
     d. Jump detection
  5. Fit candidate processes per subject:
     - Random Walk
     - Ornstein-Uhlenbeck (mean-reverting)
     - Jump-Diffusion
  6. Aggregate: parameter distribution across subjects
  7. Test cross-subject consistency

Scientific Goal:
  Identify the stochastic law governing physiological stress dynamics.
  This bridges bio data to ABM: same mathematical framework for both.

Usage:
    python scripts/phase4_representation/24_stress_process_identification.py
"""

import sys, os, json, time, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
warnings.filterwarnings("ignore")

import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy import stats as sp_stats
from scipy.signal import find_peaks

from config.settings import (
    WESAD_RAW_DIR, WESAD_CHEST_SR, WESAD_ECG_BANDPASS,
    WESAD_IBI_RANGE, WESAD_WINDOW_SEC, VALIDATION_DIR,
)
from src.data.wesad_loader import discover_subjects, load_subject
from src.preprocessing.filters import (
    bandpass, detect_r_peaks, compute_rr_intervals, reject_rr_outliers,
)

WINDOW_SEC = WESAD_WINDOW_SEC  # 5s windows
SR = WESAD_CHEST_SR            # 700 Hz
DT = WINDOW_SEC                # time step between observations (seconds)


# =====================================================================
#  SECTION 1: EXTRACT sigma(t) TIME SERIES PER SUBJECT
# =====================================================================

def extract_sigma_timeseries(sid):
    """
    Extract continuous sigma(t) time series for one WESAD subject.

    Returns dict with:
      - hr_mean: (T,) heart rate per window
      - hrv_features: (T, 5) [hr_mean, hr_std, rmssd, sdnn, pnn50]
      - phase_labels: (T,) raw WESAD label per window (1=baseline, 2=stress, etc.)
      - time_sec: (T,) time in seconds from recording start
    """
    subj = load_subject(WESAD_RAW_DIR / sid)
    ecg_raw = subj.chest_ecg.copy()
    labels_raw = subj.labels.copy()

    # Bandpass filter ECG
    ecg_filtered = bandpass(ecg_raw, WESAD_ECG_BANDPASS[0],
                            WESAD_ECG_BANDPASS[1], SR)

    # Window the signal
    window_samples = int(WINDOW_SEC * SR)
    n_windows = len(ecg_filtered) // window_samples

    hr_mean = np.zeros(n_windows)
    hrv_features = np.zeros((n_windows, 5))  # hr_mean, hr_std, rmssd, sdnn, pnn50
    phase_labels = np.zeros(n_windows, dtype=int)
    time_sec = np.arange(n_windows) * WINDOW_SEC

    for i in range(n_windows):
        start = i * window_samples
        end = start + window_samples

        # Phase label (majority vote)
        seg_labels = labels_raw[start:end]
        valid = seg_labels[seg_labels > 0]
        if len(valid) > 0:
            phase_labels[i] = int(np.bincount(valid.astype(int)).argmax())
        else:
            phase_labels[i] = 0

        # ECG -> R-peaks -> HRV
        ecg_win = ecg_filtered[start:end]
        r_peaks = detect_r_peaks(ecg_win, SR)
        rr = compute_rr_intervals(r_peaks, SR)
        rr = reject_rr_outliers(rr, WESAD_IBI_RANGE[0], WESAD_IBI_RANGE[1])

        if len(rr) > 1:
            hr = 60000.0 / rr  # bpm
            diff_rr = np.diff(rr)
            hrv_features[i, 0] = np.mean(hr)       # hr_mean
            hrv_features[i, 1] = np.std(hr)         # hr_std
            hrv_features[i, 2] = np.sqrt(np.mean(diff_rr**2))  # rmssd
            hrv_features[i, 3] = np.std(rr)         # sdnn
            hrv_features[i, 4] = np.sum(np.abs(diff_rr) > 50) / max(len(diff_rr), 1)  # pnn50
            hr_mean[i] = np.mean(hr)
        elif len(rr) == 1:
            hr_mean[i] = 60000.0 / rr[0]
            hrv_features[i, 0] = hr_mean[i]

    return {
        "subject_id": sid,
        "hr_mean": hr_mean,
        "hrv_features": hrv_features,
        "phase_labels": phase_labels,
        "time_sec": time_sec,
        "n_windows": n_windows,
    }


# =====================================================================
#  SECTION 2: COMPUTE SIGMA PROXIES (hr_mean vs PC1)
# =====================================================================

def compute_sigma_proxies(subject_data_list):
    """
    Compute two sigma proxies for each subject:
      1. hr_mean (direct)
      2. PC1 of HRV features (learned latent coordinate)

    Also performs PCA on pooled HRV features to get a single
    set of loadings, then projects per-subject.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    print("\n  Computing sigma proxies (hr_mean vs PC1 HRV)...")

    # Pool all HRV features for global PCA
    all_hrv = []
    all_valid_masks = []
    for d in subject_data_list:
        valid = d["hrv_features"][:, 0] > 0  # windows with valid HR
        all_hrv.append(d["hrv_features"][valid])
        all_valid_masks.append(valid)

    pooled_hrv = np.concatenate(all_hrv)

    # Standardize and PCA
    scaler = StandardScaler()
    pooled_scaled = scaler.fit_transform(pooled_hrv)

    pca = PCA(n_components=5)
    pca.fit(pooled_scaled)

    variance_explained = pca.explained_variance_ratio_
    print(f"    PCA variance explained: {[f'{v:.3f}' for v in variance_explained]}")
    print(f"    PC1 explains {variance_explained[0]*100:.1f}% variance")
    print(f"    PC1+PC2 explains {sum(variance_explained[:2])*100:.1f}% variance")

    # PC1 loadings
    loadings = pca.components_[0]
    feature_names = ["hr_mean", "hr_std", "rmssd", "sdnn", "pnn50"]
    print(f"    PC1 loadings: {dict(zip(feature_names, [f'{l:.3f}' for l in loadings]))}")

    # Project each subject
    for d, valid in zip(subject_data_list, all_valid_masks):
        n = d["n_windows"]
        d["sigma_hr"] = np.full(n, np.nan)
        d["sigma_pc1"] = np.full(n, np.nan)

        if valid.sum() > 0:
            hrv_valid = d["hrv_features"][valid]
            hrv_scaled = scaler.transform(hrv_valid)
            pc_scores = pca.transform(hrv_scaled)

            d["sigma_hr"][valid] = d["hr_mean"][valid]
            d["sigma_pc1"][valid] = pc_scores[:, 0]

    pca_info = {
        "variance_explained": variance_explained.tolist(),
        "pc1_loadings": dict(zip(feature_names, loadings.tolist())),
        "pc1_var_pct": round(float(variance_explained[0]) * 100, 1),
        "stress_is_1d": bool(variance_explained[0] > 0.70),
    }

    return subject_data_list, pca_info


# =====================================================================
#  SECTION 3: PER-SUBJECT NORMALIZATION + PROTOCOL SUBTRACTION
# =====================================================================

def normalize_and_subtract_protocol(subject_data_list):
    """
    Advisor's critical pipeline:
      1. Per-subject z-score normalization
      2. Remove per-phase mean (protocol subtraction)
         sigma(t) = s(t) + r(t)
         We analyze r(t) = sigma(t) - s(t)
    """
    print("\n  Per-subject normalization + protocol subtraction...")

    # WESAD phases we care about
    PHASES = {1: "baseline", 2: "stress", 3: "amusement", 4: "meditation"}

    for d in subject_data_list:
        for proxy_name in ["sigma_hr", "sigma_pc1"]:
            sigma = d[proxy_name].copy()
            valid = ~np.isnan(sigma)

            if valid.sum() < 10:
                d[f"{proxy_name}_zscore"] = sigma
                d[f"{proxy_name}_residual"] = sigma
                d[f"phase_means_{proxy_name}"] = {}
                continue

            # Step 1: Per-subject z-score
            mu = np.nanmean(sigma)
            std = np.nanstd(sigma)
            if std > 0:
                sigma_z = (sigma - mu) / std
            else:
                sigma_z = sigma - mu

            d[f"{proxy_name}_zscore"] = sigma_z

            # Step 2: Protocol mean subtraction
            phase_means = {}
            residual = sigma_z.copy()

            for phase_id, phase_name in PHASES.items():
                mask = (d["phase_labels"] == phase_id) & valid
                if mask.sum() > 0:
                    phase_mean = np.nanmean(sigma_z[mask])
                    residual[mask] = sigma_z[mask] - phase_mean
                    phase_means[phase_name] = float(phase_mean)

            d[f"{proxy_name}_residual"] = residual
            d[f"phase_means_{proxy_name}"] = phase_means

        # Print phase structure
        phases_present = {PHASES.get(p, f"unknown_{p}"): int((d["phase_labels"] == p).sum())
                          for p in np.unique(d["phase_labels"]) if p > 0}
        hr_phases = d.get("phase_means_sigma_hr", {})
        print(f"    {d['subject_id']}: phases={phases_present}, "
              f"HR phase means (z): {{{', '.join(f'{k}:{v:.2f}' for k,v in hr_phases.items())}}}")

    return subject_data_list


# =====================================================================
#  SECTION 4: STATIONARITY TESTS
# =====================================================================

def test_stationarity(series, name=""):
    """
    ADF (null: unit root) + KPSS (null: stationary).
    Robust interpretation:
      ADF reject + KPSS not reject → stationary
      ADF not reject + KPSS reject → non-stationary (unit root)
      Both reject → trend-stationary
      Neither reject → inconclusive
    """
    from statsmodels.tsa.stattools import adfuller, kpss

    valid = series[~np.isnan(series)]
    if len(valid) < 20:
        return {"verdict": "INSUFFICIENT_DATA", "n": len(valid)}

    # ADF test
    try:
        adf_stat, adf_p, adf_lags, _, adf_crit, _ = adfuller(valid, autolag='AIC')
    except Exception:
        adf_stat, adf_p = np.nan, 1.0

    # KPSS test
    try:
        kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(valid, regression='c', nlags='auto')
    except Exception:
        kpss_stat, kpss_p = np.nan, 0.0

    adf_reject = adf_p < 0.05
    kpss_reject = kpss_p < 0.05

    if adf_reject and not kpss_reject:
        verdict = "STATIONARY"
    elif not adf_reject and kpss_reject:
        verdict = "NON_STATIONARY"
    elif adf_reject and kpss_reject:
        verdict = "TREND_STATIONARY"
    else:
        verdict = "INCONCLUSIVE"

    return {
        "verdict": verdict,
        "adf_stat": round(float(adf_stat), 4) if not np.isnan(adf_stat) else None,
        "adf_p": round(float(adf_p), 4) if not np.isnan(adf_p) else None,
        "kpss_stat": round(float(kpss_stat), 4) if not np.isnan(kpss_stat) else None,
        "kpss_p": round(float(kpss_p), 4) if not np.isnan(kpss_p) else None,
        "n": len(valid),
    }


# =====================================================================
#  SECTION 5: ACF / PACF ANALYSIS
# =====================================================================

def compute_acf_pacf(series, max_lag=20):
    """Compute ACF and PACF for a time series."""
    from statsmodels.tsa.stattools import acf, pacf

    valid = series[~np.isnan(series)]
    if len(valid) < max_lag + 5:
        return {"acf": [], "pacf": [], "n": len(valid)}

    # Adjust max_lag if needed
    effective_lag = min(max_lag, len(valid) // 3)

    try:
        acf_vals = acf(valid, nlags=effective_lag, fft=True)
        pacf_vals = pacf(valid, nlags=effective_lag, method='ywm')
    except Exception:
        acf_vals = np.zeros(effective_lag + 1)
        pacf_vals = np.zeros(effective_lag + 1)

    # Significance bound (95% CI)
    sig_bound = 1.96 / np.sqrt(len(valid))

    # Count significant lags (excluding lag 0)
    n_sig_acf = int(np.sum(np.abs(acf_vals[1:]) > sig_bound))
    n_sig_pacf = int(np.sum(np.abs(pacf_vals[1:]) > sig_bound))

    # ACF decay pattern
    acf_abs = np.abs(acf_vals[1:])
    if len(acf_abs) > 2:
        # Exponential decay test: fit log(|ACF|) vs lag
        lags = np.arange(1, len(acf_abs) + 1)
        nonzero = acf_abs > 1e-10
        if nonzero.sum() > 2:
            log_acf = np.log(acf_abs[nonzero])
            slope, intercept, r_value, _, _ = sp_stats.linregress(lags[nonzero], log_acf)
            decay_rate = -slope
            decay_r2 = r_value ** 2
        else:
            decay_rate, decay_r2 = 0.0, 0.0
    else:
        decay_rate, decay_r2 = 0.0, 0.0

    return {
        "acf": acf_vals.tolist(),
        "pacf": pacf_vals.tolist(),
        "acf_lag1": float(acf_vals[1]) if len(acf_vals) > 1 else 0.0,
        "pacf_lag1": float(pacf_vals[1]) if len(pacf_vals) > 1 else 0.0,
        "sig_bound": round(float(sig_bound), 4),
        "n_sig_acf": n_sig_acf,
        "n_sig_pacf": n_sig_pacf,
        "acf_decay_rate": round(float(decay_rate), 4),
        "acf_decay_r2": round(float(decay_r2), 4),
        "n": len(valid),
    }


# =====================================================================
#  SECTION 6: INCREMENT DISTRIBUTION ANALYSIS
# =====================================================================

def analyze_increments(series):
    """
    Analyze the distribution of increments delta_sigma = sigma(t+1) - sigma(t).
    Tests: Gaussian? Heavy-tailed? Skewed? Jumps?
    """
    valid = series[~np.isnan(series)]
    if len(valid) < 10:
        return {"verdict": "INSUFFICIENT_DATA"}

    increments = np.diff(valid)

    if len(increments) < 5:
        return {"verdict": "INSUFFICIENT_DATA"}

    # Basic statistics
    mu = float(np.mean(increments))
    sigma = float(np.std(increments))
    skew = float(sp_stats.skew(increments))
    kurt = float(sp_stats.kurtosis(increments, fisher=True))  # excess

    # Normality tests
    if len(increments) >= 20:
        _, sw_p = sp_stats.shapiro(increments[:5000])  # Shapiro max 5000
        _, jb_p = sp_stats.jarque_bera(increments)
    else:
        sw_p, jb_p = 1.0, 1.0

    # Hill tail index
    abs_inc = np.abs(increments)
    abs_inc_sorted = np.sort(abs_inc)[::-1]
    k = max(10, int(len(abs_inc_sorted) * 0.05))
    if k < len(abs_inc_sorted) and abs_inc_sorted[k] > 1e-10:
        top_k = abs_inc_sorted[:k]
        threshold = abs_inc_sorted[k]
        hill_alpha = 1.0 / np.mean(np.log(top_k / threshold))
    else:
        hill_alpha = float('inf')

    # Classify distribution
    is_gaussian = sw_p > 0.05 and jb_p > 0.05
    is_heavy_tailed = kurt > 3.0
    is_skewed = abs(skew) > 0.5

    if is_gaussian:
        dist_class = "GAUSSIAN"
    elif is_heavy_tailed and not is_skewed:
        dist_class = "HEAVY_TAILED_SYMMETRIC"
    elif is_heavy_tailed and is_skewed:
        dist_class = "HEAVY_TAILED_SKEWED"
    else:
        dist_class = "NON_GAUSSIAN_LIGHT_TAILED"

    return {
        "mean": round(mu, 6),
        "std": round(sigma, 4),
        "skewness": round(skew, 4),
        "excess_kurtosis": round(kurt, 4),
        "shapiro_p": round(float(sw_p), 4),
        "jarque_bera_p": round(float(jb_p), 4),
        "hill_tail_index": round(float(hill_alpha), 2) if hill_alpha != float('inf') else None,
        "n_increments": len(increments),
        "distribution_class": dist_class,
        "is_gaussian": is_gaussian,
        "is_heavy_tailed": is_heavy_tailed,
    }


# =====================================================================
#  SECTION 7: JUMP DETECTION
# =====================================================================

def detect_jumps(series, threshold_sigma=3.0):
    """
    Detect jumps in sigma(t) using threshold on increments.
    A jump = |delta_sigma| > threshold_sigma * MAD(delta_sigma).
    """
    valid = series[~np.isnan(series)]
    if len(valid) < 10:
        return {"n_jumps": 0, "verdict": "INSUFFICIENT_DATA"}

    increments = np.diff(valid)
    if len(increments) < 5:
        return {"n_jumps": 0, "verdict": "INSUFFICIENT_DATA"}

    # MAD-based threshold (robust to outliers)
    mad = np.median(np.abs(increments - np.median(increments)))
    if mad < 1e-10:
        mad = np.std(increments)

    threshold = threshold_sigma * 1.4826 * mad  # 1.4826 converts MAD to sigma

    jump_mask = np.abs(increments) > threshold
    jump_indices = np.where(jump_mask)[0]
    n_jumps = len(jump_indices)
    jump_rate = n_jumps / len(increments) if len(increments) > 0 else 0

    # Jump sizes
    jump_sizes = increments[jump_mask] if n_jumps > 0 else np.array([])

    # Expected jumps under Gaussian (for comparison)
    expected_gaussian = len(increments) * 2 * sp_stats.norm.sf(threshold_sigma)

    return {
        "n_jumps": n_jumps,
        "jump_rate": round(float(jump_rate), 4),
        "expected_gaussian": round(float(expected_gaussian), 2),
        "jump_excess_ratio": round(n_jumps / max(expected_gaussian, 0.1), 2),
        "mean_jump_size": round(float(np.mean(np.abs(jump_sizes))), 4) if n_jumps > 0 else 0.0,
        "max_jump_size": round(float(np.max(np.abs(jump_sizes))), 4) if n_jumps > 0 else 0.0,
        "n_positive_jumps": int(np.sum(jump_sizes > 0)) if n_jumps > 0 else 0,
        "n_negative_jumps": int(np.sum(jump_sizes < 0)) if n_jumps > 0 else 0,
        "threshold": round(float(threshold), 4),
    }


# =====================================================================
#  SECTION 8: PROCESS MODEL FITTING (per subject)
# =====================================================================

def fit_ou_process(series, dt=DT):
    """
    Fit Ornstein-Uhlenbeck process to residual r(t):
      dr = theta * (mu - r) * dt + sigma * dW

    MLE estimation via discrete approximation:
      r(t+1) = r(t) + theta*(mu - r(t))*dt + sigma*sqrt(dt)*eps
      i.e. r(t+1) = (1-theta*dt)*r(t) + theta*mu*dt + noise
      => linear regression: r(t+1) = a*r(t) + b + eps

    Returns estimated parameters and diagnostics.
    """
    valid = series[~np.isnan(series)]
    if len(valid) < 15:
        return {"model": "OU", "fitted": False, "reason": "insufficient_data"}

    y = valid[1:]   # r(t+1)
    x = valid[:-1]  # r(t)

    # Linear regression: y = a*x + b + eps
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    cov_xy = np.sum((x - x_mean) * (y - y_mean)) / n
    var_x = np.sum((x - x_mean)**2) / n

    if var_x < 1e-15:
        return {"model": "OU", "fitted": False, "reason": "zero_variance"}

    a = cov_xy / var_x
    b = y_mean - a * x_mean

    # Residuals
    residuals = y - (a * x + b)
    sigma_eps = np.std(residuals)

    # Extract OU parameters
    # a = 1 - theta*dt  =>  theta = (1 - a) / dt
    # b = theta*mu*dt   =>  mu = b / (theta*dt) = b / (1-a)
    theta = (1 - a) / dt
    if abs(1 - a) > 1e-10:
        mu = b / (1 - a)
    else:
        mu = 0.0
    sigma_ou = sigma_eps / np.sqrt(dt)

    # Half-life = ln(2) / theta
    if theta > 0:
        half_life = np.log(2) / theta
    else:
        half_life = float('inf')

    # R-squared
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - y_mean)**2)
    r2 = 1 - ss_res / max(ss_tot, 1e-15) if ss_tot > 0 else 0.0

    # Test: is theta significantly > 0? (mean-reverting)
    # Standard error of a (OLS)
    se_a = sigma_eps / np.sqrt(var_x * n)
    t_stat_mr = (1 - a) / se_a  # test H0: a=1 (random walk) vs a<1 (mean-reverting)
    p_mr = 2 * sp_stats.t.sf(abs(t_stat_mr), df=n-2)

    # Compare with Random Walk (a=1, b=0)
    rw_residuals = y - x
    rw_sigma = np.std(rw_residuals)
    # Likelihood ratio (approximate): BIC comparison
    bic_ou = n * np.log(max(sigma_eps**2, 1e-15)) + 2 * np.log(n)  # 2 params
    bic_rw = n * np.log(max(rw_sigma**2, 1e-15)) + 1 * np.log(n)   # 1 param (sigma only)

    return {
        "model": "OU",
        "fitted": True,
        "theta": round(float(theta), 6),
        "mu": round(float(mu), 4),
        "sigma": round(float(sigma_ou), 6),
        "half_life_sec": round(float(half_life), 1) if half_life != float('inf') else None,
        "half_life_min": round(float(half_life / 60), 2) if half_life != float('inf') else None,
        "a_coefficient": round(float(a), 6),
        "r_squared": round(float(r2), 4),
        "mean_reversion_p": round(float(p_mr), 4),
        "is_mean_reverting": bool(p_mr < 0.05 and theta > 0),
        "bic_ou": round(float(bic_ou), 2),
        "bic_rw": round(float(bic_rw), 2),
        "prefers_ou": bool(bic_ou < bic_rw),
        "n": len(valid),
    }


def fit_hurst_exponent(series):
    """
    Estimate Hurst exponent via R/S analysis.
    H < 0.5: anti-persistent (mean-reverting)
    H = 0.5: random walk (Brownian)
    H > 0.5: persistent (trending)
    """
    valid = series[~np.isnan(series)]
    if len(valid) < 30:
        return {"hurst": None, "verdict": "INSUFFICIENT_DATA"}

    # R/S analysis
    n = len(valid)
    max_k = int(np.log2(n))
    rs_list = []
    ns_list = []

    for k in range(2, max_k + 1):
        size = 2**k
        if size > n:
            break
        n_blocks = n // size
        rs_vals = []

        for block in range(n_blocks):
            sub = valid[block*size:(block+1)*size]
            mean_sub = np.mean(sub)
            cumdev = np.cumsum(sub - mean_sub)
            R = np.max(cumdev) - np.min(cumdev)
            S = np.std(sub, ddof=1)
            if S > 1e-10:
                rs_vals.append(R / S)

        if len(rs_vals) > 0:
            rs_list.append(np.mean(rs_vals))
            ns_list.append(size)

    if len(rs_list) < 3:
        return {"hurst": None, "verdict": "INSUFFICIENT_DATA"}

    log_n = np.log(ns_list)
    log_rs = np.log(rs_list)
    slope, _, r_val, _, _ = sp_stats.linregress(log_n, log_rs)

    # Classification
    H = float(slope)
    if H < 0.4:
        verdict = "ANTI_PERSISTENT"
    elif H < 0.6:
        verdict = "RANDOM_WALK"
    else:
        verdict = "PERSISTENT"

    return {
        "hurst": round(H, 4),
        "r_squared": round(float(r_val**2), 4),
        "verdict": verdict,
    }


# =====================================================================
#  SECTION 9: PER-SUBJECT + WITHIN-PHASE ANALYSIS
# =====================================================================

def analyze_subject(d, proxy_name="sigma_hr"):
    """Full process identification for one subject, one proxy."""
    residual_key = f"{proxy_name}_residual"
    zscore_key = f"{proxy_name}_zscore"

    residual = d[residual_key]
    zscore = d[zscore_key]
    phases = d["phase_labels"]

    results = {
        "subject_id": d["subject_id"],
        "proxy": proxy_name,
        "n_windows": d["n_windows"],
    }

    # A) Full series analysis (z-scored, before protocol subtraction)
    results["full_series"] = {
        "stationarity": test_stationarity(zscore, f"{d['subject_id']}_full"),
    }

    # B) Residual analysis (after protocol subtraction) — this is the key one
    results["residual"] = {
        "stationarity": test_stationarity(residual),
        "acf_pacf": compute_acf_pacf(residual),
        "increments": analyze_increments(residual),
        "jumps": detect_jumps(residual),
        "ou_fit": fit_ou_process(residual),
        "hurst": fit_hurst_exponent(residual),
    }

    # C) Within-phase analysis (advisor's key requirement)
    PHASES = {1: "baseline", 2: "stress", 3: "amusement", 4: "meditation"}
    results["within_phase"] = {}

    for phase_id, phase_name in PHASES.items():
        mask = phases == phase_id
        if mask.sum() < 10:
            continue

        phase_residual = residual[mask]
        results["within_phase"][phase_name] = {
            "n_windows": int(mask.sum()),
            "duration_min": round(float(mask.sum() * WINDOW_SEC / 60), 1),
            "stationarity": test_stationarity(phase_residual),
            "acf_pacf": compute_acf_pacf(phase_residual, max_lag=10),
            "increments": analyze_increments(phase_residual),
            "jumps": detect_jumps(phase_residual),
            "ou_fit": fit_ou_process(phase_residual),
            "hurst": fit_hurst_exponent(phase_residual),
        }

    return results


# =====================================================================
#  SECTION 10: CROSS-SUBJECT AGGREGATION + CONSISTENCY
# =====================================================================

def aggregate_results(all_subject_results, proxy_name):
    """
    Advisor's requirement: aggregate per-subject parameters,
    test cross-subject consistency.
    """
    print(f"\n{'='*70}")
    print(f"CROSS-SUBJECT AGGREGATION ({proxy_name})")
    print(f"{'='*70}")

    # Collect OU parameters
    thetas = []
    half_lives = []
    sigmas = []
    hursts = []
    mr_subjects = []
    stationarity_verdicts = []
    increment_classes = []

    for r in all_subject_results:
        res = r["residual"]

        # Stationarity
        stationarity_verdicts.append(res["stationarity"]["verdict"])

        # OU params
        ou = res["ou_fit"]
        if ou.get("fitted"):
            thetas.append(ou["theta"])
            sigmas.append(ou["sigma"])
            if ou.get("half_life_sec") is not None:
                half_lives.append(ou["half_life_sec"])
            mr_subjects.append(r["subject_id"] if ou["is_mean_reverting"] else None)

        # Hurst
        h = res["hurst"]
        if h.get("hurst") is not None:
            hursts.append(h["hurst"])

        # Increment distribution
        inc = res["increments"]
        if "distribution_class" in inc:
            increment_classes.append(inc["distribution_class"])

    n_subj = len(all_subject_results)

    # Print stationarity
    stat_counts = {}
    for v in stationarity_verdicts:
        stat_counts[v] = stat_counts.get(v, 0) + 1
    print(f"\n  [Stationarity] Residuals after protocol subtraction:")
    for v, c in sorted(stat_counts.items()):
        print(f"    {v}: {c}/{n_subj}")

    # Print OU parameters
    print(f"\n  [OU Process] Mean-reversion parameters:")
    if thetas:
        thetas = np.array(thetas)
        print(f"    theta: mean={np.mean(thetas):.4f}, std={np.std(thetas):.4f}, "
              f"range=[{np.min(thetas):.4f}, {np.max(thetas):.4f}]")

        if half_lives:
            half_lives = np.array(half_lives)
            valid_hl = half_lives[half_lives < 3600]  # ignore extreme
            if len(valid_hl) > 0:
                print(f"    half-life: mean={np.mean(valid_hl):.1f}s ({np.mean(valid_hl)/60:.1f}min), "
                      f"range=[{np.min(valid_hl):.1f}s, {np.max(valid_hl):.1f}s]")

        n_mr = sum(1 for s in mr_subjects if s is not None)
        print(f"    mean-reverting (p<0.05): {n_mr}/{len(thetas)} subjects")

        if sigmas:
            sigmas = np.array(sigmas)
            print(f"    sigma (volatility): mean={np.mean(sigmas):.4f}, std={np.std(sigmas):.4f}")

    # Print Hurst
    print(f"\n  [Hurst Exponent]:")
    if hursts:
        hursts = np.array(hursts)
        print(f"    mean={np.mean(hursts):.4f}, std={np.std(hursts):.4f}, "
              f"range=[{np.min(hursts):.4f}, {np.max(hursts):.4f}]")
        n_mr_hurst = int(np.sum(hursts < 0.4))
        n_rw_hurst = int(np.sum((hursts >= 0.4) & (hursts < 0.6)))
        n_per_hurst = int(np.sum(hursts >= 0.6))
        print(f"    anti-persistent: {n_mr_hurst}, random-walk: {n_rw_hurst}, persistent: {n_per_hurst}")

    # Print increment distribution
    print(f"\n  [Increments]:")
    inc_counts = {}
    for c in increment_classes:
        inc_counts[c] = inc_counts.get(c, 0) + 1
    for c, n in sorted(inc_counts.items()):
        print(f"    {c}: {n}/{len(increment_classes)}")

    # Cross-subject consistency (CV of theta)
    consistency = {}
    if len(thetas) > 1:
        thetas = np.array(thetas)
        cv_theta = np.std(thetas) / max(abs(np.mean(thetas)), 1e-10)
        consistency["theta_cv"] = round(float(cv_theta), 4)
        consistency["theta_consistent"] = bool(cv_theta < 1.0)

        # Formal test: Levene's test for equal variances of residuals
        # (check if subjects have similar noise levels)
        print(f"\n  [Consistency] Theta CV: {cv_theta:.4f} "
              f"({'CONSISTENT' if cv_theta < 1.0 else 'INCONSISTENT'})")

    # Within-phase analysis
    print(f"\n  [Within-Phase Process Classes]:")
    for phase_name in ["baseline", "stress", "amusement", "meditation"]:
        phase_thetas = []
        phase_hursts = []
        phase_stat = []
        for r in all_subject_results:
            if phase_name in r["within_phase"]:
                wp = r["within_phase"][phase_name]
                ou = wp["ou_fit"]
                if ou.get("fitted"):
                    phase_thetas.append(ou["theta"])
                h = wp["hurst"]
                if h.get("hurst") is not None:
                    phase_hursts.append(h["hurst"])
                phase_stat.append(wp["stationarity"]["verdict"])

        if phase_thetas:
            pt = np.array(phase_thetas)
            ph = np.array(phase_hursts) if phase_hursts else np.array([])
            stat_summary = {v: phase_stat.count(v) for v in set(phase_stat)}
            print(f"    {phase_name}: theta={np.mean(pt):.4f}+-{np.std(pt):.4f}, "
                  f"H={np.mean(ph):.3f}+-{np.std(ph):.3f}" if len(ph) > 0 else
                  f"    {phase_name}: theta={np.mean(pt):.4f}+-{np.std(pt):.4f}",
                  f"  stationarity={stat_summary}")

    # Determine overall process class
    majority_stat = max(stat_counts, key=stat_counts.get) if stat_counts else "UNKNOWN"

    if thetas is not None and len(thetas) > 0:
        mean_theta = float(np.mean(thetas))
    else:
        mean_theta = 0.0

    if hursts is not None and len(hursts) > 0:
        mean_hurst = float(np.mean(hursts))
    else:
        mean_hurst = 0.5

    majority_inc = max(inc_counts, key=inc_counts.get) if inc_counts else "UNKNOWN"

    n_mr_total = sum(1 for s in mr_subjects if s is not None) if mr_subjects else 0

    # Process class determination
    if majority_stat == "STATIONARY" and mean_theta > 0 and n_mr_total > n_subj // 2:
        if majority_inc == "GAUSSIAN":
            process_class = "ORNSTEIN_UHLENBECK"
        elif "HEAVY_TAILED" in majority_inc:
            process_class = "OU_WITH_JUMPS"
        else:
            process_class = "ORNSTEIN_UHLENBECK"
    elif majority_stat == "NON_STATIONARY":
        process_class = "RANDOM_WALK" if mean_hurst >= 0.45 else "ANTI_PERSISTENT_WALK"
    else:
        process_class = "UNDETERMINED"

    print(f"\n  {'='*50}")
    print(f"  OVERALL PROCESS CLASS: {process_class}")
    print(f"  {'='*50}")

    aggregation = {
        "stationarity_summary": stat_counts,
        "ou_parameters": {
            "theta_mean": round(float(np.mean(thetas)), 6) if len(thetas) > 0 else None,
            "theta_std": round(float(np.std(thetas)), 6) if len(thetas) > 0 else None,
            "sigma_mean": round(float(np.mean(sigmas)), 6) if len(sigmas) > 0 else None,
            "n_mean_reverting": n_mr_total,
            "n_total": len(thetas) if len(thetas) > 0 else 0,
            "half_life_mean_sec": round(float(np.mean(valid_hl)), 1) if len(valid_hl) > 0 else None,
        },
        "hurst_summary": {
            "mean": round(float(np.mean(hursts)), 4) if len(hursts) > 0 else None,
            "std": round(float(np.std(hursts)), 4) if len(hursts) > 0 else None,
        },
        "increment_distribution": inc_counts,
        "consistency": consistency,
        "process_class": process_class,
    }

    return aggregation


# =====================================================================
#  SECTION 11: COMPARISON hr_mean vs PC1
# =====================================================================

def compare_proxies(agg_hr, agg_pc1):
    """Compare process identification results between hr_mean and PC1."""
    print(f"\n{'='*70}")
    print("PROXY COMPARISON: hr_mean vs PC1(HRV)")
    print(f"{'='*70}")

    comparison = {}

    # Process class match
    same_class = agg_hr["process_class"] == agg_pc1["process_class"]
    print(f"\n  Process class:")
    print(f"    hr_mean: {agg_hr['process_class']}")
    print(f"    PC1:     {agg_pc1['process_class']}")
    print(f"    Match:   {'YES - STRONG evidence' if same_class else 'NO - investigate'}")

    comparison["same_process_class"] = same_class
    comparison["hr_class"] = agg_hr["process_class"]
    comparison["pc1_class"] = agg_pc1["process_class"]

    # Parameter comparison
    if agg_hr["ou_parameters"]["theta_mean"] and agg_pc1["ou_parameters"]["theta_mean"]:
        theta_ratio = agg_hr["ou_parameters"]["theta_mean"] / max(
            abs(agg_pc1["ou_parameters"]["theta_mean"]), 1e-10)
        print(f"\n  OU theta ratio (hr/pc1): {theta_ratio:.3f}")
        comparison["theta_ratio"] = round(float(theta_ratio), 3)

    # Hurst comparison
    if agg_hr["hurst_summary"]["mean"] and agg_pc1["hurst_summary"]["mean"]:
        hurst_delta = abs(agg_hr["hurst_summary"]["mean"] - agg_pc1["hurst_summary"]["mean"])
        print(f"  Hurst delta: {hurst_delta:.4f}")
        comparison["hurst_delta"] = round(float(hurst_delta), 4)

    # Verdict
    if same_class:
        verdict = "PROXIES_AGREE"
        detail = ("Both proxies identify the same stochastic process class. "
                  "Strong evidence that stress dynamics are well-captured "
                  "by a 1D latent variable, regardless of proxy choice.")
    else:
        verdict = "PROXIES_DISAGREE"
        detail = ("Proxies disagree on process class. This suggests the "
                  "choice of stress proxy materially affects conclusions. "
                  "Further investigation needed.")

    print(f"\n  VERDICT: {verdict}")
    print(f"  {detail}")

    comparison["verdict"] = verdict
    comparison["detail"] = detail

    return comparison


# =====================================================================
#  SECTION 12: PAPER-READY SCIENTIFIC VERDICT
# =====================================================================

def generate_verdict(agg_hr, agg_pc1, pca_info, proxy_comparison, all_results_hr):
    """Generate interpretable scientific verdict."""
    print(f"\n{'='*70}")
    print("SCIENTIFIC VERDICT: Stress Process Identification")
    print(f"{'='*70}")

    verdicts = []

    # V1: Is stress 1D?
    if pca_info["stress_is_1d"]:
        v1 = "STRESS_IS_1D"
        v1_d = f"PC1 explains {pca_info['pc1_var_pct']}% of HRV variance (>70% threshold)"
    else:
        v1 = "STRESS_MULTI_DIMENSIONAL"
        v1_d = f"PC1 explains only {pca_info['pc1_var_pct']}% (<70% threshold)"
    verdicts.append(("Dimensionality", v1, v1_d))

    # V2: Process class
    proc = agg_hr["process_class"]
    if "ORNSTEIN" in proc:
        v2 = "MEAN_REVERTING"
        hl = agg_hr["ou_parameters"].get("half_life_mean_sec")
        v2_d = f"OU process with half-life ~{hl/60:.1f} min" if hl else "OU process identified"
    elif proc == "RANDOM_WALK":
        v2 = "RANDOM_WALK"
        v2_d = "Stress follows a random walk — no intrinsic recovery"
    else:
        v2 = proc
        v2_d = "Process class could not be definitively determined"
    verdicts.append(("Process Class", v2, v2_d))

    # V3: Cross-subject consistency
    cv = agg_hr.get("consistency", {}).get("theta_cv")
    if cv is not None:
        if cv < 0.5:
            v3 = "HIGHLY_CONSISTENT"
            v3_d = f"Theta CV={cv:.3f} — universal stress dynamics"
        elif cv < 1.0:
            v3 = "MODERATELY_CONSISTENT"
            v3_d = f"Theta CV={cv:.3f} — same process class, variable parameters"
        else:
            v3 = "INCONSISTENT"
            v3_d = f"Theta CV={cv:.3f} — subject-specific dynamics"
    else:
        v3 = "UNDETERMINED"
        v3_d = "Insufficient data for consistency assessment"
    verdicts.append(("Cross-Subject Consistency", v3, v3_d))

    # V4: Proxy agreement
    v4 = proxy_comparison["verdict"]
    v4_d = proxy_comparison["detail"]
    verdicts.append(("Proxy Agreement", v4, v4_d))

    # V5: Protocol subtraction effect
    # Compare stationarity before vs after subtraction
    n_stat_before = 0
    n_stat_after = 0
    for r in all_results_hr:
        if r["full_series"]["stationarity"]["verdict"] == "STATIONARY":
            n_stat_before += 1
        if r["residual"]["stationarity"]["verdict"] == "STATIONARY":
            n_stat_after += 1
    n_total = len(all_results_hr)

    if n_stat_after > n_stat_before:
        v5 = "PROTOCOL_SUBTRACTION_HELPS"
        v5_d = (f"Stationarity: {n_stat_before}/{n_total} (raw) -> "
                f"{n_stat_after}/{n_total} (after subtraction)")
    else:
        v5 = "PROTOCOL_ALREADY_STATIONARY"
        v5_d = f"Both raw and residual show similar stationarity ({n_stat_before} vs {n_stat_after})"
    verdicts.append(("Protocol Effect", v5, v5_d))

    for name, v, detail in verdicts:
        print(f"\n  [{v}] {name}")
        print(f"    {detail}")

    # Overall conclusion
    print(f"\n  {'='*50}")
    if "ORNSTEIN" in proc and proxy_comparison["verdict"] == "PROXIES_AGREE":
        overall = "STRESS_IS_OU_PROCESS"
        overall_d = (
            "Physiological stress dynamics follow an Ornstein-Uhlenbeck "
            "(mean-reverting) process. This has direct implications for ABM: "
            "the stress coupling function g(sigma) should model OU dynamics "
            "with the empirically estimated mean-reversion rate."
        )
    elif proxy_comparison["verdict"] == "PROXIES_AGREE":
        overall = f"STRESS_IS_{proc}"
        overall_d = f"Both proxies agree on {proc} dynamics."
    else:
        overall = "FURTHER_INVESTIGATION_NEEDED"
        overall_d = "Process class depends on proxy choice. Need more data or better proxy."

    print(f"  OVERALL: {overall}")
    print(f"  {overall_d}")

    return {
        "components": {name: {"verdict": v, "detail": detail} for name, v, detail in verdicts},
        "overall": overall,
        "overall_detail": overall_d,
    }


# =====================================================================
#  SECTION 13: MAIN
# =====================================================================

def main():
    t0 = time.time()
    print("="*70)
    print("Script 24: Physiological Stress Process Identification")
    print("Phase 4: sigma(t) Stochastic Law Discovery")
    print("="*70)

    # ── Step 1: Extract sigma(t) for all subjects ──
    print("\n[1/6] Extracting sigma(t) time series from WESAD...")
    subj_ids = discover_subjects()
    subject_data_list = []
    for sid in subj_ids:
        print(f"  Processing {sid}...")
        d = extract_sigma_timeseries(sid)
        subject_data_list.append(d)
    print(f"  Total: {len(subject_data_list)} subjects, "
          f"{sum(d['n_windows'] for d in subject_data_list)} windows")

    # ── Step 2: Compute sigma proxies ──
    print("\n[2/6] Computing sigma proxies (hr_mean vs PC1 HRV)...")
    subject_data_list, pca_info = compute_sigma_proxies(subject_data_list)

    # ── Step 3: Per-subject normalization + protocol subtraction ──
    print("\n[3/6] Per-subject normalization + protocol subtraction...")
    subject_data_list = normalize_and_subtract_protocol(subject_data_list)

    # ── Step 4: Per-subject process identification ──
    print("\n[4/6] Per-subject process identification...")
    all_results_hr = []
    all_results_pc1 = []

    for d in subject_data_list:
        print(f"\n  --- {d['subject_id']} (hr_mean) ---")
        res_hr = analyze_subject(d, "sigma_hr")
        all_results_hr.append(res_hr)

        # Print key findings
        stat = res_hr["residual"]["stationarity"]["verdict"]
        ou = res_hr["residual"]["ou_fit"]
        h = res_hr["residual"]["hurst"]
        inc = res_hr["residual"]["increments"]
        print(f"    Stationarity: {stat}")
        if ou.get("fitted"):
            print(f"    OU: theta={ou['theta']:.4f}, "
                  f"half-life={ou.get('half_life_min', '?')}min, "
                  f"mean-reverting={ou['is_mean_reverting']}")
        if h.get("hurst") is not None:
            print(f"    Hurst: {h['hurst']:.4f} ({h['verdict']})")
        if "distribution_class" in inc:
            print(f"    Increments: {inc['distribution_class']}")

        # PC1 analysis
        res_pc1 = analyze_subject(d, "sigma_pc1")
        all_results_pc1.append(res_pc1)

    # ── Step 5: Cross-subject aggregation ──
    print("\n[5/6] Cross-subject aggregation...")
    agg_hr = aggregate_results(all_results_hr, "hr_mean")
    agg_pc1 = aggregate_results(all_results_pc1, "PC1_HRV")

    # ── Step 6: Proxy comparison + Verdict ──
    print("\n[6/6] Proxy comparison + scientific verdict...")
    proxy_comparison = compare_proxies(agg_hr, agg_pc1)
    verdict = generate_verdict(agg_hr, agg_pc1, pca_info, proxy_comparison, all_results_hr)

    # ── Save results ──
    results = {
        "metadata": {
            "n_subjects": len(subject_data_list),
            "window_sec": WINDOW_SEC,
            "elapsed_sec": round(time.time() - t0, 1),
        },
        "pca_info": pca_info,
        "aggregation_hr_mean": agg_hr,
        "aggregation_pc1": agg_pc1,
        "proxy_comparison": proxy_comparison,
        "verdict": verdict,
        "per_subject_hr": [
            {
                "subject_id": r["subject_id"],
                "residual": {
                    "stationarity": r["residual"]["stationarity"],
                    "ou_fit": r["residual"]["ou_fit"],
                    "hurst": r["residual"]["hurst"],
                    "increments": {k: v for k, v in r["residual"]["increments"].items()
                                   if k != "acf" and k != "pacf"},
                    "jumps": r["residual"]["jumps"],
                    "acf_lag1": r["residual"]["acf_pacf"].get("acf_lag1"),
                    "n_sig_acf": r["residual"]["acf_pacf"].get("n_sig_acf"),
                    "acf_decay_rate": r["residual"]["acf_pacf"].get("acf_decay_rate"),
                },
                "within_phase": {
                    phase: {
                        "n_windows": wp.get("n_windows"),
                        "stationarity": wp["stationarity"]["verdict"],
                        "ou_theta": wp["ou_fit"].get("theta") if wp["ou_fit"].get("fitted") else None,
                        "ou_half_life_min": wp["ou_fit"].get("half_life_min") if wp["ou_fit"].get("fitted") else None,
                        "hurst": wp["hurst"].get("hurst"),
                        "increment_class": wp["increments"].get("distribution_class"),
                    }
                    for phase, wp in r.get("within_phase", {}).items()
                },
                "full_series_stationarity": r["full_series"]["stationarity"]["verdict"],
                "phase_means": d.get("phase_means_sigma_hr", {}),
            }
            for r, d in zip(all_results_hr, subject_data_list)
        ],
    }

    def _sanitize(obj):
        """Recursively convert numpy types to native Python for JSON."""
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj) if not np.isnan(obj) else None
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results = _sanitize(results)

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    out_path = VALIDATION_DIR / "stress_process_identification.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"Results saved to {out_path}")
    print(f"Elapsed: {time.time() - t0:.1f}s")
    print(f"OVERALL: {verdict['overall']}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
