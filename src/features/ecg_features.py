"""
ECG Feature Extraction
=======================
Extract cardiac features from preprocessed ECG signals for Stage 1.

Features:
    - Heart Rate: mean, std, slope
    - HRV time-domain: RMSSD, SDNN, pNN50
    - HRV frequency-domain: LF/HF ratio, LF power, HF power
"""

import numpy as np
from typing import Dict, Optional
from src.preprocessing.filters import (
    detect_r_peaks, compute_rr_intervals, reject_rr_outliers,
)
from scipy.signal import welch


def extract_hrv_time_domain(rr_ms: np.ndarray) -> Dict[str, float]:
    """
    Time-domain HRV metrics from R-R intervals.
    
    Args:
        rr_ms: R-R intervals in milliseconds
    
    Returns:
        dict with hr_mean, hr_std, rmssd, sdnn, pnn50, rr_mean, rr_std
    """
    if len(rr_ms) < 2:
        return {
            "hr_mean": 0, "hr_std": 0, "rmssd": 0, "sdnn": 0,
            "pnn50": 0, "rr_mean": 0, "rr_std": 0, "nn_count": 0,
        }

    hr = 60000.0 / rr_ms  # bpm
    diff_rr = np.diff(rr_ms)

    return {
        "hr_mean": float(np.mean(hr)),
        "hr_std": float(np.std(hr)),
        "rmssd": float(np.sqrt(np.mean(diff_rr ** 2))),
        "sdnn": float(np.std(rr_ms)),
        "pnn50": float(np.sum(np.abs(diff_rr) > 50) / len(diff_rr) * 100),
        "rr_mean": float(np.mean(rr_ms)),
        "rr_std": float(np.std(rr_ms)),
        "nn_count": len(rr_ms),
    }


def extract_hrv_freq_domain(rr_ms: np.ndarray, fs_interp: float = 4.0) -> Dict[str, float]:
    """
    Frequency-domain HRV via interpolated R-R series → PSD.
    
    LF: 0.04-0.15 Hz
    HF: 0.15-0.40 Hz
    """
    if len(rr_ms) < 10:
        return {"lf_power": 0, "hf_power": 0, "lf_hf_ratio": 0, "total_power": 0}

    # Interpolate to uniform sampling
    rr_sec = rr_ms / 1000.0
    cum_time = np.cumsum(rr_sec)
    cum_time = cum_time - cum_time[0]
    t_uniform = np.arange(0, cum_time[-1], 1.0 / fs_interp)

    if len(t_uniform) < 10:
        return {"lf_power": 0, "hf_power": 0, "lf_hf_ratio": 0, "total_power": 0}

    rr_interp = np.interp(t_uniform, cum_time, rr_ms)
    rr_interp -= rr_interp.mean()

    freqs, psd = welch(rr_interp, fs=fs_interp,
                       nperseg=min(256, len(rr_interp)))

    lf_mask = (freqs >= 0.04) & (freqs <= 0.15)
    hf_mask = (freqs >= 0.15) & (freqs <= 0.40)

    lf = float(np.trapz(psd[lf_mask], freqs[lf_mask])) if lf_mask.any() else 0
    hf = float(np.trapz(psd[hf_mask], freqs[hf_mask])) if hf_mask.any() else 0
    total = float(np.trapz(psd, freqs))

    return {
        "lf_power": lf,
        "hf_power": hf,
        "lf_hf_ratio": lf / hf if hf > 0 else 0,
        "total_power": total,
    }


def extract_ecg_features(
    ecg_segment: np.ndarray,
    fs: float,
    ibi_range: tuple = (250, 2000),
) -> Dict[str, float]:
    """
    Full ECG feature extraction from a signal segment.
    
    Returns combined time-domain + frequency-domain features.
    """
    r_peaks = detect_r_peaks(ecg_segment, fs)
    rr = compute_rr_intervals(r_peaks, fs)
    rr = reject_rr_outliers(rr, ibi_range[0], ibi_range[1])

    time_feats = extract_hrv_time_domain(rr)
    freq_feats = extract_hrv_freq_domain(rr)

    # Merge
    features = {}
    features.update(time_feats)
    features.update(freq_feats)

    # Add HR slope (linear trend)
    if len(rr) > 2:
        hr = 60000.0 / rr
        t = np.arange(len(hr))
        features["hr_slope"] = float(np.polyfit(t, hr, 1)[0])
    else:
        features["hr_slope"] = 0

    return features


# Feature names for consistent ordering
ECG_FEATURE_NAMES = [
    "hr_mean", "hr_std", "hr_slope",
    "rmssd", "sdnn", "pnn50",
    "rr_mean", "rr_std", "nn_count",
    "lf_power", "hf_power", "lf_hf_ratio", "total_power",
]


def features_to_vector(features: Dict[str, float]) -> np.ndarray:
    """Convert feature dict to fixed-order vector."""
    return np.array([features.get(k, 0) for k in ECG_FEATURE_NAMES])
