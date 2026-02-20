"""
EDA Feature Extraction
=======================
Extract electrodermal activity features for physiological stress detection.

Features:
    - Tonic: SCL (Skin Conductance Level) mean, std
    - Phasic: SCR (Skin Conductance Response) count, amplitude, rise time
    - Statistical: slope, range, AUC
"""

import numpy as np
from typing import Dict
from scipy.signal import butter, filtfilt, find_peaks
from src.preprocessing.filters import eda_lowpass


def decompose_eda(
    eda: np.ndarray,
    fs: float,
    tonic_cutoff: float = 0.05,
) -> tuple:
    """
    Simple tonic/phasic decomposition via highpass-lowpass split.
    
    Tonic = lowpass filtered at 0.05 Hz (SCL)
    Phasic = original − tonic (SCR)
    
    Returns:
        tonic, phasic arrays
    """
    if len(eda) < int(4 * fs / tonic_cutoff):
        # Too short for the filter, return simple mean subtraction
        tonic = np.full_like(eda, np.mean(eda))
        phasic = eda - tonic
        return tonic, phasic

    nyq = fs / 2.0
    if tonic_cutoff >= nyq:
        tonic_cutoff = nyq * 0.8
    b, a = butter(2, tonic_cutoff / nyq, btype='low')
    tonic = filtfilt(b, a, eda)
    phasic = eda - tonic
    return tonic, phasic


def detect_scr(
    phasic: np.ndarray,
    fs: float,
    min_amplitude: float = 0.01,
    min_rise_time_s: float = 0.5,
    max_rise_time_s: float = 5.0,
) -> Dict[str, object]:
    """
    Detect SCR peaks from phasic EDA.
    
    Returns:
        dict with scr_count, scr_amplitudes, scr_peak_indices,
        scr_mean_amplitude, scr_rate_per_min
    """
    peaks, props = find_peaks(phasic, height=min_amplitude,
                               distance=int(min_rise_time_s * fs))

    if len(peaks) == 0:
        return {
            "scr_count": 0,
            "scr_peak_indices": np.array([]),
            "scr_amplitudes": np.array([]),
            "scr_mean_amplitude": 0,
            "scr_rate_per_min": 0,
        }

    amplitudes = props["peak_heights"]
    duration_min = len(phasic) / fs / 60.0

    return {
        "scr_count": len(peaks),
        "scr_peak_indices": peaks,
        "scr_amplitudes": amplitudes,
        "scr_mean_amplitude": float(np.mean(amplitudes)),
        "scr_rate_per_min": len(peaks) / duration_min if duration_min > 0 else 0,
    }


def extract_eda_features(
    eda_segment: np.ndarray,
    fs: float,
    tonic_cutoff: float = 0.05,
) -> Dict[str, float]:
    """
    Full EDA feature extraction from a signal segment.
    
    Returns tonic + phasic + statistical features.
    """
    # Clean
    eda = eda_lowpass(eda_segment, fs, cutoff=1.0)

    # Decompose
    tonic, phasic = decompose_eda(eda, fs, tonic_cutoff)

    # Tonic (SCL) features
    scl_mean = float(np.mean(tonic))
    scl_std = float(np.std(tonic))
    scl_slope = 0.0
    if len(tonic) > 1:
        t = np.arange(len(tonic))
        scl_slope = float(np.polyfit(t / fs, tonic, 1)[0])

    # Phasic (SCR) features
    scr_info = detect_scr(phasic, fs)

    # Statistical features
    eda_range = float(np.ptp(eda))
    eda_auc = float(np.trapz(np.abs(phasic)) / fs)

    return {
        "scl_mean": scl_mean,
        "scl_std": scl_std,
        "scl_slope": scl_slope,
        "scr_count": scr_info["scr_count"],
        "scr_mean_amplitude": scr_info["scr_mean_amplitude"],
        "scr_rate_per_min": scr_info["scr_rate_per_min"],
        "eda_range": eda_range,
        "eda_auc": eda_auc,
        "eda_mean": float(np.mean(eda)),
        "eda_std": float(np.std(eda)),
    }


# Feature names for consistent ordering
EDA_FEATURE_NAMES = [
    "scl_mean", "scl_std", "scl_slope",
    "scr_count", "scr_mean_amplitude", "scr_rate_per_min",
    "eda_range", "eda_auc", "eda_mean", "eda_std",
]


def features_to_vector(features: Dict[str, float]) -> np.ndarray:
    """Convert feature dict to fixed-order vector."""
    return np.array([features.get(k, 0) for k in EDA_FEATURE_NAMES])
