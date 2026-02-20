"""
EEG Feature Extraction
=======================
Extract EEG features from DREAMER data for physiological stress proxy.

Features per channel (14 channels × N features):
    - Differential entropy per band (delta, theta, alpha, beta, gamma)
    - Band power ratios (alpha/beta, theta/beta)
    - Statistical: variance, kurtosis
"""

import numpy as np
from typing import Dict, List
from scipy.signal import welch
from src.preprocessing.filters import compute_differential_entropy
from config.settings import DREAMER_EEG_CHANNELS, DREAMER_EEG_SR

# Standard EEG frequency bands (Hz)
EEG_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def compute_band_powers(
    eeg_channel: np.ndarray,
    fs: float,
    bands: dict = None,
    nperseg: int = 256,
) -> Dict[str, float]:
    """
    Compute absolute band powers via Welch PSD for a single channel.
    """
    if bands is None:
        bands = EEG_BANDS

    freqs, psd = welch(eeg_channel, fs=fs,
                       nperseg=min(nperseg, len(eeg_channel)))

    powers = {}
    for name, (lo, hi) in bands.items():
        idx = (freqs >= lo) & (freqs <= hi)
        if idx.any():
            powers[name] = float(np.trapz(psd[idx], freqs[idx]))
        else:
            powers[name] = 0.0
    return powers


def extract_single_channel_features(
    eeg_channel: np.ndarray,
    fs: float,
    bands: dict = None,
) -> Dict[str, float]:
    """
    Feature extraction for one EEG channel.
    
    Returns:
        dict with de_{band}, bp_{band}, ratio_alpha_beta,
        ratio_theta_beta, variance, kurtosis
    """
    if bands is None:
        bands = EEG_BANDS

    feats = {}

    # Differential entropy per band
    for band_name, (lo, hi) in bands.items():
        de = compute_differential_entropy(eeg_channel, fs, lo, hi)
        feats[f"de_{band_name}"] = de

    # Band powers
    bp = compute_band_powers(eeg_channel, fs, bands)
    for band_name, power in bp.items():
        feats[f"bp_{band_name}"] = power

    # Ratios (arousal indicators)
    alpha = bp.get("alpha", 0)
    beta = bp.get("beta", 0)
    theta = bp.get("theta", 0)
    feats["ratio_alpha_beta"] = alpha / beta if beta > 1e-12 else 0
    feats["ratio_theta_beta"] = theta / beta if beta > 1e-12 else 0

    # Statistical
    feats["variance"] = float(np.var(eeg_channel))
    from scipy.stats import kurtosis as _kurtosis
    feats["kurtosis"] = float(_kurtosis(eeg_channel, fisher=True))

    return feats


def extract_eeg_features(
    eeg_data: np.ndarray,
    fs: float = None,
    channel_names: List[str] = None,
) -> Dict[str, float]:
    """
    Full EEG feature extraction across all channels.
    
    Args:
        eeg_data: shape (n_samples, n_channels) or (n_channels, n_samples)
        fs: sampling rate (default DREAMER_EEG_SR)
        channel_names: channel labels
    
    Returns:
        flat dict with {channel}_{feature_name} keys
    """
    if fs is None:
        fs = DREAMER_EEG_SR
    if channel_names is None:
        channel_names = DREAMER_EEG_CHANNELS

    # Ensure shape (n_channels, n_samples)
    if eeg_data.ndim == 2:
        if eeg_data.shape[0] > eeg_data.shape[1]:
            eeg_data = eeg_data.T  # transpose to (channels, samples)

    n_channels = eeg_data.shape[0] if eeg_data.ndim == 2 else 1

    all_features = {}

    for ch_idx in range(n_channels):
        ch_data = eeg_data[ch_idx] if eeg_data.ndim == 2 else eeg_data
        ch_name = channel_names[ch_idx] if ch_idx < len(channel_names) else f"ch{ch_idx}"

        ch_feats = extract_single_channel_features(ch_data, fs)
        for feat_name, feat_val in ch_feats.items():
            all_features[f"{ch_name}_{feat_name}"] = feat_val

    # Global cross-channel features
    if eeg_data.ndim == 2 and n_channels > 1:
        # Asymmetry indices (common: frontal alpha asymmetry)
        # AF3(idx 0) vs AF4(idx 1) if available
        try:
            af3_idx = channel_names.index("AF3")
            af4_idx = channel_names.index("AF4")
            af3_alpha_de = all_features.get(f"AF3_de_alpha", 0)
            af4_alpha_de = all_features.get(f"AF4_de_alpha", 0)
            all_features["frontal_alpha_asymmetry"] = af4_alpha_de - af3_alpha_de
        except (ValueError, KeyError):
            all_features["frontal_alpha_asymmetry"] = 0

    return all_features


def get_feature_names(channel_names: List[str] = None) -> List[str]:
    """Get ordered list of all EEG feature names."""
    if channel_names is None:
        channel_names = DREAMER_EEG_CHANNELS

    per_channel = []
    for band in EEG_BANDS:
        per_channel.append(f"de_{band}")
        per_channel.append(f"bp_{band}")
    per_channel += ["ratio_alpha_beta", "ratio_theta_beta", "variance", "kurtosis"]

    names = []
    for ch in channel_names:
        for feat in per_channel:
            names.append(f"{ch}_{feat}")

    names.append("frontal_alpha_asymmetry")
    return names


def features_to_vector(
    features: Dict[str, float],
    channel_names: List[str] = None,
) -> np.ndarray:
    """Convert feature dict to fixed-order vector."""
    names = get_feature_names(channel_names)
    return np.array([features.get(k, 0) for k in names])
