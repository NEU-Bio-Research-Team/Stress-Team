"""
Signal Filters & Artifact Removal
===================================
Common DSP building blocks used across WESAD and DREAMER preprocessing.
"""

import numpy as np
from scipy.signal import (
    butter, filtfilt, iirnotch, welch,
    find_peaks, sosfiltfilt, sosfilt,
)
from typing import Tuple, Optional


# ──────────────────────── Butterworth Filters ─────────────────────────

def bandpass(signal: np.ndarray, lo: float, hi: float,
             fs: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter."""
    nyq = fs / 2
    lo_n, hi_n = lo / nyq, hi / nyq
    if hi_n >= 1.0:
        hi_n = 0.99
    b, a = butter(order, [lo_n, hi_n], btype="band")
    return filtfilt(b, a, signal)


def lowpass(signal: np.ndarray, cutoff: float,
            fs: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth lowpass filter."""
    nyq = fs / 2
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, signal)


def highpass(signal: np.ndarray, cutoff: float,
             fs: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth highpass filter."""
    nyq = fs / 2
    b, a = butter(order, cutoff / nyq, btype="high")
    return filtfilt(b, a, signal)


def notch_filter(signal: np.ndarray, freq: float,
                 fs: float, Q: float = 30) -> np.ndarray:
    """Notch filter for powerline interference."""
    b, a = iirnotch(freq, Q, fs)
    return filtfilt(b, a, signal)


# ──────────────────────── ECG Processing ──────────────────────────────

def detect_r_peaks(ecg: np.ndarray, fs: float,
                   min_distance_ms: int = 250) -> np.ndarray:
    """
    Simple R-peak detection using Pan-Tompkins-like approach.
    Returns array of R-peak indices.
    """
    # 1. Bandpass 5-15 Hz
    filtered = bandpass(ecg, 5, 15, fs, order=2)
    # 2. Differentiate
    diff = np.diff(filtered)
    # 3. Square
    squared = diff ** 2
    # 4. Moving average window (150 ms)
    win_size = int(0.15 * fs)
    if win_size < 1:
        win_size = 1
    kernel = np.ones(win_size) / win_size
    integrated = np.convolve(squared, kernel, mode="same")
    # 5. Find peaks
    min_dist = int(min_distance_ms / 1000 * fs)
    threshold = np.mean(integrated) + 0.3 * np.std(integrated)
    peaks, _ = find_peaks(integrated, distance=min_dist, height=threshold)

    return peaks


def compute_rr_intervals(r_peaks: np.ndarray, fs: float) -> np.ndarray:
    """Compute R-R intervals in milliseconds."""
    if len(r_peaks) < 2:
        return np.array([])
    rr = np.diff(r_peaks) / fs * 1000  # ms
    return rr


def reject_rr_outliers(rr_ms: np.ndarray,
                        lo: float = 250, hi: float = 2000) -> np.ndarray:
    """Remove physiologically impossible R-R intervals."""
    mask = (rr_ms >= lo) & (rr_ms <= hi)
    return rr_ms[mask]


def estimate_snr(signal: np.ndarray, fs: float,
                 signal_band: Tuple[float, float] = (5, 15),
                 noise_band: Tuple[float, float] = (50, 100)) -> float:
    """Estimate SNR from power spectral density."""
    freqs, psd = welch(signal, fs=fs, nperseg=min(4096, len(signal)))
    sig_mask = (freqs >= signal_band[0]) & (freqs <= signal_band[1])
    noise_mask = (freqs >= noise_band[0]) & (freqs <= noise_band[1])

    sig_power = np.trapz(psd[sig_mask], freqs[sig_mask]) if sig_mask.any() else 0
    noise_power = np.trapz(psd[noise_mask], freqs[noise_mask]) if noise_mask.any() else 0

    if noise_power > 0:
        return 10 * np.log10(sig_power / noise_power)
    return float("inf")


# ──────────────────────── EDA Processing ──────────────────────────────

def eda_lowpass(eda: np.ndarray, fs: float,
                cutoff: float = 5.0, order: int = 4) -> np.ndarray:
    """Low-pass filter for EDA signal."""
    return lowpass(eda, cutoff, fs, order)


def eda_artifact_mask(eda: np.ndarray, acc_mag: np.ndarray,
                       threshold: float = 0.3) -> np.ndarray:
    """
    Create boolean mask for motion-contaminated EDA segments.
    True = clean, False = artifact.
    """
    min_len = min(len(eda), len(acc_mag))
    eda_s = eda[:min_len]
    acc_s = acc_mag[:min_len]

    # Compute rolling correlation (1-second windows)
    win = 100  # samples
    mask = np.ones(min_len, dtype=bool)

    for start in range(0, min_len - win, win):
        seg_eda = eda_s[start:start + win]
        seg_acc = acc_s[start:start + win]
        if np.std(seg_eda) > 0 and np.std(seg_acc) > 0:
            corr = abs(np.corrcoef(seg_eda, seg_acc)[0, 1])
            if corr > threshold:
                mask[start:start + win] = False

    return mask


# ──────────────────────── EEG Processing ──────────────────────────────

def eeg_bandpass(eeg: np.ndarray, fs: float,
                 lo: float = 0.1, hi: float = 40, order: int = 4) -> np.ndarray:
    """Bandpass filter for EEG. eeg shape: (n_samples, n_channels) or (n_samples,)."""
    if eeg.ndim == 1:
        return bandpass(eeg, lo, hi, fs, order)
    filtered = np.zeros_like(eeg)
    for ch in range(eeg.shape[1]):
        filtered[:, ch] = bandpass(eeg[:, ch], lo, hi, fs, order)
    return filtered


def eeg_notch(eeg: np.ndarray, fs: float,
              freq: float = 50, Q: float = 30) -> np.ndarray:
    """Notch filter for EEG powerline noise."""
    if eeg.ndim == 1:
        return notch_filter(eeg, freq, fs, Q)
    filtered = np.zeros_like(eeg)
    for ch in range(eeg.shape[1]):
        filtered[:, ch] = notch_filter(eeg[:, ch], freq, fs, Q)
    return filtered


def compute_differential_entropy(signal: np.ndarray, fs: float,
                                   band: Tuple[float, float],
                                   nperseg: int = 256) -> float:
    """
    Compute differential entropy for a frequency band.
    DE = 0.5 * ln(2πe * σ²) where σ² is band power.
    """
    freqs, psd = welch(signal, fs=fs, nperseg=min(nperseg, len(signal)))
    mask = (freqs >= band[0]) & (freqs <= band[1])
    band_power = np.trapz(psd[mask], freqs[mask]) if mask.any() else 1e-10
    return 0.5 * np.log(2 * np.pi * np.e * band_power + 1e-10)


# ──────────────────────── Windowing ───────────────────────────────────

def segment_signal(signal: np.ndarray, window_samples: int,
                   overlap: float = 0.0) -> np.ndarray:
    """
    Segment signal into windows.
    Returns (n_windows, window_samples[, n_channels]).
    """
    step = int(window_samples * (1 - overlap))
    if step < 1:
        step = 1

    if signal.ndim == 1:
        n_windows = (len(signal) - window_samples) // step + 1
        windows = np.zeros((n_windows, window_samples))
        for i in range(n_windows):
            start = i * step
            windows[i] = signal[start:start + window_samples]
    else:
        n_channels = signal.shape[1]
        n_windows = (signal.shape[0] - window_samples) // step + 1
        windows = np.zeros((n_windows, window_samples, n_channels))
        for i in range(n_windows):
            start = i * step
            windows[i] = signal[start:start + window_samples]

    return windows


def normalize_zscore(signal: np.ndarray, axis: int = 0) -> np.ndarray:
    """Z-score normalization."""
    mean = np.mean(signal, axis=axis, keepdims=True)
    std = np.std(signal, axis=axis, keepdims=True)
    return (signal - mean) / (std + 1e-10)
