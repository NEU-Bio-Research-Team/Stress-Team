"""
Plotting utilities for audits and EDA.
"""
import numpy as np
from pathlib import Path
from typing import Optional, List

try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _check_mpl():
    if not HAS_MPL:
        print("[plot] matplotlib not installed – skipping plot.")
        return False
    return True


def plot_label_distribution(labels: np.ndarray, label_names: dict,
                            title: str, save_path: Optional[Path] = None):
    """Bar chart of label counts."""
    if not _check_mpl():
        return
    unique, counts = np.unique(labels, return_counts=True)
    names = [label_names.get(int(u), str(u)) for u in unique]
    pcts = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(names, counts, color="steelblue")
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_title(title)
    ax.set_ylabel("Sample count")
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"[plot] Saved → {save_path}")
    plt.close(fig)


def plot_signal_segment(signal: np.ndarray, sr: float,
                        title: str = "", channel: str = "",
                        duration_sec: float = 10.0,
                        save_path: Optional[Path] = None):
    """Plot a short segment of a 1-D signal."""
    if not _check_mpl():
        return
    n = int(duration_sec * sr)
    seg = signal[:n]
    t = np.arange(len(seg)) / sr

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(t, seg, linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(channel or "Amplitude")
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_psd(freqs: np.ndarray, psd: np.ndarray,
             title: str = "", save_path: Optional[Path] = None):
    """Plot power spectral density."""
    if not _check_mpl():
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(freqs, psd, linewidth=0.8)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dB/Hz)")
    ax.set_title(title)
    ax.set_xlim(0, 60)
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_histogram(data: np.ndarray, title: str, xlabel: str,
                   bins: int = 50, save_path: Optional[Path] = None):
    if not _check_mpl():
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(data, bins=bins, color="steelblue", edgecolor="white", density=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_acf(series: np.ndarray, max_lag: int = 100,
             title: str = "", save_path: Optional[Path] = None):
    """Plot autocorrelation function."""
    if not _check_mpl():
        return
    from numpy import correlate
    s = series - series.mean()
    full = np.correlate(s, s, mode="full")
    full = full[len(full) // 2:]
    full /= full[0]
    lags = np.arange(max_lag + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(lags, full[:max_lag + 1], width=0.8, color="steelblue")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
