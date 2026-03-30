"""
Script 03 – Bio-Market Alignment Layer
=========================================
Build the coupling layer between biological stress σ(t) and
HFT market features for the ABM simulator.

This script:
    1. Loads bio stress time-series (σ(t) at 5s windows → interpolated to ms)
    2. Loads HFT market feature time-series (100ms bars)
    3. Aligns them on a common 100ms grid (HFT_COUPLING_RESOLUTION)
    4. Computes cross-correlation and lead/lag structure
    5. Outputs a merged alignment parquet for ABM calibration

Key design decisions:
    - Bio σ(t) is from WESAD (real physiological data at 5s windows).
      We upsample via cubic interpolation to 100ms, NOT by fabricating data —
      this preserves the smooth OU dynamics while matching the market grid.
    - The alignment is per-session: we pair bio sessions with market windows
      of equal duration to study the coupling structure.
    - This is a TEMPLATE — actual ABM will use synthetic σ(t) from the
      calibrated OU process, not interpolated WESAD data.

Output: data/processed/tardis/hft_features/bio_market_alignment/

Usage:
    python scripts/stage2_economics/03_bio_market_alignment.py
    python scripts/stage2_economics/03_bio_market_alignment.py --bio-resolution 100ms
"""

import sys, os, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import CubicSpline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import (
    PROCESSED_DIR, HFT_FEATURES_DIR, HFT_BARS_DIR,
    HFT_COUPLING_RESOLUTION, HFT_BIO_IBI_PRECISION_MS,
    STRESS_TIME_RESOLUTION_SEC,
    ensure_dirs,
)


# ═══════════════════════════════════════════════════════════════════════
# Bio σ(t) loading and interpolation
# ═══════════════════════════════════════════════════════════════════════

def load_wesad_stress(processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame | None:
    """
    Load WESAD stress features and construct σ(t).

    Returns DataFrame with columns:
        subject, window_idx, time_sec, stress_probability
    """
    feat_path = processed_dir / "wesad_features.csv"
    if not feat_path.exists():
        print(f"  [!] WESAD features not found: {feat_path}")
        return None

    df = pd.read_csv(feat_path)

    # Expect columns: subject, window_idx, label, and stress-related features
    if "subject" not in df.columns:
        print("  [!] Missing 'subject' column in wesad_features.csv")
        return None

    # Use label as binary stress indicator if stress_probability missing
    if "stress_probability" not in df.columns:
        if "label" in df.columns:
            df["stress_probability"] = (df["label"] == 2).astype(float)
        else:
            print("  [!] Cannot construct σ(t): no label or stress_probability")
            return None

    # Reconstruct time axis from window index
    if "window_idx" in df.columns:
        df["time_sec"] = df["window_idx"] * STRESS_TIME_RESOLUTION_SEC
    elif "time_sec" not in df.columns:
        df["time_sec"] = np.arange(len(df)) * STRESS_TIME_RESOLUTION_SEC

    return df[["subject", "time_sec", "stress_probability"]].copy()


def interpolate_stress_to_ms(
    subject_df: pd.DataFrame,
    target_resolution_ms: int = 100,
) -> pd.DataFrame:
    """
    Upsample σ(t) from 5s windows to target ms resolution using cubic spline.

    Parameters
    ----------
    subject_df : DataFrame with columns [time_sec, stress_probability]
    target_resolution_ms : target grid spacing in milliseconds

    Returns
    -------
    DataFrame with columns [time_ms, stress_sigma]
    """
    t_sec = subject_df["time_sec"].values
    sigma = subject_df["stress_probability"].values

    if len(t_sec) < 4:
        return pd.DataFrame(columns=["time_ms", "stress_sigma"])

    # Cubic spline interpolation (preserves OU smoothness)
    cs = CubicSpline(t_sec, sigma, bc_type="clamped")

    # Target grid
    t_ms_start = int(t_sec[0] * 1000)
    t_ms_end = int(t_sec[-1] * 1000)
    t_ms_grid = np.arange(t_ms_start, t_ms_end + 1, target_resolution_ms)
    t_sec_grid = t_ms_grid / 1000.0

    sigma_interp = cs(t_sec_grid)
    # Clamp to [0, 1] — interpolation can overshoot
    sigma_interp = np.clip(sigma_interp, 0.0, 1.0)

    return pd.DataFrame({
        "time_ms": t_ms_grid,
        "stress_sigma": sigma_interp,
    })


# ═══════════════════════════════════════════════════════════════════════
# Market feature loading
# ═══════════════════════════════════════════════════════════════════════

def load_market_window(
    feature_dir: Path,
    date: str,
    duration_ms: int,
) -> pd.DataFrame | None:
    """
    Load HFT features for a specific date, trimmed to duration_ms.

    Returns DataFrame with bar_start as datetime + all feature columns.
    """
    feat_path = feature_dir / f"{date}.parquet"
    if not feat_path.exists():
        return None

    df = pd.read_parquet(feat_path)
    if df.empty:
        return None

    # Convert bar_start to ms offset from start of day
    if "bar_start" in df.columns:
        df["bar_start"] = pd.to_datetime(df["bar_start"])
        t0 = df["bar_start"].iloc[0]
        df["time_ms"] = ((df["bar_start"] - t0).dt.total_seconds() * 1000).astype(int)
    else:
        # Fallback: use row index × resolution
        df["time_ms"] = np.arange(len(df)) * 100  # assume 100ms

    # Trim to duration
    df = df[df["time_ms"] <= duration_ms].copy()
    return df


# ═══════════════════════════════════════════════════════════════════════
# Cross-correlation analysis
# ═══════════════════════════════════════════════════════════════════════

def compute_cross_correlation(
    bio_series: np.ndarray,
    market_series: np.ndarray,
    max_lag: int = 50,
) -> pd.DataFrame:
    """
    Compute normalised cross-correlation between bio and market at ±max_lag.

    Positive lag = bio leads market (bio at t predicts market at t+lag).
    Negative lag = market leads bio.

    Returns DataFrame with columns [lag, xcorr].
    """
    bio = (bio_series - np.mean(bio_series)) / (np.std(bio_series) + 1e-12)
    mkt = (market_series - np.mean(market_series)) / (np.std(market_series) + 1e-12)
    n = len(bio)

    lags = range(-max_lag, max_lag + 1)
    xcorrs = []
    for lag in lags:
        if lag >= 0:
            b = bio[:n - lag] if lag > 0 else bio
            m = mkt[lag:]
        else:
            b = bio[-lag:]
            m = mkt[:n + lag]

        min_len = min(len(b), len(m))
        if min_len < 2:
            xcorrs.append(0.0)
            continue
        xcorrs.append(float(np.mean(b[:min_len] * m[:min_len])))

    return pd.DataFrame({"lag": list(lags), "xcorr": xcorrs})


# ═══════════════════════════════════════════════════════════════════════
# Alignment builder
# ═══════════════════════════════════════════════════════════════════════

def build_alignment(
    bio_stress: pd.DataFrame,
    market_dates: list[str],
    feature_resolution: str,
    coupling_res_ms: int,
    max_xcorr_lag: int = 50,
) -> dict:
    """
    Build alignment pairs: each WESAD subject session is paired with
    a randomly sampled market window of the same duration.

    Returns dict with:
        - alignment_pairs: list of (subject, date, xcorr_summary)
        - merged_parquets: written to disk
    """
    feature_dir = HFT_FEATURES_DIR / feature_resolution
    out_dir = HFT_FEATURES_DIR / "bio_market_alignment"
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = bio_stress["subject"].unique()
    summary = []

    for subj in subjects:
        subj_df = bio_stress[bio_stress["subject"] == subj].sort_values("time_sec")
        duration_sec = subj_df["time_sec"].max() - subj_df["time_sec"].min()
        duration_ms = int(duration_sec * 1000)

        if duration_ms < 10_000:  # skip very short sessions
            continue

        # Interpolate bio to coupling resolution
        bio_interp = interpolate_stress_to_ms(subj_df, coupling_res_ms)
        if bio_interp.empty:
            continue

        # Pair with each available market date
        for date_str in market_dates:
            mkt = load_market_window(feature_dir, date_str, duration_ms)
            if mkt is None or len(mkt) < 10:
                continue

            # Align on common grid by time_ms
            merged = pd.merge_asof(
                bio_interp.sort_values("time_ms"),
                mkt.sort_values("time_ms"),
                on="time_ms",
                direction="nearest",
                tolerance=coupling_res_ms,
            )

            if merged.empty:
                continue

            # Save merged alignment
            out_path = out_dir / f"{subj}_{date_str}.parquet"
            merged.to_parquet(out_path, index=False, engine="pyarrow")

            # Cross-correlation of σ(t) with key market features
            xcorr_results = {}
            for feat in ["log_return", "order_flow_imbalance",
                         "realized_vol_10", "kyle_lambda"]:
                if feat not in merged.columns:
                    continue
                valid = merged[["stress_sigma", feat]].dropna()
                if len(valid) < 20:
                    continue
                xc = compute_cross_correlation(
                    valid["stress_sigma"].values,
                    valid[feat].values,
                    max_lag=max_xcorr_lag,
                )
                peak_lag = int(xc.loc[xc["xcorr"].abs().idxmax(), "lag"])
                peak_val = float(xc["xcorr"].abs().max())
                xcorr_results[feat] = {"peak_lag": peak_lag, "peak_xcorr": peak_val}

            summary.append({
                "subject": subj,
                "market_date": date_str,
                "n_aligned_rows": len(merged),
                "duration_sec": duration_sec,
                **{f"xcorr_{k}_lag": v["peak_lag"] for k, v in xcorr_results.items()},
                **{f"xcorr_{k}_val": v["peak_xcorr"] for k, v in xcorr_results.items()},
            })

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Build bio-market alignment layer for ABM"
    )
    parser.add_argument("--bio-resolution", default=HFT_COUPLING_RESOLUTION,
                        help="Target resolution for bio interpolation")
    parser.add_argument("--market-resolution", default=f"time_{HFT_COUPLING_RESOLUTION}",
                        help="Market feature resolution directory name")
    parser.add_argument("--max-dates", type=int, default=10,
                        help="Max market dates to pair per subject (for speed)")
    parser.add_argument("--max-lag", type=int, default=50,
                        help="Max cross-correlation lag (in bars)")
    args = parser.parse_args()

    ensure_dirs()

    # Parse coupling resolution to ms
    res_str = args.bio_resolution
    if res_str.endswith("ms"):
        coupling_ms = int(res_str.replace("ms", ""))
    elif res_str.endswith("s"):
        coupling_ms = int(res_str.replace("s", "")) * 1000
    else:
        coupling_ms = 100  # default

    print(f"Bio-Market Alignment")
    print(f"  Bio resolution    : {res_str} ({coupling_ms} ms)")
    print(f"  Market resolution : {args.market_resolution}")
    print(f"  Max dates/subject : {args.max_dates}")
    print(f"  Max xcorr lag     : {args.max_lag}\n")

    # ── Load bio stress ─────────────────────────────────────────────
    bio = load_wesad_stress()
    if bio is None or bio.empty:
        print("[!] No bio stress data available. Exiting.")
        return

    n_subj = bio["subject"].nunique()
    print(f"  Bio: {n_subj} subjects, {len(bio)} windows\n")

    # ── Discover available market feature dates ─────────────────────
    feat_dir = HFT_FEATURES_DIR / args.market_resolution
    if not feat_dir.exists():
        print(f"[!] Market features not found: {feat_dir}")
        print("    Run 01 + 02 first.")
        return

    market_dates = sorted([f.stem for f in feat_dir.glob("*.parquet")])
    if not market_dates:
        print("[!] No market feature files found.")
        return

    # Limit dates for speed
    if len(market_dates) > args.max_dates:
        # Sample evenly across the range
        idx = np.linspace(0, len(market_dates) - 1, args.max_dates, dtype=int)
        market_dates = [market_dates[i] for i in idx]

    print(f"  Market: {len(market_dates)} dates selected\n")

    # ── Build alignment ─────────────────────────────────────────────
    summary = build_alignment(
        bio, market_dates, args.market_resolution, coupling_ms, args.max_lag
    )

    # ── Save summary ────────────────────────────────────────────────
    out_dir = HFT_FEATURES_DIR / "bio_market_alignment"
    out_dir.mkdir(parents=True, exist_ok=True)

    if summary:
        summary_df = pd.DataFrame(summary)
        summary_path = out_dir / "alignment_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n  Summary: {len(summary)} alignment pairs -> {summary_path}")

        # Print cross-correlation highlights
        xcorr_cols = [c for c in summary_df.columns if c.startswith("xcorr_")]
        if xcorr_cols:
            print("\n  Cross-correlation highlights:")
            for col in xcorr_cols:
                vals = summary_df[col].dropna()
                if not vals.empty:
                    print(f"    {col}: mean={vals.mean():.3f}  std={vals.std():.3f}")
    else:
        print("\n  [!] No alignment pairs produced.")

    print("\nDone.")


if __name__ == "__main__":
    main()
