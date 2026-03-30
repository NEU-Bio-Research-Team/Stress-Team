"""
Script 02 – HFT Feature Engineering
======================================
Compute high-frequency trading microstructure features from
multi-resolution bars (produced by Script 01).

Feature catalog:
    ┌────────────────────────┬──────────────────────────────────────────┐
    │ Feature                │ Description                              │
    ├────────────────────────┼──────────────────────────────────────────┤
    │ log_return             │ log(close / prev_close)                  │
    │ realized_vol_N         │ √Σr² over rolling N bars                │
    │ trade_arrival_rate     │ n_trades / bar_duration_ms × 1000       │
    │ order_flow_imbalance   │ signed_volume / volume                  │
    │ kyle_lambda            │ ΔP / signed_volume regression slope     │
    │ amihud_illiq           │ |return| / dollar_volume                │
    │ roll_spread            │ 2√(-Cov(Δp_t, Δp_{t-1}))              │
    │ vpin                   │ Σ|V_buy - V_sell| / Σ|V|  (rolling)    │
    │ inter_arrival_cv       │ std(bar_duration) / mean(bar_duration)  │
    │ hurst_exponent         │ R/S estimate over rolling window        │
    └────────────────────────┴──────────────────────────────────────────┘

Usage:
    python scripts/stage2_economics/02_hft_feature_engineering.py
    python scripts/stage2_economics/02_hft_feature_engineering.py --resolution time_100ms
    python scripts/stage2_economics/02_hft_feature_engineering.py --resolution volume --window 50
"""

import sys, os, argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import (
    TARDIS_START_DATE, TARDIS_END_DATE,
    HFT_BARS_DIR, HFT_FEATURES_DIR,
    HFT_ROLLING_WINDOWS, HFT_KYLE_LAMBDA_WINDOW,
    ensure_dirs,
)


# ═══════════════════════════════════════════════════════════════════════
# Feature functions (vectorised where possible)
# ═══════════════════════════════════════════════════════════════════════

def compute_log_returns(close: pd.Series) -> pd.Series:
    """Log returns, first value = 0."""
    return np.log(close / close.shift(1)).fillna(0.0)


def compute_realized_vol(returns: pd.Series, window: int) -> pd.Series:
    """Rolling realized volatility: sqrt of sum of squared returns."""
    return (returns ** 2).rolling(window, min_periods=1).sum().apply(np.sqrt)


def compute_trade_arrival_rate(n_trades: pd.Series,
                                bar_duration_ms: pd.Series) -> pd.Series:
    """Trades per second within each bar."""
    duration_s = bar_duration_ms / 1000.0
    return n_trades / duration_s.replace(0, np.nan)


def compute_order_flow_imbalance(signed_volume: pd.Series,
                                  volume: pd.Series) -> pd.Series:
    """OFI = signed_volume / total_volume, range [-1, 1]."""
    return signed_volume / (volume + 1e-12)


def compute_kyle_lambda(returns: pd.Series, signed_volume: pd.Series,
                         window: int) -> pd.Series:
    """
    Kyle's lambda: rolling OLS slope of ΔP on signed volume.
    λ = Cov(ΔP, SV) / Var(SV)  over rolling window.
    """
    cov = returns.rolling(window, min_periods=max(window // 2, 2)).cov(signed_volume)
    var = signed_volume.rolling(window, min_periods=max(window // 2, 2)).var()
    return cov / (var + 1e-12)


def compute_amihud_illiquidity(returns: pd.Series,
                                dollar_volume: pd.Series) -> pd.Series:
    """Amihud illiquidity: |return| / dollar volume."""
    return returns.abs() / (dollar_volume + 1e-12)


def compute_roll_spread(price_change: pd.Series, window: int) -> pd.Series:
    """
    Roll (1984) effective spread estimator:
    spread = 2 * sqrt( -Cov(Δp_t, Δp_{t-1}) )
    Negative cov → positive spread; clamp negative values to 0.
    """
    dp = price_change if price_change is not None else pd.Series(dtype=float)
    dp_lag = dp.shift(1)
    roll_cov = dp.rolling(window, min_periods=max(window // 2, 2)).cov(dp_lag)
    # Roll spread only defined when cov is negative
    neg_cov = (-roll_cov).clip(lower=0)
    return 2.0 * np.sqrt(neg_cov)


def compute_vpin(signed_volume: pd.Series, volume: pd.Series,
                  window: int) -> pd.Series:
    """
    Volume-synchronised probability of informed trading (simplified).
    VPIN = rolling_sum(|signed_volume|) / rolling_sum(volume)
    """
    abs_sv = signed_volume.abs()
    return (abs_sv.rolling(window, min_periods=1).sum()
            / (volume.rolling(window, min_periods=1).sum() + 1e-12))


def compute_inter_arrival_cv(bar_duration_ms: pd.Series,
                              window: int) -> pd.Series:
    """
    Coefficient of variation of bar durations (rolling).
    High CV → bursty trading; low CV → uniform.
    """
    mu = bar_duration_ms.rolling(window, min_periods=2).mean()
    sigma = bar_duration_ms.rolling(window, min_periods=2).std()
    return sigma / (mu + 1e-12)


def compute_hurst_rs(returns: pd.Series, window: int) -> pd.Series:
    """
    Simplified R/S Hurst exponent estimate over rolling window.
    H > 0.5 → trending, H < 0.5 → mean-reverting, H ≈ 0.5 → random walk.
    """
    def _rs_hurst(x):
        if len(x) < 10:
            return np.nan
        y = x - x.mean()
        cumdev = np.cumsum(y)
        r = cumdev.max() - cumdev.min()
        s = x.std()
        if s < 1e-12:
            return np.nan
        return np.log(r / s) / np.log(len(x))

    return returns.rolling(window, min_periods=max(window // 2, 10)).apply(
        _rs_hurst, raw=True
    )


# ═══════════════════════════════════════════════════════════════════════
# Orchestration
# ═══════════════════════════════════════════════════════════════════════

def engineer_features(bars: pd.DataFrame, windows: list[int],
                      kyle_window: int) -> pd.DataFrame:
    """
    Compute all HFT features for a single bar DataFrame.

    Parameters
    ----------
    bars        : DataFrame with OHLCV + signed_volume + bar_duration_ms + dollar_volume
    windows     : list of rolling windows for multi-scale features
    kyle_window : window for Kyle's lambda

    Returns
    -------
    DataFrame with original bar columns + engineered features.
    """
    df = bars.copy()

    # ── Base returns ────────────────────────────────────────────────
    df["log_return"] = compute_log_returns(df["close"])

    if "bar_start" in df.columns:
        df["price_change"] = df["close"].diff().fillna(0.0)
    elif "price_change" not in df.columns:
        df["price_change"] = df["close"].diff().fillna(0.0)

    # ── Per-bar instantaneous features ──────────────────────────────
    df["trade_arrival_rate"] = compute_trade_arrival_rate(
        df["n_trades"], df["bar_duration_ms"]
    )
    df["order_flow_imbalance"] = compute_order_flow_imbalance(
        df["signed_volume"], df["volume"]
    )
    df["amihud_illiq"] = compute_amihud_illiquidity(
        df["log_return"], df["dollar_volume"]
    )

    # ── Kyle's lambda ───────────────────────────────────────────────
    df["kyle_lambda"] = compute_kyle_lambda(
        df["log_return"], df["signed_volume"], kyle_window
    )

    # ── Rolling multi-scale features ────────────────────────────────
    for w in windows:
        suffix = f"_{w}"
        df[f"realized_vol{suffix}"] = compute_realized_vol(df["log_return"], w)
        df[f"roll_spread{suffix}"] = compute_roll_spread(df["price_change"], w)
        df[f"vpin{suffix}"] = compute_vpin(df["signed_volume"], df["volume"], w)
        df[f"inter_arrival_cv{suffix}"] = compute_inter_arrival_cv(
            df["bar_duration_ms"], w
        )

    # ── Hurst exponent (longest window only to save compute) ───────
    longest_w = max(windows)
    df[f"hurst_{longest_w}"] = compute_hurst_rs(df["log_return"], longest_w)

    return df


def process_resolution(resolution: str, start_dt, end_dt,
                        windows: list[int], kyle_window: int) -> int:
    """Process all daily bar files for one resolution."""
    bar_dir = HFT_BARS_DIR / resolution
    if not bar_dir.exists():
        print(f"  [!] Bar directory not found: {bar_dir}")
        return 0

    out_dir = HFT_FEATURES_DIR / resolution
    out_dir.mkdir(parents=True, exist_ok=True)

    bar_files = sorted(bar_dir.glob("*.parquet"))
    bar_files = [
        f for f in bar_files
        if start_dt <= pd.Timestamp(f.stem) <= end_dt
    ]

    if not bar_files:
        print(f"  [!] No bar files in range for {resolution}")
        return 0

    ok = 0
    for fpath in bar_files:
        date_str = fpath.stem
        out_path = out_dir / f"{date_str}.parquet"

        bars = pd.read_parquet(fpath)
        if bars.empty:
            continue

        feats = engineer_features(bars, windows, kyle_window)
        feats.to_parquet(out_path, index=False, engine="pyarrow")
        ok += 1
        print(f"    [OK] {date_str}  features={len(feats.columns)}  rows={len(feats)}")

    return ok


def main():
    parser = argparse.ArgumentParser(
        description="HFT feature engineering from multi-resolution bars"
    )
    parser.add_argument("--start", default=TARDIS_START_DATE)
    parser.add_argument("--end", default=TARDIS_END_DATE)
    parser.add_argument("--resolution", nargs="+", default=None,
                        help="Bar resolutions to process (e.g. time_100ms volume). "
                             "Default: all found in bars dir.")
    parser.add_argument("--window", nargs="+", type=int, default=None,
                        help="Rolling windows. Default from config.")
    args = parser.parse_args()

    ensure_dirs()

    start_dt = pd.Timestamp(args.start)
    end_dt = pd.Timestamp(args.end)
    windows = args.window if args.window else HFT_ROLLING_WINDOWS
    kyle_w = HFT_KYLE_LAMBDA_WINDOW

    # Auto-discover resolutions if not specified
    if args.resolution:
        resolutions = args.resolution
    else:
        if HFT_BARS_DIR.exists():
            resolutions = sorted([
                d.name for d in HFT_BARS_DIR.iterdir() if d.is_dir()
            ])
        else:
            resolutions = []

    if not resolutions:
        print("[!] No bar directories found. Run 01_build_multiresolution_bars.py first.")
        return

    print(f"HFT Feature Engineering: {args.start} -> {args.end}")
    print(f"  Resolutions: {resolutions}")
    print(f"  Windows    : {windows}")
    print(f"  Kyle λ win : {kyle_w}")
    print(f"  Output     : {HFT_FEATURES_DIR}\n")

    total = 0
    for res in resolutions:
        print(f"  [{res}]")
        n = process_resolution(res, start_dt, end_dt, windows, kyle_w)
        total += n

    print(f"\nDone: {total} feature files written")


if __name__ == "__main__":
    main()
