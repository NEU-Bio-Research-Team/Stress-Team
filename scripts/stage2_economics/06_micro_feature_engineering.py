"""
Script 06 - Micro Feature Engineering (Event-Level HFT)
=========================================================
Build real HFT microstructure features at 10ms/100ms/1s resolution
from aggTrades + bookTicker downloaded by Script 05.

Feature catalog (per resolution bin):
    === From aggTrades ===
    ofi             : Order Flow Imbalance = sum(signed_volume)
    trade_count     : Number of trades in bin
    volume          : Total volume
    dollar_volume   : Total dollar volume
    vwap            : Volume-weighted average price
    log_return      : log(vwap_t / vwap_{t-1})
    realized_var    : Sum of squared log returns (rolling)
    trade_intensity : trades / second (arrival rate)
    buy_ratio       : buy_volume / total_volume

    === From bookTicker (if available) ===
    spread_bps      : Bid-ask spread in basis points (last BBO in bin)
    mid_price       : (bid + ask) / 2
    touch_depth     : bid_qty + ask_qty at BBO
    depth_imbalance : (bid_qty - ask_qty) / (bid_qty + ask_qty)

    === Merged (aggTrades x bookTicker) ===
    effective_spread: |trade_price - mid_price| / mid_price * 10000 (bps)
    kyle_lambda     : regression slope of dp on signed_volume (rolling)
    vpin            : |sum(buy_vol) - sum(sell_vol)| / sum(vol) (rolling)

    === Rolling features ===
    realized_vol_*  : Realized volatility at multiple windows
    roll_spread_*   : Roll's implicit spread estimate
    ofi_autocorr    : OFI autocorrelation (persistence)
    amihud_illiq    : |return| / dollar_volume

Usage:
    python scripts/stage2_economics/06_micro_feature_engineering.py
    python scripts/stage2_economics/06_micro_feature_engineering.py --resolution 100
"""

import sys, os, argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import (
    EVENT_RAW_DIR, EVENT_MICRO_DIR, EVENT_CATALOG_PATH,
    MICRO_RESOLUTIONS_MS, MICRO_KYLE_WINDOW, MICRO_VPIN_WINDOW,
    ensure_dirs,
)


# =====================================================================
# aggTrades -> time-binned bars
# =====================================================================

def aggtrades_to_bars(df: pd.DataFrame, resolution_ms: int) -> pd.DataFrame:
    """
    Aggregate aggTrades into fixed-width time bars.

    Parameters
    ----------
    df : DataFrame with columns [timestamp_ms, price, quantity, side,
         dollar_volume, signed_volume]
    resolution_ms : bin width in milliseconds

    Returns DataFrame with one row per time bin.
    """
    if df.empty:
        return pd.DataFrame()

    df = df.sort_values("timestamp_ms").copy()
    t0 = df["timestamp_ms"].iloc[0]

    # Assign bin index
    df["bin_id"] = ((df["timestamp_ms"] - t0) // resolution_ms).astype(np.int64)

    grouped = df.groupby("bin_id")

    bars = pd.DataFrame({
        "bin_id": grouped.ngroup().unique() if hasattr(grouped, "ngroup") else sorted(df["bin_id"].unique()),
    })
    # Rebuild from groupby
    bars = grouped.agg(
        timestamp_ms=("timestamp_ms", "first"),
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("quantity", "sum"),
        dollar_volume=("dollar_volume", "sum"),
        trade_count=("price", "count"),
        ofi=("signed_volume", "sum"),
        buy_volume=("quantity", lambda x: x[df.loc[x.index, "side"] == "buy"].sum()),
    ).reset_index()

    bars["timestamp_ms"] = bars["timestamp_ms"].astype(np.int64)

    # Derived features
    bars["vwap"] = np.where(
        bars["volume"] > 0,
        bars["dollar_volume"] / bars["volume"],
        bars["close"],
    )
    bars["log_return"] = np.log(bars["vwap"] / bars["vwap"].shift(1)).fillna(0.0)
    bars["trade_intensity"] = bars["trade_count"] / (resolution_ms / 1000.0)
    bars["buy_ratio"] = np.where(
        bars["volume"] > 0,
        bars["buy_volume"] / bars["volume"],
        0.5,
    )

    # Time offset from start
    bars["time_offset_ms"] = (bars["bin_id"] * resolution_ms).astype(np.int64)

    return bars


# =====================================================================
# bookTicker -> time-binned BBO snapshots
# =====================================================================

def bookticker_to_bars(df: pd.DataFrame, resolution_ms: int, t0: int) -> pd.DataFrame:
    """
    Aggregate bookTicker into time bars (last BBO per bin).
    """
    if df.empty:
        return pd.DataFrame()

    df = df.sort_values("timestamp_ms").copy()
    df["bin_id"] = ((df["timestamp_ms"] - t0) // resolution_ms).astype(np.int64)

    # Last BBO per bin
    bars = df.groupby("bin_id").agg(
        best_bid_price=("best_bid_price", "last"),
        best_ask_price=("best_ask_price", "last"),
        best_bid_qty=("best_bid_qty", "last"),
        best_ask_qty=("best_ask_qty", "last"),
        mid_price=("mid_price", "last") if "mid_price" in df.columns else ("best_bid_price", "last"),
        spread_bps=("spread_bps", "last") if "spread_bps" in df.columns else ("best_bid_price", "last"),
        touch_depth=("touch_depth", "last") if "touch_depth" in df.columns else ("best_bid_qty", "last"),
        depth_imbalance=("depth_imbalance", "last") if "depth_imbalance" in df.columns else ("best_bid_qty", "last"),
        bbo_updates=("best_bid_price", "count"),
    ).reset_index()

    # Recompute if source columns were missing
    if "mid_price" not in df.columns:
        bars["mid_price"] = (bars["best_bid_price"] + bars["best_ask_price"]) / 2.0
    if "spread_bps" not in df.columns:
        bars["spread_bps"] = (bars["best_ask_price"] - bars["best_bid_price"]) / bars["mid_price"] * 10000.0
    if "touch_depth" not in df.columns:
        bars["touch_depth"] = bars["best_bid_qty"] + bars["best_ask_qty"]
    if "depth_imbalance" not in df.columns:
        total = bars["best_bid_qty"] + bars["best_ask_qty"]
        bars["depth_imbalance"] = np.where(
            total > 0,
            (bars["best_bid_qty"] - bars["best_ask_qty"]) / total,
            0.0,
        )

    return bars


# =====================================================================
# Merge trades + BBO and compute cross-features
# =====================================================================

def merge_and_enrich(
    trade_bars: pd.DataFrame,
    bbo_bars: pd.DataFrame | None,
    kyle_window: int,
    vpin_window: int,
) -> pd.DataFrame:
    """
    Merge trade bars with BBO bars and compute cross-features.
    """
    df = trade_bars.copy()

    # Merge BBO if available
    if bbo_bars is not None and not bbo_bars.empty:
        bbo_cols = [c for c in bbo_bars.columns if c != "bin_id"]
        # Rename to avoid collision if needed
        df = df.merge(bbo_bars[["bin_id"] + bbo_cols], on="bin_id", how="left")
        # Forward-fill BBO state
        for col in bbo_cols:
            if col in df.columns:
                df[col] = df[col].ffill()

        # Effective spread: |trade_price - mid_price| / mid_price * 10000
        if "mid_price" in df.columns:
            df["effective_spread_bps"] = np.where(
                df["mid_price"] > 0,
                np.abs(df["vwap"] - df["mid_price"]) / df["mid_price"] * 10000.0,
                np.nan,
            )

    # -- Rolling features --

    # Kyle's lambda: regression of price change on signed volume
    # Vectorized via E[dp*sv] - E[dp]*E[sv] / (E[sv^2] - E[sv]^2)
    df["price_change"] = df["close"].diff().fillna(0.0)
    sv = df["ofi"]
    dp = df["price_change"]
    _sv_mean   = sv.rolling(kyle_window, min_periods=5).mean()
    _dp_mean   = dp.rolling(kyle_window, min_periods=5).mean()
    _sv2_mean  = (sv ** 2).rolling(kyle_window, min_periods=5).mean()
    _dpsv_mean = (dp * sv).rolling(kyle_window, min_periods=5).mean()
    _sv_var    = _sv2_mean - _sv_mean ** 2
    _dp_sv_cov = _dpsv_mean - _dp_mean * _sv_mean
    df["kyle_lambda"] = np.where(_sv_var > 1e-12, _dp_sv_cov / _sv_var, 0.0)
    df["kyle_lambda"] = pd.to_numeric(df["kyle_lambda"], errors="coerce").fillna(0.0)

    # VPIN: |sum(buy) - sum(sell)| / sum(vol) over rolling window
    buy_vol = df["buy_volume"]
    sell_vol = df["volume"] - df["buy_volume"]
    df["vpin"] = (
        (buy_vol.rolling(vpin_window, min_periods=1).sum() -
         sell_vol.rolling(vpin_window, min_periods=1).sum()).abs() /
        df["volume"].rolling(vpin_window, min_periods=1).sum().replace(0, np.nan)
    ).fillna(0.0)

    # Realized volatility (multiple windows)
    for w in [10, 50, 200]:
        df[f"realized_vol_{w}"] = df["log_return"].rolling(
            w, min_periods=1
        ).apply(lambda x: np.sqrt(np.sum(x**2)), raw=True)

    # Roll's implicit spread (vectorized via rolling autocovariance formula)
    dp = df["price_change"]
    dp_lag = dp.shift(1)
    for w in [10, 50]:
        _dp_m   = dp.rolling(w, min_periods=2).mean()
        _lag_m  = dp_lag.rolling(w, min_periods=2).mean()
        _prod_m = (dp * dp_lag).rolling(w, min_periods=2).mean()
        _autocov = (_prod_m - _dp_m * _lag_m).fillna(0.0)
        df[f"roll_spread_{w}"] = 2.0 * np.sqrt(np.maximum(-_autocov, 0.0))

    # Amihud illiquidity
    df["amihud_illiq"] = np.where(
        df["dollar_volume"] > 0,
        np.abs(df["log_return"]) / df["dollar_volume"],
        0.0,
    )

    # OFI autocorrelation (vectorized rolling corr with lag-1)
    df["ofi_autocorr_20"] = df["ofi"].rolling(20, min_periods=2).corr(
        df["ofi"].shift(1)
    ).fillna(0.0)

    return df


# =====================================================================
# Process one event
# =====================================================================

def process_event(
    event_dir: Path,
    resolution_ms: int,
    kyle_window: int,
    vpin_window: int,
    output_dir: Path,
) -> dict | None:
    """Process one event directory into micro features."""
    agg_path = event_dir / "aggtrades.parquet"
    bkt_path = event_dir / "bookticker.parquet"

    if not agg_path.exists():
        return None

    agg_df = pd.read_parquet(agg_path)
    if agg_df.empty:
        return None

    # Build trade bars
    trade_bars = aggtrades_to_bars(agg_df, resolution_ms)
    if trade_bars.empty:
        return None

    # Build BBO bars (optional)
    bbo_bars = None
    has_bbo = False
    if bkt_path.exists():
        bkt_df = pd.read_parquet(bkt_path)
        if not bkt_df.empty:
            t0 = agg_df["timestamp_ms"].iloc[0]
            bbo_bars = bookticker_to_bars(bkt_df, resolution_ms, t0)
            has_bbo = True

    # Merge and compute features
    features = merge_and_enrich(trade_bars, bbo_bars, kyle_window, vpin_window)

    # Save
    out_name = event_dir.name
    out_path = output_dir / f"{out_name}_{resolution_ms}ms.parquet"
    features.to_parquet(out_path, index=False, engine="pyarrow")

    return {
        "event_dir": event_dir.name,
        "resolution_ms": resolution_ms,
        "n_bars": len(features),
        "n_features": len(features.columns),
        "has_bbo": has_bbo,
        "n_trades": len(agg_df),
        "output": str(out_path),
    }


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Micro HFT feature engineering from event tick data"
    )
    parser.add_argument("--resolution", nargs="+", type=int, default=None,
                        help="Resolution(s) in ms (default: from config)")
    parser.add_argument("--kyle-window", type=int, default=MICRO_KYLE_WINDOW)
    parser.add_argument("--vpin-window", type=int, default=MICRO_VPIN_WINDOW)
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip events whose output parquet already exists")
    args = parser.parse_args()

    ensure_dirs()

    resolutions = args.resolution or MICRO_RESOLUTIONS_MS

    print("Micro Feature Engineering (Tier 2)")
    print(f"  Input       : {EVENT_RAW_DIR}")
    print(f"  Output      : {EVENT_MICRO_DIR}")
    print(f"  Resolutions : {resolutions} ms")
    print(f"  Kyle window : {args.kyle_window}")
    print(f"  VPIN window : {args.vpin_window}\n")

    # Discover event directories
    event_dirs = sorted([
        d for d in EVENT_RAW_DIR.iterdir()
        if d.is_dir() and (d / "aggtrades.parquet").exists()
    ])

    if not event_dirs:
        print("[!] No event data found. Run script 05 first.")
        return

    print(f"  Found {len(event_dirs)} events\n")

    all_results = []

    for res_ms in resolutions:
        print(f"  --- Resolution: {res_ms}ms ---")
        out_dir = EVENT_MICRO_DIR / f"res_{res_ms}ms"
        out_dir.mkdir(parents=True, exist_ok=True)

        for edir in event_dirs:
            out_path = out_dir / f"{edir.name}_{res_ms}ms.parquet"
            if args.skip_existing and out_path.exists():
                print(f"    [SKIP] {edir.name}  (already exists)")
                continue
            result = process_event(
                edir, res_ms,
                args.kyle_window, args.vpin_window,
                out_dir,
            )
            if result:
                print(f"    [OK] {edir.name}  bars={result['n_bars']}  "
                      f"features={result['n_features']}  bbo={result['has_bbo']}")
                all_results.append(result)
            else:
                print(f"    [SKIP] {edir.name}  (no data)")

        print()

    # Summary
    if all_results:
        summary = pd.DataFrame(all_results)
        summary_path = EVENT_MICRO_DIR / "micro_features_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"  Summary: {len(all_results)} feature files written")
        print(f"  Saved: {summary_path}")
    else:
        print("  [!] No features produced.")

    print("\nDone.")


if __name__ == "__main__":
    main()
