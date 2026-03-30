"""
Script 01 – Build Multi-Resolution Bars from Tick Parquet
============================================================
Aggregate tick-level parquet files (from Script 00) into:
  A) Time bars  – 100ms and 1s OHLCV
  B) Volume bars – fixed BTC volume per bar
  C) Dollar bars – fixed USD notional per bar
  D) Tick bars   – fixed number of trades per bar

Output directory: data/processed/tardis/bars/<bar_type>/

Each bar contains:
    open, high, low, close, volume, dollar_volume,
    n_trades, signed_volume, vwap, bar_duration_ms

Usage:
    python scripts/stage2_economics/01_build_multiresolution_bars.py
    python scripts/stage2_economics/01_build_multiresolution_bars.py --start 2021-01-01 --end 2021-12-31
    python scripts/stage2_economics/01_build_multiresolution_bars.py --bar-types time volume dollar tick
"""

import sys, os, argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import (
    TARDIS_START_DATE, TARDIS_END_DATE,
    HFT_TICK_DIR, HFT_BARS_DIR,
    HFT_BAR_RESOLUTIONS,
    HFT_VOLUME_BAR_SIZE_BTC, HFT_DOLLAR_BAR_SIZE_USD, HFT_TICK_BAR_SIZE,
    ensure_dirs,
)


# ═══════════════════════════════════════════════════════════════════════
# Time Bars
# ═══════════════════════════════════════════════════════════════════════

def build_time_bars(ticks: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Resample tick data into time bars at the given frequency.

    Parameters
    ----------
    ticks : DataFrame with columns [datetime, price, amount, dollar_volume,
            signed_volume, timestamp_ms]
    freq  : pandas frequency string, e.g. "100ms", "1s"

    Returns
    -------
    DataFrame indexed by bar_start with OHLCV + microstructure columns.
    """
    df = ticks.set_index("datetime").sort_index()

    bars = df["price"].resample(freq).ohlc()
    bars.columns = ["open", "high", "low", "close"]

    bars["volume"] = df["amount"].resample(freq).sum()
    bars["dollar_volume"] = df["dollar_volume"].resample(freq).sum()
    bars["n_trades"] = df["price"].resample(freq).count()
    bars["signed_volume"] = df["signed_volume"].resample(freq).sum()

    # VWAP = Σ(price × amount) / Σ(amount)
    _pv = (df["price"] * df["amount"]).resample(freq).sum()
    bars["vwap"] = _pv / (bars["volume"] + 1e-12)

    # Bar duration: time span of actual trades within the bar
    _ts_min = df["timestamp_ms"].resample(freq).min()
    _ts_max = df["timestamp_ms"].resample(freq).max()
    bars["bar_duration_ms"] = _ts_max - _ts_min

    # Drop empty bars (no trades in that interval)
    bars = bars.dropna(subset=["open"])
    bars = bars[bars["n_trades"] > 0]

    bars.index.name = "bar_start"
    return bars.reset_index()


# ═══════════════════════════════════════════════════════════════════════
# Information-Driven Bars (Volume / Dollar / Tick)
# ═══════════════════════════════════════════════════════════════════════

def _build_threshold_bars(
    ticks: pd.DataFrame,
    cumsum_col: str,
    threshold: float,
) -> pd.DataFrame:
    """
    Generic threshold-bar builder.

    Accumulates `cumsum_col` across ticks; when the cumulative sum
    reaches `threshold`, a new bar is emitted and the accumulator resets.

    Parameters
    ----------
    ticks      : sorted DataFrame with [datetime, price, amount, ...]
    cumsum_col : column to accumulate ("amount", "dollar_volume", or "__one")
    threshold  : bar boundary value

    Returns
    -------
    DataFrame with one row per bar.
    """
    bars = []
    accum = 0.0
    bar_open = bar_high = bar_low = None
    bar_close = 0.0
    bar_vol = 0.0
    bar_dvol = 0.0
    bar_signed = 0.0
    bar_count = 0
    bar_start_ts = None
    bar_start_dt = None
    bar_pv = 0.0  # price × volume accumulator for VWAP

    for row in ticks.itertuples(index=False):
        price = row.price
        amount = row.amount
        dvol = row.dollar_volume
        signed = row.signed_volume
        ts_ms = row.timestamp_ms
        dt = row.datetime

        if bar_open is None:
            bar_open = price
            bar_high = price
            bar_low = price
            bar_start_ts = ts_ms
            bar_start_dt = dt

        bar_high = max(bar_high, price)
        bar_low = min(bar_low, price)
        bar_close = price
        bar_vol += amount
        bar_dvol += dvol
        bar_signed += signed
        bar_count += 1
        bar_pv += price * amount

        val = getattr(row, cumsum_col) if cumsum_col != "__one" else 1
        accum += abs(val)

        if accum >= threshold:
            bars.append({
                "bar_start": bar_start_dt,
                "open": bar_open,
                "high": bar_high,
                "low": bar_low,
                "close": bar_close,
                "volume": bar_vol,
                "dollar_volume": bar_dvol,
                "n_trades": bar_count,
                "signed_volume": bar_signed,
                "vwap": bar_pv / (bar_vol + 1e-12),
                "bar_duration_ms": ts_ms - bar_start_ts,
            })
            # Reset
            accum = 0.0
            bar_open = bar_high = bar_low = None
            bar_vol = bar_dvol = bar_signed = bar_pv = 0.0
            bar_count = 0
            bar_start_ts = bar_start_dt = None

    return pd.DataFrame(bars)


def build_volume_bars(ticks: pd.DataFrame, size_btc: float) -> pd.DataFrame:
    return _build_threshold_bars(ticks, "amount", size_btc)


def build_dollar_bars(ticks: pd.DataFrame, size_usd: float) -> pd.DataFrame:
    return _build_threshold_bars(ticks, "dollar_volume", size_usd)


def build_tick_bars(ticks: pd.DataFrame, n_ticks: int) -> pd.DataFrame:
    """Fixed number of trades per bar."""
    ticks = ticks.copy()
    ticks["__one"] = 1.0
    result = _build_threshold_bars(ticks, "__one", float(n_ticks))
    return result


# ═══════════════════════════════════════════════════════════════════════
# Main driver
# ═══════════════════════════════════════════════════════════════════════

BAR_BUILDERS = {
    "time": None,     # handled separately per resolution
    "volume": lambda t: build_volume_bars(t, HFT_VOLUME_BAR_SIZE_BTC),
    "dollar": lambda t: build_dollar_bars(t, HFT_DOLLAR_BAR_SIZE_USD),
    "tick": lambda t: build_tick_bars(t, HFT_TICK_BAR_SIZE),
}


def process_day(tick_path: Path, bar_types: list[str]) -> dict:
    """Build requested bar types for one day's tick parquet."""
    date_str = tick_path.stem  # e.g. "2021-05-20"
    ticks = pd.read_parquet(tick_path)

    if ticks.empty:
        return {"date": date_str, "status": "empty"}

    # Ensure datetime column exists
    if "datetime" not in ticks.columns:
        ticks["datetime"] = pd.to_datetime(ticks["timestamp_ms"], unit="ms")

    results = {"date": date_str, "status": "ok", "bars": {}}

    for bt in bar_types:
        if bt == "time":
            for freq in HFT_BAR_RESOLUTIONS:
                out_dir = HFT_BARS_DIR / f"time_{freq}"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{date_str}.parquet"

                bars = build_time_bars(ticks, freq)
                bars.to_parquet(out_path, index=False, engine="pyarrow")
                results["bars"][f"time_{freq}"] = len(bars)
        else:
            out_dir = HFT_BARS_DIR / bt
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{date_str}.parquet"

            builder = BAR_BUILDERS[bt]
            bars = builder(ticks)
            if not bars.empty:
                bars.to_parquet(out_path, index=False, engine="pyarrow")
            results["bars"][bt] = len(bars)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Build multi-resolution bars from tick parquet"
    )
    parser.add_argument("--start", default=TARDIS_START_DATE)
    parser.add_argument("--end", default=TARDIS_END_DATE)
    parser.add_argument("--bar-types", nargs="+",
                        default=["time", "volume", "dollar", "tick"],
                        choices=["time", "volume", "dollar", "tick"],
                        help="Which bar types to build")
    args = parser.parse_args()

    ensure_dirs()

    start_dt = pd.Timestamp(args.start)
    end_dt = pd.Timestamp(args.end)

    tick_files = sorted(HFT_TICK_DIR.glob("*.parquet"))
    if not tick_files:
        print("[!] No tick parquet files found. Run 00_reindex_ticks.py first.")
        return

    # Filter by date range
    tick_files = [
        f for f in tick_files
        if start_dt <= pd.Timestamp(f.stem) <= end_dt
    ]

    print(f"Bar construction: {args.start} -> {args.end}")
    print(f"  Tick files: {len(tick_files)}")
    print(f"  Bar types : {args.bar_types}")
    print(f"  Output    : {HFT_BARS_DIR}\n")

    ok = 0
    for fpath in tick_files:
        result = process_day(fpath, args.bar_types)
        status = result.get("status", "unknown")
        if status == "ok":
            ok += 1
            bar_info = "  ".join(
                f"{k}={v}" for k, v in result["bars"].items()
            )
            print(f"  [OK] {result['date']}  {bar_info}")
        else:
            print(f"  [FAIL] {result['date']}  {status}")

    print(f"\nDone: {ok}/{len(tick_files)} days processed")


if __name__ == "__main__":
    main()
