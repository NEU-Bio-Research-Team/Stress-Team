"""
Script 04 - Flash Crash Detection from 1-min Klines
=====================================================
Scan existing 1-minute kline data to identify flash crash events.

Strategy:
    1. Load all daily kline parquets (from script 00 output)
    2. Compute rolling minimum return over FLASH_CRASH_WINDOW_MIN
    3. Detect drops > FLASH_CRASH_DROP_PCT within the window
    4. Check for price recovery (flash vs trend crash)
    5. Deduplicate events within MIN_SEPARATION hours
    6. Output: event_catalog.csv with crash timestamps + metadata

This is Tier 1 (Macro) of the 2-tier pipeline:
    Tier 1: klines -> flash crash catalog
    Tier 2: aggTrades + bookTicker -> micro features (scripts 05-07)

Usage:
    python scripts/stage2_economics/04_detect_flash_crashes.py
    python scripts/stage2_economics/04_detect_flash_crashes.py --drop-pct 5.0 --window 10
"""

import sys, os, argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import (
    TARDIS_START_DATE, TARDIS_END_DATE,
    HFT_TICK_DIR, EVENT_CATALOG_PATH,
    FLASH_CRASH_DROP_PCT, FLASH_CRASH_WINDOW_MIN,
    FLASH_CRASH_RECOVERY_MIN, FLASH_CRASH_RECOVERY_PCT,
    FLASH_CRASH_MIN_SEPARATION_HR,
    ensure_dirs,
)


# =====================================================================
# Core detection
# =====================================================================

def load_kline_day(parquet_path: Path) -> pd.DataFrame:
    """Load a daily tick parquet and extract 1-min OHLCV-like data."""
    df = pd.read_parquet(parquet_path)
    if df.empty:
        return df

    # The tick parquets from script 00 have: timestamp_ms, price, amount, ...
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    elif "timestamp_ms" in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp_ms"], unit="ms")
    else:
        return pd.DataFrame()

    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def detect_crashes_in_series(
    times: np.ndarray,
    prices: np.ndarray,
    drop_pct: float,
    window_min: int,
    recovery_min: int,
    recovery_pct: float,
) -> list[dict]:
    """
    Detect flash crashes in a price series.

    A flash crash is defined as:
        - Price drops > drop_pct% within window_min minutes
        - Optionally: recovers recovery_pct% of the drop within recovery_min minutes

    Returns list of event dicts.
    """
    if len(prices) < 2:
        return []

    events = []
    n = len(prices)

    # Convert times to minutes from start for windowing
    t0 = times[0]
    t_min = (times - t0) / np.timedelta64(1, "m") if np.issubdtype(times.dtype, np.datetime64) \
        else (times - times[0]) / 60000.0  # ms

    for i in range(n):
        # Look forward within window_min
        window_end = t_min[i] + window_min
        mask_window = (t_min > t_min[i]) & (t_min <= window_end)
        if not mask_window.any():
            continue

        window_prices = prices[mask_window]
        min_price = window_prices.min()
        min_idx_local = np.where(mask_window)[0][np.argmin(window_prices)]

        # Compute drop
        drop = (prices[i] - min_price) / prices[i] * 100.0
        if drop < drop_pct:
            continue

        # Check recovery
        recovery_end = t_min[min_idx_local] + recovery_min
        mask_recovery = (t_min > t_min[min_idx_local]) & (t_min <= recovery_end)
        recovered_pct = 0.0
        recovery_price = min_price
        if mask_recovery.any():
            recovery_prices = prices[mask_recovery]
            recovery_price = recovery_prices.max()
            price_drop_abs = prices[i] - min_price
            if price_drop_abs > 0:
                recovered_pct = (recovery_price - min_price) / price_drop_abs * 100.0

        is_flash = recovered_pct >= recovery_pct

        events.append({
            "crash_start_time": times[i],
            "crash_bottom_time": times[min_idx_local],
            "pre_crash_price": float(prices[i]),
            "bottom_price": float(min_price),
            "recovery_price": float(recovery_price),
            "drop_pct": float(drop),
            "recovery_pct": float(recovered_pct),
            "is_flash_crash": is_flash,
            "duration_min": float(t_min[min_idx_local] - t_min[i]),
        })

    return events


def deduplicate_events(
    events: list[dict],
    min_separation_hr: float,
) -> list[dict]:
    """
    Remove overlapping events, keeping the largest drop per cluster.
    """
    if not events:
        return []

    # Sort by drop severity (largest first)
    events = sorted(events, key=lambda e: -e["drop_pct"])
    sep_td = pd.Timedelta(hours=min_separation_hr)

    kept = []
    used_times = []

    for ev in events:
        t = pd.Timestamp(ev["crash_start_time"])
        too_close = any(abs(t - ut) < sep_td for ut in used_times)
        if not too_close:
            kept.append(ev)
            used_times.append(t)

    # Re-sort chronologically
    kept.sort(key=lambda e: e["crash_start_time"])
    return kept


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Detect flash crash events from 1-min kline data"
    )
    parser.add_argument("--start", default=TARDIS_START_DATE)
    parser.add_argument("--end", default=TARDIS_END_DATE)
    parser.add_argument("--drop-pct", type=float, default=FLASH_CRASH_DROP_PCT,
                        help="Minimum drop %% to qualify as crash")
    parser.add_argument("--window", type=int, default=FLASH_CRASH_WINDOW_MIN,
                        help="Window in minutes to measure the drop")
    parser.add_argument("--recovery-min", type=int, default=FLASH_CRASH_RECOVERY_MIN)
    parser.add_argument("--recovery-pct", type=float, default=FLASH_CRASH_RECOVERY_PCT)
    parser.add_argument("--separation", type=float,
                        default=FLASH_CRASH_MIN_SEPARATION_HR,
                        help="Min hours between events (dedup)")
    args = parser.parse_args()

    ensure_dirs()

    start_dt = pd.Timestamp(args.start)
    end_dt = pd.Timestamp(args.end)

    print("Flash Crash Detection (Tier 1 - Macro)")
    print(f"  Date range    : {args.start} -> {args.end}")
    print(f"  Drop threshold: >= {args.drop_pct}% in {args.window} min")
    print(f"  Recovery check: {args.recovery_pct}% in {args.recovery_min} min")
    print(f"  Min separation: {args.separation} hours")
    print(f"  Input         : {HFT_TICK_DIR}")
    print(f"  Output        : {EVENT_CATALOG_PATH}\n")

    # Load all daily parquets
    parquet_files = sorted(HFT_TICK_DIR.glob("*.parquet"))
    parquet_files = [
        f for f in parquet_files
        if start_dt <= pd.Timestamp(f.stem) <= end_dt
    ]

    if not parquet_files:
        print("[!] No tick parquet files found. Run scripts 00 first.")
        return

    print(f"  Scanning {len(parquet_files)} daily files...\n")

    all_events = []

    for fpath in parquet_files:
        date_str = fpath.stem
        df = load_kline_day(fpath)
        if df.empty or len(df) < 10:
            continue

        times = df["datetime"].values
        prices = df["price"].values

        day_events = detect_crashes_in_series(
            times, prices,
            drop_pct=args.drop_pct,
            window_min=args.window,
            recovery_min=args.recovery_min,
            recovery_pct=args.recovery_pct,
        )

        for ev in day_events:
            ev["date"] = date_str
            all_events.append(ev)

        if day_events:
            best = max(day_events, key=lambda e: e["drop_pct"])
            tag = "FLASH" if best["is_flash_crash"] else "TREND"
            print(f"  [!] {date_str}  drop={best['drop_pct']:.1f}%  "
                  f"recovery={best['recovery_pct']:.0f}%  [{tag}]")

    # Deduplicate
    print(f"\n  Raw events found: {len(all_events)}")
    events = deduplicate_events(all_events, args.separation)
    print(f"  After dedup    : {len(events)}")

    if not events:
        print("\n[!] No flash crash events detected. Try lowering --drop-pct.")
        return

    # Save catalog
    catalog = pd.DataFrame(events)

    # Add download window columns
    catalog["download_start"] = pd.to_datetime(
        catalog["crash_start_time"]
    ) - pd.Timedelta(minutes=30)
    catalog["download_end"] = pd.to_datetime(
        catalog["crash_start_time"]
    ) + pd.Timedelta(minutes=60)
    catalog["download_date"] = pd.to_datetime(
        catalog["crash_start_time"]
    ).dt.strftime("%Y-%m-%d")

    # Sort by severity
    catalog = catalog.sort_values("drop_pct", ascending=False).reset_index(drop=True)
    catalog.index.name = "event_id"

    EVENT_CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(EVENT_CATALOG_PATH)

    print(f"\n  Event catalog saved: {EVENT_CATALOG_PATH}")
    print(f"\n  Top events:")
    print(f"  {'Date':<12} {'Drop%':>6} {'Recovery%':>10} {'Type':<6} {'Duration':>8}")
    print(f"  {'-'*46}")

    for _, row in catalog.head(20).iterrows():
        tag = "FLASH" if row["is_flash_crash"] else "TREND"
        print(f"  {row['date']:<12} {row['drop_pct']:>5.1f}% "
              f"{row['recovery_pct']:>9.0f}%  {tag:<6} "
              f"{row['duration_min']:>6.1f}m")

    # Summary statistics
    n_flash = catalog["is_flash_crash"].sum()
    n_trend = len(catalog) - n_flash
    print(f"\n  Summary: {n_flash} flash crashes, {n_trend} trend crashes")
    print(f"  Avg drop: {catalog['drop_pct'].mean():.1f}%")
    print(f"  Max drop: {catalog['drop_pct'].max():.1f}%")
    unique_dates = catalog["download_date"].nunique()
    print(f"  Unique download dates: {unique_dates}")

    print("\nDone.")


if __name__ == "__main__":
    main()
