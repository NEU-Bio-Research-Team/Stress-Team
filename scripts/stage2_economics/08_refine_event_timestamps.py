"""
Script 08 - Refine Event Timestamps To Tick Level
=================================================
Use downloaded aggTrades windows to refine the 1-minute event timestamps
from event_catalog.csv down to millisecond timestamps.

Outputs:
    data/processed/tardis/event_catalog_tick_refined.csv

Method:
    - Start from the 1-minute event catalog produced by Script 04.
    - For each event, load aggTrades from Script 05.
    - Search around the kline crash window to find:
        * tick_bottom_time: exact trade timestamp of minimum price
        * tick_start_time: last local maximum before that bottom
    - Save both the original kline timestamps and refined tick timestamps.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import EVENT_CATALOG_PATH, EVENT_RAW_DIR, PROCESSED_DIR, ensure_dirs


OUTPUT_PATH = PROCESSED_DIR / "tardis" / "event_catalog_tick_refined.csv"


def refine_event(row: pd.Series, event_dir: Path, lookback_seconds: int) -> dict:
    agg_path = event_dir / "aggtrades.parquet"
    result = {
        "event_id": int(row["event_id"]),
        "date": row["date"],
        "is_flash_crash": bool(row["is_flash_crash"]),
        "drop_pct": float(row["drop_pct"]),
        "recovery_pct": float(row["recovery_pct"]),
        "kline_crash_start_time": row["crash_start_time"],
        "kline_crash_bottom_time": row["crash_bottom_time"],
        "tick_start_time": pd.NaT,
        "tick_bottom_time": pd.NaT,
        "tick_start_price": np.nan,
        "tick_bottom_price": np.nan,
        "tick_drop_pct": np.nan,
        "tick_duration_ms": np.nan,
        "n_trades_window": 0,
    }

    if not agg_path.exists():
        result["status"] = "missing_aggtrades"
        return result

    trades = pd.read_parquet(agg_path)
    if trades.empty:
        result["status"] = "empty_aggtrades"
        return result

    trades = trades.sort_values("timestamp_ms").copy()
    trades["timestamp"] = pd.to_datetime(trades["timestamp_ms"], unit="ms", utc=True).dt.tz_localize(None)

    kline_start = pd.to_datetime(row["crash_start_time"])
    kline_bottom = pd.to_datetime(row["crash_bottom_time"])

    search_start = kline_start - pd.Timedelta(seconds=lookback_seconds)
    search_end = kline_bottom + pd.Timedelta(minutes=1)
    window = trades[(trades["timestamp"] >= search_start) & (trades["timestamp"] <= search_end)].copy()

    if window.empty:
        result["status"] = "empty_search_window"
        return result

    result["n_trades_window"] = int(len(window))

    bottom_pos = int(window["price"].values.argmin())
    bottom_row = window.iloc[bottom_pos]
    pre_bottom = window.iloc[: bottom_pos + 1]

    if pre_bottom.empty:
        result["status"] = "no_pre_bottom_window"
        return result

    start_pos = int(pre_bottom["price"].values.argmax())
    start_row = pre_bottom.iloc[start_pos]

    start_price = float(start_row["price"])
    bottom_price = float(bottom_row["price"])
    tick_drop_pct = ((start_price - bottom_price) / start_price * 100.0) if start_price > 0 else np.nan

    result.update(
        {
            "tick_start_time": start_row["timestamp"],
            "tick_bottom_time": bottom_row["timestamp"],
            "tick_start_price": start_price,
            "tick_bottom_price": bottom_price,
            "tick_drop_pct": tick_drop_pct,
            "tick_duration_ms": int(bottom_row["timestamp_ms"] - start_row["timestamp_ms"]),
            "status": "ok",
        }
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Refine crash timestamps from 1-minute to tick level")
    parser.add_argument("--catalog", type=Path, default=EVENT_CATALOG_PATH)
    parser.add_argument("--lookback-seconds", type=int, default=60,
                        help="How many seconds before kline_crash_start to search for the local peak")
    args = parser.parse_args()

    ensure_dirs()

    if not args.catalog.exists():
        raise FileNotFoundError(f"Event catalog not found: {args.catalog}")

    catalog = pd.read_csv(args.catalog)
    if catalog.empty:
        raise ValueError("Event catalog is empty")

    rows = []
    print("Tick Timestamp Refinement")
    print(f"  Catalog : {args.catalog}")
    print(f"  Events  : {len(catalog)}")
    print(f"  Output  : {OUTPUT_PATH}\n")

    for _, row in catalog.iterrows():
        event_id = int(row["event_id"])
        event_dir = EVENT_RAW_DIR / f"event_{event_id:03d}_{row['date']}"
        refined = refine_event(row, event_dir, args.lookback_seconds)
        rows.append(refined)
        print(
            f"  [{event_id:03d}] {row['date']}  status={refined['status']}  "
            f"tick_start={refined['tick_start_time']}  tick_bottom={refined['tick_bottom_time']}"
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_PATH, index=False)

    ok_count = int((out_df["status"] == "ok").sum())
    print(f"\nDone. Refined {ok_count}/{len(out_df)} events")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()