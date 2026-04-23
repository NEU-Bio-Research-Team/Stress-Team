"""
Script 10 – Augment & Regular-Grid Reindex Event_Dynamics_100ms.csv
====================================================================
NOTEARS and LiNGAM require regularly-spaced observations.  The current
Event_Dynamics_100ms.csv can have gaps when no trade lands in a 100 ms
bin.  This script:

    1. Re-indexes each event onto a strict 100 ms grid (np.arange).
    2. Forward-fills book-state columns (spread, depth, mid_price)
       because LOB state persists between updates.
    3. Zero-fills trade-flow columns (ofi, trade_intensity, volume)
       because absence of trades = zero flow.
    4. Interpolates slow-moving derived columns (kyle_lambda, vpin,
       realized_vol, leverage_proxy, order_flow_toxicity).
    5. Saves the augmented, grid-aligned file as
       Event_Dynamics_100ms_gridded.csv.

Usage:
    python scripts/stage2_economics/10_augment_dynamics_features.py
"""

import sys, os
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import PROCESSED_DIR

# ── Paths ─────────────────────────────────────────────────────────────
INPUT_CSV  = PROCESSED_DIR / "tardis" / "confounder_outputs" / "Event_Dynamics_100ms.csv"
OUTPUT_CSV = PROCESSED_DIR / "tardis" / "confounder_outputs" / "Event_Dynamics_100ms_gridded.csv"

# Column classification for fill strategy
BOOK_STATE_COLS = [
    "close", "mid_price", "spread_bps", "depth_imbalance", "touch_depth",
]
TRADE_FLOW_COLS = [
    "ofi", "trade_intensity",
]
DERIVED_COLS = [
    "kyle_lambda", "vpin", "amihud_illiq", "realized_vol_50",
    "leverage_proxy", "order_flow_toxicity",
    "velocity_pct_per_100ms", "drop_velocity_pct_per_100ms",
    "panic_acceleration_pct_per_100ms2", "drop_1s_pct", "drop_from_local_pct",
]
STEP_MS = 100


def reindex_event(edf: pd.DataFrame) -> pd.DataFrame:
    """Re-index a single event onto a regular 100 ms grid."""
    edf = edf.sort_values("timestamp_ms").copy()
    t_min = int(edf["timestamp_ms"].iloc[0])
    t_max = int(edf["timestamp_ms"].iloc[-1])

    grid = np.arange(t_min, t_max + STEP_MS, STEP_MS)

    # Set index and reindex
    edf = edf.drop_duplicates(subset="timestamp_ms", keep="last")
    edf = edf.set_index("timestamp_ms").reindex(grid)
    edf.index.name = "timestamp_ms"

    # Fill strategies
    for col in BOOK_STATE_COLS:
        if col in edf.columns:
            edf[col] = edf[col].ffill().bfill()

    for col in TRADE_FLOW_COLS:
        if col in edf.columns:
            edf[col] = edf[col].fillna(0.0)

    for col in DERIVED_COLS:
        if col in edf.columns:
            edf[col] = edf[col].interpolate(method="linear", limit_direction="both")
            edf[col] = edf[col].ffill().bfill()

    # Metadata columns
    if "event_id" in edf.columns:
        edf["event_id"] = edf["event_id"].ffill().bfill().astype(int)
    if "date" in edf.columns:
        edf["date"] = edf["date"].ffill().bfill()
    if "phase" in edf.columns:
        edf["phase"] = edf["phase"].ffill().bfill()
    if "timestamp_utc" in edf.columns:
        edf["timestamp_utc"] = pd.to_datetime(edf.index, unit="ms", utc=True).tz_localize(None)
    if "time_from_drop_start_ms" in edf.columns:
        edf["time_from_drop_start_ms"] = edf["time_from_drop_start_ms"].interpolate(method="linear")
    if "delta_from_news_ms" in edf.columns:
        edf["delta_from_news_ms"] = edf["delta_from_news_ms"].interpolate(method="linear")

    edf = edf.reset_index()
    return edf


def main():
    print("=" * 70)
    print("Script 10: Augment & Regular-Grid Reindex")
    print("=" * 70)

    if not INPUT_CSV.exists():
        print(f"[!] Input not found: {INPUT_CSV}")
        print("    Run script 09 first.")
        return

    df = pd.read_csv(INPUT_CSV)
    print(f"  Loaded {len(df)} rows, {df['event_id'].nunique()} events")

    original_gaps = 0
    total_after = 0
    gridded_events = []

    for eid, edf in df.groupby("event_id"):
        n_before = len(edf)
        gridded = reindex_event(edf)
        n_after = len(gridded)
        gaps = n_after - n_before
        original_gaps += gaps
        total_after += n_after
        gridded_events.append(gridded)
        if gaps > 0:
            print(f"  Event {int(eid):03d}: {n_before} → {n_after} rows (+{gaps} filled)")

    result = pd.concat(gridded_events, ignore_index=True)

    # Verify regularity
    for eid, edf in result.groupby("event_id"):
        diffs = edf["timestamp_ms"].diff().dropna().unique()
        if len(diffs) != 1 or diffs[0] != STEP_MS:
            print(f"  [WARN] Event {int(eid):03d}: irregular grid detected "
                  f"(unique diffs: {sorted(diffs)[:5]})")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  Total: {len(df)} → {total_after} rows ({original_gaps} gap-fills)")
    print(f"  Saved: {OUTPUT_CSV}")
    print("Done.")


if __name__ == "__main__":
    main()
