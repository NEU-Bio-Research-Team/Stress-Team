"""
Script 00 – Reindex Raw Tick Trades to Millisecond Parquet
=============================================================
Parse raw Tardis .csv.gz trade files and produce per-day parquet files
with millisecond-resolution timestamps and enriched tick-level fields.

Output per day (→ data/processed/tardis/ticks/<date>.parquet):
    timestamp_ms   : int64   – epoch milliseconds
    datetime       : datetime64[ms]
    price          : float64
    amount         : float64
    side           : str     – "buy" / "sell"
    dollar_volume  : float64 – price × amount
    signed_volume  : float64 – +amount if buy, -amount if sell
    price_change   : float64 – tick-to-tick Δprice
    inter_arrival_ms : float64 – time gap to previous trade (ms)
    is_gap         : bool    – True if inter_arrival_ms > HFT_GAP_THRESHOLD_MS
    is_duplicate   : bool    – True if inter_arrival_ms == 0 AND same price

Usage:
    python scripts/stage2_economics/00_reindex_ticks.py
    python scripts/stage2_economics/00_reindex_ticks.py --start 2021-01-01 --end 2021-12-31
"""

import sys, os, argparse, gzip, re
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import (
    TARDIS_RAW_DIR, TARDIS_START_DATE, TARDIS_END_DATE,
    TARDIS_MAY19_START, TARDIS_MAY19_END,
    HFT_TICK_DIR, HFT_GAP_THRESHOLD_MS, HFT_DUPLICATE_WINDOW_MS,
    ensure_dirs,
)


def _list_trade_files(root: Path = TARDIS_RAW_DIR) -> list[Path]:
    """Return sorted list of raw trade .csv.gz files."""
    trades_dir = root / "trades"
    if not trades_dir.exists():
        return []
    return sorted(trades_dir.glob("*.csv.gz"))


def _extract_date(filepath: Path) -> str | None:
    m = re.search(r"(\d{4}-\d{2}-\d{2})", filepath.name)
    return m.group(1) if m else None


def _in_may19_window(dt_series: pd.Series) -> pd.Series:
    """Boolean mask for the May-19 2021 crash exclusion window."""
    start = pd.Timestamp(TARDIS_MAY19_START).tz_localize(None)
    end = pd.Timestamp(TARDIS_MAY19_END).tz_localize(None)
    return (dt_series >= start) & (dt_series <= end)


def reindex_day(csv_gz_path: Path, output_dir: Path) -> dict:
    """
    Process one day's raw trades → ms-resolution parquet.

    Returns summary dict with stats for logging.
    """
    date_str = _extract_date(csv_gz_path)
    if date_str is None:
        return {"file": str(csv_gz_path), "status": "skip_no_date"}

    out_path = output_dir / f"{date_str}.parquet"
    if out_path.exists():
        return {"date": date_str, "status": "exists"}

    # ── Load raw trades ─────────────────────────────────────────────
    try:
        with gzip.open(csv_gz_path, "rt") as f:
            df = pd.read_csv(f)
    except Exception as e:
        return {"date": date_str, "status": f"error_load: {e}"}

    if df.empty or "timestamp" not in df.columns:
        return {"date": date_str, "status": "empty"}

    # ── Parse timestamps (raw unit = microseconds) ──────────────────
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="us")
    df["timestamp_ms"] = df["timestamp"] // 1000  # μs → ms

    # ── Drop May-19 crash window if present ─────────────────────────
    may19_mask = _in_may19_window(df["datetime"])
    n_may19 = may19_mask.sum()
    df = df[~may19_mask].copy()

    if df.empty:
        return {"date": date_str, "status": "empty_after_may19"}

    # ── Sort by timestamp (should already be, but ensure) ───────────
    df = df.sort_values("timestamp_ms").reset_index(drop=True)

    # ── Numeric columns ─────────────────────────────────────────────
    df["price"] = df["price"].astype(float)
    df["amount"] = df["amount"].astype(float) if "amount" in df.columns else 1.0
    df["side"] = df["side"].str.lower() if "side" in df.columns else "unknown"

    # ── Derived tick-level features ─────────────────────────────────
    df["dollar_volume"] = df["price"] * df["amount"]
    df["signed_volume"] = np.where(df["side"] == "buy", df["amount"], -df["amount"])
    df["price_change"] = df["price"].diff().fillna(0.0)

    # ── Inter-arrival time ──────────────────────────────────────────
    df["inter_arrival_ms"] = df["timestamp_ms"].diff().fillna(0.0)

    # ── Gap / duplicate flags ───────────────────────────────────────
    df["is_gap"] = df["inter_arrival_ms"] > HFT_GAP_THRESHOLD_MS
    df["is_duplicate"] = (
        (df["inter_arrival_ms"] <= HFT_DUPLICATE_WINDOW_MS)
        & (df["price_change"] == 0)
        & (df.index > 0)
    )

    # ── Select output columns ───────────────────────────────────────
    out_cols = [
        "timestamp_ms", "datetime", "price", "amount", "side",
        "dollar_volume", "signed_volume", "price_change",
        "inter_arrival_ms", "is_gap", "is_duplicate",
    ]
    out_df = df[out_cols].copy()

    # ── Write parquet ───────────────────────────────────────────────
    out_df.to_parquet(out_path, index=False, engine="pyarrow")

    return {
        "date": date_str,
        "status": "ok",
        "n_trades": len(out_df),
        "n_gaps": int(out_df["is_gap"].sum()),
        "n_duplicates": int(out_df["is_duplicate"].sum()),
        "n_may19_dropped": int(n_may19),
        "price_range": (float(out_df["price"].min()), float(out_df["price"].max())),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Reindex raw Tardis tick trades → ms-resolution parquet"
    )
    parser.add_argument("--start", default=TARDIS_START_DATE,
                        help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=TARDIS_END_DATE,
                        help="End date YYYY-MM-DD")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-process days that already have parquet output")
    args = parser.parse_args()

    ensure_dirs()

    start_dt = pd.Timestamp(args.start)
    end_dt = pd.Timestamp(args.end)

    files = _list_trade_files()
    if not files:
        print("[!] No trade files found in", TARDIS_RAW_DIR / "trades")
        return

    print(f"Tick reindex: {args.start} -> {args.end}  ({len(files)} files found)")
    print(f"Output: {HFT_TICK_DIR}\n")

    ok, skip, err = 0, 0, 0
    for fpath in files:
        date_str = _extract_date(fpath)
        if date_str is None:
            continue
        fdate = pd.Timestamp(date_str)
        if fdate < start_dt or fdate > end_dt:
            continue

        # Skip existing unless --overwrite
        if not args.overwrite and (HFT_TICK_DIR / f"{date_str}.parquet").exists():
            skip += 1
            continue

        result = reindex_day(fpath, HFT_TICK_DIR)
        status = result.get("status", "unknown")
        if status == "ok":
            ok += 1
            n = result["n_trades"]
            g = result["n_gaps"]
            print(f"  [OK] {date_str}  trades={n:>9,}  gaps={g}")
        elif status == "exists":
            skip += 1
        else:
            err += 1
            print(f"  [FAIL] {date_str}  {status}")

    print(f"\nDone: {ok} processed, {skip} skipped, {err} errors")


if __name__ == "__main__":
    main()
