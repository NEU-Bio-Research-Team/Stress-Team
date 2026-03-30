"""
Script 05 - Download Event Tick Data (aggTrades + bookTicker)
==============================================================
Download targeted tick-level data from Binance Vision for flash crash
events identified by Script 04.

Strategy (Event-Driven Sampling):
    1. Read event_catalog.csv from Script 04
    2. For each event date, download:
       - aggTrades: individual executed trades (ms precision)
       - bookTicker: best bid/offer updates (ms precision)
    3. Trim to event windows ([-30min, +30min] around crash)
    4. Save as parquet for micro-pipeline

Data sources (all FREE from data.binance.vision):
    aggTrades:  https://data.binance.vision/data/futures/um/daily/aggTrades/BTCUSDT/
    bookTicker: https://data.binance.vision/data/futures/um/daily/bookTicker/BTCUSDT/

Usage:
    python scripts/stage2_economics/05_download_event_ticks.py
    python scripts/stage2_economics/05_download_event_ticks.py --max-events 5
"""

import sys, os, argparse, io, zipfile, time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import (
    TARDIS_SYMBOL, EVENT_CATALOG_PATH,
    EVENT_RAW_DIR, EVENT_WINDOW_BEFORE_MIN, EVENT_WINDOW_AFTER_MIN,
    BOOKTICKER_COLS,
    ensure_dirs,
)

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


BASE_URL = "https://data.binance.vision/data/futures/um"

AGGTRADE_COLS = [
    "agg_trade_id", "price", "quantity", "first_trade_id",
    "last_trade_id", "transact_time", "is_buyer_maker",
]


# =====================================================================
# Download helpers
# =====================================================================

def _download_zip(url: str, timeout: int = 600) -> bytes | None:
    """Download a ZIP file from Binance Vision. Returns bytes or None."""
    if not HAS_REQUESTS:
        raise ImportError("pip install requests")
    try:
        r = requests.get(url, timeout=timeout, stream=True)
        if r.status_code == 200:
            return r.content
        if r.status_code != 404:
            print(f"    [!] HTTP {r.status_code}: {url}")
        return None
    except Exception as exc:
        print(f"    [!] Download error: {exc}")
        return None


def _csv_from_zip(zip_bytes: bytes, col_names: list[str] | None = None) -> pd.DataFrame | None:
    """Extract first CSV from ZIP bytes into DataFrame."""
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names:
                return None
            with zf.open(csv_names[0]) as fp:
                first_line = fp.readline().decode("utf-8", errors="replace")
                fp.seek(0)
                first_field = first_line.split(",")[0].strip()
                has_header = not first_field.replace(".", "").replace("-", "").isdigit()
                if has_header:
                    df = pd.read_csv(fp)
                elif col_names:
                    df = pd.read_csv(fp, header=None, names=col_names)
                else:
                    df = pd.read_csv(fp, header=None)
            return df
    except Exception as exc:
        print(f"    [!] ZIP parse error: {exc}")
        return None


# =====================================================================
# aggTrades download + transform
# =====================================================================

def download_aggtrades_day(symbol: str, date_str: str) -> pd.DataFrame | None:
    """
    Download one day of aggTrades from Binance Vision.

    Returns DataFrame with columns:
        timestamp_ms, price, quantity, side, agg_trade_id
    """
    url = (f"{BASE_URL}/daily/aggTrades/{symbol}/"
           f"{symbol}-aggTrades-{date_str}.zip")
    print(f"    Downloading aggTrades {date_str}...", end=" ", flush=True)

    zb = _download_zip(url)
    if zb is None:
        print("[FAIL] not found")
        return None

    df = _csv_from_zip(zb, col_names=AGGTRADE_COLS)
    if df is None or df.empty:
        print("[FAIL] empty")
        return None

    # Transform
    out = pd.DataFrame()

    # Detect columns
    if "transact_time" in df.columns:
        out["timestamp_ms"] = df["transact_time"].astype(np.int64)
    elif df.shape[1] >= 6:
        out["timestamp_ms"] = df.iloc[:, 5].astype(np.int64)

    if "price" in df.columns:
        out["price"] = df["price"].astype(float)
    else:
        out["price"] = df.iloc[:, 1].astype(float)

    if "quantity" in df.columns:
        out["quantity"] = df["quantity"].astype(float)
    else:
        out["quantity"] = df.iloc[:, 2].astype(float)

    if "is_buyer_maker" in df.columns:
        ibm = df["is_buyer_maker"]
    else:
        ibm = df.iloc[:, 6]

    ibm_bool = ibm.astype(str).str.lower().isin(["true", "1"])
    out["side"] = np.where(ibm_bool, "sell", "buy")

    if "agg_trade_id" in df.columns:
        out["agg_trade_id"] = df["agg_trade_id"].astype(np.int64)
    else:
        out["agg_trade_id"] = df.iloc[:, 0].astype(np.int64)

    # Derived
    out["dollar_volume"] = out["price"] * out["quantity"]
    out["signed_volume"] = np.where(out["side"] == "buy",
                                     out["quantity"], -out["quantity"])

    n = len(out)
    size_mb = zb.__sizeof__() / 1e6
    print(f"[OK] {n:,} trades ({size_mb:.0f} MB)")
    return out


# =====================================================================
# bookTicker download + transform
# =====================================================================

def download_bookticker_day(symbol: str, date_str: str) -> pd.DataFrame | None:
    """
    Download one day of bookTicker (BBO) from Binance Vision.

    Returns DataFrame with columns:
        timestamp_ms, best_bid_price, best_bid_qty, best_ask_price, best_ask_qty
    """
    # Try daily first
    url = (f"{BASE_URL}/daily/bookTicker/{symbol}/"
           f"{symbol}-bookTicker-{date_str}.zip")
    print(f"    Downloading bookTicker {date_str}...", end=" ", flush=True)

    zb = _download_zip(url)
    if zb is None:
        # bookTicker might not be available for futures on Binance Vision
        print("[N/A] bookTicker not available for this date")
        return None

    df = _csv_from_zip(zb, col_names=BOOKTICKER_COLS)
    if df is None or df.empty:
        print("[FAIL] empty")
        return None

    out = pd.DataFrame()

    # Detect timestamp column
    ts_col = None
    for candidate in ["transact_time", "transaction_time", "time", "event_time"]:
        if candidate in df.columns:
            ts_col = candidate
            break

    if ts_col:
        out["timestamp_ms"] = df[ts_col].astype(np.int64)
    elif df.shape[1] >= 5:
        # Positional: last column is usually timestamp
        out["timestamp_ms"] = df.iloc[:, -1].astype(np.int64)
    else:
        print("[FAIL] no timestamp column")
        return None

    # Price/qty columns
    for col_name, alts in [
        ("best_bid_price", ["best_bid_price", "bidPrice", "bid_price"]),
        ("best_bid_qty", ["best_bid_qty", "bidQty", "bid_qty"]),
        ("best_ask_price", ["best_ask_price", "askPrice", "ask_price"]),
        ("best_ask_qty", ["best_ask_qty", "askQty", "ask_qty"]),
    ]:
        found = False
        for alt in alts:
            if alt in df.columns:
                out[col_name] = df[alt].astype(float)
                found = True
                break
        if not found:
            # Positional fallback
            idx_map = {"best_bid_price": 0, "best_bid_qty": 1,
                       "best_ask_price": 2, "best_ask_qty": 3}
            if df.shape[1] > idx_map[col_name]:
                out[col_name] = df.iloc[:, idx_map[col_name]].astype(float)

    # Derived
    if "best_bid_price" in out.columns and "best_ask_price" in out.columns:
        out["mid_price"] = (out["best_bid_price"] + out["best_ask_price"]) / 2.0
        out["spread"] = out["best_ask_price"] - out["best_bid_price"]
        out["spread_bps"] = out["spread"] / out["mid_price"] * 10000.0

    if "best_bid_qty" in out.columns and "best_ask_qty" in out.columns:
        out["touch_depth"] = out["best_bid_qty"] + out["best_ask_qty"]
        total = out["best_bid_qty"] + out["best_ask_qty"]
        out["depth_imbalance"] = np.where(
            total > 0,
            (out["best_bid_qty"] - out["best_ask_qty"]) / total,
            0.0,
        )

    n = len(out)
    print(f"[OK] {n:,} updates")
    return out


# =====================================================================
# Event window trimming
# =====================================================================

def trim_to_window(
    df: pd.DataFrame,
    crash_time_ms: int,
    before_min: int,
    after_min: int,
) -> pd.DataFrame:
    """Trim dataframe to [crash_time - before, crash_time + after]."""
    start_ms = crash_time_ms - before_min * 60 * 1000
    end_ms = crash_time_ms + after_min * 60 * 1000
    mask = (df["timestamp_ms"] >= start_ms) & (df["timestamp_ms"] <= end_ms)
    return df[mask].copy().reset_index(drop=True)


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download tick data for flash crash events"
    )
    parser.add_argument("--catalog", default=str(EVENT_CATALOG_PATH),
                        help="Path to event_catalog.csv from script 04")
    parser.add_argument("--max-events", type=int, default=None,
                        help="Limit number of events to download")
    parser.add_argument("--symbol", default=TARDIS_SYMBOL)
    parser.add_argument("--before-min", type=int, default=EVENT_WINDOW_BEFORE_MIN)
    parser.add_argument("--after-min", type=int, default=EVENT_WINDOW_AFTER_MIN)
    parser.add_argument("--skip-bookticker", action="store_true",
                        help="Skip bookTicker download (use aggTrades only)")
    args = parser.parse_args()

    ensure_dirs()

    # Load event catalog
    catalog_path = Path(args.catalog)
    if not catalog_path.exists():
        print(f"[!] Event catalog not found: {catalog_path}")
        print("    Run script 04 first.")
        return

    catalog = pd.read_csv(catalog_path)
    print("Event-Driven Tick Download (Tier 2)")
    print(f"  Catalog    : {catalog_path}")
    print(f"  Events     : {len(catalog)}")
    print(f"  Symbol     : {args.symbol}")
    print(f"  Window     : [-{args.before_min}min, +{args.after_min}min]")
    print(f"  Output     : {EVENT_RAW_DIR}\n")

    if args.max_events:
        catalog = catalog.head(args.max_events)
        print(f"  (limited to {args.max_events} events)\n")

    # Deduplicate download dates
    dates_needed = sorted(catalog["download_date"].unique())
    print(f"  Unique dates to download: {len(dates_needed)}\n")

    # Track results
    results = []
    bookticker_available = True

    for i, (_, event) in enumerate(catalog.iterrows()):
        event_id = i
        date_str = str(event["download_date"])
        crash_time = pd.Timestamp(event["crash_start_time"])
        crash_time_ms = int(crash_time.timestamp() * 1000)

        print(f"  Event {event_id}: {date_str} "
              f"(drop={event['drop_pct']:.1f}%)")

        out_dir = EVENT_RAW_DIR / f"event_{event_id:03d}_{date_str}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # -- aggTrades --
        agg_path = out_dir / "aggtrades.parquet"
        agg_ok = False
        if agg_path.exists():
            print(f"    aggTrades: already exists, skipping")
            agg_ok = True
        else:
            agg_df = download_aggtrades_day(args.symbol, date_str)
            if agg_df is not None and not agg_df.empty:
                # Trim to event window
                agg_trimmed = trim_to_window(
                    agg_df, crash_time_ms, args.before_min, args.after_min
                )
                if not agg_trimmed.empty:
                    agg_trimmed.to_parquet(agg_path, index=False, engine="pyarrow")
                    print(f"    -> Saved {len(agg_trimmed):,} trades "
                          f"(trimmed from {len(agg_df):,})")
                    agg_ok = True
                else:
                    # Save full day if trim produced nothing
                    agg_df.to_parquet(agg_path, index=False, engine="pyarrow")
                    print(f"    -> Saved {len(agg_df):,} trades (full day, trim empty)")
                    agg_ok = True

        # -- bookTicker --
        bkt_path = out_dir / "bookticker.parquet"
        bkt_ok = False
        if args.skip_bookticker:
            bkt_ok = False
        elif bkt_path.exists():
            print(f"    bookTicker: already exists, skipping")
            bkt_ok = True
        elif bookticker_available:
            bkt_df = download_bookticker_day(args.symbol, date_str)
            if bkt_df is not None and not bkt_df.empty:
                bkt_trimmed = trim_to_window(
                    bkt_df, crash_time_ms, args.before_min, args.after_min
                )
                if not bkt_trimmed.empty:
                    bkt_trimmed.to_parquet(bkt_path, index=False, engine="pyarrow")
                    print(f"    -> Saved {len(bkt_trimmed):,} BBO updates "
                          f"(trimmed from {len(bkt_df):,})")
                    bkt_ok = True
                else:
                    bkt_df.to_parquet(bkt_path, index=False, engine="pyarrow")
                    print(f"    -> Saved {len(bkt_df):,} BBO updates (full day)")
                    bkt_ok = True
            else:
                print("    [!] bookTicker not available on Binance Vision "
                      "for Futures. Will use aggTrades only.")
                bookticker_available = False

        # Save event metadata
        meta = {
            "event_id": event_id,
            "date": date_str,
            "crash_start_time": str(event["crash_start_time"]),
            "drop_pct": event["drop_pct"],
            "is_flash_crash": event.get("is_flash_crash", False),
            "has_aggtrades": agg_ok,
            "has_bookticker": bkt_ok,
        }
        meta_df = pd.DataFrame([meta])
        meta_df.to_csv(out_dir / "event_meta.csv", index=False)

        results.append(meta)
        print()

        # Polite delay
        time.sleep(0.5)

    # Summary
    results_df = pd.DataFrame(results)
    n_agg = results_df["has_aggtrades"].sum() if "has_aggtrades" in results_df else 0
    n_bkt = results_df["has_bookticker"].sum() if "has_bookticker" in results_df else 0

    print(f"\nSummary:")
    print(f"  Events processed : {len(results)}")
    print(f"  aggTrades OK     : {n_agg}")
    print(f"  bookTicker OK    : {n_bkt}")
    if not bookticker_available:
        print(f"  [NOTE] bookTicker not available on Binance Vision for Futures.")
        print(f"         Micro-pipeline will compute implied spread from aggTrades.")
    print(f"\nDone. Data saved to {EVENT_RAW_DIR}")


if __name__ == "__main__":
    main()
