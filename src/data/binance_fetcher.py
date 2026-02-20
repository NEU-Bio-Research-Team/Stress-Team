"""
Binance Vision Data Fetcher (FREE – no API key required)
=========================================================
Downloads historical Binance USDT-M Futures data from data.binance.vision,
Binance's official public data repository.

    - No API key
    - No rate limits (be polite though)
    - No cost

Data types:
    - klines  (1m OHLCV): lightweight (~1 MB/month zipped), sufficient for
      returns, volatility, stylized facts, ABM calibration targets.
    - aggTrades: tick-level aggregated trades (large, 50-200 MB/day).
    - liquidationSnapshot: forced liquidation events.

Output is saved in the same directory structure and column format as the
Tardis pipeline (data/raw/tardis/{data_type}/), so every downstream script
works identically regardless of data source.

References:
    https://data.binance.vision/
    https://github.com/binance/binance-public-data
"""

import io
import gzip
import time
import calendar
import zipfile
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timedelta

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    TARDIS_RAW_DIR, TARDIS_SYMBOL,
    TARDIS_START_DATE, TARDIS_END_DATE,
)

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# ── Binance Vision base URL ─────────────────────────────────────────
BASE_URL = "https://data.binance.vision/data/futures/um"

# ── Known column layouts (Binance CSVs may or may not have headers) ──
KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "count",
    "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]
AGGTRADE_COLS = [
    "agg_trade_id", "price", "quantity", "first_trade_id",
    "last_trade_id", "transact_time", "is_buyer_maker",
]
LIQUIDATION_COLS = [
    "symbol", "side", "order_type", "time_in_force",
    "original_quantity", "price", "average_price",
    "order_status", "time", "filled_quantity",
]


# ────────────────────────── helpers ──────────────────────────────────

def _month_range(start: str, end: str) -> List[str]:
    """Generate YYYY-MM strings covering *start* to *end*."""
    s = datetime.strptime(start[:7], "%Y-%m")
    e = datetime.strptime(end[:7], "%Y-%m")
    months: List[str] = []
    while s <= e:
        months.append(s.strftime("%Y-%m"))
        s = (s.replace(day=28) + timedelta(days=4)).replace(day=1)
    return months


def _date_range(start: str, end: str) -> List[str]:
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    dates: List[str] = []
    while s <= e:
        dates.append(s.strftime("%Y-%m-%d"))
        s += timedelta(days=1)
    return dates


def _month_bounds(month: str, start: str, end: str):
    """Clip a YYYY-MM month to [start, end] and return (first_day, last_day)."""
    y, m = int(month[:4]), int(month[5:])
    n_days = calendar.monthrange(y, m)[1]
    first = f"{month}-01"
    last = f"{month}-{n_days:02d}"
    return max(first, start), min(last, end)


def _download_zip(url: str, timeout: int = 300) -> Optional[bytes]:
    """GET a ZIP file; return raw bytes or None on 404 / error."""
    if not HAS_REQUESTS:
        raise ImportError("Install requests:  pip install requests")
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            return r.content
        if r.status_code != 404:
            print(f"  [!] HTTP {r.status_code}: {url}")
        return None
    except Exception as exc:
        print(f"  [!] {exc}")
        return None


def _csv_from_zip(
    zip_bytes: bytes,
    col_names: Optional[List[str]] = None,
) -> Optional[pd.DataFrame]:
    """Extract the first CSV inside a ZIP and return a DataFrame."""
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names:
                return None
            with zf.open(csv_names[0]) as fp:
                # Peek first line to decide if headers exist
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
        print(f"  [!] ZIP read error: {exc}")
        return None


def _out_path(directory: Path, exchange: str, dtype: str,
              date_str: str, symbol: str) -> Path:
    return directory / f"{exchange}_{dtype}_{date_str}_{symbol}.csv.gz"


# ────────────────── column transforms ────────────────────────────────

def _transform_klines(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Binance 1m klines → Tardis-compatible trade-like CSV.

    Saved columns (superset – downstream auto-detects format):
        timestamp  – open_time in µs
        price      – close price  (compat with trades pipeline)
        amount     – bar volume   (compat with trades pipeline)
        side       – "buy" placeholder
        open, high, low, close, volume
        n_trades, buy_volume, sell_volume, order_flow
    """
    out = pd.DataFrame()

    # Positional access in case column names differ between downloads
    ot = df.iloc[:, 0].astype(np.int64)                # open_time ms
    out["timestamp"]  = ot * 1000                       # → µs
    out["open"]       = df.iloc[:, 1].astype(float)
    out["high"]       = df.iloc[:, 2].astype(float)
    out["low"]        = df.iloc[:, 3].astype(float)
    out["close"]      = df.iloc[:, 4].astype(float)
    out["volume"]     = df.iloc[:, 5].astype(float)
    out["n_trades"]   = df.iloc[:, 8].astype(int)
    out["buy_volume"] = df.iloc[:, 9].astype(float)       # taker buy vol

    out["sell_volume"] = out["volume"] - out["buy_volume"]
    out["order_flow"]  = out["buy_volume"] - out["sell_volume"]

    # Backward-compat aliases for the trades pipeline
    out["price"]  = out["close"]
    out["amount"] = out["volume"]
    out["side"]   = "buy"           # not meaningful for klines

    return out


def _transform_aggtrades(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Binance aggTrades → Tardis trade format."""
    out = pd.DataFrame()

    # Detect column layout
    if "transact_time" in df.columns:
        ts_col, px_col, qty_col, ibm_col = (
            "transact_time", "price", "quantity", "is_buyer_maker")
    elif "time" in df.columns:
        ts_col, px_col, qty_col, ibm_col = "time", "price", "qty", "isBuyerMaker"
    else:
        # Positional fallback (headerless CSV)
        ts_col, px_col, qty_col, ibm_col = 5, 1, 2, 6

    out["timestamp"] = df.iloc[:, ts_col] * 1000 if isinstance(ts_col, int) \
        else df[ts_col].astype(np.int64) * 1000          # ms → µs
    out["price"]  = (df.iloc[:, px_col] if isinstance(px_col, int)
                     else df[px_col]).astype(float)
    out["amount"] = (df.iloc[:, qty_col] if isinstance(qty_col, int)
                     else df[qty_col]).astype(float)

    ibm = df.iloc[:, ibm_col] if isinstance(ibm_col, int) else df[ibm_col]
    ibm_bool = ibm.astype(str).str.lower().isin(["true", "1"])
    # isBuyerMaker=True → taker was SELLER → side="sell"
    out["side"] = np.where(ibm_bool, "sell", "buy")

    return out


def _transform_liquidations(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()

    ts_col = next((c for c in ["time", "transact_time"] if c in df.columns), None)
    if ts_col:
        out["timestamp"] = df[ts_col].astype(np.int64) * 1000
    else:
        out["timestamp"] = df.iloc[:, 8].astype(np.int64) * 1000  # positional

    px_col = "price" if "price" in df.columns else "average_price"
    out["price"] = df[px_col].astype(float) if px_col in df.columns \
        else df.iloc[:, 5].astype(float)

    qty_col = next((c for c in ["original_quantity", "filled_quantity",
                                "quantity", "qty"] if c in df.columns), None)
    out["quantity"] = df[qty_col].astype(float) if qty_col else df.iloc[:, 4].astype(float)
    out["amount"]   = out["quantity"]

    if "side" in df.columns:
        out["side"] = df["side"].str.lower()
    else:
        out["side"] = df.iloc[:, 1].str.lower()

    return out


# ══════════════════════ public download API ══════════════════════════

def fetch_klines(
    symbol: str = TARDIS_SYMBOL,
    interval: str = "1m",
    start_date: str = TARDIS_START_DATE,
    end_date: str = TARDIS_END_DATE,
    output_dir: Optional[Path] = None,
    exchange_label: str = "binance-futures",
) -> int:
    """
    Download 1-minute klines from data.binance.vision.

    Lightweight: ~1 MB / month zipped → ~150 MB total for 4+ years.
    Sufficient for returns, volatility, stylized facts, ABM calibration.
    """
    output_dir = output_dir or (TARDIS_RAW_DIR / "trades")
    output_dir.mkdir(parents=True, exist_ok=True)

    months = _month_range(start_date, end_date)
    print(f"[binance-vision] Downloading {interval} klines for {symbol}")
    print(f"  Range : {start_date} → {end_date}  ({len(months)} months)")
    print(f"  Output: {output_dir}\n")

    saved = 0

    for month in months:
        m_start, m_end = _month_bounds(month, start_date, end_date)
        days = _date_range(m_start, m_end)

        # Skip if every daily file already exists
        existing = [d for d in days
                    if _out_path(output_dir, exchange_label, "trades",
                                 d, symbol).exists()]
        if len(existing) == len(days):
            print(f"  {month}: ✓ already complete ({len(days)} files)")
            saved += len(days)
            continue

        # ── try monthly ZIP first
        url = (f"{BASE_URL}/monthly/klines/{symbol}/{interval}/"
               f"{symbol}-{interval}-{month}.zip")
        print(f"  {month}: ", end="", flush=True)
        zb = _download_zip(url)

        if zb is not None:
            df = _csv_from_zip(zb, col_names=KLINE_COLS)
            if df is not None and not df.empty:
                transformed = _transform_klines(df)
                transformed["_date"] = pd.to_datetime(
                    transformed["timestamp"], unit="us"
                ).dt.strftime("%Y-%m-%d")

                for d in days:
                    out = _out_path(output_dir, exchange_label, "trades",
                                    d, symbol)
                    if out.exists():
                        saved += 1
                        continue
                    day_df = transformed[transformed["_date"] == d].drop(
                        columns=["_date"])
                    if day_df.empty:
                        continue
                    day_df.to_csv(out, index=False, compression="gzip")
                    saved += 1
                print(f"✓ monthly ({len(days)} days)")
                continue

        # ── fallback: daily ZIPs
        print("monthly N/A → daily ", end="", flush=True)
        day_ok = 0
        for d in days:
            out = _out_path(output_dir, exchange_label, "trades", d, symbol)
            if out.exists():
                saved += 1
                day_ok += 1
                continue
            durl = (f"{BASE_URL}/daily/klines/{symbol}/{interval}/"
                    f"{symbol}-{interval}-{d}.zip")
            db = _download_zip(durl)
            if db is not None:
                ddf = _csv_from_zip(db, col_names=KLINE_COLS)
                if ddf is not None and not ddf.empty:
                    day_t = _transform_klines(ddf)
                    day_t.to_csv(out, index=False, compression="gzip")
                    saved += 1
                    day_ok += 1
            time.sleep(0.05)
        print(f"✓ ({day_ok}/{len(days)} days)")

    print(f"\n[binance-vision] Klines done – {saved} day-files in {output_dir}")
    return saved


def fetch_aggtrades(
    symbol: str = TARDIS_SYMBOL,
    start_date: str = TARDIS_START_DATE,
    end_date: str = TARDIS_END_DATE,
    output_dir: Optional[Path] = None,
    exchange_label: str = "binance-futures",
) -> int:
    """
    Download aggregated trades from data.binance.vision.

    WARNING: Files are large (50-200 MB/day compressed).
    Full 4-year range may require 100+ GB of disk.
    Use fetch_klines() for a lightweight alternative.
    """
    output_dir = output_dir or (TARDIS_RAW_DIR / "trades")
    output_dir.mkdir(parents=True, exist_ok=True)

    months = _month_range(start_date, end_date)
    print(f"[binance-vision] Downloading aggTrades for {symbol}")
    print(f"  Range : {start_date} → {end_date}  ({len(months)} months)")
    print(f"  ⚠  aggTrades are LARGE.  Use --mode klines for a lighter option.\n")

    saved = 0

    for month in months:
        m_start, m_end = _month_bounds(month, start_date, end_date)
        days = _date_range(m_start, m_end)

        existing = [d for d in days
                    if _out_path(output_dir, exchange_label, "trades",
                                 d, symbol).exists()]
        if len(existing) == len(days):
            print(f"  {month}: ✓ complete")
            saved += len(days)
            continue

        url = (f"{BASE_URL}/monthly/aggTrades/{symbol}/"
               f"{symbol}-aggTrades-{month}.zip")
        print(f"  {month}: ", end="", flush=True)
        zb = _download_zip(url)

        if zb is not None:
            df = _csv_from_zip(zb, col_names=AGGTRADE_COLS)
            if df is not None and not df.empty:
                transformed = _transform_aggtrades(df)
                transformed["_date"] = pd.to_datetime(
                    transformed["timestamp"], unit="us"
                ).dt.strftime("%Y-%m-%d")
                for d in days:
                    out = _out_path(output_dir, exchange_label, "trades",
                                    d, symbol)
                    if out.exists():
                        saved += 1
                        continue
                    day_df = transformed[transformed["_date"] == d].drop(
                        columns=["_date"])
                    if day_df.empty:
                        continue
                    day_df.to_csv(out, index=False, compression="gzip")
                    saved += 1
                print(f"✓ ({len(days)} days)")
                continue

        # daily fallback
        print("monthly N/A → daily ", end="", flush=True)
        day_ok = 0
        for d in days:
            out = _out_path(output_dir, exchange_label, "trades", d, symbol)
            if out.exists():
                saved += 1
                day_ok += 1
                continue
            durl = (f"{BASE_URL}/daily/aggTrades/{symbol}/"
                    f"{symbol}-aggTrades-{d}.zip")
            db = _download_zip(durl)
            if db is not None:
                ddf = _csv_from_zip(db, col_names=AGGTRADE_COLS)
                if ddf is not None and not ddf.empty:
                    day_t = _transform_aggtrades(ddf)
                    day_t.to_csv(out, index=False, compression="gzip")
                    saved += 1
                    day_ok += 1
            time.sleep(0.1)
        print(f"({day_ok}/{len(days)} days)")

    print(f"\n[binance-vision] aggTrades done – {saved} day-files")
    return saved


def fetch_liquidations(*args, **kwargs) -> int:
    """
    Liquidation snapshot data is NOT available on data.binance.vision.

    Binance Vision only publishes: aggTrades, klines, trades.
    Liquidation (forceOrder) data is a premium dataset:
      - Tardis.dev provides it (paid, requires API key).
      - The downstream pipeline handles missing liquidation files
        gracefully – liquidation features default to zero.

    If you need liquidation data without Tardis, a manual workaround is
    to stream `/fapi/v1/allForceOrders` via the REST API, but it only
    returns the last 100-1000 events (not full history).
    """
    print()
    print("[binance-vision] Liquidation data is NOT available on data.binance.vision.")
    print("  Binance Vision only provides: aggTrades, klines, trades.")
    print("  Liquidation features will be set to zero in the pipeline.")
    print("  To get liquidation history, use Tardis (paid) with --source tardis.")
    return 0


# ════════════════════ unified entry point ════════════════════════════

def fetch(
    mode: str = "klines",
    symbol: str = TARDIS_SYMBOL,
    start_date: str = TARDIS_START_DATE,
    end_date: str = TARDIS_END_DATE,
    output_dir: Optional[Path] = None,
    include_liquidations: bool = False,  # not available on Binance Vision
):
    """
    Download BTC futures data from Binance Vision (FREE, no API key).

    Parameters
    ----------
    mode : str
        ``"klines"`` – 1-min OHLCV bars (lightweight, ~150 MB total).
        ``"aggtrades"`` – tick-level aggregated trades (100+ GB total).
    symbol, start_date, end_date : str
        Standard Binance symbol and YYYY-MM-DD range.
    include_liquidations : bool
        No-op. Liquidation snapshot data is NOT available on
        data.binance.vision. Use --source tardis for liquidation history.
    """
    root = output_dir or TARDIS_RAW_DIR
    print()
    print("=" * 60)
    print("  Binance Vision Data Fetcher  (FREE – no API key)")
    print("=" * 60)
    print(f"  Symbol : {symbol}")
    print(f"  Range  : {start_date} → {end_date}")
    print(f"  Mode   : {mode}")
    print(f"  Output : {root}")
    print("=" * 60)
    print()

    if mode == "klines":
        fetch_klines(symbol=symbol, start_date=start_date,
                     end_date=end_date, output_dir=root / "trades")
    elif mode == "aggtrades":
        fetch_aggtrades(symbol=symbol, start_date=start_date,
                        end_date=end_date, output_dir=root / "trades")
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'klines' or 'aggtrades'.")

    # Liquidation snapshot data does NOT exist on data.binance.vision.
    # Skip silently – the pipeline handles empty liquidation dirs gracefully.
    print()
    print("[binance-vision] NOTE: liquidation data not available from Binance Vision.")
    print("  Liquidation features will be zero. Use --source tardis for full data.")
    print()
    print("[binance-vision] All downloads complete.")
    print(f"  Next step: python scripts/03_audit_tardis.py")
    print(f"             python scripts/06_preprocess_tardis.py")
