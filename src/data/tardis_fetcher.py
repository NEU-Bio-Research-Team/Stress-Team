"""
Tardis BTC Futures Data Fetcher
================================
Downloads historical Binance USDT-M Futures data from tardis.dev.

Data types:
    - trades: individual executed trades
    - incremental_book_L2: level-2 order book incremental updates
    - liquidations: forced liquidation events (available since 2021-09-01)

Usage:
    Set TARDIS_API_KEY environment variable, then:
        python scripts/00_fetch_tardis.py

References:
    https://docs.tardis.dev/historical-data-details/binance-futures
"""

import os
import gzip
import json
import time
import hashlib
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timedelta

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    TARDIS_API_KEY, TARDIS_RAW_DIR, TARDIS_EXCHANGE,
    TARDIS_SYMBOL, TARDIS_DATA_TYPES, TARDIS_START_DATE,
    TARDIS_END_DATE, TARDIS_UNSAFE_DATE,
)

try:
    from tardis_dev import datasets as tardis_datasets
    HAS_TARDIS = True
except ImportError:
    HAS_TARDIS = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


def _date_range(start: str, end: str) -> List[str]:
    """Generate list of date strings YYYY-MM-DD from start to end (inclusive)."""
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    dates = []
    while s <= e:
        dates.append(s.strftime("%Y-%m-%d"))
        s += timedelta(days=1)
    return dates


def fetch_via_library(
    api_key: Optional[str] = None,
    exchange: str = TARDIS_EXCHANGE,
    symbol: str = TARDIS_SYMBOL,
    data_types: Optional[List[str]] = None,
    start_date: str = TARDIS_START_DATE,
    end_date: str = TARDIS_END_DATE,
    output_dir: Optional[Path] = None,
):
    """
    Download using the official tardis-dev Python library.
    Saves files in output_dir/{data_type}/ organized by date.
    """
    if not HAS_TARDIS:
        raise ImportError("Install tardis-dev: pip install tardis-dev")

    api_key = api_key or TARDIS_API_KEY
    if not api_key:
        raise ValueError(
            "TARDIS_API_KEY not set. Set via environment variable or pass api_key=."
        )

    data_types = data_types or TARDIS_DATA_TYPES
    output_dir = output_dir or TARDIS_RAW_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter data types by availability date
    effective_types = []
    for dt in data_types:
        if dt == "liquidations" and start_date < "2021-09-01":
            print(f"[tardis] NOTE: {dt} only available from 2021-09-01.")
            print(f"         Will download from 2021-09-01 for liquidations.")
        effective_types.append(dt)

    print(f"[tardis] Downloading {symbol} from {exchange}")
    print(f"         Date range: {start_date} → {end_date}")
    print(f"         Data types: {effective_types}")
    print(f"         Output: {output_dir}")

    for dt in effective_types:
        dt_start = start_date
        if dt == "liquidations" and dt_start < "2021-09-01":
            dt_start = "2021-09-01"

        dt_dir = output_dir / dt
        dt_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[tardis] Fetching {dt} ({dt_start} → {end_date}) ...")
        try:
            tardis_datasets.download(
                exchange=exchange,
                data_types=[dt],
                from_date=dt_start,
                to_date=end_date,
                symbols=[symbol],
                api_key=api_key,
                download_dir=str(dt_dir),
            )
            print(f"[tardis] ✓ {dt} download complete.")
        except Exception as e:
            print(f"[tardis] ✗ {dt} download failed: {e}")


def fetch_via_http(
    api_key: Optional[str] = None,
    exchange: str = TARDIS_EXCHANGE,
    symbol: str = TARDIS_SYMBOL,
    data_types: Optional[List[str]] = None,
    start_date: str = TARDIS_START_DATE,
    end_date: str = TARDIS_END_DATE,
    output_dir: Optional[Path] = None,
):
    """
    Download via HTTP API (fallback if tardis-dev library not installed).
    Uses https://datasets.tardis.dev/v1/
    """
    if not HAS_REQUESTS:
        raise ImportError("Install requests: pip install requests")

    api_key = api_key or TARDIS_API_KEY
    if not api_key:
        raise ValueError("TARDIS_API_KEY not set.")

    data_types = data_types or TARDIS_DATA_TYPES
    output_dir = output_dir or TARDIS_RAW_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    BASE_URL = "https://datasets.tardis.dev/v1"

    for dt in data_types:
        dt_start = start_date
        if dt == "liquidations" and dt_start < "2021-09-01":
            dt_start = "2021-09-01"

        dt_dir = output_dir / dt
        dt_dir.mkdir(parents=True, exist_ok=True)

        dates = _date_range(dt_start, end_date)
        print(f"[tardis-http] Fetching {dt}: {len(dates)} days ...")

        for date_str in dates:
            out_file = dt_dir / f"{exchange}_{dt}_{date_str}_{symbol}.csv.gz"
            if out_file.exists():
                continue   # skip already downloaded

            url = f"{BASE_URL}/{exchange}/{dt}/{date_str}/{symbol}.csv.gz"
            headers = {"Authorization": f"Bearer {api_key}"}

            try:
                resp = requests.get(url, headers=headers, stream=True, timeout=120)
                if resp.status_code == 200:
                    with open(out_file, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=1 << 20):
                            f.write(chunk)
                elif resp.status_code == 404:
                    pass  # no data for this date
                else:
                    print(f"  [!] {date_str} {dt}: HTTP {resp.status_code}")
            except Exception as e:
                print(f"  [!] {date_str} {dt}: {e}")

            time.sleep(0.1)  # rate limit

        print(f"[tardis-http] ✓ {dt} done.")


def fetch(api_key: Optional[str] = None, **kwargs):
    """Auto-select best download method."""
    if HAS_TARDIS:
        fetch_via_library(api_key=api_key, **kwargs)
    else:
        fetch_via_http(api_key=api_key, **kwargs)


def list_downloaded_files(output_dir: Optional[Path] = None) -> dict:
    """List downloaded files organized by data type."""
    output_dir = output_dir or TARDIS_RAW_DIR
    result = {}
    if not output_dir.exists():
        return result
    for dt_dir in sorted(output_dir.iterdir()):
        if dt_dir.is_dir():
            files = sorted(dt_dir.glob("*.csv.gz"))
            result[dt_dir.name] = {
                "count": len(files),
                "first": files[0].name if files else None,
                "last": files[-1].name if files else None,
                "total_bytes": sum(f.stat().st_size for f in files),
            }
    return result
