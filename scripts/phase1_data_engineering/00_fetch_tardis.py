"""
Script 00 – Fetch BTC Futures Data
====================================
Downloads historical Binance USDT-M Futures data.

Sources (chosen automatically):
  • Binance Vision  (FREE, default, no API key)
  • Tardis.dev      (requires TARDIS_API_KEY, paid)

Usage:
    # Free – Binance Vision klines (default, ~150 MB total)
    python scripts/00_fetch_tardis.py

    # Free – Binance Vision tick-level aggTrades (very large)
    python scripts/00_fetch_tardis.py --mode aggtrades

    # Paid – Tardis (if you have an API key)
    python scripts/00_fetch_tardis.py --source tardis

    # Custom date range
    python scripts/00_fetch_tardis.py --start 2021-01-01 --end 2021-12-31
"""

import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config.settings import (
    TARDIS_START_DATE, TARDIS_END_DATE, TARDIS_DATA_TYPES,
    TARDIS_EXCHANGE, TARDIS_SYMBOL, TARDIS_API_KEY, ensure_dirs,
)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch BTC futures data (Binance Vision FREE or Tardis paid)")
    parser.add_argument("--start", default=TARDIS_START_DATE,
                        help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=TARDIS_END_DATE,
                        help="End date YYYY-MM-DD")
    parser.add_argument("--source", default="auto",
                        choices=["auto", "binance", "tardis"],
                        help="Data source: auto (default) picks Binance if no "
                             "TARDIS_API_KEY is set")
    parser.add_argument("--mode", default="klines",
                        choices=["klines", "aggtrades"],
                        help="Download mode (Binance source only). "
                             "klines=lightweight 1m OHLCV, "
                             "aggtrades=tick-level (very large)")
    parser.add_argument("--symbol", default=TARDIS_SYMBOL)
    args = parser.parse_args()

    ensure_dirs()

    # ── Decide source ────────────────────────────────────────────────
    use_tardis = False
    if args.source == "tardis":
        use_tardis = True
    elif args.source == "auto":
        use_tardis = bool(TARDIS_API_KEY)

    if use_tardis:
        # ── Tardis path (paid) ───────────────────────────────────────
        if not TARDIS_API_KEY:
            print("ERROR: --source tardis requires TARDIS_API_KEY.")
            print("  Windows:  set TARDIS_API_KEY=your_key_here")
            print("  Linux:    export TARDIS_API_KEY=your_key_here")
            print("\nTip: omit --source to use Binance Vision (free).")
            sys.exit(1)

        from src.data.tardis_fetcher import fetch as tardis_fetch
        print(f"[tardis] Fetching for {args.symbol} on {TARDIS_EXCHANGE}")
        print(f"  Date range: {args.start} → {args.end}")
        tardis_fetch(
            exchange=TARDIS_EXCHANGE,
            symbol=args.symbol,
            data_types=TARDIS_DATA_TYPES,
            start_date=args.start,
            end_date=args.end,
        )
    else:
        # ── Binance Vision path (FREE) ───────────────────────────────
        from src.data.binance_fetcher import fetch as binance_fetch
        binance_fetch(
            mode=args.mode,
            symbol=args.symbol,
            start_date=args.start,
            end_date=args.end,
        )

    print("\nDone. Files saved to data/raw/tardis/")


if __name__ == "__main__":
    main()
