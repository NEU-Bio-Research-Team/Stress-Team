"""
Script 12 - Append Trade-Only BBO Proxies To Gridded Event Dynamics
===================================================================
When true BBO / bookTicker data is unavailable, add transparent fallback
proxies so downstream Phase-1 and Phase-3 pipelines can continue.

Proxy mapping:
    mid_price        <- close
    spread_bps       <- amihud_illiq * close * 10000
    depth_imbalance  <- clipped ofi / (trade_intensity * 100)
    touch_depth      <- inverse absolute Kyle lambda (winsorized)

These proxies are intended as trade-only fallbacks, not substitutes for real
book-state data. If true BBO columns already exist, they are preserved unless
--force is passed.

Usage:
    python scripts/stage2_economics/12_bbo_proxies.py
    python scripts/stage2_economics/12_bbo_proxies.py --force
"""

import sys, os, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import PROCESSED_DIR


INPUT_CSV = PROCESSED_DIR / "tardis" / "confounder_outputs" / "Event_Dynamics_100ms_gridded.csv"


def _should_fill(df: pd.DataFrame, col: str, force: bool) -> bool:
    if force or col not in df.columns:
        return True
    return bool(df[col].isna().all())


def _touch_depth_proxy(kyle_lambda: pd.Series) -> pd.Series:
    impact = pd.to_numeric(kyle_lambda, errors="coerce").abs().replace(0, np.nan)
    proxy = 1.0 / impact
    valid = proxy.dropna()
    if valid.empty:
        return pd.Series(np.zeros(len(proxy)), index=proxy.index)

    lo = float(valid.quantile(0.01))
    hi = float(valid.quantile(0.99))
    proxy = proxy.clip(lower=lo, upper=hi)
    return proxy.fillna(float(valid.median()))


def main():
    parser = argparse.ArgumentParser(
        description="Append trade-only BBO proxy columns to gridded event dynamics"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing BBO columns if they are already present",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Script 12: Append Trade-Only BBO Proxies")
    print("=" * 70)

    if not INPUT_CSV.exists():
        print(f"[!] Input not found: {INPUT_CSV}")
        print("    Run script 10 first.")
        return

    df = pd.read_csv(INPUT_CSV)
    print(f"  Loaded {len(df)} rows")

    fills = []

    if _should_fill(df, "mid_price", args.force):
        df["mid_price"] = pd.to_numeric(df["close"], errors="coerce")
        fills.append("mid_price <- close")

    if _should_fill(df, "spread_bps", args.force):
        close = pd.to_numeric(df["close"], errors="coerce")
        illiq = pd.to_numeric(df["amihud_illiq"], errors="coerce").fillna(0.0)
        df["spread_bps"] = (illiq * close.abs() * 10000.0).fillna(0.0).clip(lower=0.0)
        fills.append("spread_bps <- amihud_illiq * close * 10000")

    if _should_fill(df, "depth_imbalance", args.force):
        ofi = pd.to_numeric(df["ofi"], errors="coerce").fillna(0.0)
        intensity = pd.to_numeric(df["trade_intensity"], errors="coerce").fillna(0.0)
        scale = (intensity * 100.0).replace(0, np.nan)
        depth_proxy = (ofi / scale).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        df["depth_imbalance"] = depth_proxy.clip(-1.0, 1.0)
        fills.append("depth_imbalance <- clipped ofi / (trade_intensity * 100)")

    if _should_fill(df, "touch_depth", args.force):
        df["touch_depth"] = _touch_depth_proxy(df["kyle_lambda"])
        fills.append("touch_depth <- winsorized inverse absolute kyle_lambda")

    if fills:
        df.to_csv(INPUT_CSV, index=False)
        print("  Added proxy columns:")
        for fill in fills:
            print(f"    - {fill}")
        print(f"  Saved: {INPUT_CSV}")
    else:
        print("  No proxy columns needed; existing BBO columns were kept.")

    print("Done.")


if __name__ == "__main__":
    main()