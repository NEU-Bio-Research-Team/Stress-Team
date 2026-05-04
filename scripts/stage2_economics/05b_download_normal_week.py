"""
Script 05b - Download Normal-Week Baseline Data
================================================
Downloads BTCUSDT futures aggTrades for configured normal-market windows,
stores per-day parquet files, and computes baseline prior stats used by
scripts 11/18 for calibration ablations.

Outputs:
    data/processed/tardis/normal_baseline/
      - <label>_<YYYY-MM-DD>/aggtrades.parquet
      - baseline_download_manifest.csv
      - baseline_prior_stats.json

Usage:
    python scripts/stage2_economics/05b_download_normal_week.py
    python scripts/stage2_economics/05b_download_normal_week.py --max-days 2
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import NORMAL_WEEK_DIR, NORMAL_WEEK_WINDOWS, TARDIS_SYMBOL, ensure_dirs


try:
    import requests
except ImportError as exc:  # pragma: no cover - dependency check
    raise ImportError("Missing dependency 'requests'. Install via requirements.txt") from exc


BASE_URL = "https://data.binance.vision/data/futures/um/daily/aggTrades"
AGGTRADE_COLS = [
    "agg_trade_id", "price", "quantity", "first_trade_id",
    "last_trade_id", "transact_time", "is_buyer_maker",
]
QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]


def daterange(start_date: str, end_date: str) -> list[str]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    return [str(day.date()) for day in pd.date_range(start, end, freq="D")]


def _download_zip(url: str, timeout: int = 600) -> bytes | None:
    try:
        response = requests.get(url, timeout=timeout)
    except Exception as exc:
        print(f"    [!] Download error: {exc}")
        return None

    if response.status_code == 200:
        return response.content
    if response.status_code != 404:
        print(f"    [!] HTTP {response.status_code}: {url}")
    return None


def _csv_from_zip(zip_bytes: bytes) -> pd.DataFrame | None:
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            csv_names = [name for name in zf.namelist() if name.endswith(".csv")]
            if not csv_names:
                return None
            with zf.open(csv_names[0]) as fp:
                first_line = fp.readline().decode("utf-8", errors="replace")
                fp.seek(0)
                first_field = first_line.split(",")[0].strip()
                has_header = not first_field.replace(".", "").replace("-", "").isdigit()
                if has_header:
                    return pd.read_csv(fp)
                return pd.read_csv(fp, header=None, names=AGGTRADE_COLS)
    except Exception as exc:
        print(f"    [!] ZIP parse error: {exc}")
        return None


def download_aggtrades_day(symbol: str, date_str: str, timeout: int = 600) -> pd.DataFrame | None:
    url = f"{BASE_URL}/{symbol}/{symbol}-aggTrades-{date_str}.zip"
    print(f"    Downloading aggTrades {date_str}...", end=" ", flush=True)

    zip_bytes = _download_zip(url, timeout=timeout)
    if zip_bytes is None:
        print("[FAIL] not found")
        return None

    df = _csv_from_zip(zip_bytes)
    if df is None or df.empty:
        print("[FAIL] empty")
        return None

    out = pd.DataFrame()
    out["timestamp_ms"] = pd.to_numeric(df.get("transact_time", df.iloc[:, 5]), errors="coerce").astype("Int64")
    out["price"] = pd.to_numeric(df.get("price", df.iloc[:, 1]), errors="coerce")
    out["quantity"] = pd.to_numeric(df.get("quantity", df.iloc[:, 2]), errors="coerce")

    ibm = df.get("is_buyer_maker", df.iloc[:, 6])
    ibm_bool = ibm.astype(str).str.lower().isin(["true", "1"])
    out["side"] = np.where(ibm_bool, "sell", "buy")

    out = out.dropna(subset=["timestamp_ms", "price", "quantity"]).copy()
    out["timestamp_ms"] = out["timestamp_ms"].astype(np.int64)
    out["dollar_volume"] = out["price"] * out["quantity"]
    out["signed_volume"] = np.where(out["side"] == "buy", out["quantity"], -out["quantity"])

    print(f"[OK] {len(out):,} trades")
    return out


def _phase_quantiles(series: pd.Series) -> dict[str, float | None]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    s = s[s != 0]
    if s.empty:
        return {f"p{int(q * 100):02d}": None for q in QUANTILES}
    return {f"p{int(q * 100):02d}": round(float(s.quantile(q)), 8) for q in QUANTILES}


def _phase_stats(series: pd.Series) -> dict[str, float | int | None]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"mean": None, "std": None, "median": None, "n": 0}
    return {
        "mean": round(float(s.mean()), 8),
        "std": round(float(s.std()), 8),
        "median": round(float(s.median()), 8),
        "n": int(len(s)),
    }


def _pareto_alpha_mle(values: np.ndarray) -> float | None:
    x = values[np.isfinite(values)]
    x = x[x > 0]
    if len(x) < 10:
        return None
    x_min = np.min(x)
    if x_min <= 0:
        return None
    alpha = len(x) / np.sum(np.log(x / x_min))
    if not np.isfinite(alpha):
        return None
    return round(float(alpha), 4)


def _bars_100ms(agg_df: pd.DataFrame) -> pd.DataFrame:
    if agg_df.empty:
        return pd.DataFrame()

    df = agg_df.sort_values("timestamp_ms").copy()
    t0 = int(df["timestamp_ms"].iloc[0])
    df["bin_id"] = ((df["timestamp_ms"] - t0) // 100).astype(np.int64)
    grouped = df.groupby("bin_id")

    bars = grouped.agg(
        close=("price", "last"),
        volume=("quantity", "sum"),
        dollar_volume=("dollar_volume", "sum"),
        trade_count=("price", "count"),
        ofi=("signed_volume", "sum"),
        buy_volume=("quantity", lambda x: x[df.loc[x.index, "side"] == "buy"].sum()),
    ).reset_index()

    bars["trade_intensity"] = bars["trade_count"] / 0.1
    bars["vwap"] = np.where(bars["volume"] > 0, bars["dollar_volume"] / bars["volume"], bars["close"])
    bars["log_return"] = np.log(bars["vwap"] / bars["vwap"].shift(1)).fillna(0.0)

    bars["price_change"] = bars["close"].diff().fillna(0.0)
    sv = bars["ofi"]
    dp = bars["price_change"]
    kyle_window = 50
    sv_mean = sv.rolling(kyle_window, min_periods=5).mean()
    dp_mean = dp.rolling(kyle_window, min_periods=5).mean()
    sv2_mean = (sv ** 2).rolling(kyle_window, min_periods=5).mean()
    dpsv_mean = (dp * sv).rolling(kyle_window, min_periods=5).mean()
    sv_var = sv2_mean - sv_mean ** 2
    dp_sv_cov = dpsv_mean - dp_mean * sv_mean
    bars["kyle_lambda"] = np.where(sv_var > 1e-12, dp_sv_cov / sv_var, 0.0)
    bars["kyle_lambda"] = pd.to_numeric(bars["kyle_lambda"], errors="coerce").fillna(0.0)

    buy_vol = bars["buy_volume"]
    sell_vol = bars["volume"] - bars["buy_volume"]
    bars["vpin"] = (
        (buy_vol.rolling(50, min_periods=1).sum() - sell_vol.rolling(50, min_periods=1).sum()).abs()
        / bars["volume"].rolling(50, min_periods=1).sum().replace(0, np.nan)
    ).fillna(0.0)

    bars["realized_vol_50"] = bars["log_return"].rolling(50, min_periods=1).apply(
        lambda x: np.sqrt(np.sum(x ** 2)), raw=True
    )
    bars["amihud_illiq"] = np.where(
        bars["dollar_volume"] > 0,
        np.abs(bars["log_return"]) / bars["dollar_volume"],
        0.0,
    )

    return bars


def compute_label_stats(label: str, frames: list[pd.DataFrame]) -> dict[str, object]:
    if not frames:
        return {
            "label": label,
            "n_days": 0,
            "n_trades": 0,
            "n_bars_100ms": 0,
            "ofi_percentiles": {f"p{int(q * 100):02d}": None for q in QUANTILES},
            "trade_intensity": {"mean": None, "std": None, "median": None, "n": 0},
            "kyle_lambda": {"mean": None, "std": None, "median": None, "n": 0},
            "realized_vol": {"mean": None, "std": None, "median": None, "n": 0},
            "spread_bps": {"mean": None, "std": None, "median": None, "n": 0},
            "depth_imbalance": {"mean": None, "std": None, "median": None, "n": 0},
            "vpin": {"mean": None, "std": None, "median": None, "n": 0},
            "amihud": {"mean": None, "std": None, "median": None, "n": 0},
            "noise_trader_lambda": None,
            "order_size_pareto_alpha": None,
        }

    bars = pd.concat([_bars_100ms(frame) for frame in frames], ignore_index=True)
    ofi_values = pd.to_numeric(bars.get("ofi"), errors="coerce")

    trade_intensity_stats = _phase_stats(bars.get("trade_intensity", pd.Series(dtype=float)))
    return {
        "label": label,
        "n_days": len(frames),
        "n_trades": int(sum(len(frame) for frame in frames)),
        "n_bars_100ms": int(len(bars)),
        "ofi_percentiles": _phase_quantiles(ofi_values),
        "trade_intensity": trade_intensity_stats,
        "kyle_lambda": _phase_stats(bars.get("kyle_lambda", pd.Series(dtype=float))),
        "realized_vol": _phase_stats(bars.get("realized_vol_50", pd.Series(dtype=float))),
        "spread_bps": {"mean": None, "std": None, "median": None, "n": 0},
        "depth_imbalance": {"mean": None, "std": None, "median": None, "n": 0},
        "vpin": _phase_stats(bars.get("vpin", pd.Series(dtype=float))),
        "amihud": _phase_stats(bars.get("amihud_illiq", pd.Series(dtype=float))),
        "noise_trader_lambda": trade_intensity_stats["mean"],
        "order_size_pareto_alpha": _pareto_alpha_mle(np.abs(ofi_values.fillna(0.0).to_numpy(dtype=float))),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download normal-week baseline aggTrades and compute prior stats")
    parser.add_argument("--normal-dir", type=Path, default=NORMAL_WEEK_DIR)
    parser.add_argument("--symbol", type=str, default=TARDIS_SYMBOL)
    parser.add_argument("--max-days", type=int, default=None,
                        help="Optional cap for smoke tests")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip downloading if aggtrades.parquet already exists")
    parser.add_argument("--timeout", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    args.normal_dir.mkdir(parents=True, exist_ok=True)
    stats_path = args.normal_dir / "baseline_prior_stats.json"

    windows = list(NORMAL_WEEK_WINDOWS)
    print("Normal Baseline Download (Script 05b)")
    print(f"  Windows    : {len(windows)}")
    print(f"  Output dir : {args.normal_dir}")
    print(f"  Stats file : {stats_path}")

    manifest_rows: list[dict[str, object]] = []
    stats_payload: dict[str, object] = {
        "metadata": {
            "description": "Normal-market baseline anchors for calibration",
            "symbol": args.symbol,
            "source": "binance_vision_aggtrades",
            "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "windows": windows,
        }
    }

    days_processed = 0
    for window in windows:
        label = str(window["label"])
        day_frames: list[pd.DataFrame] = []
        dates = daterange(str(window["start"]), str(window["end"]))

        print(f"\n  [{label}] {window['start']} -> {window['end']} ({len(dates)} days)")
        for date_str in dates:
            if args.max_days is not None and days_processed >= args.max_days:
                break

            out_dir = args.normal_dir / f"{label}_{date_str}"
            out_dir.mkdir(parents=True, exist_ok=True)
            agg_path = out_dir / "aggtrades.parquet"

            if args.skip_existing and agg_path.exists() and agg_path.stat().st_size > 0:
                agg_df = pd.read_parquet(agg_path)
                print(f"    [SKIP] {date_str} existing rows={len(agg_df):,}")
                status = "cached"
            else:
                agg_df = download_aggtrades_day(args.symbol, date_str, timeout=args.timeout)
                if agg_df is None or agg_df.empty:
                    manifest_rows.append({
                        "label": label,
                        "date": date_str,
                        "status": "missing",
                        "n_rows": 0,
                        "output": "",
                    })
                    continue
                agg_df.to_parquet(agg_path, index=False, engine="pyarrow")
                status = "downloaded"

            day_frames.append(agg_df)
            manifest_rows.append({
                "label": label,
                "date": date_str,
                "status": status,
                "n_rows": int(len(agg_df)),
                "output": str(agg_path),
            })
            days_processed += 1

        stats_payload[label] = compute_label_stats(label, day_frames)

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = args.normal_dir / "baseline_download_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(stats_payload, handle, indent=2)

    print("\nDone.")
    print(f"  Manifest: {manifest_path}")
    print(f"  Stats   : {stats_path}")


if __name__ == "__main__":
    main()
