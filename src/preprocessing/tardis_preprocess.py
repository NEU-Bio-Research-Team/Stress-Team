"""
Tardis BTC Futures Preprocessing
==================================
Stage 2 data curation: transform raw Tardis downloads into
calibration targets for the ABM market simulator.

Pipeline:
    1. Load raw trades + incremental_book_L2
    2. Orderbook reconstruction & validation
    3. Feature extraction (spread, depth, volatility, order flow, midprice)
    4. Stylized facts computation
    5. Event identification (flash crashes, liquidation cascades)
    6. Export processed features → data/processed/tardis/
"""

import sys
import gzip
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    TARDIS_RAW_DIR, TARDIS_MAY19_START, TARDIS_MAY19_END,
    VOLATILITY_WINDOWS, PROCESSED_DIR,
)
from src.data.tardis_orderbook import (
    reconstruct_day_to_df, load_trades_day,
)


def _list_files(data_type: str, root: Path = TARDIS_RAW_DIR) -> List[Path]:
    dt_dir = root / data_type
    if not dt_dir.exists():
        return []
    return sorted(dt_dir.glob("*.csv.gz"))


def _extract_date(filepath: Path) -> Optional[str]:
    import re
    m = re.search(r"(\d{4}-\d{2}-\d{2})", filepath.name)
    return m.group(1) if m else None


def compute_daily_features(
    trades_path: Path,
    book_path: Optional[Path] = None,
    resample_freq: str = "1min",
) -> Optional[pd.DataFrame]:
    """
    Compute market microstructure features for one day.
    
    Returns DataFrame with columns:
        datetime, midprice, spread, bid_depth, ask_depth,
        volume, n_trades, buy_volume, sell_volume, order_flow,
        return_1m, volatility_1m, volatility_5m, volatility_1h
    """
    # Load data (supports both raw trades and pre-aggregated klines)
    try:
        with gzip.open(trades_path, "rt") as f:
            trades = pd.read_csv(f)
    except Exception as e:
        print(f"  [!] Failed: {trades_path.name}: {e}")
        return None

    if trades.empty or "price" not in trades.columns:
        return None

    # ── Auto-detect format: klines vs raw trades ────────────────────
    is_klines = all(c in trades.columns for c in ["open", "high", "low", "close"])

    if is_klines:
        # DATA IS ALREADY 1-MINUTE OHLCV (from Binance Vision klines)
        trades["datetime"] = pd.to_datetime(trades["timestamp"], unit="us")
        ohlcv = trades.set_index("datetime").copy()

        # Ensure required columns exist with correct types
        for col in ["open", "high", "low", "close", "volume"]:
            ohlcv[col] = ohlcv[col].astype(float)
        if "n_trades" not in ohlcv.columns:
            ohlcv["n_trades"] = 1
        if "buy_volume" not in ohlcv.columns:
            ohlcv["buy_volume"] = ohlcv["volume"] / 2
        if "sell_volume" not in ohlcv.columns:
            ohlcv["sell_volume"] = ohlcv["volume"] - ohlcv["buy_volume"]
        if "order_flow" not in ohlcv.columns:
            ohlcv["order_flow"] = ohlcv["buy_volume"] - ohlcv["sell_volume"]

    else:
        # RAW TRADES – resample to 1-minute bars
        trades["price"] = trades["price"].astype(float)
        if "amount" in trades.columns:
            trades["amount"] = trades["amount"].astype(float)
        else:
            trades["amount"] = 1.0

        trades["datetime"] = pd.to_datetime(trades["timestamp"], unit="us")
        trades = trades.set_index("datetime")

        if "side" in trades.columns:
            trades["is_buy"] = trades["side"].str.lower() == "buy"
        else:
            trades["is_buy"] = True

        ohlcv = trades["price"].resample(resample_freq).agg(
            ["first", "last", "min", "max", "count"]
        ).rename(columns={"first": "open", "last": "close",
                           "min": "low", "max": "high", "count": "n_trades"})
        ohlcv["volume"] = trades["amount"].resample(resample_freq).sum()
        ohlcv["buy_volume"] = trades.loc[trades["is_buy"], "amount"].resample(
            resample_freq).sum().reindex(ohlcv.index).fillna(0)
        ohlcv["sell_volume"] = ohlcv["volume"] - ohlcv["buy_volume"]
        ohlcv["order_flow"] = ohlcv["buy_volume"] - ohlcv["sell_volume"]

    # Midprice = (open + close) / 2 as proxy
    ohlcv["midprice"] = (ohlcv["open"] + ohlcv["close"]) / 2

    # Returns: standard close-to-close log return
    # NOTE: using close.shift(1) not open.shift(1) — the latter computes a
    # 2-bar return (prev open → current close) which creates spurious lag-1 ACF
    ohlcv["return_1m"] = np.log(ohlcv["close"] / ohlcv["close"].shift(1))

    # Volatility at different scales
    for win_sec in VOLATILITY_WINDOWS:
        win_periods = max(2, win_sec // 60)  # convert to minutes, min 2 for valid std
        col = f"volatility_{win_sec}s"
        ohlcv[col] = ohlcv["return_1m"].rolling(win_periods).std() * np.sqrt(win_periods)

    # Add orderbook features if available
    if book_path is not None and book_path.exists():
        try:
            book_df = reconstruct_day_to_df(book_path, sample_ms=60000)
            if not book_df.empty:
                book_df = book_df.set_index("datetime")
                book_resampled = book_df[["spread", "bid_depth_1pct", "ask_depth_1pct"]]\
                    .resample(resample_freq).mean()
                ohlcv = ohlcv.join(book_resampled, how="left")
        except Exception as e:
            print(f"  [!] Book reconstruction failed: {e}")

    ohlcv = ohlcv.dropna(subset=["close"])
    return ohlcv.reset_index()


def identify_flash_crashes(
    df: pd.DataFrame,
    threshold_pct: float = 5.0,
    window_minutes: int = 60,
) -> pd.DataFrame:
    """
    Identify flash crash events (> threshold% drop within window).
    """
    if "midprice" not in df.columns or df.empty:
        return pd.DataFrame()

    events = []
    prices = df["midprice"].values
    times = df["datetime"].values if "datetime" in df.columns else np.arange(len(df))

    for i in range(window_minutes, len(prices)):
        window_high = np.max(prices[i - window_minutes:i])
        drop_pct = (window_high - prices[i]) / window_high * 100
        if drop_pct > threshold_pct:
            events.append({
                "datetime": times[i],
                "drop_pct": round(drop_pct, 2),
                "price_at_event": prices[i],
                "window_high": window_high,
            })

    return pd.DataFrame(events)


def compute_stylized_facts(returns: np.ndarray) -> Dict:
    """
    Compute core stylized facts from return series.
    
    Returns dict with:
        excess_kurtosis, skewness,
        return_acf (lags 1-10),
        abs_return_acf (lags 1-10),
        squared_return_acf (lags 1-10),
        hurst_exponent (R/S method)
    """
    from scipy.stats import kurtosis, skew

    returns = returns[~np.isnan(returns)]
    n = len(returns)

    result = {
        "n_returns": n,
        "mean": float(np.mean(returns)),
        "std": float(np.std(returns)),
        "skewness": float(skew(returns)),
        "excess_kurtosis": float(kurtosis(returns, fisher=True)),
    }

    # ACF of returns
    def _acf(series, max_lag=10):
        s = series - series.mean()
        c0 = np.sum(s ** 2)
        if c0 == 0:
            return np.zeros(max_lag)
        acf_vals = []
        for lag in range(1, max_lag + 1):
            c = np.sum(s[:-lag] * s[lag:])
            acf_vals.append(c / c0)
        return np.array(acf_vals)

    result["return_acf"] = _acf(returns, 10).tolist()
    result["abs_return_acf"] = _acf(np.abs(returns), 10).tolist()
    result["squared_return_acf"] = _acf(returns ** 2, 10).tolist()

    # Hurst exponent (R/S method)
    try:
        def _hurst(ts):
            n = len(ts)
            if n < 20:
                return 0.5
            max_k = min(n // 2, 200)
            sizes = list(range(10, max_k, 5))
            RS = []
            for size in sizes:
                n_chunks = n // size
                if n_chunks == 0:
                    continue
                rs_vals = []
                for i in range(n_chunks):
                    chunk = ts[i * size:(i + 1) * size]
                    mean_adj = chunk - chunk.mean()
                    cumsum = np.cumsum(mean_adj)
                    R = cumsum.max() - cumsum.min()
                    S = chunk.std()
                    if S > 0:
                        rs_vals.append(R / S)
                if rs_vals:
                    RS.append((size, np.mean(rs_vals)))
            if len(RS) < 2:
                return 0.5
            log_sizes = np.log([r[0] for r in RS])
            log_RS = np.log([r[1] for r in RS])
            H = np.polyfit(log_sizes, log_RS, 1)[0]
            return float(H)

        result["hurst_exponent"] = _hurst(returns)
    except Exception:
        result["hurst_exponent"] = None

    # Core stylized facts assessment
    result["fact_fat_tails"] = result["excess_kurtosis"] > 3
    result["fact_vol_clustering"] = (
        abs(result["abs_return_acf"][0]) > 0.05
        if result["abs_return_acf"] else False
    )
    result["fact_no_return_autocorr"] = (
        abs(result["return_acf"][0]) < 0.05
        if result["return_acf"] else True
    )

    return result


def preprocess_all(
    root: Path = TARDIS_RAW_DIR,
    output_dir: Optional[Path] = None,
    max_days: int = 0,  # 0 = all available
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict:
    """
    Full preprocessing pipeline for Tardis BTC futures data.
    start_date / end_date: optional YYYY-MM-DD filter applied to filenames.
    """
    output_dir = output_dir or (PROCESSED_DIR / "tardis")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TARDIS BTC PREPROCESSING")
    print("=" * 60)

    trades_files = _list_files("trades", root)
    book_files = _list_files("incremental_book_L2", root)

    # Apply optional date range filter
    if start_date or end_date:
        def _in_range(p: Path) -> bool:
            d = _extract_date(p)
            if d is None:
                return True
            if start_date and d < start_date:
                return False
            if end_date and d > end_date:
                return False
            return True
        trades_files = [f for f in trades_files if _in_range(f)]
        book_files   = [f for f in book_files   if _in_range(f)]

    if not trades_files:
        print("[!] No trades files found. Run 00_fetch_tardis.py first.")
        return {}

    book_dates = {_extract_date(f): f for f in book_files}

    all_features = []
    all_crashes = []
    all_returns = []
    days_processed = 0

    n_files = len(trades_files) if max_days == 0 else min(max_days, len(trades_files))

    for path in trades_files[:n_files]:
        date = _extract_date(path)
        if date is None:
            continue

        # Skip May 19 2021
        if date == "2021-05-19":
            print(f"  [!] Skipping {date} (known data integrity issue)")
            continue

        print(f"  Processing {date} ...", end="")
        book_path = book_dates.get(date)
        features = compute_daily_features(path, book_path)

        if features is not None and not features.empty:
            features["date"] = date
            all_features.append(features)

            # Collect returns for stylized facts
            rets = features["return_1m"].dropna().values
            all_returns.append(rets)

            # Flash crashes
            crashes = identify_flash_crashes(features)
            if not crashes.empty:
                crashes["date"] = date
                all_crashes.append(crashes)
                print(f" {len(crashes)} crash events!", end="")

            print(f" ✓ ({len(features)} bars)")
            days_processed += 1
        else:
            print(" (empty)")

    if not all_features:
        print("[!] No features computed.")
        return {}

    # Combine
    features_df = pd.concat(all_features, ignore_index=True)
    features_path = output_dir / "btc_features_1min.parquet"
    features_df.to_parquet(features_path, index=False)
    print(f"\n[tardis] Features → {features_path} ({len(features_df)} rows)")

    # Crashes
    if all_crashes:
        crashes_df = pd.concat(all_crashes, ignore_index=True)
        crashes_path = output_dir / "flash_crashes.csv"
        crashes_df.to_csv(crashes_path, index=False)
        print(f"[tardis] Flash crashes → {crashes_path} ({len(crashes_df)} events)")

    # Stylized facts
    all_rets = np.concatenate(all_returns) if all_returns else np.array([])
    if len(all_rets) > 100:
        facts = compute_stylized_facts(all_rets)
        facts_path = output_dir / "stylized_facts.json"
        import json
        with open(facts_path, "w") as f:
            json.dump(facts, f, indent=2, default=str)
        print(f"[tardis] Stylized facts → {facts_path}")
        print(f"  Excess kurtosis: {facts['excess_kurtosis']:.2f}")
        print(f"  Fat tails: {facts['fact_fat_tails']}")
        print(f"  Vol clustering: {facts['fact_vol_clustering']}")
        print(f"  No return autocorr: {facts['fact_no_return_autocorr']}")
        if facts.get("hurst_exponent") is not None:
            print(f"  Hurst exponent: {facts['hurst_exponent']:.3f}")

    return {
        "days_processed": days_processed,
        "n_bars": len(features_df),
        "n_crashes": sum(len(c) for c in all_crashes),
    }


if __name__ == "__main__":
    from config.settings import ensure_dirs
    ensure_dirs()
    preprocess_all()
