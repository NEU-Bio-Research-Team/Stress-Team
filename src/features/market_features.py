"""
Market Microstructure Feature Extraction
==========================================
Extract market features from Tardis BTC futures data for ABM calibration.

Feature groups:
    - Price: returns, multi-scale volatility, log returns
    - Orderbook: spread, depth, imbalance (bid/ask ratio)
    - Volume: VWAP, trade intensity, buy/sell ratio
    - Liquidation: count, volume, cascade intensity
    - Stylized facts: kurtosis, Hurst exponent, ACF structure
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


# ──── Price features ────────────────────────────────────────────────

def compute_returns(
    prices: np.ndarray,
    log_returns: bool = True,
) -> np.ndarray:
    """Compute log-returns or simple returns."""
    prices = np.asarray(prices, dtype=float)
    if log_returns:
        return np.diff(np.log(prices + 1e-12))
    return np.diff(prices) / (prices[:-1] + 1e-12)


def compute_realized_volatility(
    returns: np.ndarray,
    window: int = None,
) -> float:
    """Realized volatility as sqrt of sum of squared returns."""
    if window is not None:
        returns = returns[-window:]
    return float(np.sqrt(np.sum(returns ** 2)))


def compute_multi_scale_volatility(
    returns: np.ndarray,
    scales: tuple = (1, 5, 15, 60),
) -> Dict[str, float]:
    """
    Volatility at multiple time scales (in bars).
    Each scale aggregates returns by that factor.
    """
    result = {}
    for s in scales:
        if len(returns) >= s:
            agg = returns[:len(returns) // s * s].reshape(-1, s).sum(axis=1)
            result[f"vol_{s}"] = float(np.std(agg))
        else:
            result[f"vol_{s}"] = 0
    return result


# ──── Orderbook features ───────────────────────────────────────────

def compute_spread_features(
    best_bid: np.ndarray,
    best_ask: np.ndarray,
) -> Dict[str, float]:
    """Spread statistics from best bid/ask series."""
    spread = best_ask - best_bid
    mid = (best_ask + best_bid) / 2.0
    rel_spread = spread / (mid + 1e-12)

    return {
        "spread_mean": float(np.mean(spread)),
        "spread_std": float(np.std(spread)),
        "spread_median": float(np.median(spread)),
        "spread_max": float(np.max(spread)),
        "rel_spread_mean": float(np.mean(rel_spread)),
        "rel_spread_std": float(np.std(rel_spread)),
    }


def compute_depth_imbalance(
    bid_depth: np.ndarray,
    ask_depth: np.ndarray,
) -> Dict[str, float]:
    """
    Orderbook depth imbalance features.
    
    Imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth)
    Range: [-1, 1], positive = more buying pressure
    """
    total = bid_depth + ask_depth
    imbalance = np.where(total > 0, (bid_depth - ask_depth) / total, 0)

    return {
        "depth_bid_mean": float(np.mean(bid_depth)),
        "depth_ask_mean": float(np.mean(ask_depth)),
        "depth_total_mean": float(np.mean(total)),
        "imbalance_mean": float(np.mean(imbalance)),
        "imbalance_std": float(np.std(imbalance)),
    }


# ──── Volume features ─────────────────────────────────────────────

def compute_volume_features(trades_df: pd.DataFrame) -> Dict[str, float]:
    """
    Volume features from trades DataFrame.
    
    Expects columns: price, amount, side (buy/sell), timestamp
    """
    if trades_df.empty:
        return {
            "total_volume": 0, "trade_count": 0, "vwap": 0,
            "buy_ratio": 0, "avg_trade_size": 0,
            "trade_intensity": 0,
        }

    total_volume = float(trades_df["amount"].sum())
    trade_count = len(trades_df)

    # VWAP
    vwap = float((trades_df["price"] * trades_df["amount"]).sum() /
                  (total_volume + 1e-12))

    # Buy/sell ratio
    if "side" in trades_df.columns:
        buy_vol = trades_df.loc[trades_df["side"] == "buy", "amount"].sum()
        buy_ratio = float(buy_vol / (total_volume + 1e-12))
    else:
        buy_ratio = 0.5

    # Time span
    if "timestamp" in trades_df.columns:
        t_span = (trades_df["timestamp"].max() - trades_df["timestamp"].min())
        if hasattr(t_span, 'total_seconds'):
            t_span = t_span.total_seconds()
        t_span_min = t_span / 60.0 if t_span > 0 else 1
    else:
        t_span_min = 1

    return {
        "total_volume": total_volume,
        "trade_count": trade_count,
        "vwap": vwap,
        "buy_ratio": buy_ratio,
        "avg_trade_size": total_volume / trade_count,
        "trade_intensity": trade_count / t_span_min,
    }


def compute_order_flow_imbalance(
    trades_df: pd.DataFrame,
    window_s: int = 60,
) -> pd.Series:
    """
    Compute Order Flow Imbalance (OFI) as buy_volume − sell_volume
    aggregated per window.
    """
    if trades_df.empty or "side" not in trades_df.columns:
        return pd.Series(dtype=float)

    df = trades_df.copy()
    df["signed_amount"] = df["amount"].where(df["side"] == "buy", -df["amount"])

    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
        ofi = df["signed_amount"].resample(f"{window_s}s").sum()
    else:
        # Fall back to positional windowing
        n = len(df) // max(1, window_s)
        ofi = pd.Series([df["signed_amount"].iloc[i * window_s:(i + 1) * window_s].sum()
                          for i in range(n)])
    return ofi


# ──── Liquidation features ─────────────────────────────────────────

def compute_liquidation_features(
    liq_df: pd.DataFrame,
) -> Dict[str, float]:
    """
    Features from liquidation events.
    
    Expects columns: price, quantity, side, timestamp
    """
    if liq_df is None or liq_df.empty:
        return {
            "liq_count": 0, "liq_total_volume": 0,
            "liq_long_ratio": 0, "liq_mean_size": 0,
            "liq_cascade_max": 0,
        }

    liq_count = len(liq_df)
    qty_col = "quantity" if "quantity" in liq_df.columns else "amount"
    total_vol = float(liq_df[qty_col].sum()) if qty_col in liq_df.columns else 0

    # Long vs short liquidations
    if "side" in liq_df.columns:
        long_count = (liq_df["side"].str.lower() == "sell").sum()  # long liq = forced sell
        long_ratio = float(long_count / liq_count) if liq_count > 0 else 0
    else:
        long_ratio = 0.5

    # Cascade detection: count max liquidations in any 5-minute window
    cascade_max = 0
    if "timestamp" in liq_df.columns and liq_count > 0:
        ts = pd.to_datetime(liq_df["timestamp"])
        cascade_max = int(ts.dt.floor("5min").value_counts().max())

    return {
        "liq_count": liq_count,
        "liq_total_volume": total_vol,
        "liq_long_ratio": long_ratio,
        "liq_mean_size": total_vol / liq_count if liq_count > 0 else 0,
        "liq_cascade_max": cascade_max,
    }


# ──── Stylized fact metrics ────────────────────────────────────────

def compute_stylized_metrics(returns: np.ndarray) -> Dict[str, float]:
    """
    Key stylized-fact metrics for ABM validation.
    """
    from scipy.stats import kurtosis as _kurtosis

    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]

    if len(r) < 30:
        return {
            "kurt": 0, "skew": 0, "abs_acf1": 0,
            "sq_acf1": 0, "ret_acf1": 0, "hurst": 0.5,
        }

    # Fat tails
    kurt = float(_kurtosis(r, fisher=True))
    skew = float(pd.Series(r).skew())

    # Autocorrelation structure
    r_centered = r - r.mean()
    var = np.var(r)
    if var < 1e-15:
        return {"kurt": kurt, "skew": skew, "abs_acf1": 0,
                "sq_acf1": 0, "ret_acf1": 0, "hurst": 0.5}

    # |r| ACF lag-1 (volatility clustering)
    abs_r = np.abs(r_centered)
    abs_acf1 = float(np.corrcoef(abs_r[:-1], abs_r[1:])[0, 1])

    # r² ACF lag-1
    sq_r = r_centered ** 2
    sq_acf1 = float(np.corrcoef(sq_r[:-1], sq_r[1:])[0, 1])

    # Return ACF lag-1 (should be ≈0)
    ret_acf1 = float(np.corrcoef(r_centered[:-1], r_centered[1:])[0, 1])

    # Hurst exponent (R/S method)
    hurst = _estimate_hurst(r)

    return {
        "kurt": kurt,
        "skew": skew,
        "abs_acf1": abs_acf1,
        "sq_acf1": sq_acf1,
        "ret_acf1": ret_acf1,
        "hurst": hurst,
    }


def _estimate_hurst(ts: np.ndarray, max_k: int = 20) -> float:
    """Simplified Hurst exponent via R/S analysis."""
    n = len(ts)
    if n < 20:
        return 0.5

    rs_list = []
    ns_list = []

    for k in range(2, min(max_k + 1, n // 4)):
        size = n // k
        if size < 4:
            break
        rs_vals = []
        for i in range(k):
            chunk = ts[i * size:(i + 1) * size]
            mean_c = np.mean(chunk)
            cumdev = np.cumsum(chunk - mean_c)
            r = np.max(cumdev) - np.min(cumdev)
            s = np.std(chunk)
            if s > 1e-12:
                rs_vals.append(r / s)
        if rs_vals:
            rs_list.append(np.mean(rs_vals))
            ns_list.append(size)

    if len(rs_list) < 3:
        return 0.5

    log_n = np.log(ns_list)
    log_rs = np.log(rs_list)
    hurst = float(np.polyfit(log_n, log_rs, 1)[0])
    return np.clip(hurst, 0, 1)


# ──── Aggregate extraction ─────────────────────────────────────────

MARKET_FEATURE_NAMES = [
    "vol_1", "vol_5", "vol_15", "vol_60",
    "spread_mean", "spread_std", "spread_median", "spread_max",
    "rel_spread_mean", "rel_spread_std",
    "depth_bid_mean", "depth_ask_mean", "depth_total_mean",
    "imbalance_mean", "imbalance_std",
    "total_volume", "trade_count", "vwap", "buy_ratio",
    "avg_trade_size", "trade_intensity",
    "liq_count", "liq_total_volume", "liq_long_ratio",
    "liq_mean_size", "liq_cascade_max",
    "kurt", "skew", "abs_acf1", "sq_acf1", "ret_acf1", "hurst",
]


def features_to_vector(features: Dict[str, float]) -> np.ndarray:
    """Convert feature dict to fixed-order vector."""
    return np.array([features.get(k, 0) for k in MARKET_FEATURE_NAMES])
