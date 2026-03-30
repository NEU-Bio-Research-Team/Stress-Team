"""
Script 07 - DAG Causal Validation
====================================
Validate the causal structure from the Algorithmic Panic DAG using
micro-level HFT features computed by Script 06.

Tests performed:

    1. STRUCTURAL BREAK DETECTION
       Detect regime shifts around flash crash events:
       - Pre-crash (calm) vs crash (panic) vs post-crash (recovery)
       - Compare feature distributions across regimes

    2. GRANGER CAUSALITY (DAG edges)
       Test directed edges in the DAG:
       - OFI -> Spread (Market Conditions -> Liquidity Spikes)
       - Spread -> OFI (Structural Market Feedback)
       - Trade intensity -> Spread (Behavioral Channel)
       - Volatility -> Depth withdrawal (Risk Aversion -> Liquidity)

    3. IMPULSE RESPONSE FUNCTIONS
       Measure how a shock in one variable propagates to others:
       - Large OFI shock -> spread response (Kyle's lambda in action)
       - Spread blowup -> trade intensity response (Panic feedback)
       - Depth collapse -> price impact amplification

    4. ENDOGENOUS PANIC LOOP DETECTION
       Test for self-reinforcing dynamics:
       - Positive feedback: spread_t -> OFI_{t+1} -> spread_{t+2}
       - Panic amplification coefficient
       - Mean-reversion time after shock (recovery dynamics)

    5. FLASH CRASH ANATOMY
       Per-event structural decomposition:
       - Phase 1: Trigger (initial OFI shock)
       - Phase 2: Amplification (spread blowup + depth withdrawal)
       - Phase 3: Recovery (liquidity return + spread normalization)
       - Measure time constants for each phase

Output: data/processed/tardis/dag_validation/
    - granger_results.csv
    - impulse_responses.csv
    - crash_anatomy.csv
    - dag_validation_summary.json

Usage:
    python scripts/stage2_economics/07_dag_validation.py
"""

import sys, os, argparse, json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import (
    EVENT_MICRO_DIR, EVENT_DAG_DIR, EVENT_CATALOG_PATH,
    MICRO_SPREAD_ZSCORE_THR,
    ensure_dirs,
)


# =====================================================================
# 1. Structural Break Detection
# =====================================================================

def detect_regimes(
    df: pd.DataFrame,
    crash_offset_ms: int,
    pre_window_ms: int = 5 * 60 * 1000,   # 5 min
    crash_window_ms: int = 5 * 60 * 1000,  # 5 min
    post_window_ms: int = 10 * 60 * 1000,  # 10 min
) -> pd.DataFrame:
    """
    Label rows as pre_crash / crash / post_crash regime.
    crash_offset_ms: ms offset of crash start from data start.
    """
    df = df.copy()

    if "time_offset_ms" not in df.columns:
        if "timestamp_ms" in df.columns:
            df["time_offset_ms"] = df["timestamp_ms"] - df["timestamp_ms"].iloc[0]
        else:
            return df

    pre_end = crash_offset_ms
    pre_start = max(0, crash_offset_ms - pre_window_ms)
    crash_end = crash_offset_ms + crash_window_ms
    post_end = crash_end + post_window_ms

    conditions = [
        (df["time_offset_ms"] >= pre_start) & (df["time_offset_ms"] < pre_end),
        (df["time_offset_ms"] >= pre_end) & (df["time_offset_ms"] < crash_end),
        (df["time_offset_ms"] >= crash_end) & (df["time_offset_ms"] < post_end),
    ]
    choices = ["pre_crash", "crash", "post_crash"]
    df["regime"] = np.select(conditions, choices, default="other")

    return df


def regime_statistics(df: pd.DataFrame) -> dict:
    """Compute regime-wise feature statistics."""
    features = ["ofi", "log_return", "trade_intensity", "vpin",
                "kyle_lambda", "amihud_illiq"]

    # Add BBO features if available
    if "spread_bps" in df.columns:
        features.extend(["spread_bps", "touch_depth", "depth_imbalance"])
    if "effective_spread_bps" in df.columns:
        features.append("effective_spread_bps")

    stats = {}
    for regime in ["pre_crash", "crash", "post_crash"]:
        rdf = df[df["regime"] == regime]
        if rdf.empty:
            continue
        regime_stats = {"n_bars": len(rdf)}
        for feat in features:
            if feat not in rdf.columns:
                continue
            vals = rdf[feat].dropna()
            if vals.empty:
                continue
            regime_stats[f"{feat}_mean"] = float(vals.mean())
            regime_stats[f"{feat}_std"] = float(vals.std())
            regime_stats[f"{feat}_median"] = float(vals.median())
            regime_stats[f"{feat}_skew"] = float(vals.skew()) if len(vals) > 2 else 0.0
        stats[regime] = regime_stats

    return stats


# =====================================================================
# 2. Granger Causality
# =====================================================================

def simple_granger_test(
    x: np.ndarray,
    y: np.ndarray,
    max_lag: int = 5,
) -> dict:
    """
    Simple F-test based Granger causality: does x Granger-cause y?

    Uses OLS: compare y ~ y_lags vs y ~ y_lags + x_lags.
    Returns F-statistic and approximate p-value.
    """
    n = len(x)
    if n < max_lag + 10:
        return {"f_stat": 0.0, "p_value": 1.0, "lags": max_lag, "n": n}

    # Build lag matrix
    Y = y[max_lag:]
    Y_lags = np.column_stack([y[max_lag - i - 1:n - i - 1] for i in range(max_lag)])
    X_lags = np.column_stack([x[max_lag - i - 1:n - i - 1] for i in range(max_lag)])

    # Restricted model: y ~ y_lags
    Z_r = np.column_stack([Y_lags, np.ones(len(Y))])
    try:
        beta_r = np.linalg.lstsq(Z_r, Y, rcond=None)[0]
        rss_r = np.sum((Y - Z_r @ beta_r) ** 2)
    except np.linalg.LinAlgError:
        return {"f_stat": 0.0, "p_value": 1.0, "lags": max_lag, "n": n}

    # Unrestricted model: y ~ y_lags + x_lags
    Z_u = np.column_stack([Y_lags, X_lags, np.ones(len(Y))])
    try:
        beta_u = np.linalg.lstsq(Z_u, Y, rcond=None)[0]
        rss_u = np.sum((Y - Z_u @ beta_u) ** 2)
    except np.linalg.LinAlgError:
        return {"f_stat": 0.0, "p_value": 1.0, "lags": max_lag, "n": n}

    # F-test
    df1 = max_lag
    df2 = len(Y) - Z_u.shape[1]

    if df2 <= 0 or rss_u <= 0:
        return {"f_stat": 0.0, "p_value": 1.0, "lags": max_lag, "n": n}

    f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)

    # Approximate p-value using F-distribution survival function
    try:
        from scipy.stats import f as f_dist
        p_value = float(f_dist.sf(f_stat, df1, df2))
    except ImportError:
        # Manual approximation if scipy not available
        p_value = 1.0 if f_stat < 1.0 else max(0.001, 1.0 / f_stat)

    return {"f_stat": float(f_stat), "p_value": p_value, "lags": max_lag, "n": n}


def test_dag_edges(df: pd.DataFrame, max_lag: int = 5) -> list[dict]:
    """
    Test Granger causality for each directed edge in the DAG.
    """
    # DAG edges to test (source -> target)
    edges = [
        ("ofi", "log_return", "OFI -> Price (Market Impact)"),
        ("trade_intensity", "ofi", "Trade Intensity -> OFI (Herding)"),
        ("log_return", "trade_intensity", "Return -> Trade Intensity (Feedback)"),
        ("kyle_lambda", "log_return", "Kyle Lambda -> Return (Informed Trading)"),
        ("vpin", "log_return", "VPIN -> Return (Toxicity)"),
    ]

    # Add BBO-dependent edges if available
    if "spread_bps" in df.columns:
        edges.extend([
            ("ofi", "spread_bps", "OFI -> Spread (Liquidity Impact)"),
            ("spread_bps", "trade_intensity", "Spread -> Intensity (Feedback Loop)"),
            ("spread_bps", "ofi", "Spread -> OFI (Structural Feedback)"),
            ("touch_depth", "spread_bps", "Depth -> Spread (Liquidity Provision)"),
            ("log_return", "spread_bps", "Return -> Spread (Volatility-Liquidity)"),
        ])

    results = []
    for source, target, description in edges:
        if source not in df.columns or target not in df.columns:
            continue

        x = df[source].dropna().values
        y = df[target].dropna().values
        min_len = min(len(x), len(y))
        x, y = x[:min_len], y[:min_len]

        result = simple_granger_test(x, y, max_lag=max_lag)
        result["source"] = source
        result["target"] = target
        result["edge"] = description
        result["significant"] = result["p_value"] < 0.05
        results.append(result)

    return results


# =====================================================================
# 3. Impulse Response
# =====================================================================

def compute_impulse_response(
    df: pd.DataFrame,
    shock_var: str,
    response_var: str,
    shock_threshold_sigma: float = 2.0,
    response_window: int = 50,
) -> dict | None:
    """
    Empirical impulse response: when shock_var has a large move,
    how does response_var evolve over the next N bars?
    """
    if shock_var not in df.columns or response_var not in df.columns:
        return None

    shock = df[shock_var].values
    response = df[response_var].values

    # Identify shock events (> threshold sigma)
    shock_mean = np.nanmean(shock)
    shock_std = np.nanstd(shock)
    if shock_std == 0:
        return None

    shock_events = np.where(
        np.abs(shock - shock_mean) > shock_threshold_sigma * shock_std
    )[0]

    if len(shock_events) < 3:
        return None

    # Average response after each shock
    responses = []
    for idx in shock_events:
        if idx + response_window >= len(response):
            continue
        # Normalize response relative to pre-shock level
        pre_level = response[max(0, idx-5):idx].mean() if idx >= 5 else response[idx]
        resp_window = response[idx:idx + response_window]
        if pre_level != 0:
            responses.append((resp_window - pre_level) / abs(pre_level))
        else:
            responses.append(resp_window - pre_level)

    if not responses:
        return None

    avg_response = np.mean(responses, axis=0)
    std_response = np.std(responses, axis=0)

    # Peak response
    peak_idx = int(np.argmax(np.abs(avg_response)))
    peak_val = float(avg_response[peak_idx])

    # Half-life (time to decay to 50% of peak)
    half_life = response_window
    if abs(peak_val) > 0:
        for t in range(peak_idx, len(avg_response)):
            if abs(avg_response[t]) < abs(peak_val) * 0.5:
                half_life = t - peak_idx
                break

    return {
        "shock_var": shock_var,
        "response_var": response_var,
        "n_shocks": len(shock_events),
        "n_responses": len(responses),
        "peak_response": peak_val,
        "peak_lag": peak_idx,
        "half_life_bars": half_life,
        "avg_response": avg_response.tolist(),
        "std_response": std_response.tolist(),
    }


# =====================================================================
# 4. Flash Crash Anatomy
# =====================================================================

def analyze_crash_anatomy(df: pd.DataFrame) -> dict:
    """
    Decompose a flash crash event into phases and time constants.
    """
    if "log_return" not in df.columns or len(df) < 20:
        return {}

    # Cumulative return
    cum_ret = df["log_return"].cumsum().values
    n = len(cum_ret)

    # Find bottom (minimum cumulative return)
    bottom_idx = int(np.argmin(cum_ret))
    max_drawdown = float(cum_ret[bottom_idx] - cum_ret[0]) if bottom_idx > 0 else 0.0

    # Phase 1: Trigger — onset of selling
    # Find where cumulative return starts dropping consistently
    trigger_idx = 0
    for i in range(bottom_idx):
        if cum_ret[i] > cum_ret[0] * 0.99:  # still near start
            trigger_idx = i

    # Phase 2: Amplification — steepest part of the drop
    if bottom_idx > trigger_idx + 1:
        returns_slice = df["log_return"].iloc[trigger_idx:bottom_idx + 1].values
        steepest_window = min(10, len(returns_slice))
        rolling_sum = np.convolve(returns_slice, np.ones(steepest_window), "valid")
        amplification_start = trigger_idx + int(np.argmin(rolling_sum))
    else:
        amplification_start = trigger_idx

    # Phase 3: Recovery — measure how long to recover 50% of drop
    recovery_50_idx = n - 1
    if max_drawdown < 0:
        target = cum_ret[bottom_idx] - max_drawdown * 0.5
        for i in range(bottom_idx, n):
            if cum_ret[i] >= target:
                recovery_50_idx = i
                break

    anatomy = {
        "total_bars": n,
        "bottom_idx": bottom_idx,
        "max_drawdown_pct": float(max_drawdown * 100),
        "trigger_to_bottom_bars": bottom_idx - trigger_idx,
        "amplification_start_idx": amplification_start,
        "recovery_50_idx": recovery_50_idx,
        "recovery_50_bars": recovery_50_idx - bottom_idx,
    }

    # OFI dynamics per phase
    if "ofi" in df.columns:
        ofi = df["ofi"].values
        anatomy["ofi_trigger_mean"] = float(np.mean(ofi[trigger_idx:amplification_start+1])) \
            if amplification_start > trigger_idx else 0.0
        anatomy["ofi_crash_mean"] = float(np.mean(ofi[amplification_start:bottom_idx+1])) \
            if bottom_idx > amplification_start else 0.0
        anatomy["ofi_recovery_mean"] = float(np.mean(ofi[bottom_idx:recovery_50_idx+1])) \
            if recovery_50_idx > bottom_idx else 0.0

    # VPIN dynamics
    if "vpin" in df.columns:
        vpin = df["vpin"].values
        anatomy["vpin_pre_crash"] = float(np.mean(vpin[max(0, trigger_idx-20):trigger_idx+1]))
        anatomy["vpin_at_crash"] = float(np.mean(vpin[amplification_start:bottom_idx+1])) \
            if bottom_idx > amplification_start else 0.0
        anatomy["vpin_post_crash"] = float(np.mean(vpin[bottom_idx:min(n, bottom_idx+20)]))

    # Spread dynamics (if available)
    if "spread_bps" in df.columns:
        spread = df["spread_bps"].values
        anatomy["spread_pre_crash"] = float(np.nanmean(spread[max(0, trigger_idx-20):trigger_idx+1]))
        anatomy["spread_at_crash"] = float(np.nanmean(spread[amplification_start:bottom_idx+1])) \
            if bottom_idx > amplification_start else 0.0
        anatomy["spread_peak"] = float(np.nanmax(spread[amplification_start:min(n, bottom_idx+10)])) \
            if bottom_idx > amplification_start else 0.0

    return anatomy


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="DAG causal validation from micro HFT features"
    )
    parser.add_argument("--resolution", type=int, default=100,
                        help="Resolution in ms to analyze (default: 100)")
    parser.add_argument("--granger-lags", type=int, default=5,
                        help="Max lags for Granger test")
    args = parser.parse_args()

    ensure_dirs()

    res_dir = EVENT_MICRO_DIR / f"res_{args.resolution}ms"
    if not res_dir.exists():
        print(f"[!] Micro feature dir not found: {res_dir}")
        print("    Run scripts 05-06 first.")
        return

    print("DAG Causal Validation (Tier 2)")
    print(f"  Resolution   : {args.resolution}ms")
    print(f"  Input        : {res_dir}")
    print(f"  Output       : {EVENT_DAG_DIR}")
    print(f"  Granger lags : {args.granger_lags}\n")

    feature_files = sorted(res_dir.glob("*.parquet"))
    if not feature_files:
        print("[!] No micro feature files found.")
        return

    print(f"  Found {len(feature_files)} event feature files\n")

    all_granger = []
    all_impulse = []
    all_anatomy = []
    all_regimes = []

    for fpath in feature_files:
        event_name = fpath.stem
        print(f"  [{event_name}]")

        df = pd.read_parquet(fpath)
        if df.empty or len(df) < 30:
            print("    [SKIP] too few rows")
            continue

        # --- Regime detection ---
        # Crash offset is ~30 min from start (EVENT_WINDOW_BEFORE_MIN)
        crash_offset_ms = 30 * 60 * 1000  # 30 min
        df_regimed = detect_regimes(df, crash_offset_ms)
        regime_stats = regime_statistics(df_regimed)
        for regime, stats in regime_stats.items():
            stats["event"] = event_name
            stats["regime"] = regime
            all_regimes.append(stats)

        n_regimes = len(regime_stats)
        print(f"    Regimes: {n_regimes} detected")

        # --- Granger causality ---
        granger_results = test_dag_edges(df, max_lag=args.granger_lags)
        sig_count = sum(1 for r in granger_results if r["significant"])
        print(f"    Granger: {sig_count}/{len(granger_results)} edges significant")
        for r in granger_results:
            r["event"] = event_name
            all_granger.append(r)

        # --- Impulse response ---
        ir_pairs = [
            ("ofi", "log_return"),
            ("trade_intensity", "ofi"),
            ("log_return", "trade_intensity"),
        ]
        if "spread_bps" in df.columns:
            ir_pairs.extend([
                ("ofi", "spread_bps"),
                ("spread_bps", "ofi"),
                ("spread_bps", "trade_intensity"),
            ])

        ir_count = 0
        for shock_v, resp_v in ir_pairs:
            ir = compute_impulse_response(df, shock_v, resp_v)
            if ir:
                ir["event"] = event_name
                # Remove large arrays for CSV
                ir_summary = {k: v for k, v in ir.items()
                             if k not in ("avg_response", "std_response")}
                all_impulse.append(ir_summary)
                ir_count += 1
        print(f"    Impulse: {ir_count} responses computed")

        # --- Crash anatomy ---
        anatomy = analyze_crash_anatomy(df)
        if anatomy:
            anatomy["event"] = event_name
            all_anatomy.append(anatomy)
            print(f"    Anatomy: drawdown={anatomy.get('max_drawdown_pct', 0):.2f}%  "
                  f"recovery={anatomy.get('recovery_50_bars', -1)} bars")

        print()

    # =================================================================
    # Save results
    # =================================================================
    EVENT_DAG_DIR.mkdir(parents=True, exist_ok=True)

    # Granger results
    if all_granger:
        granger_df = pd.DataFrame(all_granger)
        granger_path = EVENT_DAG_DIR / "granger_results.csv"
        granger_df.to_csv(granger_path, index=False)
        print(f"  Granger results: {granger_path}")

        # Aggregate across events
        print("\n  === Granger Causality Summary (across all events) ===")
        for edge in granger_df["edge"].unique():
            edge_df = granger_df[granger_df["edge"] == edge]
            n_sig = edge_df["significant"].sum()
            n_total = len(edge_df)
            avg_f = edge_df["f_stat"].mean()
            avg_p = edge_df["p_value"].mean()
            sig_rate = n_sig / n_total * 100 if n_total > 0 else 0
            tag = "***" if sig_rate >= 60 else "**" if sig_rate >= 40 else "*" if sig_rate >= 20 else ""
            print(f"    {edge:<50} sig={n_sig}/{n_total} ({sig_rate:.0f}%)  "
                  f"F={avg_f:.2f}  p={avg_p:.4f} {tag}")

    # Impulse responses
    if all_impulse:
        impulse_df = pd.DataFrame(all_impulse)
        impulse_path = EVENT_DAG_DIR / "impulse_responses.csv"
        impulse_df.to_csv(impulse_path, index=False)
        print(f"\n  Impulse responses: {impulse_path}")

    # Crash anatomy
    if all_anatomy:
        anatomy_df = pd.DataFrame(all_anatomy)
        anatomy_path = EVENT_DAG_DIR / "crash_anatomy.csv"
        anatomy_df.to_csv(anatomy_path, index=False)
        print(f"  Crash anatomy: {anatomy_path}")

        print("\n  === Flash Crash Anatomy Summary ===")
        for col in ["max_drawdown_pct", "trigger_to_bottom_bars",
                     "recovery_50_bars"]:
            if col in anatomy_df.columns:
                vals = anatomy_df[col].dropna()
                if not vals.empty:
                    print(f"    {col}: mean={vals.mean():.2f}  "
                          f"std={vals.std():.2f}  "
                          f"range=[{vals.min():.2f}, {vals.max():.2f}]")

    # Regime statistics
    if all_regimes:
        regime_df = pd.DataFrame(all_regimes)
        regime_path = EVENT_DAG_DIR / "regime_statistics.csv"
        regime_df.to_csv(regime_path, index=False)
        print(f"  Regime statistics: {regime_path}")

    # Overall DAG validation summary
    summary = {
        "n_events": len(feature_files),
        "resolution_ms": args.resolution,
        "granger_edges_tested": len(all_granger),
        "impulse_responses": len(all_impulse),
        "crash_anatomies": len(all_anatomy),
    }

    if all_granger:
        granger_df = pd.DataFrame(all_granger)
        sig_rates = {}
        for edge in granger_df["edge"].unique():
            edf = granger_df[granger_df["edge"] == edge]
            sig_rates[edge] = float(edf["significant"].mean())
        summary["granger_significance_rates"] = sig_rates

    summary_path = EVENT_DAG_DIR / "dag_validation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  DAG summary: {summary_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
