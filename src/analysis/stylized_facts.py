"""
Stylized Facts Validation
==========================
Dedicated module for computing and validating the 5 core stylized facts
that the ABM must reproduce:

    SF-1: Fat tails (excess kurtosis > 3 for returns)
    SF-2: Volatility clustering (significant ACF of |r| at multiple lags)
    SF-3: Leverage effect (negative correlation between r and future vol)
    SF-4: Volume-volatility correlation (positive corr)
    SF-5: Absence of return autocorrelation (ACF of r ≈ 0)

Reference: Cont (2001) "Empirical properties of asset returns: stylized
facts and statistical issues"
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
import json


def sf1_fat_tails(
    returns: np.ndarray,
    threshold_kurt: float = 3.0,
) -> Dict[str, object]:
    """
    SF-1: Fat tails.
    
    Financial returns exhibit excess kurtosis >> 3 (Gaussian).
    Typical values: 5-50 for minute-level crypto.
    """
    from scipy.stats import kurtosis as _kurtosis, jarque_bera

    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]

    if len(r) < 30:
        return {"name": "SF-1 Fat Tails", "pass": False, "reason": "insufficient data"}

    kurt = float(_kurtosis(r, fisher=True))  # excess kurtosis
    skew = float(pd.Series(r).skew())

    # Jarque-Bera test for normality
    jb_stat, jb_pval = jarque_bera(r)

    # Distribution fit: tail index via Hill estimator
    tail_idx = _hill_estimator(np.abs(r))

    return {
        "name": "SF-1 Fat Tails",
        "excess_kurtosis": kurt,
        "skewness": skew,
        "jarque_bera_stat": float(jb_stat),
        "jarque_bera_pval": float(jb_pval),
        "hill_tail_index": tail_idx,
        "normality_rejected": jb_pval < 0.05,
        "pass": kurt > threshold_kurt,
        "note": f"Kurtosis={kurt:.2f} ({'>' if kurt > threshold_kurt else '<='} {threshold_kurt})",
    }


def _hill_estimator(abs_returns: np.ndarray, fraction: float = 0.05) -> float:
    """Hill estimator for tail index (α) using top fraction of data."""
    sorted_vals = np.sort(abs_returns)[::-1]
    k = max(10, int(len(sorted_vals) * fraction))
    if k >= len(sorted_vals):
        return 0
    top_k = sorted_vals[:k]
    threshold = sorted_vals[k]
    if threshold <= 0:
        return 0
    alpha = 1.0 / np.mean(np.log(top_k / threshold))
    return float(alpha)


def sf2_volatility_clustering(
    returns: np.ndarray,
    max_lag: int = 50,
    significance_level: float = 0.05,
) -> Dict[str, object]:
    """
    SF-2: Volatility clustering.
    
    |returns| show significant positive autocorrelation that decays slowly.
    Check: ACF of |r| at lags 1,5,10,20,50.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]

    if len(r) < max_lag + 10:
        return {"name": "SF-2 Volatility Clustering", "pass": False, "reason": "insufficient data"}

    abs_r = np.abs(r - np.mean(r))
    sq_r = (r - np.mean(r)) ** 2

    check_lags = [1, 5, 10, 20, min(50, max_lag)]
    se = 1.0 / np.sqrt(len(r))  # standard error under null

    acf_abs = {}
    acf_sq = {}
    for lag in check_lags:
        if lag < len(abs_r):
            acf_abs[lag] = float(np.corrcoef(abs_r[:-lag], abs_r[lag:])[0, 1])
            acf_sq[lag] = float(np.corrcoef(sq_r[:-lag], sq_r[lag:])[0, 1])

    # Significant if ACF > 2*SE at multiple lags
    sig_count = sum(1 for v in acf_abs.values() if v > 2 * se)

    return {
        "name": "SF-2 Volatility Clustering",
        "acf_abs_returns": acf_abs,
        "acf_squared_returns": acf_sq,
        "standard_error": float(se),
        "significant_lags": sig_count,
        "total_lags_checked": len(check_lags),
        "pass": sig_count >= 3,
        "note": f"{sig_count}/{len(check_lags)} lags significant",
    }


def sf3_leverage_effect(
    returns: np.ndarray,
    max_lag: int = 20,
) -> Dict[str, object]:
    """
    SF-3: Leverage effect.
    
    Negative returns tend to increase future volatility more than
    positive returns of the same magnitude.
    
    Measured as: corr(r_t, |r_{t+k}|^2) < 0 for small k.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]

    if len(r) < max_lag + 10:
        return {"name": "SF-3 Leverage Effect", "pass": False, "reason": "insufficient data"}

    # Cross-correlation: r_t vs |r_{t+lag}|^2
    leverage_corr = {}
    for lag in [1, 5, 10]:
        if lag < len(r):
            leverage_corr[lag] = float(
                np.corrcoef(r[:-lag], r[lag:] ** 2)[0, 1]
            )

    # Asymmetric volatility: compare vol after negative vs positive returns
    neg_mask = r[:-1] < 0
    pos_mask = r[:-1] > 0

    vol_after_neg = float(np.std(r[1:][neg_mask])) if neg_mask.sum() > 10 else 0
    vol_after_pos = float(np.std(r[1:][pos_mask])) if pos_mask.sum() > 10 else 0

    # Leverage effect: most lag correlations should be negative
    neg_count = sum(1 for v in leverage_corr.values() if v < 0)

    return {
        "name": "SF-3 Leverage Effect",
        "leverage_correlations": leverage_corr,
        "vol_after_negative": vol_after_neg,
        "vol_after_positive": vol_after_pos,
        "asymmetry_ratio": vol_after_neg / (vol_after_pos + 1e-12),
        "negative_corr_count": neg_count,
        "pass": neg_count >= 2 or vol_after_neg > vol_after_pos,
        "note": f"Asymmetry ratio = {vol_after_neg / (vol_after_pos + 1e-12):.3f}",
    }


def sf4_volume_volatility_correlation(
    returns: np.ndarray,
    volumes: np.ndarray,
) -> Dict[str, object]:
    """
    SF-4: Volume-volatility positive correlation.
    
    Trading volume is positively correlated with market volatility.
    """
    r = np.asarray(returns, dtype=float)
    v = np.asarray(volumes, dtype=float)

    min_len = min(len(r), len(v))
    r = r[:min_len]
    v = v[:min_len]

    valid = np.isfinite(r) & np.isfinite(v) & (v > 0)
    r = r[valid]
    v = v[valid]

    if len(r) < 30:
        return {"name": "SF-4 Volume-Volatility", "pass": False, "reason": "insufficient data"}

    abs_r = np.abs(r)
    corr_abs = float(np.corrcoef(abs_r, v)[0, 1])
    corr_sq = float(np.corrcoef(r ** 2, v)[0, 1])

    # Log volume vs abs returns
    log_v = np.log(v + 1)
    corr_log = float(np.corrcoef(abs_r, log_v)[0, 1])

    return {
        "name": "SF-4 Volume-Volatility Correlation",
        "corr_absret_vol": corr_abs,
        "corr_sqret_vol": corr_sq,
        "corr_absret_logvol": corr_log,
        "pass": corr_abs > 0,
        "note": f"corr(|r|, V) = {corr_abs:.4f}",
    }


def sf5_absence_return_autocorrelation(
    returns: np.ndarray,
    max_lag: int = 20,
    economic_threshold: float = 0.05,
) -> Dict[str, object]:
    """
    SF-5: Absence of return autocorrelation.
    
    Raw returns show very little (≈0) linear autocorrelation,
    in contrast to |returns| which show strong ACF.
    
    Uses ECONOMIC significance (|ACF| < 0.05) rather than pure
    statistical significance (±2/√N). With N > 1M, even ACF = 0.002
    is "statistically significant" but economically meaningless.
    Ref: Cont (2001), Table 1 — return ACF is "insignificant" meaning
    economically negligible, not necessarily zero at machine precision.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]

    if len(r) < max_lag + 10:
        return {"name": "SF-5 No Return ACF", "pass": False, "reason": "insufficient data"}

    se = 1.0 / np.sqrt(len(r))

    acf = {}
    for lag in [1, 2, 5, 10, 20]:
        if lag < len(r):
            acf[lag] = float(np.corrcoef(r[:-lag], r[lag:])[0, 1])

    # ── Dual significance criterion ──
    # 1) Statistical: |ACF| < 2*SE  (classical Bartlett bound)
    # 2) Economic:    |ACF| < 0.05  (Cont 2001: ACF is "insignificant")
    # For large N (>1M), statistical SE ~ 0.0006, so even tiny correlations
    # are "statistically significant" but have zero economic meaning.
    # We use the ECONOMIC criterion for the pass/fail decision.
    econ_pass = sum(1 for v in acf.values() if abs(v) < economic_threshold)
    stat_pass = sum(1 for v in acf.values() if abs(v) < 2 * se)

    # Compare ACF magnitude to |returns| ACF (should be orders smaller)
    abs_r = np.abs(r - r.mean())
    abs_acf1 = float(np.corrcoef(abs_r[:-1], abs_r[1:])[0, 1]) if len(r) > 1 else 0
    max_ret_acf = max(abs(v) for v in acf.values()) if acf else 0
    acf_ratio = max_ret_acf / (abs_acf1 + 1e-12)  # should be << 1

    return {
        "name": "SF-5 Absence of Return ACF",
        "return_acf": acf,
        "standard_error": float(se),
        "economic_threshold": economic_threshold,
        "econ_insignificant": econ_pass,
        "stat_insignificant": stat_pass,
        "total_lags": len(acf),
        "max_abs_acf": max_ret_acf,
        "abs_return_acf_lag1": abs_acf1,
        "return_vs_absreturn_acf_ratio": acf_ratio,
        # Pass if ALL lags are economically insignificant (|ACF| < 0.05)
        "pass": econ_pass == len(acf),
        "note": f"max|ACF|={max_ret_acf:.4f} (threshold {economic_threshold}); "
                f"{econ_pass}/{len(acf)} economically insignificant; "
                f"{stat_pass}/{len(acf)} statistically insignificant (SE={se:.6f}); "
                f"|r| ACF₁={abs_acf1:.3f} >> return ACF (ratio={acf_ratio:.3f})",
    }


def run_all_stylized_facts(
    returns: np.ndarray,
    volumes: Optional[np.ndarray] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, object]:
    """
    Run all 5 stylized fact checks on a return series.
    
    Args:
        returns: array of log-returns
        volumes: optional volume array (for SF-4)
        output_dir: save JSON report here
    
    Returns:
        summary dict with all results
    """
    from config.settings import REPORTS_DIR

    if output_dir is None:
        output_dir = REPORTS_DIR / "stylized_facts"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    results.append(sf1_fat_tails(returns))
    results.append(sf2_volatility_clustering(returns))
    results.append(sf3_leverage_effect(returns))

    if volumes is not None:
        results.append(sf4_volume_volatility_correlation(returns, volumes))
    else:
        results.append({
            "name": "SF-4 Volume-Volatility Correlation",
            "pass": False,
            "reason": "No volume data provided",
        })

    results.append(sf5_absence_return_autocorrelation(returns))

    summary = {
        "total_facts": len(results),
        "passed": sum(1 for r in results if r.get("pass", False)),
        "failed": sum(1 for r in results if not r.get("pass", False)),
        "details": results,
    }

    # Console output
    print(f"\n{'='*60}")
    print(f"Stylized Facts Validation Report")
    print(f"{'='*60}")
    print(f"Total checks: {summary['total_facts']}")
    print(f"Passed:       {summary['passed']}")
    print(f"Failed:       {summary['failed']}")

    for r in results:
        status = "✓ PASS" if r.get("pass", False) else "✗ FAIL"
        print(f"\n  [{status}] {r['name']}")
        if "note" in r:
            print(f"          {r['note']}")
        if "reason" in r:
            print(f"          Reason: {r['reason']}")

    # Save
    out_path = output_dir / "stylized_facts_report.json"

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)

    print(f"\nReport saved: {out_path}")
    return summary
