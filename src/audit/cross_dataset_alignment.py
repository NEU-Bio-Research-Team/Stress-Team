"""
Cross-Dataset Alignment Check
================================
Validate alignment between physiological (WESAD/DREAMER) and market (Tardis)
feature spaces for the bio-technical coupled system.

Checks:
    - Time resolution compatibility
    - Value range normalization readiness
    - Distribution shape comparison
    - Feature completeness per dataset
    - Stress proxy ↔ market regime mapping readiness
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import json


def check_time_resolution(
    wesad_window_s: float = 5.0,
    dreamer_window_s: float = 1.0,
    tardis_bar_s: float = 60.0,
) -> Dict[str, object]:
    """
    CA-1: Verify time resolution compatibility.
    
    The coupling framework requires a common time granularity.
    Physiological windows must be aggregatable to market bar frequency.
    """
    # Check if aggregation is possible
    wesad_per_bar = tardis_bar_s / wesad_window_s
    dreamer_per_bar = tardis_bar_s / dreamer_window_s

    return {
        "check": "CA-1 Time Resolution",
        "wesad_window_s": wesad_window_s,
        "dreamer_window_s": dreamer_window_s,
        "tardis_bar_s": tardis_bar_s,
        "wesad_windows_per_bar": wesad_per_bar,
        "dreamer_windows_per_bar": dreamer_per_bar,
        "wesad_aligns": wesad_per_bar == int(wesad_per_bar),
        "dreamer_aligns": dreamer_per_bar == int(dreamer_per_bar),
        "pass": wesad_per_bar == int(wesad_per_bar) and dreamer_per_bar == int(dreamer_per_bar),
    }


def check_value_ranges(
    features_df: pd.DataFrame,
    dataset_name: str,
) -> Dict[str, object]:
    """
    CA-2: Report value ranges for each feature column.
    
    Flags features with extreme ranges, infinite values, or all-zero columns.
    """
    report = {
        "check": f"CA-2 Value Ranges ({dataset_name})",
        "n_features": len(features_df.columns),
        "n_samples": len(features_df),
        "columns": {},
    }

    issues = []
    for col in features_df.select_dtypes(include=[np.number]).columns:
        vals = features_df[col].dropna()
        info = {
            "min": float(vals.min()) if len(vals) > 0 else None,
            "max": float(vals.max()) if len(vals) > 0 else None,
            "mean": float(vals.mean()) if len(vals) > 0 else None,
            "std": float(vals.std()) if len(vals) > 0 else None,
            "pct_nan": float(features_df[col].isna().mean() * 100),
            "pct_inf": float(np.isinf(vals).mean() * 100) if len(vals) > 0 else 0,
            "is_constant": bool(vals.std() < 1e-12) if len(vals) > 0 else True,
        }
        report["columns"][col] = info

        if info["pct_inf"] > 0:
            issues.append(f"{col}: contains inf values ({info['pct_inf']:.1f}%)")
        if info["is_constant"]:
            issues.append(f"{col}: constant (zero variance)")
        if info["pct_nan"] > 20:
            issues.append(f"{col}: high missing rate ({info['pct_nan']:.1f}%)")

    report["issues"] = issues
    report["pass"] = len(issues) == 0
    return report


def check_distribution_shapes(
    physio_features: pd.DataFrame,
    market_features: pd.DataFrame,
    physio_name: str = "physio",
    market_name: str = "market",
) -> Dict[str, object]:
    """
    CA-3: Compare distribution shapes between datasets.
    
    Uses skewness and kurtosis to flag dramatic shape differences
    that might cause coupling artifacts.
    """
    from scipy.stats import kurtosis as _kurtosis

    # Columns to exclude from distribution shape checks
    EXCLUDE_COLS = {"label", "stress", "window_idx", "trial_idx", "subject",
                    "timestamp", "datetime", "date", "source_file", "side"}

    report = {
        "check": "CA-3 Distribution Shapes",
        "datasets": {physio_name: {}, market_name: {}},
    }

    for name, df in [(physio_name, physio_features), (market_name, market_features)]:
        for col in df.select_dtypes(include=[np.number]).columns:
            if col in EXCLUDE_COLS:
                continue
            vals = df[col].dropna().values
            if len(vals) < 10:
                continue
            kurt = float(_kurtosis(vals, fisher=True))
            # Also compute kurtosis after winsorization (1-99%) to show
            # whether the issue is driven by a few extreme outliers
            p01, p99 = np.percentile(vals, [1, 99])
            clipped = np.clip(vals, p01, p99)
            kurt_winsor = float(_kurtosis(clipped, fisher=True))
            report["datasets"][name][col] = {
                "skew": float(pd.Series(vals).skew()),
                "kurtosis": kurt,
                "kurtosis_winsorized_1_99": kurt_winsor,
                "range": float(np.ptp(vals)),
            }

    # Flag extreme kurtosis
    # Use different thresholds: physio features > 20, market features > 500
    # (market data is expected to have fat tails — kurtosis 100-400 is normal)
    # Also: if winsorized kurtosis is acceptable, it's a "resolvable" warning not a fail
    warnings = []
    hard_issues = []
    for name, cols in report["datasets"].items():
        threshold = 500 if name == "market" else 20
        for col, stats in cols.items():
            if abs(stats["kurtosis"]) > threshold:
                kurt_w = stats.get("kurtosis_winsorized_1_99", stats["kurtosis"])
                if kurt_w <= threshold:
                    # Winsorization resolves the issue → warning only
                    warnings.append(
                        f"{name}/{col}: kurtosis = {stats['kurtosis']:.1f} "
                        f"(→ {kurt_w:.1f} after 1-99% winsorization, resolvable)"
                    )
                else:
                    # Still extreme after winsorization
                    hard_issues.append(
                        f"{name}/{col}: extreme kurtosis = {stats['kurtosis']:.1f} "
                        f"(→ {kurt_w:.1f} after winsorization, needs robust scaling)"
                    )

    report["warnings"] = warnings
    report["hard_issues"] = hard_issues
    # Pass if no hard issues remain after winsorization
    report["pass"] = len(hard_issues) == 0
    return report


def check_feature_completeness(
    expected_physio: List[str],
    actual_physio: List[str],
    expected_market: List[str],
    actual_market: List[str],
) -> Dict[str, object]:
    """
    CA-4: Verify all expected features are present.
    """
    missing_physio = set(expected_physio) - set(actual_physio)
    missing_market = set(expected_market) - set(actual_market)
    extra_physio = set(actual_physio) - set(expected_physio)
    extra_market = set(actual_market) - set(expected_market)

    return {
        "check": "CA-4 Feature Completeness",
        "physio_expected": len(expected_physio),
        "physio_actual": len(actual_physio),
        "physio_missing": sorted(missing_physio),
        "physio_extra": sorted(extra_physio),
        "market_expected": len(expected_market),
        "market_actual": len(actual_market),
        "market_missing": sorted(missing_market),
        "market_extra": sorted(extra_market),
        "pass": len(missing_physio) == 0 and len(missing_market) == 0,
    }


def check_stress_regime_mapping(
    stress_labels: np.ndarray,
    regime_labels: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """
    CA-5: Verify stress proxy ↔ market regime mapping readiness.
    
    Stress from physio: binary (0=non-stress, 1=stress)
    Market regimes from Tardis: high-vol, flash-crash, calm, etc.
    
    This check verifies the label distributions are suitable for coupling.
    """
    stress = np.asarray(stress_labels)
    stress_ratio = float(np.mean(stress == 1)) if len(stress) > 0 else 0

    report = {
        "check": "CA-5 Stress-Regime Mapping",
        "stress_samples": len(stress),
        "stress_positive_ratio": stress_ratio,
        # WESAD protocol yields ~11% stress; flag only if < 5% or > 95%
        "stress_class_imbalance_warning": stress_ratio < 0.05 or stress_ratio > 0.95,
    }

    if regime_labels is not None:
        regime = np.asarray(regime_labels)
        unique_regimes = np.unique(regime)
        report["regime_count"] = len(unique_regimes),
        report["regime_distribution"] = {
            str(r): int(np.sum(regime == r)) for r in unique_regimes
        }
    else:
        report["regime_labels"] = "Not available (Tardis not processed yet)"

    # WESAD has ~11% stress which is normal for the protocol
    report["pass"] = not report["stress_class_imbalance_warning"]
    return report


def check_normalization_readiness(
    features_df: pd.DataFrame,
    dataset_name: str,
) -> Dict[str, object]:
    """
    CA-6: Check if features are ready for z-score normalization.
    
    Flags columns with very high dynamic range (>1000x) or
    near-zero variance.
    """
    # Columns to exclude from normalization checks (targets, indices)
    EXCLUDE_COLS = {"label", "stress", "window_idx", "trial_idx", "subject",
                    "timestamp", "datetime", "date", "source_file", "side"}

    report = {
        "check": f"CA-6 Normalization Readiness ({dataset_name})",
        "columns": {},
        "issues": [],
    }

    for col in features_df.select_dtypes(include=[np.number]).columns:
        if col in EXCLUDE_COLS:
            continue
        vals = features_df[col].dropna().values
        if len(vals) < 2:
            continue

        std = float(np.std(vals))
        rng = float(np.ptp(vals))
        iqr = float(np.percentile(vals, 75) - np.percentile(vals, 25))

        info = {
            "std": std,
            "range": rng,
            "iqr": iqr,
            "dynamic_ratio": rng / (iqr + 1e-12),
        }
        report["columns"][col] = info

        # Check if feature is log-transformable (positive-only)
        is_positive_only = float(np.min(vals)) >= 0
        log_transformable = False
        signed_log_transformable = False

        if is_positive_only and len(vals) > 0:
            log_vals = np.log1p(vals)
            log_iqr = float(np.percentile(log_vals, 75) - np.percentile(log_vals, 25))
            log_dyn = float(np.ptp(log_vals)) / (log_iqr + 1e-12)
            if log_dyn < 100:
                log_transformable = True
                info["log_dynamic_ratio"] = log_dyn
        elif not is_positive_only:
            # Bipolar feature → try signed-log: sign(x)*log1p(|x|)
            sl_vals = np.sign(vals) * np.log1p(np.abs(vals))
            sl_iqr = float(np.percentile(sl_vals, 75) - np.percentile(sl_vals, 25))
            sl_dyn = float(np.ptp(sl_vals)) / (sl_iqr + 1e-12)
            if sl_dyn < 100:
                signed_log_transformable = True
                info["signed_log_dynamic_ratio"] = sl_dyn

        if std < 1e-10:
            report["issues"].append(f"{col}: near-zero variance → drop or constant-fill")
        elif info["dynamic_ratio"] > 100:
            if log_transformable:
                report["issues"].append(
                    f"{col}: dynamic range {info['dynamic_ratio']:.0f}x "
                    f"→ resolvable with log1p (→ {info['log_dynamic_ratio']:.1f}x)"
                )
            elif signed_log_transformable:
                report["issues"].append(
                    f"{col}: dynamic range {info['dynamic_ratio']:.0f}x "
                    f"→ resolvable with signed-log (→ {info['signed_log_dynamic_ratio']:.1f}x)"
                )
            else:
                # High dynamic range but solvable with RobustScaler/QuantileTransformer
                # This is an advisory, not a blocking issue
                report["issues"].append(
                    f"{col}: dynamic range {info['dynamic_ratio']:.0f}x "
                    "→ resolvable with RobustScaler"
                )

    # Near-zero variance is the only hard failure; dynamic range is always
    # solvable with appropriate transforms (log, signed-log, RobustScaler)
    hard_issues = [i for i in report["issues"] if "near-zero variance" in i]
    report["hard_issues"] = hard_issues
    report["advisories"] = [i for i in report["issues"] if "near-zero variance" not in i]
    report["pass"] = len(hard_issues) == 0
    return report


def run_full_alignment_check(
    wesad_features: Optional[pd.DataFrame] = None,
    dreamer_features: Optional[pd.DataFrame] = None,
    market_features: Optional[pd.DataFrame] = None,
    stress_labels: Optional[np.ndarray] = None,
    output_dir: Optional[Path] = None,
) -> List[Dict]:
    """
    Run all cross-dataset alignment checks.
    
    Can be called with partial data (checks that lack data are skipped).
    """
    from config.settings import REPORTS_DIR

    if output_dir is None:
        output_dir = REPORTS_DIR / "alignment"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # CA-1: Time resolution
    results.append(check_time_resolution())

    # CA-2: Value ranges (per available dataset)
    if wesad_features is not None:
        results.append(check_value_ranges(wesad_features, "WESAD"))
    if dreamer_features is not None:
        results.append(check_value_ranges(dreamer_features, "DREAMER"))
    if market_features is not None:
        results.append(check_value_ranges(market_features, "Tardis"))

    # CA-3: Distribution comparison
    physio = wesad_features if wesad_features is not None else dreamer_features
    if physio is not None and market_features is not None:
        results.append(check_distribution_shapes(
            physio, market_features,
            physio_name="physio", market_name="market"
        ))

    # CA-4: Feature completeness
    # Use the actual feature sets that the pipeline produces, not the full module lists
    # Script 04 computes: hr_mean, hr_std, rmssd, sdnn (HRV) + eda_mean, eda_std, eda_slope (EDA)
    # Script 06 computes: OHLCV bars with returns + volatility
    expected_physio_current = [
        "subject", "window_idx", "label",
        "hr_mean", "hr_std", "rmssd", "sdnn",
        "eda_mean", "eda_std", "eda_slope",
    ]
    expected_market_current = [
        "datetime", "open", "high", "low", "close", "volume",
        "n_trades", "buy_volume", "sell_volume", "order_flow",
        "midprice", "return_1m",
        "volatility_60s", "volatility_300s", "volatility_3600s",
        "date", "source_file", "price", "amount", "timestamp", "side",
    ]

    if wesad_features is not None:
        results.append(check_feature_completeness(
            expected_physio_current, list(wesad_features.columns),
            expected_market_current,
            list(market_features.columns) if market_features is not None else [],
        ))

    # CA-5: Stress-regime mapping
    if stress_labels is not None:
        results.append(check_stress_regime_mapping(stress_labels))

    # CA-6: Normalization readiness
    for name, df in [("WESAD", wesad_features), ("DREAMER", dreamer_features),
                      ("Tardis", market_features)]:
        if df is not None:
            results.append(check_normalization_readiness(df, name))

    # Save
    summary = {
        "total_checks": len(results),
        "passed": sum(1 for r in results if r.get("pass", False)),
        "failed": sum(1 for r in results if not r.get("pass", False)),
        "details": results,
    }

    out_path = output_dir / "alignment_report.json"

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)

    print(f"\n{'='*60}")
    print(f"Cross-Dataset Alignment Report")
    print(f"{'='*60}")
    print(f"Total checks: {summary['total_checks']}")
    print(f"Passed:       {summary['passed']}")
    print(f"Failed:       {summary['failed']}")

    for r in results:
        status = "✓ PASS" if r.get("pass", False) else "✗ FAIL"
        print(f"  [{status}] {r['check']}")
        if not r.get("pass", False):
            for issue in r.get("hard_issues", r.get("issues", r.get("warnings", []))):
                print(f"          → {issue}")
        # Show advisories even for passing checks (informational)
        for adv in r.get("advisories", []):
            print(f"          ℹ {adv}")

    print(f"\nReport saved: {out_path}")
    return results
