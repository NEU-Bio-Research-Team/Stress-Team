"""
Effect Size Analysis
=====================
Compute Cohen's d and other effect-size metrics for each feature
between stress and non-stress classes.

Critical diagnostic: if ALL features have |d| < 0.2, the signal
is too weak for any model to learn.

Interpretation (Cohen 1988):
  |d| < 0.2  → negligible
  0.2 ≤ |d| < 0.5 → small
  0.5 ≤ |d| < 0.8 → medium
  |d| ≥ 0.8  → large
"""

import numpy as np
from typing import Dict, Optional, List, Any


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Cohen's d (pooled standard deviation version).

    d = (μ₁ - μ₂) / s_pooled
    """
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0

    m1, m2 = group1.mean(), group2.mean()
    s1, s2 = group1.std(ddof=1), group2.std(ddof=1)

    # Pooled standard deviation
    s_pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))

    if s_pooled < 1e-10:
        return 0.0

    return float((m1 - m2) / s_pooled)


def compute_effect_sizes(
    X: np.ndarray,
    y: np.ndarray,
    feature_cols: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute Cohen's d for each feature between stress (y=1) and non-stress (y=0).

    Args:
        X: (N, D) feature matrix
        y: (N,) binary labels
        feature_cols: list of feature names (defaults to f0, f1, ...)

    Returns:
        dict {feature_name: cohens_d_value}
    """
    if feature_cols is None:
        feature_cols = [f"f{i}" for i in range(X.shape[1])]

    stress_mask = y == 1
    non_stress_mask = y == 0

    result = {}
    for i, col in enumerate(feature_cols):
        d = cohens_d(X[stress_mask, i], X[non_stress_mask, i])
        result[col] = round(d, 4)

    return result


def effect_size_summary(effect_sizes: Dict[str, float]) -> Dict[str, Any]:
    """
    Summarize effect sizes across all features.

    Returns:
        dict with counts per category, strongest/weakest features.
    """
    from collections import Counter

    categories = Counter()
    for col, d in effect_sizes.items():
        ad = abs(d)
        if ad >= 0.8:
            categories["large"] += 1
        elif ad >= 0.5:
            categories["medium"] += 1
        elif ad >= 0.2:
            categories["small"] += 1
        else:
            categories["negligible"] += 1

    sorted_by_abs = sorted(effect_sizes.items(), key=lambda x: abs(x[1]), reverse=True)

    return {
        "category_counts": dict(categories),
        "n_features": len(effect_sizes),
        "strongest_features": sorted_by_abs[:5],
        "weakest_features": sorted_by_abs[-5:],
        "max_abs_d": max(abs(d) for d in effect_sizes.values()) if effect_sizes else 0,
        "mean_abs_d": np.mean([abs(d) for d in effect_sizes.values()]) if effect_sizes else 0,
        "signal_assessment": (
            "STRONG" if max(abs(d) for d in effect_sizes.values()) >= 0.8 else
            "MODERATE" if max(abs(d) for d in effect_sizes.values()) >= 0.5 else
            "WEAK" if max(abs(d) for d in effect_sizes.values()) >= 0.2 else
            "NO_SIGNAL"
        ) if effect_sizes else "NO_DATA",
    }


def compute_feature_correlation(X: np.ndarray, feature_cols: Optional[List[str]] = None):
    """
    Compute correlation matrix to detect redundant features.

    Returns:
        corr_matrix: (D, D) Pearson correlation matrix
        high_corr_pairs: list of (feat_i, feat_j, corr) for |corr| > 0.9
    """
    if feature_cols is None:
        feature_cols = [f"f{i}" for i in range(X.shape[1])]

    corr = np.corrcoef(X.T)
    corr = np.nan_to_num(corr, nan=0.0)

    high_corr_pairs = []
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            if abs(corr[i, j]) > 0.9:
                high_corr_pairs.append((feature_cols[i], feature_cols[j], round(corr[i, j], 4)))

    return corr, high_corr_pairs
