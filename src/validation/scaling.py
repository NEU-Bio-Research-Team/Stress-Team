"""
In-Pipeline Scaling Transforms
================================
Feature-specific normalization applied INSIDE the CV loop.
Never fit on test data. Never save pre-normalized files.

Strategy per dataset (from alignment report CA-6):
  WESAD:
    - hr_mean, hr_std, rmssd, sdnn → log1p + StandardScaler
    - eda_mean → StandardScaler
    - eda_std, eda_slope → RobustScaler (extreme dynamic range)
  DREAMER:
    - All 70 DE features → StandardScaler (all < 22× dynamic range)
"""

import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Callable, Optional, List


class Log1pTransformer(BaseEstimator, TransformerMixin):
    """Apply log1p to non-negative features, handling negatives gracefully."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.sign(X) * np.log1p(np.abs(X))


class SignedLogTransformer(BaseEstimator, TransformerMixin):
    """Signed-log: sign(x) * log1p(|x|). For bipolar features like order_flow."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.sign(X) * np.log1p(np.abs(X))


# ─────────────────────── Scaler Factories ───────────────────────
# These return CALLABLES that produce fresh scaler instances for each CV fold.


def get_wesad_scaler_factory() -> Callable:
    """
    Return factory for WESAD-specific column transformer.
    Uses RobustScaler for all features (safe default for extreme ranges).
    """
    def factory():
        return RobustScaler()
    return factory


def get_wesad_column_scaler_factory(feature_cols: List[str]) -> Callable:
    """
    Return factory for WESAD column-specific transformer.

    hrv features (hr_mean, hr_std, rmssd, sdnn) → log1p + StandardScaler
    eda features (eda_mean, eda_std, eda_slope) → RobustScaler
    """
    hrv_cols = ["hr_mean", "hr_std", "rmssd", "sdnn"]
    eda_cols = ["eda_mean", "eda_std", "eda_slope"]

    hrv_idx = [feature_cols.index(c) for c in hrv_cols if c in feature_cols]
    eda_idx = [feature_cols.index(c) for c in eda_cols if c in feature_cols]

    def factory():
        transformers = []
        if hrv_idx:
            transformers.append((
                "hrv",
                Pipeline([("log1p", Log1pTransformer()), ("scale", StandardScaler())]),
                hrv_idx,
            ))
        if eda_idx:
            transformers.append((
                "eda",
                RobustScaler(),
                eda_idx,
            ))
        return ColumnTransformer(transformers, remainder="passthrough")

    return factory


def get_dreamer_scaler_factory() -> Callable:
    """
    Return factory for DREAMER scaler.
    All DE features < 22× dynamic range → StandardScaler sufficient.
    """
    def factory():
        return StandardScaler()
    return factory


def get_scaler_factory(dataset: str, feature_cols: Optional[List[str]] = None) -> Callable:
    """
    Get the appropriate scaler factory for a dataset.

    Args:
        dataset: "wesad" or "dreamer"
        feature_cols: feature column names (needed for column-level WESAD scaling)

    Returns:
        callable that returns a fresh scaler instance
    """
    if dataset == "wesad":
        if feature_cols:
            return get_wesad_column_scaler_factory(feature_cols)
        return get_wesad_scaler_factory()
    elif dataset == "dreamer":
        return get_dreamer_scaler_factory()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
