"""Custom scikit-learn transformers used by the churn pipeline."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class IQRCapper(BaseEstimator, TransformerMixin):
    """Cap numeric outliers using IQR bounds learned from training data."""

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        q1 = np.nanpercentile(values, 25, axis=0)
        q3 = np.nanpercentile(values, 75, axis=0)
        iqr = q3 - q1
        self.lower_bounds_ = q1 - 1.5 * iqr
        self.upper_bounds_ = q3 + 1.5 * iqr
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=float)
        return np.clip(values, self.lower_bounds_, self.upper_bounds_)
