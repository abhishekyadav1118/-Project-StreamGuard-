"""
StreamGuard: Custom Transformers for OTT Churn & CLV Machine Learning Pipeline.
Defines domain-specific feature engineering and IQR clipping transformers
for leak-free integration with scikit-learn pipelines.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class OTTFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Generates domain-specific features for OTT subscriber intelligence:
    1. Engagement_Metric = Watch_Hours * (Login_Frequency / 30.0)
    2. Watch_to_Fee_Ratio = Watch_Hours / max(Monthly_Fee, 0.01)
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        watch = pd.to_numeric(X["Watch_Hours"], errors="coerce").fillna(0)
        login = pd.to_numeric(X["Login_Frequency"], errors="coerce").fillna(0)
        fee = pd.to_numeric(X["Monthly_Fee"], errors="coerce").fillna(10)

        # 1. Engagement Index: total viewing scaled by monthly login frequency
        X["Engagement_Metric"] = np.round(watch * (login / 30.0), 3)

        # 2. Engagement Efficiency: hours watched per dollar spent
        X["Watch_to_Fee_Ratio"] = np.round(watch / np.maximum(fee, 0.01), 3)
        return X


class IQRClipper(BaseEstimator, TransformerMixin):
    """
    Clips numerical values to IQR fences (Q1 - multiplier*IQR, Q3 + multiplier*IQR)
    learned strictly on the training set during fit() to prevent data leakage.
    """

    def __init__(self, columns=None, multiplier=1.5):
        self.columns = columns
        self.multiplier = multiplier

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        cols = self.columns or X.columns.tolist()
        self.bounds_ = {}
        for c in cols:
            s = pd.to_numeric(X[c], errors="coerce").dropna()
            if len(s) == 0:
                continue
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            self.bounds_[c] = (q1 - self.multiplier * iqr, q3 + self.multiplier * iqr)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for c, (lower, upper) in self.bounds_.items():
            if c in X.columns:
                X[c] = pd.to_numeric(X[c], errors="coerce").clip(lower, upper)
        return X
