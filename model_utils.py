
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """Feature engineering shared by training and prediction."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        x = X.copy()

        x["AvgMonthlySpend"] = (
            x["TotalCharges"] / (x["tenure"] + 1)
        )

        x["TenureGroup"] = pd.cut(
            x["tenure"],
            bins=[-0.001, 12, 24, 48, np.inf],
            labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"],
            include_lowest=True
        ).astype("object")

        x["ChargeLevel"] = pd.cut(
            x["MonthlyCharges"],
            bins=[-np.inf, 40, 70, np.inf],
            labels=["Low", "Medium", "High"]
        ).astype("object")

        return x
