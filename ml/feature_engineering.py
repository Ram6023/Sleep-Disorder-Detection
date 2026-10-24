"""
Feature engineering module for Sleep Disorder Detection.

Implements a Scikit-Learn compatible transformer to derive physiological and
lifestyle indicators without introducing data leakage.
"""

from typing import List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer for engineering domain-specific features
    from sleep, stress, activity, and cardiovascular measurements.

    Features engineered:
    - BP_Pulse_Pressure: Systolic BP - Diastolic BP.
    - BP_Mean_Arterial: Diastolic BP + (Pulse Pressure / 3).
    - Sleep_to_Stress_Ratio: Sleep Duration / (Stress Level + 1e-5).
    - Sleep_Quality_Index: Sleep Duration * Quality of Sleep.
    - Activity_to_Step_Ratio: Daily Steps / (Physical Activity Level + 1e-5).
    """

    def __init__(self, add_ratios: bool = True) -> None:
        self.add_ratios = add_ratios
        self.feature_names_in_: Optional[List[str]] = None
        self.engineered_feature_names_: List[str] = [
            "BP_Pulse_Pressure",
            "BP_Mean_Arterial",
            "Sleep_to_Stress_Ratio",
            "Sleep_Quality_Index",
            "Activity_to_Step_Ratio",
        ]

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[pd.Series] = None) -> "FeatureEngineer":
        """
        Fits the transformer by recording input feature names.

        Args:
            X: Input dataframe or array.
            y: Ignored (present for scikit-learn API compatibility).

        Returns:
            self
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Transforms features by computing derived clinical lifestyle indicators.

        Args:
            X: Input feature DataFrame or array.

        Returns:
            DataFrame containing original and newly engineered columns.
        """
        if not isinstance(X, pd.DataFrame):
            if self.feature_names_in_ is not None and len(self.feature_names_in_) == X.shape[1]:
                df = pd.DataFrame(X, columns=self.feature_names_in_).copy()
            else:
                raise ValueError("X must be a pandas DataFrame or match the feature names seen during fit.")
        else:
            df = X.copy()

        if not self.add_ratios:
            return df

        # 1. Pulse Pressure (Systolic - Diastolic)
        if "BP_Systolic" in df.columns and "BP_Diastolic" in df.columns:
            pulse_pressure = df["BP_Systolic"] - df["BP_Diastolic"]
            df["BP_Pulse_Pressure"] = pulse_pressure
            # Mean Arterial Pressure (MAP) = Diastolic + (1/3 * Pulse Pressure)
            df["BP_Mean_Arterial"] = df["BP_Diastolic"] + (pulse_pressure / 3.0)

        # 2. Sleep vs. Stress interactions
        if "Sleep Duration" in df.columns and "Stress Level" in df.columns:
            df["Sleep_to_Stress_Ratio"] = df["Sleep Duration"] / (df["Stress Level"].astype(float) + 1e-5)

        # 3. Overall Sleep Score (Duration * Subjective Quality)
        if "Sleep Duration" in df.columns and "Quality of Sleep" in df.columns:
            df["Sleep_Quality_Index"] = df["Sleep Duration"] * df["Quality of Sleep"].astype(float)

        # 4. Movement efficiency (Steps per minute of physical activity)
        if "Daily Steps" in df.columns and "Physical Activity Level" in df.columns:
            df["Activity_to_Step_Ratio"] = df["Daily Steps"].astype(float) / (
                df["Physical Activity Level"].astype(float) + 1e-5
            )

        return df

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """
        Returns feature names after transformation.
        """
        base = input_features if input_features is not None else (self.feature_names_in_ or [])
        if self.add_ratios:
            return list(base) + [f for f in self.engineered_feature_names_ if f not in base]
        return list(base)
