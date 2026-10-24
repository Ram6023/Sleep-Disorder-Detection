"""
Preprocessing and transformation pipeline module for Sleep Disorder Detection.

Provides reusable Scikit-Learn Pipeline and ColumnTransformer configurations:
- Training-fitted SimpleImputer for numerical and categorical attributes.
- Robust StandardScaler for continuous indicators.
- OneHotEncoder(handle_unknown="ignore") for unseen categorical levels at inference.
- Full end-to-end composite pipeline integrating FeatureEngineer, ColumnTransformer, and downstream models.
"""

from typing import List, Optional, Tuple, Dict, Any, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from ml.feature_engineering import FeatureEngineer

# Default column definitions after clean_and_prepare_data
BASE_NUMERIC_FEATURES: List[str] = [
    "Age",
    "Sleep Duration",
    "Quality of Sleep",
    "Physical Activity Level",
    "Stress Level",
    "Heart Rate",
    "Daily Steps",
    "BP_Systolic",
    "BP_Diastolic",
]

ENGINEERED_NUMERIC_FEATURES: List[str] = [
    "BP_Pulse_Pressure",
    "BP_Mean_Arterial",
    "Sleep_to_Stress_Ratio",
    "Sleep_Quality_Index",
    "Activity_to_Step_Ratio",
]

CATEGORICAL_FEATURES: List[str] = [
    "Gender",
    "Occupation",
    "BMI Category",
]

# Strict 3-class target mapping
TARGET_LABEL_MAP: Dict[str, int] = {
    "None": 0,
    "Insomnia": 1,
    "Sleep Apnea": 2,
}

INVERSE_TARGET_MAP: Dict[int, str] = {v: k for k, v in TARGET_LABEL_MAP.items()}


class TargetLabelEncoder:
    """
    Encoder for multi-class target labels preserving 'None' as class 0.
    """

    def __init__(self, mapping: Optional[Dict[str, int]] = None) -> None:
        self.mapping = mapping or TARGET_LABEL_MAP
        self.inverse_mapping = {v: k for k, v in self.mapping.items()}
        self.classes_ = np.array(list(self.mapping.keys()))

    def fit(self, y: pd.Series) -> "TargetLabelEncoder":
        return self

    def transform(self, y: pd.Series) -> np.ndarray:
        clean_y = y.astype(str).str.strip()
        unknown = set(clean_y.unique()) - set(self.mapping.keys())
        if unknown:
            raise ValueError(f"Unknown target labels: {unknown}. Expected {list(self.mapping.keys())}")
        return clean_y.map(self.mapping).to_numpy(dtype=int)

    def fit_transform(self, y: pd.Series) -> np.ndarray:
        return self.fit(y).transform(y)

    def inverse_transform(self, y: Union[np.ndarray, List[int]]) -> np.ndarray:
        return np.array([self.inverse_mapping[int(val)] for val in y])


def build_column_transformer(
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
) -> ColumnTransformer:
    """
    Constructs a ColumnTransformer with training-fitted imputers, standard scalers,
    and unknown-handling OneHotEncoders.

    Args:
        numeric_features: List of numeric column names.
        categorical_features: List of categorical column names.

    Returns:
        Configured ColumnTransformer instance.
    """
    num_cols = numeric_features if numeric_features is not None else (BASE_NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES)
    cat_cols = categorical_features if categorical_features is not None else CATEGORICAL_FEATURES

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def build_preprocessor(
    include_feature_engineering: bool = True,
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
) -> Pipeline:
    """
    Builds the complete feature processing pipeline.

    Args:
        include_feature_engineering: If True, FeatureEngineer is placed as the first step.
        numeric_features: Optional list of numeric columns.
        categorical_features: Optional list of categorical columns.

    Returns:
        Scikit-learn Pipeline instance.
    """
    steps: List[Tuple[str, Any]] = []

    if include_feature_engineering:
        steps.append(("feature_engineer", FeatureEngineer(add_ratios=True)))
        num_cols = numeric_features if numeric_features is not None else (BASE_NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES)
    else:
        num_cols = numeric_features if numeric_features is not None else BASE_NUMERIC_FEATURES

    cat_cols = categorical_features if categorical_features is not None else CATEGORICAL_FEATURES
    col_transformer = build_column_transformer(numeric_features=num_cols, categorical_features=cat_cols)
    steps.append(("col_transformer", col_transformer))

    return Pipeline(steps=steps)


def create_full_pipeline(
    model: Optional[BaseEstimator] = None,
    include_feature_engineering: bool = True,
) -> Pipeline:
    """
    Constructs an end-to-end pipeline combining preprocessing and a downstream estimator.

    Args:
        model: Optional scikit-learn classifier.
        include_feature_engineering: Whether to compute derived lifestyle features.

    Returns:
        Composite Pipeline ready for training / cross-validation.
    """
    preprocessor = build_preprocessor(include_feature_engineering=include_feature_engineering)
    steps: List[Tuple[str, Any]] = [("preprocessor", preprocessor)]

    if model is not None:
        steps.append(("classifier", model))

    return Pipeline(steps=steps)
