"""
Data preprocessing module for Sleep Disorder Detection.

Responsible for:
- Validating raw schema and expected columns.
- Excluding identifier columns (Person ID).
- Cleaning and normalizing categorical entries (e.g. 'Normal Weight' -> 'Normal').
- Parsing complex composite strings (Blood Pressure -> BP_Systolic, BP_Diastolic).
- Preserving literal target string 'None' as a valid target category.
- Validating target classes.
- Providing train-test splitting utilities prior to pipeline fitting.
"""

from typing import Tuple, Optional, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

EXPECTED_COLUMNS: List[str] = [
    "Person ID",
    "Gender",
    "Age",
    "Occupation",
    "Sleep Duration",
    "Quality of Sleep",
    "Physical Activity Level",
    "Stress Level",
    "BMI Category",
    "Blood Pressure",
    "Heart Rate",
    "Daily Steps",
    "Sleep Disorder",
]

TARGET_COL: str = "Sleep Disorder"
ID_COL: str = "Person ID"

TARGET_CLASSES: List[str] = ["None", "Insomnia", "Sleep Apnea"]


def validate_columns(df: pd.DataFrame, is_inference: bool = False) -> None:
    """
    Validates that the required columns are present in the dataframe.

    Args:
        df: Input DataFrame to validate.
        is_inference: If True, TARGET_COL and ID_COL are optional,
                      and Blood Pressure can be provided as compound 'Blood Pressure'
                      or pre-split ('BP_Systolic', 'BP_Diastolic').

    Raises:
        ValueError: If any required column is missing.
    """
    if is_inference:
        core_features = [
            "Gender",
            "Age",
            "Occupation",
            "Sleep Duration",
            "Quality of Sleep",
            "Physical Activity Level",
            "Stress Level",
            "BMI Category",
            "Heart Rate",
            "Daily Steps",
        ]
        missing = [c for c in core_features if c not in df.columns]

        has_compound_bp = "Blood Pressure" in df.columns
        has_split_bp = ("BP_Systolic" in df.columns) and ("BP_Diastolic" in df.columns)
        if not (has_compound_bp or has_split_bp):
            missing.append("Blood Pressure (or BP_Systolic and BP_Diastolic)")

        if missing:
            raise ValueError(f"Missing required column(s) in dataset: {missing}")
    else:
        expected = EXPECTED_COLUMNS
        missing = [c for c in expected if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required column(s) in dataset: {missing}")


def normalize_bmi_category(series: pd.Series) -> pd.Series:
    """
    Normalizes BMI Category entries, specifically consolidating 'Normal Weight' into 'Normal'.

    Args:
        series: Raw BMI Category pandas Series.

    Returns:
        Normalized BMI Category Series.
    """
    cleaned = series.astype(str).str.strip()
    return cleaned.replace({"Normal Weight": "Normal"})


def parse_blood_pressure(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Parses 'Systolic/Diastolic' Blood Pressure string into two numeric float series.

    Args:
        series: Raw Blood Pressure strings (e.g., '126/83').

    Returns:
        Tuple of (BP_Systolic, BP_Diastolic) numeric Series.

    Raises:
        ValueError: If Blood Pressure values cannot be parsed into systolic/diastolic components.
    """
    if series.isnull().any():
        raise ValueError("Blood Pressure series contains null values that cannot be parsed.")

    # Split string by slash
    split_bp = series.astype(str).str.strip().str.split("/", expand=True)
    if split_bp.shape[1] != 2:
        raise ValueError("Blood Pressure strings must follow the 'Systolic/Diastolic' format (e.g., '120/80').")

    try:
        systolic = pd.to_numeric(split_bp[0].str.strip(), errors="raise").astype(float)
        diastolic = pd.to_numeric(split_bp[1].str.strip(), errors="raise").astype(float)
    except Exception as e:
        raise ValueError(f"Failed to parse numeric Blood Pressure values: {e}")

    # Validate reasonable human physiological bounds
    if (systolic <= 0).any() or (diastolic <= 0).any():
        raise ValueError("Blood pressure values must be positive numbers.")

    return systolic, diastolic


def clean_and_prepare_data(
    data: pd.DataFrame,
    is_inference: bool = False,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Performs initial data verification, column exclusion, typo correction, and feature extraction.

    Transforms:
    1. Validates presence of expected columns.
    2. Drops Person ID from feature set.
    3. Normalizes BMI Category ('Normal Weight' -> 'Normal').
    4. Splits Blood Pressure into BP_Systolic and BP_Diastolic, drops Blood Pressure.
    5. Preserves literal 'None' string in Sleep Disorder target.
    6. Validates target categories against TARGET_CLASSES.

    Args:
        data: Input raw DataFrame.
        is_inference: True if processing data for prediction without target label.

    Returns:
        Tuple of (X, y) where X is the feature DataFrame and y is the target Series (or None).

    Raises:
        ValueError: If schema or target validation fails.
    """
    df = data.copy()

    # 1. Validate Schema
    validate_columns(df, is_inference=is_inference)

    # 2. Exclude Person ID
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])

    # 3. Clean Target (if present)
    y: Optional[pd.Series] = None
    if TARGET_COL in df.columns and not is_inference:
        # Preserve literal 'None' string. In case of actual NaN, map to 'None'
        target_series = df[TARGET_COL].fillna("None").astype(str).str.strip()
        invalid_classes = set(target_series.unique()) - set(TARGET_CLASSES)
        if invalid_classes:
            raise ValueError(
                f"Unexpected target classes encountered: {invalid_classes}. Expected only {TARGET_CLASSES}."
            )
        y = target_series
        df = df.drop(columns=[TARGET_COL])

    # 4. Normalize BMI Category
    if "BMI Category" in df.columns:
        df["BMI Category"] = normalize_bmi_category(df["BMI Category"])

    # 5. Parse Blood Pressure
    if "Blood Pressure" in df.columns:
        systolic, diastolic = parse_blood_pressure(df["Blood Pressure"])
        df["BP_Systolic"] = systolic
        df["BP_Diastolic"] = diastolic
        df = df.drop(columns=["Blood Pressure"])
    elif "BP_Systolic" in df.columns and "BP_Diastolic" in df.columns:
        df["BP_Systolic"] = pd.to_numeric(df["BP_Systolic"], errors="raise").astype(float)
        df["BP_Diastolic"] = pd.to_numeric(df["BP_Diastolic"], errors="raise").astype(float)
        if (df["BP_Systolic"] <= 0).any() or (df["BP_Diastolic"] <= 0).any():
            raise ValueError("Blood pressure values must be positive numbers.")

    return df, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits data into train and test sets prior to fitting any transformer or pipeline.
    This guarantees zero data leakage from test to train.

    Args:
        X: Feature DataFrame.
        y: Target label Series.
        test_size: Fraction of samples to hold out for testing.
        random_state: Reproducibility seed.
        stratify: Whether to preserve class proportions across splits.

    Returns:
        X_train, X_test, y_train, y_test
    """
    strat = y if stratify else None
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=strat,
    )
