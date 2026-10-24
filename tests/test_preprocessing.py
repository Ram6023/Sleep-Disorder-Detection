"""
Unit tests for data preprocessing, feature engineering, and transformation pipelines.
"""

import os
import pytest
import pandas as pd
import numpy as np

from ml.preprocess import (
    clean_and_prepare_data,
    validate_columns,
    normalize_bmi_category,
    parse_blood_pressure,
    split_data,
    EXPECTED_COLUMNS,
    TARGET_CLASSES,
)
from ml.feature_engineering import FeatureEngineer
from ml.pipeline import (
    build_preprocessor,
    create_full_pipeline,
    TargetLabelEncoder,
    TARGET_LABEL_MAP,
)


@pytest.fixture
def sample_raw_data() -> pd.DataFrame:
    """Provides a synthetic valid subset matching raw dataset schema."""
    return pd.DataFrame(
        {
            "Person ID": [1, 2, 3, 4, 5],
            "Gender": ["Male", "Female", "Male", "Female", "Male"],
            "Age": [27, 29, 35, 45, 52],
            "Occupation": ["Software Engineer", "Nurse", "Doctor", "Teacher", "Lawyer"],
            "Sleep Duration": [6.1, 6.5, 7.5, 5.8, 7.2],
            "Quality of Sleep": [6, 5, 8, 4, 7],
            "Physical Activity Level": [42, 40, 70, 30, 60],
            "Stress Level": [6, 7, 4, 8, 5],
            "BMI Category": ["Overweight", "Normal Weight", "Normal", "Obese", "Normal"],
            "Blood Pressure": ["126/83", "132/87", "120/80", "140/90", "125/82"],
            "Heart Rate": [77, 80, 70, 85, 72],
            "Daily Steps": [4200, 4000, 8000, 3000, 7000],
            "Sleep Disorder": ["None", "Sleep Apnea", "None", "Insomnia", "Sleep Apnea"],
        }
    )


def test_expected_columns_verification(sample_raw_data: pd.DataFrame):
    """Verifies that all 13 expected columns are checked and missing columns raise ValueError."""
    # Successful validation on complete data
    validate_columns(sample_raw_data)

    # Missing column raises ValueError
    incomplete_df = sample_raw_data.drop(columns=["Age"])
    with pytest.raises(ValueError, match="Missing required column"):
        validate_columns(incomplete_df)


def test_person_id_exclusion(sample_raw_data: pd.DataFrame):
    """Verifies that Person ID is strictly excluded from feature matrix X."""
    X, y = clean_and_prepare_data(sample_raw_data)
    assert "Person ID" not in X.columns
    assert X.shape[1] == 12  # 13 minus Person ID and Sleep Disorder, plus BP split (13 - 1 - 1 + 1 = 12)


def test_bmi_category_normalization():
    """Verifies 'Normal Weight' is consolidated into 'Normal' while preserving other categories."""
    raw_bmi = pd.Series(["Normal Weight", "Normal", "Overweight", "Obese", " Normal Weight "])
    normalized = normalize_bmi_category(raw_bmi)
    expected = ["Normal", "Normal", "Overweight", "Obese", "Normal"]
    assert list(normalized) == expected


def test_blood_pressure_parsing():
    """Verifies parsing composite BP string into numeric systolic and diastolic values."""
    bp_series = pd.Series(["120/80", "135/85", "140/90"])
    systolic, diastolic = parse_blood_pressure(bp_series)

    assert systolic.dtype == float
    assert diastolic.dtype == float
    assert list(systolic) == [120.0, 135.0, 140.0]
    assert list(diastolic) == [80.0, 85.0, 90.0]


def test_blood_pressure_invalid_format_raises_error():
    """Verifies that malformed BP strings raise an informative ValueError."""
    with pytest.raises(ValueError, match="Systolic/Diastolic"):
        parse_blood_pressure(pd.Series(["120-80"]))

    with pytest.raises(ValueError, match="positive numbers"):
        parse_blood_pressure(pd.Series(["-120/80"]))


def test_target_preserves_none_as_valid_class(sample_raw_data: pd.DataFrame):
    """Verifies that literal string 'None' is treated as a valid target category."""
    X, y = clean_and_prepare_data(sample_raw_data)
    assert y is not None
    assert "None" in y.values
    assert set(y.unique()).issubset(set(TARGET_CLASSES))
    assert not y.isnull().any()


def test_invalid_target_class_raises_error(sample_raw_data: pd.DataFrame):
    """Verifies that unauthorized target categories raise ValueError."""
    bad_data = sample_raw_data.copy()
    bad_data.loc[0, "Sleep Disorder"] = "UnknownCondition"
    with pytest.raises(ValueError, match="Unexpected target classes encountered"):
        clean_and_prepare_data(bad_data)


def test_split_before_fitting_preserves_leakage_barrier(sample_raw_data: pd.DataFrame):
    """Verifies that split_data partitions data with no overlapping indexes."""
    X, y = clean_and_prepare_data(sample_raw_data)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.4, random_state=42, stratify=False)

    # Disjoint indexes (no data leakage)
    assert set(X_train.index).isdisjoint(set(X_test.index))
    assert len(X_train) + len(X_test) == len(sample_raw_data)


def test_feature_engineering_transformer(sample_raw_data: pd.DataFrame):
    """Verifies that FeatureEngineer computes pulse pressure, ratios, and interactions safely."""
    X, _ = clean_and_prepare_data(sample_raw_data)
    fe = FeatureEngineer(add_ratios=True)
    X_eng = fe.fit_transform(X)

    assert "BP_Pulse_Pressure" in X_eng.columns
    assert "BP_Mean_Arterial" in X_eng.columns
    assert "Sleep_to_Stress_Ratio" in X_eng.columns
    assert "Sleep_Quality_Index" in X_eng.columns
    assert "Activity_to_Step_Ratio" in X_eng.columns

    # Verify mathematical accuracy for first row: 126/83, Pulse Pressure = 43
    assert X_eng.loc[0, "BP_Pulse_Pressure"] == 43.0
    assert not X_eng.isnull().any().any()


def test_reusable_pipeline_and_unknown_category_handling(sample_raw_data: pd.DataFrame):
    """
    Tests fitting pipeline on training set and transforming test set containing unseen categorical values.
    Verifies OneHotEncoder(handle_unknown='ignore') executes without failure.
    """
    X, y = clean_and_prepare_data(sample_raw_data)
    X_train = X.iloc[:3].copy()  # Occupations: Software Engineer, Nurse, Doctor
    X_test = X.iloc[3:].copy()   # Occupations: Teacher, Lawyer (unseen in X_train!)

    pipeline = build_preprocessor(include_feature_engineering=True)
    X_train_trans = pipeline.fit_transform(X_train)
    X_test_trans = pipeline.transform(X_test)

    assert isinstance(X_train_trans, np.ndarray)
    assert isinstance(X_test_trans, np.ndarray)
    assert not np.isnan(X_train_trans).any()
    assert not np.isnan(X_test_trans).any()
    assert X_train_trans.shape[1] == X_test_trans.shape[1]


def test_target_label_encoder():
    """Verifies consistent encoding and exact inverse transformation for all target classes."""
    encoder = TargetLabelEncoder()
    targets = pd.Series(["None", "Insomnia", "Sleep Apnea", "None"])
    encoded = encoder.fit_transform(targets)

    assert list(encoded) == [0, 1, 2, 0]
    decoded = encoder.inverse_transform(encoded)
    assert list(decoded) == list(targets)


def test_real_dataset_loading_and_pipeline():
    """Integration test loading the real dataset from data/ directory."""
    data_path = os.path.join("data", "Sleep_health_and_lifestyle_dataset.csv")
    if not os.path.exists(data_path):
        pytest.skip("Dataset file not found at data/ path.")

    raw_df = pd.read_csv(data_path)
    X, y = clean_and_prepare_data(raw_df)

    assert len(X) == 374
    assert len(y) == 374
    assert "Person ID" not in X.columns
    assert "Normal Weight" not in X["BMI Category"].values
    assert set(y.unique()) == {"None", "Insomnia", "Sleep Apnea"}

    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42, stratify=True)
    pipeline = build_preprocessor(include_feature_engineering=True)

    X_train_proc = pipeline.fit_transform(X_train)
    X_test_proc = pipeline.transform(X_test)

    assert X_train_proc.shape[0] == len(X_train)
    assert X_test_proc.shape[0] == len(X_test)
    assert not np.isnan(X_train_proc).any()
    assert not np.isnan(X_test_proc).any()
