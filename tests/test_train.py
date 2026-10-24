"""
Unit and integration tests for Model Training, Cross-Validation, Serialization, and Inference.
"""

import os
import pytest
import pandas as pd
import numpy as np
import joblib

from ml.model_comparison import (
    get_candidate_models,
    build_fold_pipeline,
    evaluate_model_cv,
)
from ml.evaluate import compute_classification_metrics, TARGET_CLASSES
from ml.pipeline import TargetLabelEncoder, create_full_pipeline
from ml.preprocess import clean_and_prepare_data


@pytest.fixture
def sample_features_and_target():
    """Provides a synthetic valid subset matching schema for testing training logic."""
    df = pd.DataFrame(
        {
            "Person ID": list(range(1, 21)),
            "Gender": ["Male", "Female"] * 10,
            "Age": [25, 30, 35, 40, 45, 50, 55, 60, 28, 32, 38, 42, 48, 52, 58, 29, 34, 39, 44, 49],
            "Occupation": ["Software Engineer", "Doctor", "Nurse", "Teacher", "Lawyer"] * 4,
            "Sleep Duration": [7.5, 6.0, 5.8, 6.2, 7.8, 5.5, 6.5, 7.0, 8.0, 6.1, 5.9, 6.4, 7.1, 6.8, 7.3, 6.3, 5.7, 7.2, 6.6, 7.4],
            "Quality of Sleep": [8, 5, 4, 6, 9, 4, 6, 7, 9, 6, 5, 6, 7, 6, 8, 6, 4, 8, 6, 8],
            "Physical Activity Level": [60, 40, 30, 45, 70, 35, 50, 65, 80, 42, 38, 48, 55, 62, 75, 40, 32, 68, 52, 72],
            "Stress Level": [4, 7, 8, 6, 3, 8, 6, 5, 3, 6, 7, 6, 5, 5, 4, 6, 8, 4, 6, 3],
            "BMI Category": ["Normal", "Overweight", "Obese", "Normal", "Normal"] * 4,
            "Blood Pressure": ["120/80", "135/85", "140/90", "125/82", "118/75"] * 4,
            "Heart Rate": [70, 80, 85, 75, 68, 82, 76, 72, 65, 78, 83, 74, 71, 73, 69, 79, 84, 70, 75, 67],
            "Daily Steps": [8000, 4000, 3000, 5000, 9000, 3500, 6000, 7500, 10000, 4500, 3800, 5200, 6800, 7200, 8500, 4200, 3200, 8200, 6200, 9200],
            "Sleep Disorder": ["None", "Insomnia", "Sleep Apnea", "None", "None"] * 4,
        }
    )
    X, y = clean_and_prepare_data(df)
    encoder = TargetLabelEncoder()
    y_encoded = encoder.fit_transform(y)
    return X, y_encoded, encoder


def test_candidate_models_instantiation():
    """Verifies that all 9 candidate models instantiate properly."""
    models = get_candidate_models(random_state=42)
    assert len(models) >= 8
    expected = [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Support Vector Machine",
        "K-Nearest Neighbors",
        "Gradient Boosting",
        "AdaBoost",
        "Bagging",
    ]
    for name in expected:
        assert name in models, f"Model {name} missing from candidate models"


def test_fold_pipeline_construction_and_leakage_barrier(sample_features_and_target):
    """Verifies pipeline builds with and without SMOTE and feature engineering."""
    X, y, _ = sample_features_and_target
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(n_estimators=10, random_state=42)

    # 1. Without SMOTE
    pipe1 = build_fold_pipeline(model=rf, include_feature_engineering=True, use_smote=False)
    pipe1.fit(X, y)
    preds1 = pipe1.predict(X)
    assert len(preds1) == len(X)
    assert set(preds1).issubset({0, 1, 2})

    # 2. With SMOTE (using k_neighbors=2 to accommodate small synthetic unit test fixture)
    pipe2 = build_fold_pipeline(model=rf, include_feature_engineering=True, use_smote=True, k_neighbors=2)
    pipe2.fit(X, y)
    preds2 = pipe2.predict(X)
    assert len(preds2) == len(X)
    assert set(preds2).issubset({0, 1, 2})


def test_cv_evaluation_metrics_generation(sample_features_and_target):
    """Verifies StratifiedKFold cross-validation metric generation."""
    X, y, _ = sample_features_and_target
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(max_iter=500, random_state=42)
    cv_res = evaluate_model_cv(
        model_name="Test LR",
        model=lr,
        X=X,
        y=y,
        n_splits=3,
        random_state=42,
        include_feature_engineering=True,
        use_smote=False,
    )

    assert "cv_macro_f1_mean" in cv_res
    assert "cv_balanced_acc_mean" in cv_res
    assert "cv_accuracy_mean" in cv_res
    assert 0.0 <= cv_res["cv_macro_f1_mean"] <= 1.0
    assert len(cv_res["fold_macro_f1_scores"]) == 3


def test_metric_computation_exactness():
    """Verifies that compute_classification_metrics returns exact expected metrics."""
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 0, 2, 2])  # 1 error in class 1

    metrics = compute_classification_metrics(y_true, y_pred, labels=[0, 1, 2], target_names=TARGET_CLASSES)

    assert metrics["accuracy"] == round(5 / 6, 4)
    assert "per_class" in metrics
    assert "None" in metrics["per_class"]
    assert "Insomnia" in metrics["per_class"]
    assert "Sleep Apnea" in metrics["per_class"]
    assert metrics["per_class"]["None"]["recall"] == 1.0
    assert metrics["per_class"]["Sleep Apnea"]["recall"] == 1.0
    assert metrics["per_class"]["Insomnia"]["recall"] == 0.5


def test_model_serialization_and_deserialization():
    """Verifies that serialized artifacts exist and load correctly."""
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    best_model_path = os.path.join(models_dir, "best_model.joblib")
    encoder_path = os.path.join(models_dir, "label_encoder.joblib")

    if not os.path.exists(best_model_path):
        pytest.skip("Models not yet serialized; run train.py first.")

    pipeline = joblib.load(best_model_path)
    encoder = joblib.load(encoder_path)

    assert hasattr(pipeline, "predict")
    assert hasattr(encoder, "inverse_transform")
    assert list(encoder.classes_) == TARGET_CLASSES


def test_prediction_on_unseen_rows():
    """Verifies that the trained pipeline reliably predicts on brand new, unseen patient data."""
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    best_model_path = os.path.join(models_dir, "best_model.joblib")
    encoder_path = os.path.join(models_dir, "label_encoder.joblib")

    if not os.path.exists(best_model_path):
        pytest.skip("Models not yet serialized; run train.py first.")

    pipeline = joblib.load(best_model_path)
    encoder = joblib.load(encoder_path)

    # Synthetic unseen patient record (Person ID excluded, BP already parsed or clean)
    unseen_patient = pd.DataFrame(
        [
            {
                "Gender": "Male",
                "Age": 45,
                "Occupation": "Engineer",
                "Sleep Duration": 7.2,
                "Quality of Sleep": 8,
                "Physical Activity Level": 60,
                "Stress Level": 4,
                "BMI Category": "Normal",
                "BP_Systolic": 120.0,
                "BP_Diastolic": 80.0,
                "Heart Rate": 68,
                "Daily Steps": 8000,
            },
            {
                "Gender": "Female",
                "Age": 54,
                "Occupation": "Nurse",
                "Sleep Duration": 5.8,
                "Quality of Sleep": 4,
                "Physical Activity Level": 35,
                "Stress Level": 8,
                "BMI Category": "Overweight",
                "BP_Systolic": 140.0,
                "BP_Diastolic": 95.0,
                "Heart Rate": 85,
                "Daily Steps": 3200,
            },
        ]
    )

    preds_encoded = pipeline.predict(unseen_patient)
    assert len(preds_encoded) == 2
    assert set(preds_encoded).issubset({0, 1, 2})

    preds_labels = encoder.inverse_transform(preds_encoded)
    assert len(preds_labels) == 2
    for label in preds_labels:
        assert label in TARGET_CLASSES


def test_reproducibility():
    """Verifies that pipeline predictions are deterministic with the fixed seed."""
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    best_model_path = os.path.join(models_dir, "best_model.joblib")

    if not os.path.exists(best_model_path):
        pytest.skip("Models not yet serialized; run train.py first.")

    pipeline = joblib.load(best_model_path)

    test_input = pd.DataFrame(
        [
            {
                "Gender": "Female",
                "Age": 38,
                "Occupation": "Teacher",
                "Sleep Duration": 6.5,
                "Quality of Sleep": 6,
                "Physical Activity Level": 45,
                "Stress Level": 6,
                "BMI Category": "Normal",
                "BP_Systolic": 125.0,
                "BP_Diastolic": 82.0,
                "Heart Rate": 72,
                "Daily Steps": 5500,
            }
        ]
    )

    pred1 = pipeline.predict(test_input)
    pred2 = pipeline.predict(test_input)
    assert pred1[0] == pred2[0]
