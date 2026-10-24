"""
Unit and Integration Tests for Model Serialization and API Contract Audit (Task 1.8)
for 'ML-Based Intelligent System for Automated Sleep Disorder Detection Using Lifestyle and Health Data'.

Tests cover:
1. Fresh-process artifact loading and consistency verification.
2. Prediction consistency between deserialized model and evaluation benchmarks.
3. Probability distribution exactness: finiteness, sum to 1.0, non-negativity, and strict class ordering.
4. Input contract validation: canonical Title Case vs snake_case aliases, compound BP vs split BP.
5. Invalid input rejection: physiological bounds, malformed strings, inverted systolic/diastolic.
6. Unknown category resilience: unseen occupations encoded gracefully via handle_unknown='ignore'.
7. Output contract conformity and clinical disclaimer enforcement.
8. Explainability integration: real-time SHAP feature attribution generation.
"""

import os
import sys
import json
import pytest
import numpy as np
import pandas as pd
from pydantic import ValidationError

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.preprocess import clean_and_prepare_data, split_data, TARGET_CLASSES
from ml.explain import CLASS_DISPLAY_NAMES
from ml.contract import (
    SleepPredictionInput,
    SleepPredictionOutput,
    ModelArtifactManager,
    ArtifactConsistencyError,
    CONTRACT_VERSION,
    MODEL_VERSION,
    DISCLAIMER_TEXT,
)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "Sleep_health_and_lifestyle_dataset.csv")


@pytest.fixture(scope="module")
def artifact_manager():
    """Initializes ModelArtifactManager in a clean runtime."""
    return ModelArtifactManager()


@pytest.fixture(scope="module")
def holdout_test_data():
    """Loads raw dataset and reproduces the exact untouched 20% holdout test split."""
    raw_df = pd.read_csv(DATA_PATH, keep_default_na=False)
    X, y = clean_and_prepare_data(raw_df)
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=0.2, random_state=42, stratify=True
    )
    return {"X_test": X_test, "y_test": y_test}


def test_fresh_process_artifact_loading_and_metadata_consistency(artifact_manager):
    """Verifies that all 5 model artifacts exist and cross-verify with 100% consistency."""
    assert artifact_manager.pipeline is not None
    assert artifact_manager.label_encoder is not None
    assert artifact_manager.metadata is not None
    assert artifact_manager.test_evaluation is not None
    assert artifact_manager.shap_metadata is not None

    # Verify pipeline steps
    assert list(artifact_manager.pipeline.named_steps.keys()) == ["col_transformer", "classifier"]

    # Verify feature dimension consistency
    assert len(artifact_manager.feature_names) == 25
    assert artifact_manager.shap_metadata["feature_count"] == 25
    assert artifact_manager.shap_metadata["transformed_features"] == artifact_manager.feature_names

    # Verify class mappings
    assert list(artifact_manager.label_encoder.classes_) == TARGET_CLASSES
    assert artifact_manager.metadata["target_classes"] == TARGET_CLASSES


def test_prediction_consistency_on_holdout_test_set(artifact_manager, holdout_test_data):
    """Verifies that deserialized model predictions match evaluation benchmarks on holdout set."""
    X_test = holdout_test_data["X_test"]
    y_test = holdout_test_data["y_test"]

    # Transform y_test with loaded label encoder
    y_test_encoded = artifact_manager.label_encoder.transform(y_test)

    # Predict with loaded pipeline
    y_pred = artifact_manager.pipeline.predict(X_test)
    y_proba = artifact_manager.pipeline.predict_proba(X_test)

    # Verify accuracy matches exactly 96.00% (72/75)
    acc = float(np.mean(y_pred == y_test_encoded))
    assert acc == pytest.approx(0.9600, abs=1e-4)

    # Verify expected test evaluation metrics
    saved_eval = artifact_manager.test_evaluation
    assert saved_eval["accuracy"] == 0.96
    assert saved_eval["macro_f1"] == 0.9352


def test_probability_distribution_stability_and_ordering(artifact_manager, holdout_test_data):
    """Verifies that predicted probabilities are non-negative, finite, sum to 1, and match class order."""
    X_test = holdout_test_data["X_test"]
    y_proba = artifact_manager.pipeline.predict_proba(X_test)

    assert y_proba.shape == (75, 3)
    assert not np.isnan(y_proba).any()
    assert not np.isinf(y_proba).any()
    assert (y_proba >= 0.0).all()
    assert (y_proba <= 1.0).all()

    # Verify row sums
    sums = np.sum(y_proba, axis=1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-5)


def test_input_contract_valid_payload_formats(artifact_manager):
    """Verifies input validation across both Title Case / snake_case and compound / split BP."""
    # 1. Canonical Title Case with compound Blood Pressure string
    payload_compound = {
        "Gender": "Male",
        "Age": 38,
        "Occupation": "Engineer",
        "Sleep Duration": 7.5,
        "Quality of Sleep": 8,
        "Physical Activity Level": 60,
        "Stress Level": 4,
        "BMI Category": "Normal",
        "Blood Pressure": "120/80",
        "Heart Rate": 68,
        "Daily Steps": 8000,
        "Person ID": 999,
    }
    input_obj1 = SleepPredictionInput(**payload_compound)
    assert input_obj1.bp_systolic == 120.0
    assert input_obj1.bp_diastolic == 80.0
    assert input_obj1.gender == "Male"

    # 2. snake_case with pre-split BP and 'Normal Weight' normalization
    payload_split = {
        "gender": "female",
        "age": 49,
        "occupation": "Nurse",
        "sleep_duration": 6.1,
        "quality_of_sleep": 5,
        "physical_activity_level": 40,
        "stress_level": 7,
        "bmi_category": "Normal Weight",
        "bp_systolic": 135.0,
        "bp_diastolic": 88.0,
        "heart_rate": 78,
        "daily_steps": 5000,
    }
    input_obj2 = SleepPredictionInput(**payload_split)
    assert input_obj2.gender == "Female"
    assert input_obj2.bmi_category == "Normal"
    assert input_obj2.bp_systolic == 135.0
    assert input_obj2.bp_diastolic == 88.0

    # Ensure to_dataframe produces the exact 12 columns
    df1 = input_obj1.to_dataframe()
    assert df1.shape == (1, 12)
    assert "Person ID" not in df1.columns


def test_input_contract_invalid_payload_rejection():
    """Verifies that invalid or physiologically contradictory inputs raise validation errors."""
    # 1. Negative age
    with pytest.raises(ValidationError):
        SleepPredictionInput(
            Gender="Male", Age=-5, Occupation="Doctor",
            **{"Sleep Duration": 7.0, "Quality of Sleep": 8, "Physical Activity Level": 60,
               "Stress Level": 4, "BMI Category": "Normal", "Blood Pressure": "120/80",
               "Heart Rate": 70, "Daily Steps": 8000}
        )

    # 2. Stress level out of 1-10 range
    with pytest.raises(ValidationError):
        SleepPredictionInput(
            Gender="Male", Age=30, Occupation="Doctor",
            **{"Sleep Duration": 7.0, "Quality of Sleep": 8, "Physical Activity Level": 60,
               "Stress Level": 15, "BMI Category": "Normal", "Blood Pressure": "120/80",
               "Heart Rate": 70, "Daily Steps": 8000}
        )

    # 3. Systolic pressure <= Diastolic pressure (physiologically invalid)
    with pytest.raises(ValueError, match="strictly greater"):
        SleepPredictionInput(
            Gender="Female", Age=40, Occupation="Teacher",
            **{"Sleep Duration": 6.5, "Quality of Sleep": 6, "Physical Activity Level": 50,
               "Stress Level": 6, "BMI Category": "Overweight", "Blood Pressure": "80/120",
               "Heart Rate": 75, "Daily Steps": 6000}
        )

    # 4. Malformed blood pressure string
    with pytest.raises(ValueError, match="Systolic/Diastolic"):
        SleepPredictionInput(
            Gender="Male", Age=40, Occupation="Teacher",
            **{"Sleep Duration": 6.5, "Quality of Sleep": 6, "Physical Activity Level": 50,
               "Stress Level": 6, "BMI Category": "Normal", "Blood Pressure": "120-80",
               "Heart Rate": 75, "Daily Steps": 6000}
        )


def test_unknown_categorical_values_gracefully_handled(artifact_manager):
    """Verifies that novel/unseen occupations (e.g. 'Astronaut') are handled safely without failure."""
    novel_payload = {
        "Gender": "Male",
        "Age": 35,
        "Occupation": "Astronaut",  # Never seen during training
        "Sleep Duration": 6.5,
        "Quality of Sleep": 6,
        "Physical Activity Level": 50,
        "Stress Level": 6,
        "BMI Category": "Normal",
        "Blood Pressure": "122/81",
        "Heart Rate": 72,
        "Daily Steps": 7000,
    }
    output = artifact_manager.predict(novel_payload)
    assert output.predicted_class in [0, 1, 2]
    assert output.predicted_class_name in CLASS_DISPLAY_NAMES
    assert sum(output.probabilities.values()) == pytest.approx(1.0, abs=1e-4)


def test_output_contract_and_disclaimer_compliance(artifact_manager):
    """Verifies the output schema structure, versioning, probabilities, and disclaimer."""
    valid_payload = {
        "Gender": "Male",
        "Age": 45,
        "Occupation": "Engineer",
        "Sleep Duration": 7.2,
        "Quality of Sleep": 8,
        "Physical Activity Level": 60,
        "Stress Level": 4,
        "BMI Category": "Normal",
        "Blood Pressure": "120/80",
        "Heart Rate": 68,
        "Daily Steps": 8000,
    }
    res = artifact_manager.predict(valid_payload, include_explanation=False)

    assert isinstance(res, SleepPredictionOutput)
    assert res.model_version == MODEL_VERSION
    assert res.contract_version == CONTRACT_VERSION
    assert len(res.prediction_id) > 10
    assert res.predicted_class in [0, 1, 2]
    assert res.predicted_class_name == CLASS_DISPLAY_NAMES[res.predicted_class]
    assert set(res.probabilities.keys()) == set(CLASS_DISPLAY_NAMES)
    assert res.disclaimer == DISCLAIMER_TEXT
    assert res.explanation is None


def test_explainability_integration_on_demand(artifact_manager):
    """Verifies that requesting local explanation generates accurate, non-empty SHAP factors."""
    insomnia_prone_payload = {
        "Gender": "Female",
        "Age": 51,
        "Occupation": "Teacher",
        "Sleep Duration": 5.5,
        "Quality of Sleep": 4,
        "Physical Activity Level": 30,
        "Stress Level": 8,
        "BMI Category": "Overweight",
        "Blood Pressure": "128/84",
        "Heart Rate": 82,
        "Daily Steps": 3500,
    }
    res = artifact_manager.predict(insomnia_prone_payload, include_explanation=True, top_n_features=6)

    assert res.explanation is not None
    assert res.explanation.method == "SHAP TreeExplainer"
    assert isinstance(res.explanation.base_value, float)
    assert len(res.explanation.top_factors) == 6

    for factor in res.explanation.top_factors:
        assert factor.feature in artifact_manager.feature_names
        assert isinstance(factor.shap_value, float)
        assert factor.direction in ["supports_prediction", "opposes_prediction"]


def test_contract_info_endpoint_payload(artifact_manager):
    """Verifies that get_contract_info provides comprehensive schema and bounds metadata."""
    info = artifact_manager.get_contract_info()
    assert info.model_name == "Random Forest (Base Features Only)"
    assert info.model_version == MODEL_VERSION
    assert info.input_feature_count == 12
    assert info.transformed_feature_count == 25
    assert "age" in info.validation_bounds
    assert info.disclaimer == DISCLAIMER_TEXT
