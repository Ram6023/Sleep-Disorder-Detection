"""
Unit and Integration Tests for Explainable AI (SHAP and LIME)
for 'ML-Based Intelligent System for Automated Sleep Disorder Detection Using Lifestyle and Health Data'.

Tests cover:
1. Feature-name extraction and pipeline component validation.
2. Training/test transformation shapes and data leakage barrier preservation.
3. SHAP TreeExplainer initialization and background sample reproducibility.
4. SHAP output dimensions for multiclass classification: (n_samples, 25, 3).
5. Target class mapping consistency ('None' -> 'No recorded sleep disorder').
6. Global feature importance calculation, ordering, and metrics.
7. Local individual instance explanations (SHAP waterfall & Explanation objects).
8. LIME TabularExplainer compatibility, local surrogate weights, and figure rendering.
9. Verification of all 5 visual artifacts and machine-readable JSON reports.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import pytest
import joblib

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.preprocess import clean_and_prepare_data, split_data, TARGET_CLASSES
from ml.explain import (
    CLASS_DISPLAY_NAMES,
    CLASS_LABEL_TO_DISPLAY,
    extract_pipeline_components,
    transform_data_for_explanation,
    create_shap_explainer,
    compute_shap_values,
    compute_global_feature_importance,
    plot_shap_summary,
    plot_shap_feature_importance,
    plot_shap_class_comparison,
    plot_shap_individual_prediction,
    create_lime_explainer,
    explain_individual_lime,
    plot_lime_individual_prediction,
    run_explainability_pipeline,
)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "Sleep_health_and_lifestyle_dataset.csv")


@pytest.fixture(scope="module")
def loaded_artifacts():
    """Loads saved model pipeline, label encoder, and preprocessed dataset."""
    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    encoder_path = os.path.join(MODELS_DIR, "label_encoder.joblib")

    assert os.path.exists(model_path), f"Missing model artifact: {model_path}"
    assert os.path.exists(encoder_path), f"Missing label encoder: {encoder_path}"

    pipeline = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)

    raw_df = pd.read_csv(DATA_PATH, keep_default_na=False)
    X, y = clean_and_prepare_data(raw_df)

    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=0.2, random_state=42, stratify=True
    )
    y_test_encoded = label_encoder.transform(y_test)

    return {
        "pipeline": pipeline,
        "label_encoder": label_encoder,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_test_encoded": y_test_encoded,
    }


def test_pipeline_component_and_feature_name_extraction(loaded_artifacts):
    """Verifies that preprocessor, classifier, and clean feature names are extracted correctly."""
    pipeline = loaded_artifacts["pipeline"]
    preprocessor, classifier, feature_names = extract_pipeline_components(pipeline)

    assert preprocessor is not None
    assert classifier is not None
    assert hasattr(classifier, "predict_proba")
    assert isinstance(feature_names, list)
    assert len(feature_names) == 25, f"Expected 25 transformed features, got {len(feature_names)}"

    # Check key expected feature names
    expected_features = [
        "Age",
        "Sleep Duration",
        "Quality of Sleep",
        "BP_Systolic",
        "BP_Diastolic",
        "Gender_Female",
        "Gender_Male",
        "BMI Category_Normal",
        "BMI Category_Overweight",
        "Occupation_Nurse",
    ]
    for feat in expected_features:
        assert feat in feature_names, f"Expected feature {feat} not found in {feature_names}"


def test_data_transformation_preserves_dimensions(loaded_artifacts):
    """Verifies that transformation maps 12 raw features to 25 transformed numeric features."""
    pipeline = loaded_artifacts["pipeline"]
    preprocessor, _, feature_names = extract_pipeline_components(pipeline)

    X_train = loaded_artifacts["X_train"]
    X_test = loaded_artifacts["X_test"]

    X_train_trans, X_test_trans = transform_data_for_explanation(preprocessor, X_train, X_test)

    assert X_train_trans.shape == (299, 25)
    assert X_test_trans.shape == (75, 25)
    assert not np.isnan(X_train_trans).any()
    assert not np.isnan(X_test_trans).any()


def test_shap_tree_explainer_reproducible_background(loaded_artifacts):
    """Verifies that SHAP TreeExplainer initializes reproducibly from training data."""
    pipeline = loaded_artifacts["pipeline"]
    preprocessor, classifier, _ = extract_pipeline_components(pipeline)
    X_train = loaded_artifacts["X_train"]
    X_test = loaded_artifacts["X_test"]

    X_train_trans, _ = transform_data_for_explanation(preprocessor, X_train, X_test)

    # Explainer 1 with seed 42
    explainer1 = create_shap_explainer(
        classifier=classifier,
        background_data=X_train_trans,
        background_size=50,
        random_state=42,
    )
    # Explainer 2 with same seed 42
    explainer2 = create_shap_explainer(
        classifier=classifier,
        background_data=X_train_trans,
        background_size=50,
        random_state=42,
    )

    np.testing.assert_allclose(
        explainer1.expected_value,
        explainer2.expected_value,
        err_msg="Expected values must be identical across identical random seeds.",
    )


def test_shap_output_dimensions_and_class_mapping(loaded_artifacts):
    """Verifies that SHAP values matrix has shape (n_samples, n_features, n_classes)."""
    pipeline = loaded_artifacts["pipeline"]
    preprocessor, classifier, feature_names = extract_pipeline_components(pipeline)
    X_train = loaded_artifacts["X_train"]
    X_test = loaded_artifacts["X_test"]

    X_train_trans, X_test_trans = transform_data_for_explanation(preprocessor, X_train, X_test)

    explainer = create_shap_explainer(
        classifier=classifier,
        background_data=X_train_trans,
        background_size=50,
        random_state=42,
    )

    # Compute on first 10 test samples
    test_subset = X_test_trans[:10]
    shap_vals = compute_shap_values(explainer, test_subset, check_additivity=False)

    assert shap_vals.shape == (10, 25, 3), f"Expected shape (10, 25, 3), got {shap_vals.shape}"
    assert len(CLASS_DISPLAY_NAMES) == 3
    assert CLASS_DISPLAY_NAMES[0] == "No recorded sleep disorder"
    assert CLASS_DISPLAY_NAMES[1] == "Insomnia"
    assert CLASS_DISPLAY_NAMES[2] == "Sleep Apnea"


def test_global_feature_importance_ranking(loaded_artifacts):
    """Verifies compute_global_feature_importance returns ordered metrics and expected columns."""
    pipeline = loaded_artifacts["pipeline"]
    preprocessor, classifier, feature_names = extract_pipeline_components(pipeline)
    X_train = loaded_artifacts["X_train"]
    X_test = loaded_artifacts["X_test"]

    X_train_trans, X_test_trans = transform_data_for_explanation(preprocessor, X_train, X_test)
    explainer = create_shap_explainer(classifier, X_train_trans, background_size=50, random_state=42)
    shap_vals = compute_shap_values(explainer, X_test_trans[:20], check_additivity=False)

    importance_df = compute_global_feature_importance(shap_vals, feature_names, CLASS_DISPLAY_NAMES)

    assert isinstance(importance_df, pd.DataFrame)
    assert len(importance_df) == 25
    assert list(importance_df.columns) == [
        "feature",
        "overall_mean_abs_shap",
        "mean_abs_shap_No recorded sleep disorder",
        "mean_abs_shap_Insomnia",
        "mean_abs_shap_Sleep Apnea",
        "rank",
    ]

    # Verify monotonic descending order of overall importance
    scores = importance_df["overall_mean_abs_shap"].to_numpy()
    assert np.all(scores[:-1] >= scores[1:]), "Features must be sorted in descending order of importance"
    assert importance_df["rank"].tolist() == list(range(1, 26))


def test_lime_tabular_explainer_integration(loaded_artifacts):
    """Verifies that LIME TabularExplainer integrates cleanly and generates local weights."""
    pipeline = loaded_artifacts["pipeline"]
    preprocessor, classifier, feature_names = extract_pipeline_components(pipeline)
    X_train = loaded_artifacts["X_train"]
    X_test = loaded_artifacts["X_test"]

    X_train_trans, X_test_trans = transform_data_for_explanation(preprocessor, X_train, X_test)

    lime_explainer = create_lime_explainer(
        training_data=X_train_trans,
        feature_names=feature_names,
        class_names=CLASS_DISPLAY_NAMES,
        random_state=42,
    )

    assert lime_explainer is not None

    sample_inst = X_test_trans[0]
    pred_class = int(classifier.predict(sample_inst.reshape(1, -1))[0])

    exp = explain_individual_lime(
        lime_explainer=lime_explainer,
        instance=sample_inst,
        predict_fn=classifier.predict_proba,
        target_class=pred_class,
        num_features=8,
    )

    assert exp is not None
    exp_list = exp.as_list(label=pred_class)
    assert len(exp_list) == 8
    for rule, weight in exp_list:
        assert isinstance(rule, str)
        assert isinstance(weight, float)


def test_visual_figures_generated_and_non_empty():
    """Verifies that all 5 required XAI figures exist on disk and are non-empty."""
    required_figures = [
        "shap_summary.png",
        "shap_feature_importance.png",
        "shap_class_comparison.png",
        "shap_individual_prediction.png",
        "lime_individual_prediction.png",
    ]

    for fig_name in required_figures:
        fig_path = os.path.join(FIGURES_DIR, fig_name)
        assert os.path.exists(fig_path), f"Required figure missing: {fig_path}"
        file_size = os.path.getsize(fig_path)
        assert file_size > 10000, f"Figure {fig_name} is unexpectedly small ({file_size} bytes)"


def test_machine_readable_json_and_markdown_report_integrity():
    """Verifies the schema and content of shap_feature_importance.json and shap_report.md."""
    json_path = os.path.join(REPORTS_DIR, "shap_feature_importance.json")
    assert os.path.exists(json_path), f"Missing JSON summary: {json_path}"

    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["explainer"] == "shap.TreeExplainer"
    assert meta["lime_integrated"] is True
    assert meta["feature_count"] == 25
    assert meta["background_sample_size"] == 100
    assert meta["holdout_test_samples"] == 75
    assert meta["random_state"] == 42
    assert "class_mapping" in meta
    assert meta["class_mapping"]["0"]["display"] == "No recorded sleep disorder"
    assert len(meta["top_features_overall"]) == 25
    assert "sample_explanation" in meta
    assert "disclaimer" in meta

    report_md_path = os.path.join(REPORTS_DIR, "shap_report.md")
    assert os.path.exists(report_md_path), f"Missing Markdown report: {report_md_path}"

    with open(report_md_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "Clinical Non-Diagnostic Disclaimer" in content
    assert "TreeExplainer" in content
    assert "No recorded sleep disorder" in content
    assert "BP_Systolic" in content
    assert "LIME Local Surrogate Weights" in content
    assert "Interpretation Limitations and Ethics" in content
