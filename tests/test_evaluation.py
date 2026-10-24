"""
Unit and integration tests for Model Evaluation and Multiclass ROC-AUC (One-vs-Rest).
Covers:
- Exact metric verification on holdout data.
- Multiclass ROC-AUC computation with predict_proba.
- Alignment of probability columns with encoded class order.
- Dimension validation and error handling.
- Unsupported edge cases (single class in slice, missing classes).
- Publication-quality ROC curve plot generation.
- Backward compatibility when y_proba is omitted.
"""

import os
import pytest
import numpy as np
import pandas as pd
import joblib
import matplotlib.figure

from ml.evaluate import (
    compute_classification_metrics,
    compute_multiclass_roc_auc,
    plot_roc_curve_multiclass,
    TARGET_CLASSES,
)
from ml.preprocess import clean_and_prepare_data, split_data


@pytest.fixture
def holdout_test_data():
    """Provides the exact untouched 20% holdout test dataset and trained artifacts."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(project_root, "data", "Sleep_health_and_lifestyle_dataset.csv")
    model_path = os.path.join(project_root, "models", "best_model.joblib")
    encoder_path = os.path.join(project_root, "models", "label_encoder.joblib")

    if not os.path.exists(model_path) or not os.path.exists(encoder_path):
        pytest.skip("Models not yet serialized; train.py must be run first.")

    raw_df = pd.read_csv(data_path, keep_default_na=False)
    X, y = clean_and_prepare_data(raw_df)

    # Exact untouched holdout partition
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=0.2, random_state=42, stratify=True
    )

    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)

    y_test_encoded = encoder.transform(y_test)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)

    return {
        "X_test": X_test,
        "y_test_encoded": y_test_encoded,
        "y_test_pred": y_test_pred,
        "y_test_proba": y_test_proba,
        "model": model,
        "encoder": encoder,
    }


def test_holdout_metrics_and_roc_auc_exactness(holdout_test_data):
    """
    Verifies that the tuned Random Forest pipeline achieves verified performance
    on the untouched 20% holdout test set (75 samples).
    """
    y_test_encoded = holdout_test_data["y_test_encoded"]
    y_test_pred = holdout_test_data["y_test_pred"]
    y_test_proba = holdout_test_data["y_test_proba"]
    model = holdout_test_data["model"]

    metrics = compute_classification_metrics(
        y_true=y_test_encoded,
        y_pred=y_test_pred,
        labels=[0, 1, 2],
        target_names=TARGET_CLASSES,
        y_proba=y_test_proba,
        model_classes=getattr(model, "classes_", None),
    )

    # Core classification metrics
    assert metrics["accuracy"] == 0.9600
    assert metrics["balanced_accuracy"] == 0.9347
    assert metrics["macro_f1"] == 0.9352
    assert metrics["weighted_f1"] == 0.9599

    # Multiclass ROC-AUC (OvR)
    assert metrics["macro_roc_auc"] == 0.9957
    assert metrics["weighted_roc_auc"] == 0.9973
    assert metrics["roc_auc_supported"] is True

    # Per-class ROC-AUC
    assert metrics["per_class"]["None"]["roc_auc"] == 1.0000
    assert metrics["per_class"]["Insomnia"]["roc_auc"] == 0.9944
    assert metrics["per_class"]["Sleep Apnea"]["roc_auc"] == 0.9926

    # Confusion matrix verification (44 None, 13 Insomnia, 15 Sleep Apnea correct)
    cm = metrics["confusion_matrix"]
    assert cm == [
        [44, 0, 0],
        [0, 13, 2],
        [0, 1, 15],
    ]


def test_probability_dimensions_and_input_validation():
    """Verifies that invalid probability shapes or values raise descriptive ValueErrors."""
    y_true = np.array([0, 1, 2, 0])

    # 1. 1D probability array
    with pytest.raises(ValueError, match="2D array"):
        compute_multiclass_roc_auc(y_true, np.array([0.1, 0.5, 0.4, 0.8]))

    # 2. Sample count mismatch
    with pytest.raises(ValueError, match="Sample count mismatch"):
        compute_multiclass_roc_auc(y_true, np.zeros((3, 3)))

    # 3. Column count mismatch (2 columns instead of 3 expected classes)
    with pytest.raises(ValueError, match="Probability column count"):
        compute_multiclass_roc_auc(y_true, np.zeros((4, 2)), labels=[0, 1, 2])

    # 4. NaN or Inf values
    nan_proba = np.array([
        [0.7, 0.2, 0.1],
        [np.nan, 0.5, 0.5],
        [0.1, 0.8, 0.1],
        [0.9, 0.05, 0.05],
    ])
    with pytest.raises(ValueError, match="NaN or Infinite"):
        compute_multiclass_roc_auc(y_true, nan_proba)


def test_class_ordering_and_realignment():
    """
    Verifies that when model_classes order differs from labels order,
    compute_multiclass_roc_auc re-aligns columns correctly.
    """
    y_true = np.array([0, 1, 2, 0, 1, 2])
    # Aligned probabilities for [0, 1, 2]
    proba_aligned = np.array([
        [0.9, 0.05, 0.05],
        [0.1, 0.8, 0.1],
        [0.1, 0.2, 0.7],
        [0.85, 0.1, 0.05],
        [0.05, 0.9, 0.05],
        [0.15, 0.15, 0.7],
    ])

    res_aligned = compute_multiclass_roc_auc(
        y_true, proba_aligned, labels=[0, 1, 2], model_classes=[0, 1, 2]
    )

    # Permuted probabilities where columns are [2, 0, 1]
    proba_permuted = proba_aligned[:, [2, 0, 1]]
    res_permuted = compute_multiclass_roc_auc(
        y_true, proba_permuted, labels=[0, 1, 2], model_classes=[2, 0, 1]
    )

    assert res_aligned["macro_roc_auc"] == res_permuted["macro_roc_auc"]
    assert res_aligned["weighted_roc_auc"] == res_permuted["weighted_roc_auc"]
    assert res_aligned["per_class_roc_auc"] == res_permuted["per_class_roc_auc"]

    # Mismatched model_classes set raises ValueError
    with pytest.raises(ValueError, match="does not match expected labels"):
        compute_multiclass_roc_auc(
            y_true, proba_aligned, labels=[0, 1, 2], model_classes=[0, 1, 99]
        )


def test_unsupported_edge_case_single_class_overall():
    """
    Verifies that when y_true contains only 1 class overall,
    ROC-AUC is marked unsupported without crashing or inventing values.
    """
    y_true_single = np.array([0, 0, 0, 0])
    proba_single = np.array([
        [0.8, 0.1, 0.1],
        [0.9, 0.05, 0.05],
        [0.7, 0.2, 0.1],
        [0.85, 0.1, 0.05],
    ])

    res = compute_multiclass_roc_auc(y_true_single, proba_single, labels=[0, 1, 2])
    assert res["is_supported"] is False
    assert res["macro_roc_auc"] is None
    assert res["weighted_roc_auc"] is None
    for cls in TARGET_CLASSES:
        assert res["per_class_roc_auc"][cls] is None
    assert len(res["unsupported_classes"]) == 3


def test_unsupported_edge_case_missing_class_slice():
    """
    Verifies that when one class is completely absent from y_true,
    the missing class is marked unsupported, while existing classes are scored properly.
    """
    # Classes 0 and 1 present, but Class 2 has 0 instances
    y_true_partial = np.array([0, 0, 1, 1, 0, 1])
    proba_partial = np.array([
        [0.9, 0.1, 0.0],
        [0.8, 0.2, 0.0],
        [0.1, 0.9, 0.0],
        [0.2, 0.8, 0.0],
        [0.85, 0.15, 0.0],
        [0.3, 0.7, 0.0],
    ])

    res = compute_multiclass_roc_auc(
        y_true_partial, proba_partial, labels=[0, 1, 2], target_names=TARGET_CLASSES
    )

    # Class 2 (Sleep Apnea) has 0 positive cases in OvR slice
    assert res["per_class_roc_auc"]["Sleep Apnea"] is None
    assert "Sleep Apnea" in res["unsupported_classes"]

    # Classes None and Insomnia are fully supported
    assert res["per_class_roc_auc"]["None"] == 1.0
    assert res["per_class_roc_auc"]["Insomnia"] == 1.0
    assert res["macro_roc_auc"] == 1.0
    assert res["is_supported"] is True


def test_plot_roc_curve_multiclass_figure_generation(holdout_test_data, tmp_path):
    """
    Verifies that plot_roc_curve_multiclass creates a valid Matplotlib Figure
    and saves a high-resolution PNG to disk.
    """
    y_test_encoded = holdout_test_data["y_test_encoded"]
    y_test_proba = holdout_test_data["y_test_proba"]

    save_file = tmp_path / "test_roc_plot.png"
    fig = plot_roc_curve_multiclass(
        y_true=y_test_encoded,
        y_proba=y_test_proba,
        labels=[0, 1, 2],
        classes=TARGET_CLASSES,
        title="Test ROC Curve Plot",
        save_path=str(save_file),
    )

    assert isinstance(fig, matplotlib.figure.Figure)
    assert save_file.exists()
    assert save_file.stat().st_size > 1000  # Non-trivial image file


def test_compute_classification_metrics_backward_compatibility():
    """Verifies that compute_classification_metrics works when y_proba is None."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 2])

    metrics = compute_classification_metrics(y_true, y_pred, labels=[0, 1, 2])

    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "macro_roc_auc" not in metrics
    assert "confusion_matrix" in metrics
