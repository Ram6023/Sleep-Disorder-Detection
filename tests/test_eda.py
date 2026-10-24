"""
Unit and integration tests for the Exploratory Data Analysis (EDA) module.
"""

import os
import pytest
import pandas as pd
import numpy as np

from ml.eda import (
    load_dataset,
    analyze_dataset_structure,
    analyze_target_and_imbalance,
    compute_numerical_descriptive_stats,
    analyze_categorical_associations,
    compute_correlations,
    generate_markdown_report,
    CLASS_ORDER,
    FIGURES_DIR,
    REPORT_PATH,
)
from ml.preprocess import clean_and_prepare_data
from ml.feature_engineering import FeatureEngineer


@pytest.fixture
def raw_dataset() -> pd.DataFrame:
    """Loads the real dataset via EDA loader."""
    return load_dataset()


def test_load_dataset_integrity(raw_dataset: pd.DataFrame):
    """Verifies that the dataset loads with 374 rows, 13 columns, and literal 'None' preserved."""
    assert raw_dataset.shape == (374, 13)
    assert "Person ID" in raw_dataset.columns
    assert "Sleep Disorder" in raw_dataset.columns
    # Check that 'None' is a string, not NaN
    assert "None" in raw_dataset["Sleep Disorder"].values
    assert not raw_dataset["Sleep Disorder"].isnull().any()


def test_analyze_dataset_structure(raw_dataset: pd.DataFrame):
    """Verifies structure analysis output metrics."""
    struct = analyze_dataset_structure(raw_dataset)
    assert struct["n_rows"] == 374
    assert struct["n_columns"] == 13
    assert struct["total_missing_cells"] == 0
    assert struct["exact_duplicates"] == 0
    assert struct["feature_duplicates"] >= 0


def test_analyze_target_and_imbalance(raw_dataset: pd.DataFrame):
    """Verifies class counts and proportions preserving 'None'."""
    res = analyze_target_and_imbalance(raw_dataset)
    counts = res["class_counts"]
    pcts = res["class_percentages"]

    assert counts["None"] == 219
    assert counts["Insomnia"] == 77
    assert counts["Sleep Apnea"] == 78
    assert round(pcts["None"], 1) == 58.6
    assert round(pcts["Insomnia"], 1) == 20.6
    assert round(pcts["Sleep Apnea"], 1) == 20.9
    assert res["majority_class"] == "None"


def test_numerical_descriptive_stats(raw_dataset: pd.DataFrame):
    """Verifies summary statistics computation for base and engineered features."""
    X, y = clean_and_prepare_data(raw_dataset)
    fe = FeatureEngineer(add_ratios=True)
    X_eng = fe.fit_transform(X)

    desc = compute_numerical_descriptive_stats(X, y, X_eng)
    assert "Age" in desc["overall_stats"]
    assert "Sleep Duration" in desc["overall_stats"]
    assert "BP_Pulse_Pressure" in desc["overall_stats"]
    assert "Sleep_to_Stress_Ratio" in desc["overall_stats"]

    # Check that grouped stats are provided for all 3 classes
    for cls in CLASS_ORDER:
        assert cls in desc["grouped_stats"]
        assert "Sleep Duration" in desc["grouped_stats"][cls]


def test_analyze_categorical_associations(raw_dataset: pd.DataFrame):
    """Verifies cross-tabulations of categorical variables."""
    X, y = clean_and_prepare_data(raw_dataset)
    cat_res = analyze_categorical_associations(X, y)

    assert "Gender" in cat_res
    assert "BMI Category" in cat_res
    assert "Occupation" in cat_res
    assert "Normal" in cat_res["BMI Category"]["counts"]


def test_compute_correlations(raw_dataset: pd.DataFrame):
    """Verifies Pearson and Spearman correlation matrices."""
    X, y = clean_and_prepare_data(raw_dataset)
    fe = FeatureEngineer(add_ratios=True)
    X_eng = fe.fit_transform(X)

    corr = compute_correlations(X_eng, y)
    pearson = corr["pearson"]
    spearman = corr["spearman"]

    assert (pearson >= -1.0).all().all() and (pearson <= 1.0).all().all()
    assert (spearman >= -1.0).all().all() and (spearman <= 1.0).all().all()
    assert "Target_Encoded" in pearson.columns


def test_eda_artifacts_generated():
    """Verifies that all 8 report figures and the Markdown report exist."""
    assert os.path.exists(REPORT_PATH), f"Report not found at {REPORT_PATH}"
    assert os.path.getsize(REPORT_PATH) > 1000

    expected_figures = [
        "01_class_distribution.png",
        "02_numerical_distributions.png",
        "03_outliers_boxplots.png",
        "04_categorical_by_disorder.png",
        "05_features_by_sleep_disorder.png",
        "06_correlation_matrix.png",
        "07_engineered_features_analysis.png",
        "08_pairplot_key_features.png",
    ]
    for fig_name in expected_figures:
        fig_path = os.path.join(FIGURES_DIR, fig_name)
        assert os.path.exists(fig_path), f"Figure missing: {fig_path}"
        assert os.path.getsize(fig_path) > 10000, f"Figure unexpectedly small: {fig_path}"
