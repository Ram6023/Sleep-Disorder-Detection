"""
Model Training, Cross-Validation, Hyperparameter Tuning, and Evaluation Pipeline
for 'ML-Based Intelligent System for Automated Sleep Disorder Detection Using Lifestyle and Health Data'.

Orchestrates Task 1.5:
1. Loads dataset, validates schema, and verifies target distribution.
2. Partitions into stratified train (80%) and holdout test (20%) sets.
3. Evaluates 9 candidate classifiers using Stratified 5-Fold Cross-Validation.
4. Assesses impact of feature engineering (addressing BP collinearity) and SMOTE.
5. Performs hyperparameter tuning (GridSearchCV with f1_macro) on training folds only.
6. Evaluates the best pipeline on unseen holdout test data.
7. Serializes artifacts (pipeline, label encoder, metadata, metrics) to models/.
8. Saves visual comparison plots and confusion matrices to reports/figures/.
"""

import os
import sys
import json
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import joblib
import warnings

warnings.filterwarnings("ignore", message=".*Unknown solver options: iprint.*")

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.preprocess import clean_and_prepare_data, split_data, TARGET_CLASSES, ID_COL
from ml.pipeline import TargetLabelEncoder, build_preprocessor, create_full_pipeline
from ml.model_comparison import compare_all_models, build_fold_pipeline, get_candidate_models
from ml.evaluate import (
    compute_classification_metrics,
    plot_confusion_matrix,
    plot_model_comparison_bars,
    plot_roc_curve_multiclass,
)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "Sleep_health_and_lifestyle_dataset.csv")


def load_and_validate_data(data_path: str = DATA_PATH) -> Tuple[pd.DataFrame, pd.Series]:
    """Loads CSV, validates classes, and prepares feature matrix X and target y."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found at: {data_path}")

    # Use keep_default_na=False so 'None' is parsed as literal string
    raw_df = pd.read_csv(data_path, keep_default_na=False)

    # Validate classes
    target_series = raw_df["Sleep Disorder"].astype(str).str.strip()
    class_counts = target_series.value_counts().to_dict()
    print(f"Verified dataset target counts: {class_counts}")

    expected_classes = set(TARGET_CLASSES)
    if set(class_counts.keys()) != expected_classes:
        raise ValueError(f"Target classes {set(class_counts.keys())} do not match expected {expected_classes}")

    X, y = clean_and_prepare_data(raw_df)
    assert ID_COL not in X.columns, f"Leaked {ID_COL} into feature matrix X"
    return X, y


def run_cross_validation_comparison(
    X_train: pd.DataFrame,
    y_train_encoded: np.ndarray,
    random_state: int = 42,
) -> pd.DataFrame:
    """Executes stratified 5-fold CV across 9 candidate models under multiple feature regimes."""
    print("\n" + "=" * 70)
    print("RUNNING 5-FOLD STRATIFIED CROSS-VALIDATION MODEL COMPARISON")
    print("=" * 70)

    comparison_df = compare_all_models(
        X_train=X_train,
        y_train=y_train_encoded,
        n_splits=5,
        random_state=random_state,
        test_feature_engineering=True,
        test_smote=True,
    )

    print("\nTop 10 Cross-Validation Performers (Ranked by Macro F1):")
    cols_to_show = ["model_name", "cv_macro_f1_mean", "cv_macro_f1_std", "cv_balanced_acc_mean", "cv_accuracy_mean"]
    print(comparison_df[cols_to_show].head(10).to_string(index=False))

    return comparison_df


def tune_best_model(
    top_base_model_name: str,
    include_feature_engineering: bool,
    use_smote: bool,
    X_train: pd.DataFrame,
    y_train_encoded: np.ndarray,
    random_state: int = 42,
) -> Any:
    """
    Performs hyperparameter tuning via GridSearchCV on training folds only.
    """
    print("\n" + "=" * 70)
    print(f"HYPERPARAMETER TUNING: {top_base_model_name}")
    print("=" * 70)

    param_grid: Dict[str, List[Any]] = {}

    if "Random Forest" in top_base_model_name:
        estimator = RandomForestClassifier(random_state=random_state)
        param_grid = {
            "classifier__n_estimators": [100, 150, 200],
            "classifier__max_depth": [6, 8, 10, None],
            "classifier__min_samples_split": [2, 4, 6],
            "classifier__min_samples_leaf": [1, 2],
            "classifier__class_weight": ["balanced", None],
        }
    elif "Gradient Boosting" in top_base_model_name:
        estimator = GradientBoostingClassifier(random_state=random_state)
        param_grid = {
            "classifier__n_estimators": [80, 120, 160],
            "classifier__learning_rate": [0.04, 0.08, 0.12],
            "classifier__max_depth": [3, 4, 5],
            "classifier__subsample": [0.8, 1.0],
        }
    elif "Logistic Regression" in top_base_model_name:
        estimator = LogisticRegression(max_iter=2000, random_state=random_state)
        param_grid = {
            "classifier__C": [0.01, 0.1, 1.0, 5.0, 10.0],
            "classifier__penalty": ["l2"],
            "classifier__class_weight": ["balanced", None],
        }
    else:
        # Default tuning for RandomForest
        estimator = RandomForestClassifier(random_state=random_state)
        param_grid = {
            "classifier__n_estimators": [100, 150],
            "classifier__max_depth": [6, 8, None],
        }

    pipeline = build_fold_pipeline(
        model=estimator,
        include_feature_engineering=include_feature_engineering,
        use_smote=use_smote,
        random_state=random_state,
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=skf,
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train, y_train_encoded)

    print(f"Best CV Macro F1: {grid_search.best_score_:.4f}")
    print(f"Optimal Hyperparameters: {grid_search.best_params_}")

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def train_and_evaluate_pipeline(random_state: int = 42) -> Dict[str, Any]:
    """
    Main orchestration routine for Task 1.5.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # 1. Load data
    X, y = load_and_validate_data()
    print(f"Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features.")

    # 2. Holdout split (80/20 Stratified)
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=0.2, random_state=random_state, stratify=True
    )
    print(f"Partitioned data: Train = {len(X_train)} samples, Test = {len(X_test)} samples.")

    # 3. Fit target label encoder on training set
    label_encoder = TargetLabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # 4. Model comparison across all 9 classifiers
    comparison_df = run_cross_validation_comparison(
        X_train=X_train,
        y_train_encoded=y_train_encoded,
        random_state=random_state,
    )

    # Save comparison metrics
    comparison_csv_path = os.path.join(MODELS_DIR, "metrics_comparison.csv")
    comparison_df.to_csv(comparison_csv_path, index=False)
    comparison_df.to_csv(os.path.join(REPORTS_DIR, "model_comparison.csv"), index=False)

    comparison_json_path = os.path.join(MODELS_DIR, "metrics_comparison.json")
    with open(comparison_json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_df.to_dict(orient="records"), f, indent=2)

    # Generate comparison plot
    comp_plot_path = os.path.join(FIGURES_DIR, "model_comparison_cv.png")
    # Take top 8 models for clean visualization
    top_models_plot_df = comparison_df.head(8).copy()
    plot_model_comparison_bars(top_models_plot_df, save_path=comp_plot_path)
    print(f"Saved model comparison plot to: {comp_plot_path}")

    # 5. Collinearity & Feature Engineering Assessment
    fe_impact = comparison_df.groupby("configuration")["cv_macro_f1_mean"].mean()
    print("\nFeature Regime Performance Impact (Mean Macro F1 across models):")
    for cfg, score in fe_impact.items():
        print(f"  - {cfg}: {score:.4f}")

    # Select best performing model configuration
    best_row = comparison_df.iloc[0]
    top_model_name = best_row["base_model_name"]
    top_fe_flag = bool(best_row["feature_engineering"])
    top_smote_flag = bool(best_row["use_smote"])

    print(f"\nSelected Best Candidate Architecture: {best_row['model_name']}")
    print(f"  - CV Macro F1: {best_row['cv_macro_f1_mean']} ± {best_row['cv_macro_f1_std']}")
    print(f"  - CV Balanced Accuracy: {best_row['cv_balanced_acc_mean']}")
    print(f"  - CV Accuracy: {best_row['cv_accuracy_mean']}")

    # 6. Hyperparameter Tuning
    tuned_pipeline, best_params, best_cv_score = tune_best_model(
        top_base_model_name=top_model_name,
        include_feature_engineering=top_fe_flag,
        use_smote=top_smote_flag,
        X_train=X_train,
        y_train_encoded=y_train_encoded,
        random_state=random_state,
    )

    # 7. Final Training on Full Training Set
    print("\nFitting finalized tuned pipeline on full training dataset...")
    tuned_pipeline.fit(X_train, y_train_encoded)

    # 8. Unseen Holdout Test Set Evaluation
    print("\n" + "=" * 70)
    print("EVALUATING FINAL PIPELINE ON UNSEEN HOLDOUT TEST SET")
    print("=" * 70)
    y_test_pred_encoded = tuned_pipeline.predict(X_test)
    y_test_pred_labels = label_encoder.inverse_transform(y_test_pred_encoded)
    y_test_proba = tuned_pipeline.predict_proba(X_test)

    test_metrics = compute_classification_metrics(
        y_true=y_test_encoded,
        y_pred=y_test_pred_encoded,
        labels=[0, 1, 2],
        target_names=TARGET_CLASSES,
        y_proba=y_test_proba,
        model_classes=getattr(tuned_pipeline, "classes_", None),
    )

    print(f"Holdout Test Accuracy:          {test_metrics['accuracy']:.4f}")
    print(f"Holdout Test Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"Holdout Test Macro F1:          {test_metrics['macro_f1']:.4f}")
    print(f"Holdout Test Weighted F1:       {test_metrics['weighted_f1']:.4f}")
    if test_metrics.get("macro_roc_auc") is not None:
        print(f"Holdout Test Macro ROC-AUC:     {test_metrics['macro_roc_auc']:.4f}")
        print(f"Holdout Test Weighted ROC-AUC:  {test_metrics['weighted_roc_auc']:.4f}")
    print("\nPer-Class Breakdown on Holdout Test Set:")
    for cls, met in test_metrics["per_class"].items():
        roc_str = f" | ROC-AUC: {met['roc_auc']:.4f}" if met.get("roc_auc") is not None else ""
        print(f"  - {cls:12s} | Precision: {met['precision']:.4f} | Recall: {met['recall']:.4f} | F1: {met['f1']:.4f}{roc_str} | Support: {met['support']}")

    print("\nClassification Report:")
    print(test_metrics["classification_report_str"])

    # 9. Save Confusion Matrix and ROC Curve Plots
    cm_plot_path = os.path.join(FIGURES_DIR, "best_model_confusion_matrix.png")
    plot_confusion_matrix(
        cm=test_metrics["confusion_matrix"],
        classes=TARGET_CLASSES,
        title=f"Holdout Test Confusion Matrix ({top_model_name})",
        save_path=cm_plot_path,
        normalize=True,
    )
    print(f"Saved confusion matrix plot to: {cm_plot_path}")

    roc_plot_path = os.path.join(FIGURES_DIR, "roc_auc_ovr.png")
    plot_roc_curve_multiclass(
        y_true=y_test_encoded,
        y_proba=y_test_proba,
        labels=[0, 1, 2],
        classes=TARGET_CLASSES,
        title=f"Multiclass ROC Curve - One-vs-Rest ({top_model_name})",
        save_path=roc_plot_path,
    )
    print(f"Saved multiclass ROC curve plot to: {roc_plot_path}")

    # 10. Save Model Artifacts
    best_model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    joblib.dump(tuned_pipeline, best_model_path)
    print(f"Serialized trained pipeline to: {best_model_path}")

    encoder_path = os.path.join(MODELS_DIR, "label_encoder.joblib")
    joblib.dump(label_encoder, encoder_path)
    print(f"Serialized label encoder to: {encoder_path}")

    metadata = {
        "selected_model": top_model_name,
        "full_model_name": best_row["model_name"],
        "feature_engineering": top_fe_flag,
        "use_smote": top_smote_flag,
        "best_hyperparameters": best_params,
        "cv_macro_f1_mean": best_row["cv_macro_f1_mean"],
        "cv_macro_f1_std": best_row["cv_macro_f1_std"],
        "cv_balanced_acc_mean": best_row["cv_balanced_acc_mean"],
        "cv_accuracy_mean": best_row["cv_accuracy_mean"],
        "test_metrics": {
            "accuracy": test_metrics["accuracy"],
            "balanced_accuracy": test_metrics["balanced_accuracy"],
            "macro_f1": test_metrics["macro_f1"],
            "weighted_f1": test_metrics["weighted_f1"],
            "per_class": test_metrics["per_class"],
        },
        "target_classes": TARGET_CLASSES,
        "input_features": list(X.columns),
        "random_state": random_state,
    }

    metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved model metadata to: {metadata_path}")

    test_metrics_path = os.path.join(MODELS_DIR, "test_evaluation.json")
    with open(test_metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    # Also maintain artifacts under ml_pipeline/artifacts for Phase 2/3 compatibility
    ml_artifacts_dir = os.path.join(PROJECT_ROOT, "ml_pipeline", "artifacts")
    os.makedirs(ml_artifacts_dir, exist_ok=True)
    joblib.dump(tuned_pipeline, os.path.join(ml_artifacts_dir, "best_model.joblib"))
    joblib.dump(label_encoder, os.path.join(ml_artifacts_dir, "label_encoder.joblib"))
    with open(os.path.join(ml_artifacts_dir, "model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    with open(os.path.join(ml_artifacts_dir, "model_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(comparison_df.to_dict(orient="records"), f, indent=2)

    print("\n" + "=" * 70)
    print("MODEL TRAINING AND EVALUATION PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)

    return {
        "comparison_df": comparison_df,
        "selected_model": top_model_name,
        "tuned_pipeline": tuned_pipeline,
        "best_params": best_params,
        "test_metrics": test_metrics,
        "metadata": metadata,
    }


if __name__ == "__main__":
    train_and_evaluate_pipeline()
