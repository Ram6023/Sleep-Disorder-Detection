"""
Model Comparison and Cross-Validation Module
for 'ML-Based Intelligent System for Automated Sleep Disorder Detection Using Lifestyle and Health Data'.

Compares 9 classifiers using Stratified 5-Fold Cross-Validation:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine (SVC)
5. K-Nearest Neighbors (KNN)
6. Gradient Boosting (GBT)
7. AdaBoost
8. Bagging
9. XGBoost (if available)

Enforces strict featurization ordering (zero data leakage) by fitting all transformers,
scalers, and optional SMOTE resamplers strictly within training folds.
"""

import os
import sys
import warnings
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message=".*Unknown solver options: iprint.*")

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
)
from sklearn.pipeline import Pipeline as SklearnPipeline

try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except (ImportError, Exception):
    XGBOOST_AVAILABLE = False

# Ensure project root is accessible
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.pipeline import build_preprocessor, create_full_pipeline, TargetLabelEncoder


def get_candidate_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Instantiates the 9 candidate classifiers with reproducible random seeds.
    """
    models: Dict[str, Any] = {
        "Logistic Regression": LogisticRegression(
            max_iter=1500,
            random_state=random_state,
            class_weight="balanced",
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_split=5,
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_split=4,
            random_state=random_state,
        ),
        "Support Vector Machine": SVC(
            C=1.0,
            kernel="rbf",
            probability=True,
            random_state=random_state,
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.08,
            max_depth=4,
            random_state=random_state,
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=random_state,
        ),
        "Bagging": BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=6, random_state=random_state),
            n_estimators=50,
            random_state=random_state,
        ),
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=100,
            learning_rate=0.08,
            max_depth=4,
            random_state=random_state,
            eval_metric="mlogloss",
        )

    return models


def build_fold_pipeline(
    model: Any,
    include_feature_engineering: bool = True,
    use_smote: bool = False,
    random_state: int = 42,
    k_neighbors: int = 5,
) -> Any:
    """
    Constructs an end-to-end pipeline for cross-validation and inference.
    If use_smote is True, uses imblearn.pipeline so SMOTE is applied only during fit (training folds).
    """
    preprocessor = build_preprocessor(include_feature_engineering=include_feature_engineering)
    base_steps = list(preprocessor.steps)

    if use_smote and IMBLEARN_AVAILABLE:
        # Use imblearn Pipeline so resampling is restricted to training data during fit()
        pipeline = ImbPipeline(
            steps=base_steps + [
                ("smote", SMOTE(random_state=random_state, k_neighbors=k_neighbors)),
                ("classifier", model),
            ]
        )
    else:
        pipeline = SklearnPipeline(
            steps=base_steps + [
                ("classifier", model),
            ]
        )

    return pipeline


def evaluate_model_cv(
    model_name: str,
    model: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
    include_feature_engineering: bool = True,
    use_smote: bool = False,
) -> Dict[str, Any]:
    """
    Evaluates a single model using StratifiedKFold cross-validation.
    Ensures preprocessors and optional SMOTE are fitted solely within each training fold.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_macro_f1: List[float] = []
    fold_balanced_acc: List[float] = []
    fold_acc: List[float] = []
    fold_weighted_f1: List[float] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_fold_train, X_fold_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]

        pipeline = build_fold_pipeline(
            model=model,
            include_feature_engineering=include_feature_engineering,
            use_smote=use_smote,
            random_state=random_state,
        )

        pipeline.fit(X_fold_train, y_fold_train)
        y_val_pred = pipeline.predict(X_fold_val)

        fold_macro_f1.append(float(f1_score(y_fold_val, y_val_pred, average="macro", zero_division=0)))
        fold_balanced_acc.append(float(balanced_accuracy_score(y_fold_val, y_val_pred)))
        fold_acc.append(float(accuracy_score(y_fold_val, y_val_pred)))
        fold_weighted_f1.append(float(f1_score(y_fold_val, y_val_pred, average="weighted", zero_division=0)))

    return {
        "model_name": model_name,
        "feature_engineering": include_feature_engineering,
        "use_smote": use_smote,
        "cv_macro_f1_mean": round(float(np.mean(fold_macro_f1)), 4),
        "cv_macro_f1_std": round(float(np.std(fold_macro_f1)), 4),
        "cv_balanced_acc_mean": round(float(np.mean(fold_balanced_acc)), 4),
        "cv_balanced_acc_std": round(float(np.std(fold_balanced_acc)), 4),
        "cv_accuracy_mean": round(float(np.mean(fold_acc)), 4),
        "cv_accuracy_std": round(float(np.std(fold_acc)), 4),
        "cv_weighted_f1_mean": round(float(np.mean(fold_weighted_f1)), 4),
        "cv_weighted_f1_std": round(float(np.std(fold_weighted_f1)), 4),
        "fold_macro_f1_scores": [round(s, 4) for s in fold_macro_f1],
    }


def compare_all_models(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
    test_feature_engineering: bool = True,
    test_smote: bool = True,
) -> pd.DataFrame:
    """
    Executes cross-validation comparison across all candidate models under different configurations.
    Returns a structured DataFrame sorted by cv_macro_f1_mean.
    """
    candidate_models = get_candidate_models(random_state=random_state)
    results: List[Dict[str, Any]] = []

    configs: List[Tuple[bool, bool, str]] = [(True, False, "Base + Engineered")]
    if test_feature_engineering:
        configs.append((False, False, "Base Features Only"))
    if test_smote and IMBLEARN_AVAILABLE:
        configs.append((True, True, "Base + Engineered + SMOTE"))

    for fe_flag, smote_flag, config_desc in configs:
        for name, model in candidate_models.items():
            label = f"{name} ({config_desc})"
            res = evaluate_model_cv(
                model_name=label,
                model=model,
                X=X_train,
                y=y_train,
                n_splits=n_splits,
                random_state=random_state,
                include_feature_engineering=fe_flag,
                use_smote=smote_flag,
            )
            res["base_model_name"] = name
            res["configuration"] = config_desc
            results.append(res)

    comparison_df = pd.DataFrame(results).sort_values(by="cv_macro_f1_mean", ascending=False).reset_index(drop=True)
    return comparison_df
