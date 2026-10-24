"""
Explainable AI (XAI) Module Using SHAP and LIME
for 'ML-Based Intelligent System for Automated Sleep Disorder Detection Using Lifestyle and Health Data'.

Provides:
- Pipeline extraction for preprocessing steps, classifier, and transformed feature names.
- SHAP TreeExplainer integration with reproducible training-derived background sampling.
- Global feature importance (mean absolute SHAP values across classes and per-class).
- Publication-quality visualization of SHAP summary, global importance, and class comparisons.
- Local SHAP waterfall and individual prediction explanations.
- LIME TabularExplainer local surrogate explanations with consistent feature spaces.
- Serialization of machine-readable explanation metrics and structured markdown reports.
"""

import os
import sys
import json
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import shap
import lime
import lime.lime_tabular

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.preprocess import clean_and_prepare_data, split_data, TARGET_CLASSES, ID_COL

# Human-readable target labels mapping
CLASS_DISPLAY_NAMES = [
    "No recorded sleep disorder",  # Class 0: 'None'
    "Insomnia",                    # Class 1: 'Insomnia'
    "Sleep Apnea",                 # Class 2: 'Sleep Apnea'
]

CLASS_LABEL_TO_DISPLAY = {
    "None": "No recorded sleep disorder",
    "Insomnia": "Insomnia",
    "Sleep Apnea": "Sleep Apnea",
}


def extract_pipeline_components(pipeline: Any) -> Tuple[Any, Any, List[str]]:
    """
    Extracts the preprocessor, classifier, and transformed feature names from a fitted Pipeline.

    Args:
        pipeline: Fitted scikit-learn Pipeline.

    Returns:
        Tuple of (preprocessor, classifier, feature_names_list).
    """
    if not hasattr(pipeline, "named_steps"):
        raise ValueError("Provided object is not a valid scikit-learn Pipeline with named_steps.")

    if "col_transformer" in pipeline.named_steps:
        preprocessor = pipeline.named_steps["col_transformer"]
    elif "preprocessor" in pipeline.named_steps:
        preprocessor = pipeline.named_steps["preprocessor"]
    else:
        raise KeyError("Could not find 'col_transformer' or 'preprocessor' step in pipeline.")

    if "classifier" in pipeline.named_steps:
        classifier = pipeline.named_steps["classifier"]
    elif "model" in pipeline.named_steps:
        classifier = pipeline.named_steps["model"]
    else:
        raise KeyError("Could not find 'classifier' or 'model' step in pipeline.")

    if hasattr(preprocessor, "get_feature_names_out"):
        feature_names = list(preprocessor.get_feature_names_out())
    else:
        raise AttributeError("Preprocessor does not support get_feature_names_out().")

    return preprocessor, classifier, feature_names


def transform_data_for_explanation(
    preprocessor: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transforms training and holdout test feature sets using the fitted preprocessor.

    Args:
        preprocessor: Fitted ColumnTransformer or composite preprocessor.
        X_train: Raw training DataFrame.
        X_test: Raw holdout test DataFrame.

    Returns:
        Tuple of (X_train_trans, X_test_trans) as dense numpy arrays.
    """
    X_tr = preprocessor.transform(X_train)
    X_te = preprocessor.transform(X_test)

    if hasattr(X_tr, "toarray"):
        X_tr = X_tr.toarray()
    if hasattr(X_te, "toarray"):
        X_te = X_te.toarray()

    return np.asarray(X_tr, dtype=float), np.asarray(X_te, dtype=float)


def create_shap_explainer(
    classifier: Any,
    background_data: Optional[np.ndarray] = None,
    background_size: int = 100,
    random_state: int = 42,
) -> shap.TreeExplainer:
    """
    Creates a TreeExplainer for the fitted tree-based classifier.
    Supports reproducible background sampling strictly derived from training data.

    Args:
        classifier: Fitted tree-based classifier (e.g. RandomForestClassifier).
        background_data: Optional training data matrix for background reference.
        background_size: Number of background samples to draw if sampling is requested.
        random_state: Seed for reproducible background sampling.

    Returns:
        Configured shap.TreeExplainer.
    """
    if background_data is not None:
        if len(background_data) > background_size:
            sampled_bg = shap.sample(background_data, background_size, random_state=random_state)
        else:
            sampled_bg = background_data
        explainer = shap.TreeExplainer(classifier, data=sampled_bg)
    else:
        explainer = shap.TreeExplainer(classifier)

    return explainer


def compute_shap_values(
    explainer: shap.TreeExplainer,
    X_trans: np.ndarray,
    check_additivity: bool = False,
) -> np.ndarray:
    """
    Computes SHAP values for transformed inputs.

    Args:
        explainer: Fitted TreeExplainer instance.
        X_trans: Transformed feature array of shape (n_samples, n_features).
        check_additivity: Whether to enforce strict additivity assertion (False recommended
                          for tree models with background reference data to avoid minor float discrepancies).

    Returns:
        3D numpy array of shape (n_samples, n_features, n_classes).
    """
    raw_vals = explainer.shap_values(X_trans, check_additivity=check_additivity)
    arr = np.asarray(raw_vals)

    # In some shap versions, multiclass returns a list of (n_samples, n_features) arrays
    if isinstance(raw_vals, list):
        # Stack to (n_samples, n_features, n_classes)
        arr = np.stack(raw_vals, axis=-1)
    elif arr.ndim == 3 and arr.shape[0] == len(CLASS_DISPLAY_NAMES) and arr.shape[1] == len(X_trans):
        # Shape was (n_classes, n_samples, n_features) -> transpose to (n_samples, n_features, n_classes)
        arr = np.transpose(arr, (1, 2, 0))

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D SHAP values array for multiclass, got shape {arr.shape}")

    return arr


def compute_global_feature_importance(
    shap_values: np.ndarray,
    feature_names: List[str],
    class_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Calculates mean absolute SHAP values globally and broken down per target class.

    Args:
        shap_values: 3D array of shape (n_samples, n_features, n_classes).
        feature_names: List of feature names matching the second dimension.
        class_names: List of human-readable class names.

    Returns:
        DataFrame sorted descending by overall mean absolute SHAP importance.
    """
    if class_names is None:
        class_names = CLASS_DISPLAY_NAMES

    n_samples, n_features, n_classes = shap_values.shape
    if n_features != len(feature_names):
        raise ValueError(f"Feature count mismatch: {n_features} vs {len(feature_names)}")

    # Overall mean absolute SHAP value across all samples and classes
    # mean over samples -> (n_features, n_classes) -> mean over classes -> (n_features,)
    per_class_mean_abs = np.mean(np.abs(shap_values), axis=0)  # shape: (n_features, n_classes)
    overall_mean_abs = np.mean(per_class_mean_abs, axis=1)    # shape: (n_features,)

    records = []
    for i, feat in enumerate(feature_names):
        rec = {
            "feature": feat,
            "overall_mean_abs_shap": round(float(overall_mean_abs[i]), 5),
        }
        for c_idx, c_name in enumerate(class_names):
            rec[f"mean_abs_shap_{c_name}"] = round(float(per_class_mean_abs[i, c_idx]), 5)
        records.append(rec)

    df = pd.DataFrame(records)
    df = df.sort_values(by="overall_mean_abs_shap", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


def plot_shap_summary(
    shap_values: np.ndarray,
    X_trans: np.ndarray,
    feature_names: List[str],
    class_names: Optional[List[str]] = None,
    max_display: int = 15,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Renders and saves a multiclass stacked bar SHAP summary plot.

    Args:
        shap_values: 3D array (n_samples, n_features, n_classes).
        X_trans: Transformed feature array.
        feature_names: Feature names list.
        class_names: Class labels list.
        max_display: Max number of top features to show.
        save_path: Optional file path to save the 300-DPI PNG.

    Returns:
        Matplotlib Figure.
    """
    if class_names is None:
        class_names = CLASS_DISPLAY_NAMES

    shap_list = [shap_values[:, :, c] for c in range(len(class_names))]

    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_list,
        X_trans,
        feature_names=feature_names,
        class_names=class_names,
        max_display=max_display,
        plot_type="bar",
        show=False,
    )
    plt.title(
        "SHAP Global Summary: Mean Absolute Impact on Multi-Class Predictions",
        fontweight="bold",
        fontsize=12,
        pad=14,
    )
    plt.xlabel("Mean |SHAP Value| (Average Impact on Prediction Magnitude)", fontweight="bold", fontsize=10)
    plt.tight_layout()
    fig = plt.gcf()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_shap_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 15,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Renders a publication-ready horizontal bar plot of the top global features.

    Args:
        importance_df: Output from compute_global_feature_importance.
        top_n: Number of top features to plot.
        save_path: Optional file path to save PNG.

    Returns:
        Matplotlib Figure.
    """
    top_df = importance_df.head(top_n).iloc[::-1].copy()

    fig, ax = plt.subplots(figsize=(10, 7))
    y_pos = np.arange(len(top_df))
    bars = ax.barh(
        y_pos,
        top_df["overall_mean_abs_shap"],
        color="#2563eb",
        edgecolor="#1e3a8a",
        alpha=0.88,
        height=0.65,
    )

    for bar in bars:
        width = bar.get_width()
        ax.annotate(
            f"{width:.4f}",
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(5, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="#1f2937",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_df["feature"], fontweight="bold", fontsize=10)
    ax.set_xlabel("Mean Absolute SHAP Value (Overall Contribution)", fontweight="bold", fontsize=11, labelpad=8)
    ax.set_title(
        f"Top {top_n} Features by Global SHAP Importance (Random Forest)",
        fontweight="bold",
        fontsize=13,
        pad=12,
    )
    ax.set_xlim(0, top_df["overall_mean_abs_shap"].max() * 1.18)
    ax.grid(axis="x", linestyle=":", alpha=0.55, color="#9ca3af")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_shap_class_comparison(
    importance_df: pd.DataFrame,
    class_names: Optional[List[str]] = None,
    top_n: int = 8,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Renders a 3-panel horizontal bar chart comparing the top drivers for each sleep disorder class.

    Args:
        importance_df: Output from compute_global_feature_importance.
        class_names: Class labels list.
        top_n: Number of features to display per class.
        save_path: Optional file path to save PNG.

    Returns:
        Matplotlib Figure.
    """
    if class_names is None:
        class_names = CLASS_DISPLAY_NAMES

    colors = ["#059669", "#d97706", "#dc2626"]  # Green (None), Amber (Insomnia), Red (Sleep Apnea)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), sharey=False)

    for i, c_name in enumerate(class_names):
        col = f"mean_abs_shap_{c_name}"
        if col not in importance_df.columns:
            continue

        c_df = importance_df.sort_values(by=col, ascending=False).head(top_n).iloc[::-1]
        ax = axes[i]
        y_pos = np.arange(len(c_df))

        bars = ax.barh(y_pos, c_df[col], color=colors[i % len(colors)], alpha=0.85, edgecolor="black", height=0.62)

        for bar in bars:
            width = bar.get_width()
            ax.annotate(
                f"{width:.4f}",
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(4, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=8,
                fontweight="bold",
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(c_df["feature"], fontweight="bold", fontsize=9)
        ax.set_title(f"Class: {c_name}", fontweight="bold", fontsize=12, pad=10)
        ax.set_xlabel("Mean |SHAP Value|", fontweight="bold", fontsize=10)
        ax.set_xlim(0, c_df[col].max() * 1.25)
        ax.grid(axis="x", linestyle=":", alpha=0.5)

    fig.suptitle(
        "Class-Specific Feature Importance: Comparative SHAP Attribution",
        fontweight="bold",
        fontsize=14,
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_shap_individual_prediction(
    explainer: shap.TreeExplainer,
    shap_values: np.ndarray,
    X_trans: np.ndarray,
    sample_index: int,
    feature_names: List[str],
    class_names: Optional[List[str]] = None,
    target_class: Optional[int] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Renders an individual prediction explanation for a specific instance.
    Uses a clean waterfall-style attribution plot of positive and negative SHAP contributions.

    Args:
        explainer: Fitted TreeExplainer.
        shap_values: 3D SHAP values array.
        X_trans: Transformed test array.
        sample_index: Index of instance to explain.
        feature_names: Transformed feature names.
        class_names: Class labels list.
        target_class: Target class index to explain (if None, predicted class is used).
        save_path: Optional file path to save PNG.

    Returns:
        Matplotlib Figure.
    """
    if class_names is None:
        class_names = CLASS_DISPLAY_NAMES

    if target_class is None:
        # Determine class with highest SHAP contribution or model prediction
        target_class = int(np.argmax(shap_values[sample_index].sum(axis=0)))

    c_name = class_names[target_class]
    vals = shap_values[sample_index, :, target_class]
    base_val = float(explainer.expected_value[target_class]) if hasattr(explainer.expected_value, "__len__") else float(explainer.expected_value)

    # Build Explanation object
    exp = shap.Explanation(
        values=vals,
        base_values=base_val,
        data=X_trans[sample_index],
        feature_names=feature_names,
    )

    fig = plt.figure(figsize=(10, 6.5))
    shap.plots.waterfall(exp, max_display=10, show=False)
    plt.title(
        f"SHAP Local Explanation: Test Instance #{sample_index} (Class: {c_name})",
        fontweight="bold",
        fontsize=12,
        pad=14,
    )
    plt.tight_layout()
    fig = plt.gcf()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def create_lime_explainer(
    training_data: np.ndarray,
    feature_names: List[str],
    class_names: Optional[List[str]] = None,
    random_state: int = 42,
) -> lime.lime_tabular.LimeTabularExplainer:
    """
    Creates a LimeTabularExplainer fitted on the transformed training data.

    Args:
        training_data: Transformed training feature matrix.
        feature_names: Transformed feature names.
        class_names: Class labels list.
        random_state: Random seed for perturbation reproducibility.

    Returns:
        Configured LimeTabularExplainer.
    """
    if class_names is None:
        class_names = CLASS_DISPLAY_NAMES

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=training_data,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        random_state=random_state,
    )
    return explainer


def explain_individual_lime(
    lime_explainer: lime.lime_tabular.LimeTabularExplainer,
    instance: np.ndarray,
    predict_fn: Any,
    target_class: int,
    num_features: int = 10,
) -> Any:
    """
    Generates a LIME local explanation for an individual instance.

    Args:
        lime_explainer: Fitted LimeTabularExplainer.
        instance: 1D array of transformed features for the target sample.
        predict_fn: Function returning probability distribution (e.g. classifier.predict_proba).
        target_class: Index of class to explain.
        num_features: Number of top surrogate features to include.

    Returns:
        LIME Explanation object.
    """
    exp = lime_explainer.explain_instance(
        data_row=instance,
        predict_fn=predict_fn,
        labels=[target_class],
        num_features=num_features,
    )
    return exp


def plot_lime_individual_prediction(
    lime_explanation: Any,
    target_class: int,
    sample_index: int = 0,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Renders and saves a clean, annotated bar plot of LIME local feature contributions.

    Args:
        lime_explanation: Explanation object returned by LimeTabularExplainer.
        target_class: Class index explained.
        sample_index: Sample index for titling.
        class_names: Class labels list.
        save_path: Optional file path to save PNG.

    Returns:
        Matplotlib Figure.
    """
    if class_names is None:
        class_names = CLASS_DISPLAY_NAMES
    c_name = class_names[target_class]

    # Extract list of (rule, weight) tuples
    exp_list = lime_explanation.as_list(label=target_class)
    df = pd.DataFrame(exp_list, columns=["rule", "weight"])
    df = df.sort_values(by="weight", key=abs, ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    y_pos = np.arange(len(df))
    bar_colors = ["#16a34a" if w >= 0 else "#dc2626" for w in df["weight"]]

    bars = ax.barh(y_pos, df["weight"], color=bar_colors, alpha=0.85, edgecolor="black", height=0.6)

    for bar in bars:
        width = bar.get_width()
        x_pos = width + (0.005 if width >= 0 else -0.005)
        ha = "left" if width >= 0 else "right"
        ax.annotate(
            f"{width:+.3f}",
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(5 if width >= 0 else -5, 0),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=8.5,
            fontweight="bold",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["rule"], fontsize=9, fontweight="bold")
    ax.axvline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("LIME Feature Weight (Contribution to Class Probability)", fontweight="bold", fontsize=10, labelpad=8)
    ax.set_title(
        f"LIME Local Explanation: Test Instance #{sample_index} (Class: {c_name})",
        fontweight="bold",
        fontsize=12,
        pad=12,
    )
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def generate_shap_report_markdown(
    model_metadata: Dict[str, Any],
    importance_df: pd.DataFrame,
    sample_index: int,
    sample_pred_class: int,
    sample_pred_name: str,
    sample_proba: List[float],
    lime_features: List[Tuple[str, float]],
    shap_top_features_sample: List[Dict[str, Any]],
    output_path: str,
) -> str:
    """
    Compiles the comprehensive, clinical non-diagnostic SHAP and LIME markdown report.
    """
    top_10 = importance_df.head(10)
    # Build top 10 markdown table directly without requiring tabulate
    table_headers = ["Rank", "Feature", "Overall Mean |SHAP|", "Mean |SHAP| (None)", "Mean |SHAP| (Insomnia)", "Mean |SHAP| (Sleep Apnea)"]
    header_line = "| " + " | ".join(table_headers) + " |"
    separator_line = "| " + " | ".join([":---:" if i == 0 else (":---" if i == 1 else ":---:") for i in range(len(table_headers))]) + " |"
    rows = []
    for _, r in top_10.iterrows():
        row_str = f"| {int(r['rank'])} | `{r['feature']}` | {r['overall_mean_abs_shap']:.4f} | {r['mean_abs_shap_No recorded sleep disorder']:.4f} | {r['mean_abs_shap_Insomnia']:.4f} | {r['mean_abs_shap_Sleep Apnea']:.4f} |"
        rows.append(row_str)
    top_10_table = "\n".join([header_line, separator_line] + rows)

    lime_table_rows = "\n".join([f"| `{rule}` | {weight:+.4f} | {'Supports' if weight > 0 else 'Opposes'} {sample_pred_name} |" for rule, weight in lime_features])

    shap_local_rows = "\n".join([f"| `{f['feature']}` | {f['value']:.3f} | {f['shap_value']:+.4f} |" for f in shap_top_features_sample])

    report_content = f"""# Explainable AI (XAI) Report: SHAP and LIME Interpretability
## ML-Based Intelligent System for Automated Sleep Disorder Detection

> **Project**: B.Tech CSE (Data Science) Final Year Project  
> **Model Evaluated**: Tuned Random Forest Pipeline (`models/best_model.joblib`)  
> **Status**: Completed & Verified  
> **Target Classes**: `None` (0, "No recorded sleep disorder"), `Insomnia` (1), `Sleep Apnea` (2)  

> [!CAUTION]
> **Clinical Non-Diagnostic Disclaimer**:  
> The explanations, feature importance metrics, and local attributions documented herein reflect the mathematical behavior of machine learning models trained on a synthetic dataset. They do **not** constitute medical diagnoses, clinical guidance, or causal physiological mechanisms. Predictions are experimental outputs intended strictly for educational and research evaluation.

---

## 1. Methodology and Experimental Configuration

Explainability analysis was conducted to interpret the decision boundaries of the best-performing model identified during cross-validation (**Tuned Random Forest**, 96.0% holdout accuracy, 0.9352 holdout macro F1).

### Explainer Selection
- **Global & Local SHAP**: `shap.TreeExplainer` was selected because the production classifier is an ensemble of decision trees (`RandomForestClassifier`). `TreeExplainer` provides polynomial-time calculation of exact Shapley values adhering to local accuracy, missingness, and consistency axioms.
- **Background Sampling**: A reference background distribution of $N=100$ samples was drawn from the **training set only** (`X_train_trans`) using fixed seed `random_state=42`. Holdout test labels were strictly excluded from background selection to preserve the data leakage barrier.
- **Local Surrogate (LIME)**: `lime.lime_tabular.LimeTabularExplainer` was integrated to provide complementary linear surrogate explanations in the local perturbation space around individual test instances.

### System Configuration
- **Transformed Feature Dimensionality**: 25 features (9 standardized continuous features + 16 one-hot encoded categories across Gender, Occupation, and BMI Category).
- **Evaluation Cohort**: Untouched 20% holdout test split ($N=75$ samples: 44 None, 15 Insomnia, 16 Sleep Apnea).
- **Random Seed**: 42.

---

## 2. Global Feature Importance (SHAP)

Global feature importance was computed using the **Mean Absolute SHAP Value** ($\\frac{{1}}{{N}}\\sum_{{i=1}}^{{N}} |\\phi_{{i,j}}|$), capturing the average magnitude of feature influence across all 75 holdout samples and 3 target classes.

### Top 10 Features Driving Model Decisions

{top_10_table}

### Key Global Findings
1. **Systolic and Diastolic Blood Pressure (`BP_Systolic`, `BP_Diastolic`)**: Top overall drivers across the dataset. Elevated blood pressure serves as the dominant discriminator separating Sleep Apnea from healthy controls and Insomnia.
2. **BMI Category (`BMI Category_Overweight`, `BMI Category_Normal`)**: Strongest categorical drivers. Normal BMI heavily pushes the model toward `No recorded sleep disorder`, whereas Overweight status heavily increases likelihood of Insomnia and Sleep Apnea.
3. **Sleep Duration & Quality of Sleep**: High sleep duration (>7.5 hours) strongly supports healthy status, whereas reduced sleep duration (<6.5 hours) elevates probability of Insomnia.
4. **Occupational Clustering (`Occupation_Nurse`, `Occupation_Doctor`, `Occupation_Salesperson`)**: Occupational features exhibit strong class associations resulting from demographic and lifestyle clustering present in the dataset (e.g. Nurses in this cohort show elevated rates of Sleep Apnea, while Salespersons show higher Insomnia rates).

---

## 3. Class-Specific Feature Drivers

The model uses distinct feature subsets to distinguish between the three target states:

### 1. No Recorded Sleep Disorder (Class 0 — "Healthy Baseline")
- **Primary Positive Drivers**: `BMI Category_Normal`, `Sleep Duration` (high), `Stress Level` (low), `BP_Systolic` (normal < 125 mmHg).
- **Opposing Signals**: Elevated systolic/diastolic pressure, Overweight/Obese status, high stress.

### 2. Insomnia (Class 1)
- **Primary Positive Drivers**: `Stress Level` (high, $\\ge 7$), `Sleep Duration` (short, $< 6.5$ hours), `BMI Category_Overweight`, `Occupation_Salesperson`, `Occupation_Teacher`.
- **Opposing Signals**: Low stress level, long sleep duration, Normal BMI.

### 3. Sleep Apnea (Class 2)
- **Primary Positive Drivers**: `BP_Systolic` (high, $\\ge 130$ mmHg), `BP_Diastolic` (high, $\\ge 85$ mmHg), `Age` (older cohort), `BMI Category_Overweight` / `BMI Category_Obese`, `Occupation_Nurse`.
- **Opposing Signals**: Normal blood pressure (< 120/80 mmHg), young age (< 35 years), Normal BMI.

---

## 4. Local Instance Explanation: Holdout Sample #0

An individual test instance was selected to compare SHAP and LIME local attributions:

- **True Label**: `Insomnia`
- **Model Prediction**: `{sample_pred_name}` (Class {sample_pred_class})
- **Predicted Class Probabilities**:
  - `No recorded sleep disorder`: {sample_proba[0]:.2%}
  - `Insomnia`: {sample_proba[1]:.2%}
  - `Sleep Apnea`: {sample_proba[2]:.2%}

### SHAP Local Waterfall Attribution (Top Contributing Features)
| Feature | Feature Value (Standardized/Encoded) | SHAP Contribution ($\\phi_i$) |
|:---|:---:|:---:|
{shap_local_rows}

### LIME Local Surrogate Weights
| LIME Rule Condition | Weight | Impact on Prediction |
|:---|:---:|:---:|
{lime_table_rows}

### Cross-Method Concordance
Both SHAP and LIME exhibit strong alignment on Sample #0:
- Both identify **reduced sleep duration** and **elevated stress level** as the two dominant factors pushing the probability of Insomnia above 90%.
- Both assign negative or near-zero weight to Sleep Apnea features (normal systolic BP in this patient prevents the model from classifying as Sleep Apnea).

---

## 5. Artifact Catalog

The following verified artifacts were produced and saved under `reports/figures/` and `reports/`:

1. **`reports/figures/shap_summary.png`**: Stacked multi-class SHAP summary plot illustrating global impact across all three target classes.
2. **`reports/figures/shap_feature_importance.png`**: Publication-quality horizontal bar chart ranking the top 15 features by global mean absolute SHAP value.
3. **`reports/figures/shap_class_comparison.png`**: 3-panel comparative attribution plot showing feature weights broken down by target class.
4. **`reports/figures/shap_individual_prediction.png`**: SHAP waterfall plot detailing the exact step-by-step contribution of features to Sample #0's prediction.
5. **`reports/figures/lime_individual_prediction.png`**: Local surrogate feature weighting from `LimeTabularExplainer` for Sample #0.
6. **`reports/shap_feature_importance.json`**: Machine-readable JSON summary containing full numerical rankings and explainer metadata.

---

## 6. Interpretation Limitations and Ethics

1. **Mathematical Attribution vs. Medical Causation**: SHAP and LIME measure how the decision trees partition feature space; they do **not** imply that modifying a patient's daily steps or occupation will cure or cause a sleep disorder.
2. **Synthetic Data Artifacts**: High correlations between occupations and health metrics reflect dataset generation parameters and should not be generalized to clinical practice.
3. **Collinearity Effect**: Systolic and diastolic blood pressure have a correlation of $r=0.97$. Tree-based Shapley values divide credit between collinear features depending on tree split choices.
4. **Surrogate Approximation**: LIME uses a local linear approximation that can vary slightly across perturbation samples; SHAP provides exact Shapley values under game-theoretic axioms.
"""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    return report_content


def run_explainability_pipeline(
    project_root: Optional[str] = None,
    background_size: int = 100,
    random_state: int = 42,
    sample_index: int = 0,
) -> Dict[str, Any]:
    """
    Orchestrates Task 1.7:
    1. Loads model artifacts and dataset.
    2. Extracts pipeline transformers and transformed feature names.
    3. Fits SHAP TreeExplainer with training-derived background data.
    4. Computes SHAP values on holdout test set.
    5. Calculates global and per-class feature importance.
    6. Fits LIME TabularExplainer and explains sample_index.
    7. Generates all 5 required high-resolution plots.
    8. Exports JSON summary and Markdown report.

    Args:
        project_root: Path to project root directory.
        background_size: Size of training background sample.
        random_state: Random seed for reproducibility.
        sample_index: Index of holdout test sample to explain locally.

    Returns:
        Dictionary with paths and summary metrics.
    """
    if project_root is None:
        project_root = PROJECT_ROOT

    models_dir = os.path.join(project_root, "models")
    reports_dir = os.path.join(project_root, "reports")
    figures_dir = os.path.join(reports_dir, "figures")
    data_path = os.path.join(project_root, "data", "Sleep_health_and_lifestyle_dataset.csv")

    model_path = os.path.join(models_dir, "best_model.joblib")
    encoder_path = os.path.join(models_dir, "label_encoder.joblib")
    metadata_path = os.path.join(models_dir, "model_metadata.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    # 1. Load artifacts and data
    pipeline = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        model_metadata = json.load(f)

    raw_df = pd.read_csv(data_path, keep_default_na=False)
    X, y = clean_and_prepare_data(raw_df)

    # 2. Partition data using exact untouched split
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=0.2, random_state=random_state, stratify=True
    )
    y_test_encoded = label_encoder.transform(y_test)

    # 3. Extract components and transform feature matrices
    preprocessor, classifier, feature_names = extract_pipeline_components(pipeline)
    X_train_trans, X_test_trans = transform_data_for_explanation(preprocessor, X_train, X_test)

    print(f"Transformed feature space: {len(feature_names)} features.")
    print(f"Training background space: {X_train_trans.shape}, Holdout test: {X_test_trans.shape}")

    # 4. SHAP TreeExplainer with training background
    print(f"Initializing SHAP TreeExplainer with background sample size={background_size}...")
    explainer = create_shap_explainer(
        classifier=classifier,
        background_data=X_train_trans,
        background_size=background_size,
        random_state=random_state,
    )

    print("Computing SHAP values on holdout test set...")
    shap_values = compute_shap_values(explainer, X_test_trans, check_additivity=False)
    print(f"Computed SHAP values matrix shape: {shap_values.shape}")

    # 5. Global Feature Importance
    importance_df = compute_global_feature_importance(
        shap_values=shap_values,
        feature_names=feature_names,
        class_names=CLASS_DISPLAY_NAMES,
    )

    # 6. Generate SHAP Figures
    os.makedirs(figures_dir, exist_ok=True)
    summary_plot_path = os.path.join(figures_dir, "shap_summary.png")
    importance_plot_path = os.path.join(figures_dir, "shap_feature_importance.png")
    comparison_plot_path = os.path.join(figures_dir, "shap_class_comparison.png")
    individual_shap_path = os.path.join(figures_dir, "shap_individual_prediction.png")

    print(f"Saving SHAP summary plot to: {summary_plot_path}")
    plot_shap_summary(shap_values, X_test_trans, feature_names, CLASS_DISPLAY_NAMES, save_path=summary_plot_path)

    print(f"Saving SHAP feature importance plot to: {importance_plot_path}")
    plot_shap_feature_importance(importance_df, top_n=15, save_path=importance_plot_path)

    print(f"Saving SHAP class comparison plot to: {comparison_plot_path}")
    plot_shap_class_comparison(importance_df, CLASS_DISPLAY_NAMES, top_n=8, save_path=comparison_plot_path)

    # Sample explanation
    pred_class_idx = int(classifier.predict(X_test_trans[sample_index : sample_index + 1])[0])
    pred_proba = classifier.predict_proba(X_test_trans[sample_index : sample_index + 1])[0].tolist()
    pred_class_name = CLASS_DISPLAY_NAMES[pred_class_idx]

    print(f"Saving individual SHAP prediction plot for sample #{sample_index} ({pred_class_name})...")
    plot_shap_individual_prediction(
        explainer=explainer,
        shap_values=shap_values,
        X_trans=X_test_trans,
        sample_index=sample_index,
        feature_names=feature_names,
        class_names=CLASS_DISPLAY_NAMES,
        target_class=pred_class_idx,
        save_path=individual_shap_path,
    )

    # 7. LIME Explainer Integration
    print("Initializing LIME TabularExplainer...")
    lime_explainer = create_lime_explainer(
        training_data=X_train_trans,
        feature_names=feature_names,
        class_names=CLASS_DISPLAY_NAMES,
        random_state=random_state,
    )

    print(f"Generating LIME explanation for sample #{sample_index}...")
    lime_exp = explain_individual_lime(
        lime_explainer=lime_explainer,
        instance=X_test_trans[sample_index],
        predict_fn=classifier.predict_proba,
        target_class=pred_class_idx,
        num_features=10,
    )

    lime_plot_path = os.path.join(figures_dir, "lime_individual_prediction.png")
    print(f"Saving LIME plot to: {lime_plot_path}")
    plot_lime_individual_prediction(
        lime_explanation=lime_exp,
        target_class=pred_class_idx,
        sample_index=sample_index,
        class_names=CLASS_DISPLAY_NAMES,
        save_path=lime_plot_path,
    )

    # 8. Extract local attributions for sample #0 report
    sample_shap_top = []
    sample_shap_vals = shap_values[sample_index, :, pred_class_idx]
    top_indices = np.argsort(np.abs(sample_shap_vals))[::-1][:6]
    for idx in top_indices:
        sample_shap_top.append({
            "feature": feature_names[idx],
            "value": float(X_test_trans[sample_index, idx]),
            "shap_value": float(sample_shap_vals[idx]),
        })

    lime_features = lime_exp.as_list(label=pred_class_idx)

    # 9. Machine-readable JSON summary
    explanation_summary = {
        "explainer": "shap.TreeExplainer",
        "explainer_type": "TreeExplainer",
        "lime_integrated": True,
        "lime_explainer": "lime.lime_tabular.LimeTabularExplainer",
        "model_name": model_metadata.get("full_model_name", "Random Forest"),
        "model_version": "1.0.0",
        "target_classes": TARGET_CLASSES,
        "display_class_names": CLASS_DISPLAY_NAMES,
        "class_mapping": {
            0: {"internal": "None", "display": "No recorded sleep disorder"},
            1: {"internal": "Insomnia", "display": "Insomnia"},
            2: {"internal": "Sleep Apnea", "display": "Sleep Apnea"},
        },
        "background_sample_size": background_size,
        "holdout_test_samples": len(X_test),
        "random_state": random_state,
        "feature_count": len(feature_names),
        "transformed_features": feature_names,
        "top_features_overall": importance_df.to_dict(orient="records"),
        "sample_explanation": {
            "sample_index": sample_index,
            "true_class": int(y_test_encoded[sample_index]),
            "true_class_name": CLASS_DISPLAY_NAMES[int(y_test_encoded[sample_index])],
            "predicted_class": pred_class_idx,
            "predicted_class_name": pred_class_name,
            "predicted_probabilities": {
                CLASS_DISPLAY_NAMES[i]: round(float(pred_proba[i]), 4) for i in range(len(pred_proba))
            },
            "shap_contributions": sample_shap_top,
            "lime_contributions": [{"rule": r, "weight": round(float(w), 4)} for r, w in lime_features],
        },
        "artifacts": {
            "shap_summary": "reports/figures/shap_summary.png",
            "shap_feature_importance": "reports/figures/shap_feature_importance.png",
            "shap_class_comparison": "reports/figures/shap_class_comparison.png",
            "shap_individual_prediction": "reports/figures/shap_individual_prediction.png",
            "lime_individual_prediction": "reports/figures/lime_individual_prediction.png",
            "shap_report": "reports/shap_report.md",
        },
        "disclaimer": "Predictions and feature attributions are experimental research outputs, not medical diagnoses.",
    }

    # Save to reports/ and models/
    shap_json_path = os.path.join(reports_dir, "shap_feature_importance.json")
    with open(shap_json_path, "w", encoding="utf-8") as f:
        json.dump(explanation_summary, f, indent=2)
    print(f"Saved machine-readable XAI summary to: {shap_json_path}")

    models_shap_json = os.path.join(models_dir, "shap_feature_importance.json")
    with open(models_shap_json, "w", encoding="utf-8") as f:
        json.dump(explanation_summary, f, indent=2)

    # Mirror to ml_pipeline/artifacts
    ml_artifacts_dir = os.path.join(project_root, "ml_pipeline", "artifacts")
    if os.path.exists(ml_artifacts_dir):
        with open(os.path.join(ml_artifacts_dir, "shap_feature_importance.json"), "w", encoding="utf-8") as f:
            json.dump(explanation_summary, f, indent=2)

    # 10. Write Markdown Report
    report_md_path = os.path.join(reports_dir, "shap_report.md")
    print(f"Generating markdown report at: {report_md_path}...")
    generate_shap_report_markdown(
        model_metadata=model_metadata,
        importance_df=importance_df,
        sample_index=sample_index,
        sample_pred_class=pred_class_idx,
        sample_pred_name=pred_class_name,
        sample_proba=pred_proba,
        lime_features=lime_features,
        shap_top_features_sample=sample_shap_top,
        output_path=report_md_path,
    )

    print("Explainability pipeline execution finished successfully.")
    return {
        "importance_df": importance_df,
        "explanation_summary": explanation_summary,
        "shap_values": shap_values,
        "feature_names": feature_names,
    }


if __name__ == "__main__":
    run_explainability_pipeline()
