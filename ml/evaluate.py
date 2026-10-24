"""
Evaluation and Metrics Module
for 'ML-Based Intelligent System for Automated Sleep Disorder Detection Using Lifestyle and Health Data'.

Provides:
- Comprehensive classification metrics (Accuracy, Balanced Accuracy, Macro/Weighted F1, Precision, Recall).
- Per-class performance metrics for None, Insomnia, and Sleep Apnea.
- Confusion matrix computation and high-resolution visualization.
- Multi-model comparison bar chart generation.
"""

import os
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)

TARGET_CLASSES = ["None", "Insomnia", "Sleep Apnea"]


def compute_multiclass_roc_auc(
    y_true: Union[np.ndarray, List[Any], pd.Series],
    y_proba: Union[np.ndarray, List[List[float]], pd.DataFrame],
    labels: Optional[List[Any]] = None,
    target_names: Optional[List[str]] = None,
    model_classes: Optional[Union[np.ndarray, List[Any]]] = None,
) -> Dict[str, Any]:
    """
    Computes One-vs-Rest (OvR) multiclass ROC-AUC metrics with mathematical verification.

    Validates:
    - Probability matrix dimensions and alignment with classes.
    - Class presence in y_true (handles unsupported single-class or absent-class slices).
    - Macro and weighted ROC-AUC across supported classes.

    Args:
        y_true: Ground truth target labels (encoded integers or strings).
        y_proba: Predicted probability matrix of shape (n_samples, n_classes).
        labels: Expected class labels corresponding to probability columns.
        target_names: Human-readable names for classes.
        model_classes: Optional classes_ array from model for alignment verification.

    Returns:
        Dictionary with macro_roc_auc, weighted_roc_auc, per_class_roc_auc, and support status.
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_proba, dtype=float)

    if labels is None:
        if model_classes is not None:
            labels = list(model_classes)
        elif len(y_t) > 0 and isinstance(y_t[0], (int, np.integer)):
            labels = [0, 1, 2]
        else:
            labels = TARGET_CLASSES

    if target_names is None:
        target_names = [TARGET_CLASSES[i] if i < len(TARGET_CLASSES) else str(lbl) for i, lbl in enumerate(labels)]

    # Validate shape
    if y_p.ndim != 2:
        raise ValueError(f"y_proba must be a 2D array, got shape {y_p.shape}")
    if len(y_t) != y_p.shape[0]:
        raise ValueError(f"Sample count mismatch: len(y_true)={len(y_t)} vs y_proba.shape[0]={y_p.shape[0]}")
    if y_p.shape[1] != len(labels):
        raise ValueError(
            f"Probability column count ({y_p.shape[1]}) does not match number of labels ({len(labels)})"
        )

    # Check model_classes alignment if provided
    if model_classes is not None:
        mc_list = list(model_classes)
        if mc_list != list(labels):
            if set(mc_list) == set(labels):
                col_indices = [mc_list.index(lbl) for lbl in labels]
                y_p = y_p[:, col_indices]
            else:
                raise ValueError(f"model_classes {mc_list} does not match expected labels {labels}")

    # Check for NaN / Inf
    if np.isnan(y_p).any() or np.isinf(y_p).any():
        raise ValueError("y_proba contains NaN or Infinite values.")

    unique_classes = np.unique(y_t)
    per_class_auc: Dict[str, Optional[float]] = {}
    unsupported_classes: List[str] = []

    # If only 1 class overall is present in y_true, ROC-AUC is mathematically undefined
    if len(unique_classes) < 2:
        for name in target_names:
            per_class_auc[name] = None
        return {
            "macro_roc_auc": None,
            "weighted_roc_auc": None,
            "per_class_roc_auc": per_class_auc,
            "is_supported": False,
            "unsupported_classes": list(target_names),
            "reason": "At least 2 distinct classes must be present in y_true to evaluate ROC-AUC.",
        }

    # Per-class One-vs-Rest calculation
    for i, name in enumerate(target_names):
        lbl = labels[i]
        y_binary = (y_t == lbl).astype(int)
        # Binary ROC-AUC requires both positive (1) and negative (0) samples in y_binary
        if len(np.unique(y_binary)) < 2:
            per_class_auc[name] = None
            unsupported_classes.append(name)
        else:
            score = float(roc_auc_score(y_binary, y_p[:, i]))
            per_class_auc[name] = round(score, 4)

    # Macro & Weighted ROC-AUC across all classes
    macro_auc: Optional[float] = None
    weighted_auc: Optional[float] = None
    is_supported: bool = False

    all_present = all(lbl in unique_classes for lbl in labels)
    if all_present:
        try:
            macro_auc = round(float(roc_auc_score(y_t, y_p, labels=labels, multi_class="ovr", average="macro")), 4)
            weighted_auc = round(float(roc_auc_score(y_t, y_p, labels=labels, multi_class="ovr", average="weighted")), 4)
            is_supported = True
        except Exception:
            valid = [s for s in per_class_auc.values() if s is not None]
            if len(valid) >= 2:
                macro_auc = round(float(np.mean(valid)), 4)
                is_supported = True
    else:
        valid = [s for s in per_class_auc.values() if s is not None]
        if len(valid) >= 2:
            macro_auc = round(float(np.mean(valid)), 4)
            is_supported = True

    return {
        "macro_roc_auc": macro_auc,
        "weighted_roc_auc": weighted_auc,
        "per_class_roc_auc": per_class_auc,
        "is_supported": is_supported,
        "unsupported_classes": unsupported_classes,
    }


def compute_classification_metrics(
    y_true: Union[np.ndarray, List[Any], pd.Series],
    y_pred: Union[np.ndarray, List[Any], pd.Series],
    labels: Optional[List[Any]] = None,
    target_names: Optional[List[str]] = None,
    y_proba: Optional[Union[np.ndarray, List[List[float]], pd.DataFrame]] = None,
    model_classes: Optional[Union[np.ndarray, List[Any]]] = None,
) -> Dict[str, Any]:
    """
    Computes rigorous multi-class classification metrics including optional multiclass ROC-AUC.

    Args:
        y_true: Ground truth target labels.
        y_pred: Predicted target labels.
        labels: Optional label order / IDs.
        target_names: Human-readable names for classes.
        y_proba: Optional predicted probability scores for ROC-AUC.
        model_classes: Optional model classes_ array for probability column alignment.

    Returns:
        Dictionary containing overall, per-class, and ROC-AUC performance metrics.
    """
    if labels is None:
        labels = [0, 1, 2] if (isinstance(y_true[0], (int, np.integer))) else TARGET_CLASSES
    if target_names is None:
        target_names = TARGET_CLASSES

    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    acc = float(accuracy_score(y_t, y_p))
    bal_acc = float(balanced_accuracy_score(y_t, y_p))
    macro_f1 = float(f1_score(y_t, y_p, labels=labels, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_t, y_p, labels=labels, average="weighted", zero_division=0))
    macro_prec = float(precision_score(y_t, y_p, labels=labels, average="macro", zero_division=0))
    weighted_prec = float(precision_score(y_t, y_p, labels=labels, average="weighted", zero_division=0))
    macro_rec = float(recall_score(y_t, y_p, labels=labels, average="macro", zero_division=0))
    weighted_rec = float(recall_score(y_t, y_p, labels=labels, average="weighted", zero_division=0))

    cm = confusion_matrix(y_t, y_p, labels=labels)

    # Per-class metrics
    per_class_f1 = f1_score(y_t, y_p, labels=labels, average=None, zero_division=0)
    per_class_prec = precision_score(y_t, y_p, labels=labels, average=None, zero_division=0)
    per_class_rec = recall_score(y_t, y_p, labels=labels, average=None, zero_division=0)

    per_class: Dict[str, Dict[str, Any]] = {}
    for i, name in enumerate(target_names):
        lbl = labels[i]
        support = int((y_t == lbl).sum())
        per_class[name] = {
            "precision": round(float(per_class_prec[i]), 4),
            "recall": round(float(per_class_rec[i]), 4),
            "f1": round(float(per_class_f1[i]), 4),
            "support": support,
        }

    report_dict = classification_report(
        y_t,
        y_p,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    report_str = classification_report(
        y_t,
        y_p,
        labels=labels,
        target_names=target_names,
        zero_division=0,
    )

    metrics_res: Dict[str, Any] = {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "macro_precision": round(macro_prec, 4),
        "weighted_precision": round(weighted_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "weighted_recall": round(weighted_rec, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "classification_report_dict": report_dict,
        "classification_report_str": report_str,
    }

    if y_proba is not None:
        roc_res = compute_multiclass_roc_auc(
            y_true=y_t,
            y_proba=y_proba,
            labels=labels,
            target_names=target_names,
            model_classes=model_classes,
        )
        metrics_res["macro_roc_auc"] = roc_res["macro_roc_auc"]
        metrics_res["weighted_roc_auc"] = roc_res["weighted_roc_auc"]
        metrics_res["per_class_roc_auc"] = roc_res["per_class_roc_auc"]
        metrics_res["roc_auc_supported"] = roc_res["is_supported"]
        for name in target_names:
            if name in per_class and name in roc_res["per_class_roc_auc"]:
                per_class[name]["roc_auc"] = roc_res["per_class_roc_auc"][name]

    return metrics_res


def plot_confusion_matrix(
    cm: Union[np.ndarray, List[List[int]]],
    classes: List[str] = TARGET_CLASSES,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    normalize: bool = False,
) -> plt.Figure:
    """
    Renders and optionally saves an annotated confusion matrix heatmap.
    """
    cm_arr = np.asarray(cm)
    if normalize:
        cm_norm = cm_arr.astype("float") / cm_arr.sum(axis=1)[:, np.newaxis]
        annot = np.array(
            [[f"{val}\n({norm:.1%})" for val, norm in zip(row_val, row_norm)]
             for row_val, row_norm in zip(cm_arr, cm_norm)]
        )
    else:
        annot = np.array([[f"{val}" for val in row] for row in cm_arr])

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_arr,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=True,
        ax=ax,
        linewidths=1.0,
        linecolor="gray",
    )
    ax.set_title(title, fontweight="bold", fontsize=13, pad=12)
    ax.set_xlabel("Predicted Label", fontweight="bold", labelpad=8)
    ax.set_ylabel("True Label", fontweight="bold", labelpad=8)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_model_comparison_bars(
    comparison_df: pd.DataFrame,
    metric_cols: Tuple[str, ...] = ("cv_macro_f1_mean", "cv_balanced_acc_mean", "cv_accuracy_mean"),
    metric_labels: Tuple[str, ...] = ("Macro F1", "Balanced Accuracy", "Accuracy"),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plots a multi-metric bar chart comparing all candidate classifiers.
    """
    df = comparison_df.sort_values(by=metric_cols[0], ascending=True).copy()

    fig, ax = plt.subplots(figsize=(12, 7))
    y_pos = np.arange(len(df))
    bar_height = 0.25

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for i, (col, label) in enumerate(zip(metric_cols, metric_labels)):
        offset = (i - 1) * bar_height
        bars = ax.barh(
            y_pos + offset,
            df[col],
            height=bar_height,
            label=label,
            color=colors[i % len(colors)],
            alpha=0.85,
            edgecolor="black",
        )
        # Add labels to bars
        for bar in bars:
            width = bar.get_width()
            ax.annotate(
                f"{width:.3f}",
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(4, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=8,
                fontweight="bold",
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["model_name"], fontweight="bold")
    ax.set_xlabel("5-Fold Cross-Validation Score", fontweight="bold")
    ax.set_title("Classifier Comparison across Stratified 5-Fold Cross-Validation", fontweight="bold", fontsize=13, pad=12)
    ax.set_xlim(0, 1.1)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_roc_curve_multiclass(
    y_true: Union[np.ndarray, List[Any], pd.Series],
    y_proba: Union[np.ndarray, List[List[float]], pd.DataFrame],
    labels: Optional[List[Any]] = None,
    classes: Optional[List[str]] = None,
    title: str = "Multiclass Receiver Operating Characteristic (One-vs-Rest)",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Renders and optionally saves a publication-quality One-vs-Rest ROC curve plot.

    Args:
        y_true: Ground truth target labels.
        y_proba: Predicted probability matrix (n_samples, n_classes).
        labels: Class labels matching probability columns (e.g., [0, 1, 2]).
        classes: Human-readable names for classes (e.g., ['None', 'Insomnia', 'Sleep Apnea']).
        title: Plot title.
        save_path: Optional file path to save the 300-DPI PNG.

    Returns:
        Matplotlib Figure object.
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_proba, dtype=float)

    if labels is None:
        labels = [0, 1, 2] if (len(y_t) > 0 and isinstance(y_t[0], (int, np.integer))) else TARGET_CLASSES
    if classes is None:
        classes = [TARGET_CLASSES[i] if i < len(TARGET_CLASSES) else str(lbl) for i, lbl in enumerate(labels)]

    fig, ax = plt.subplots(figsize=(8, 6.5))

    # Baseline chance diagonal
    ax.plot([0, 1], [0, 1], linestyle="--", lw=1.5, color="#6b7280", label="Chance (AUC = 0.5000)", alpha=0.85)

    palette = ["#1d4ed8", "#d97706", "#059669", "#7c3aed", "#dc2626"]
    valid_fprs: List[np.ndarray] = []
    valid_tprs: List[np.ndarray] = []
    valid_aucs: List[float] = []

    for i, name in enumerate(classes):
        lbl = labels[i]
        y_binary = (y_t == lbl).astype(int)
        if len(np.unique(y_binary)) >= 2:
            fpr, tpr, _ = roc_curve(y_binary, y_p[:, i])
            auc_score = roc_auc_score(y_binary, y_p[:, i])
            color = palette[i % len(palette)]
            ax.plot(
                fpr,
                tpr,
                lw=2.2,
                color=color,
                label=f"{name} (AUC = {auc_score:.4f})",
            )
            valid_fprs.append(fpr)
            valid_tprs.append(tpr)
            valid_aucs.append(auc_score)
        else:
            # Class not present in holdout slice
            ax.plot([], [], label=f"{name} (Unsupported - single class in slice)", linestyle=":")

    # Plot Macro-average ROC curve if at least 2 valid classes
    if len(valid_fprs) >= 2:
        macro_auc = float(np.mean(valid_aucs))
        all_fpr = np.unique(np.concatenate(valid_fprs))
        mean_tpr = np.zeros_like(all_fpr)
        for fpr, tpr in zip(valid_fprs, valid_tprs):
            mean_tpr += np.interp(all_fpr, fpr, tpr)
        mean_tpr /= len(valid_fprs)
        ax.plot(
            all_fpr,
            mean_tpr,
            label=f"Macro-average (AUC = {macro_auc:.4f})",
            color="#111827",
            linestyle=":",
            linewidth=2.5,
        )

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontweight="bold", fontsize=11, labelpad=8)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontweight="bold", fontsize=11, labelpad=8)
    ax.set_title(title, fontweight="bold", fontsize=13, pad=12)
    ax.legend(loc="lower right", frameon=True, facecolor="#ffffff", edgecolor="#d1d5db", fontsize=10, shadow=False)
    ax.grid(True, linestyle=":", alpha=0.5, color="#9ca3af")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def evaluate_saved_model(
    project_root: Optional[str] = None,
    save_plots: bool = True,
    save_json: bool = True,
) -> Dict[str, Any]:
    """
    Evaluates the serialized model on the untouched holdout test set.
    Reuses existing artifacts and produces updated test_evaluation.json and ROC curve plots.
    """
    import joblib
    import sys
    if project_root is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from ml.preprocess import clean_and_prepare_data, split_data, TARGET_CLASSES, ID_COL

    models_dir = os.path.join(project_root, "models")
    reports_dir = os.path.join(project_root, "reports")
    figures_dir = os.path.join(reports_dir, "figures")
    data_path = os.path.join(project_root, "data", "Sleep_health_and_lifestyle_dataset.csv")

    model_path = os.path.join(models_dir, "best_model.joblib")
    encoder_path = os.path.join(models_dir, "label_encoder.joblib")

    if not os.path.exists(model_path) or not os.path.exists(encoder_path):
        raise FileNotFoundError("Serialized model or label encoder not found in models/ directory.")

    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)

    raw_df = pd.read_csv(data_path, keep_default_na=False)
    X, y = clean_and_prepare_data(raw_df)

    # Recreate the exact untouched 80/20 stratified holdout split
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=0.2, random_state=42, stratify=True
    )
    y_test_encoded = label_encoder.transform(y_test)

    # Predictions & Probabilities
    y_pred_encoded = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Compute comprehensive metrics with ROC-AUC
    metrics = compute_classification_metrics(
        y_true=y_test_encoded,
        y_pred=y_pred_encoded,
        labels=[0, 1, 2],
        target_names=TARGET_CLASSES,
        y_proba=y_proba,
        model_classes=getattr(model, "classes_", None),
    )

    if save_plots:
        os.makedirs(figures_dir, exist_ok=True)
        # 1. Confusion Matrix
        cm_path = os.path.join(figures_dir, "best_model_confusion_matrix.png")
        plot_confusion_matrix(
            cm=metrics["confusion_matrix"],
            classes=TARGET_CLASSES,
            title="Holdout Test Confusion Matrix (Random Forest)",
            save_path=cm_path,
            normalize=True,
        )
        # 2. ROC Curve
        roc_path = os.path.join(figures_dir, "roc_auc_ovr.png")
        plot_roc_curve_multiclass(
            y_true=y_test_encoded,
            y_proba=y_proba,
            labels=[0, 1, 2],
            classes=TARGET_CLASSES,
            title="Receiver Operating Characteristic (ROC) - One-vs-Rest (OvR)",
            save_path=roc_path,
        )

    if save_json:
        # Update models/test_evaluation.json
        test_eval_path = os.path.join(models_dir, "test_evaluation.json")
        with open(test_eval_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        # Update models/model_metadata.json
        metadata_path = os.path.join(models_dir, "model_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            meta["test_metrics"]["macro_roc_auc"] = metrics["macro_roc_auc"]
            meta["test_metrics"]["weighted_roc_auc"] = metrics["weighted_roc_auc"]
            for cls_name, cls_stats in metrics["per_class"].items():
                if cls_name in meta["test_metrics"]["per_class"]:
                    meta["test_metrics"]["per_class"][cls_name]["roc_auc"] = cls_stats.get("roc_auc")
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            # Mirror to ml_pipeline/artifacts
            artifacts_dir = os.path.join(project_root, "ml_pipeline", "artifacts")
            if os.path.exists(artifacts_dir):
                with open(os.path.join(artifacts_dir, "test_evaluation.json"), "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2)
                art_meta_path = os.path.join(artifacts_dir, "model_metadata.json")
                if os.path.exists(art_meta_path):
                    with open(art_meta_path, "w", encoding="utf-8") as f:
                        json.dump(meta, f, indent=2)

    return metrics


if __name__ == "__main__":
    print("Executing holdout evaluation with ROC-AUC...")
    res = evaluate_saved_model()
    print("Holdout Test Metrics Summary:")
    print(f"  - Accuracy:          {res['accuracy']:.4f}")
    print(f"  - Balanced Accuracy: {res['balanced_accuracy']:.4f}")
    print(f"  - Macro F1:          {res['macro_f1']:.4f}")
    print(f"  - Weighted F1:       {res['weighted_f1']:.4f}")
    print(f"  - Macro ROC-AUC:     {res['macro_roc_auc']:.4f}")
    print(f"  - Weighted ROC-AUC:  {res['weighted_roc_auc']:.4f}")
    print("Per-class ROC-AUC:")
    for cls_name, cls_dict in res["per_class"].items():
        print(f"  - {cls_name:12s}: ROC-AUC = {cls_dict.get('roc_auc', 'N/A')}")
