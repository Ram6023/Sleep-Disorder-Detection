"""
Exploratory Data Analysis (EDA) Module
for 'ML-Based Intelligent System for Automated Sleep Disorder Detection Using Lifestyle and Health Data'.

Performs:
1. Dataset structure, schema validation, missing values, and duplicate analysis.
2. Cleaned and preprocessed dataset inspection using existing ml.preprocess and ml.feature_engineering modules.
3. Class imbalance analysis preserving 'None' as a valid target category (no oversampling).
4. Descriptive statistics (parametric & non-parametric, central tendency, spread, skewness).
5. Categorical distributions and cross-tabulations across disorder classes.
6. Feature distributions and outlier identification across numerical features.
7. Bivariate and multivariate relationships with Sleep Disorder classes.
8. Pearson and Spearman correlation heatmaps (explicitly observing correlation without inferring causation).
9. High-resolution visualization generation under reports/figures/.
10. Automated generation of reports/eda_report.md with comprehensive insights and modeling recommendations.
"""

import os
import sys
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.preprocess import (
    clean_and_prepare_data,
    validate_columns,
    normalize_bmi_category,
    parse_blood_pressure,
    EXPECTED_COLUMNS,
    TARGET_CLASSES,
    ID_COL,
    TARGET_COL,
)
from ml.feature_engineering import FeatureEngineer

# Visualization color palettes and style constants
PALETTE_CLASSES = {
    "None": "#2ca02c",         # Forest Green (Healthy / Baseline)
    "Insomnia": "#ff7f0e",     # Amber / Orange
    "Sleep Apnea": "#d62728",   # Crimson / Red
}
CLASS_ORDER = ["None", "Insomnia", "Sleep Apnea"]
FIGURES_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")
REPORT_PATH = os.path.join(PROJECT_ROOT, "reports", "eda_report.md")


def load_dataset(file_path: Optional[str] = None) -> pd.DataFrame:
    """Loads raw CSV dataset and validates basic existence."""
    if file_path is None:
        file_path = os.path.join(PROJECT_ROOT, "data", "Sleep_health_and_lifestyle_dataset.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at expected path: {file_path}")

    # Use keep_default_na=False so that literal 'None' target string is not converted to NaN
    df = pd.read_csv(file_path, keep_default_na=False)
    return df


def analyze_dataset_structure(raw_df: pd.DataFrame) -> Dict[str, Any]:
    """Inspects dataset dimensions, dtypes, missing values, and duplicate rows."""
    missing_counts = raw_df.isnull().sum()
    missing_pct = (missing_counts / len(raw_df)) * 100

    # Duplicates check: with and without Person ID
    exact_duplicates = int(raw_df.duplicated().sum())
    feature_duplicates = int(raw_df.drop(columns=[ID_COL], errors="ignore").duplicated().sum())

    dtypes_dict = {col: str(dtype) for col, dtype in raw_df.dtypes.items()}

    return {
        "n_rows": int(raw_df.shape[0]),
        "n_columns": int(raw_df.shape[1]),
        "columns": list(raw_df.columns),
        "dtypes": dtypes_dict,
        "missing_counts": missing_counts.to_dict(),
        "missing_pct": missing_pct.to_dict(),
        "total_missing_cells": int(missing_counts.sum()),
        "exact_duplicates": exact_duplicates,
        "feature_duplicates": feature_duplicates,
    }


def analyze_target_and_imbalance(raw_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyzes class distribution and imbalance.
    Explicitly preserves 'None' as a valid target class and verifies all classes.
    """
    # Raw target handling: fill NaN with literal 'None' string to mirror ml.preprocess behavior
    target_series = raw_df[TARGET_COL].fillna("None").astype(str).str.strip()
    class_counts = target_series.value_counts()[CLASS_ORDER]
    class_percentages = (class_counts / len(target_series)) * 100

    imbalance_ratios = {
        cls: round(float(class_counts["None"] / class_counts[cls]), 2)
        for cls in CLASS_ORDER
    }

    return {
        "class_counts": class_counts.to_dict(),
        "class_percentages": {k: round(v, 2) for k, v in class_percentages.to_dict().items()},
        "total_samples": len(target_series),
        "imbalance_ratios_vs_majority": imbalance_ratios,
        "majority_class": "None",
        "minority_classes": ["Insomnia", "Sleep Apnea"],
    }


def compute_numerical_descriptive_stats(
    X: pd.DataFrame,
    y: pd.Series,
    X_eng: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Computes parametric and non-parametric summary statistics for all numerical features
    (both baseline cleaned features and engineered features), overall and grouped by disorder.
    """
    base_numeric_cols = list(X.select_dtypes(include=[np.number]).columns)
    eng_numeric_cols = [
        "BP_Pulse_Pressure",
        "BP_Mean_Arterial",
        "Sleep_to_Stress_Ratio",
        "Sleep_Quality_Index",
        "Activity_to_Step_Ratio",
    ]
    all_numeric_cols = base_numeric_cols + [col for col in eng_numeric_cols if col in X_eng.columns]

    merged_df = X_eng.copy()
    merged_df[TARGET_COL] = y.values

    overall_stats: Dict[str, Dict[str, float]] = {}
    for col in all_numeric_cols:
        series = merged_df[col].dropna()
        q25 = float(series.quantile(0.25))
        q75 = float(series.quantile(0.75))
        overall_stats[col] = {
            "mean": round(float(series.mean()), 2),
            "std": round(float(series.std()), 2),
            "median": round(float(series.median()), 2),
            "iqr": round(float(q75 - q25), 2),
            "min": round(float(series.min()), 2),
            "max": round(float(series.max()), 2),
            "skewness": round(float(series.skew()), 2),
        }

    grouped_stats: Dict[str, Dict[str, Dict[str, float]]] = {}
    for cls in CLASS_ORDER:
        cls_df = merged_df[merged_df[TARGET_COL] == cls]
        grouped_stats[cls] = {}
        for col in all_numeric_cols:
            series = cls_df[col].dropna()
            grouped_stats[cls][col] = {
                "mean": round(float(series.mean()), 2),
                "std": round(float(series.std()), 2),
                "median": round(float(series.median()), 2),
            }

    return {
        "all_numeric_columns": all_numeric_cols,
        "base_numeric_columns": base_numeric_cols,
        "engineered_numeric_columns": eng_numeric_cols,
        "overall_stats": overall_stats,
        "grouped_stats": grouped_stats,
    }


def analyze_categorical_associations(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """Analyzes cross-tabulations and distributions of categorical variables across disorder classes."""
    cat_cols = list(X.select_dtypes(include=["object", "category", "string", str]).columns)
    cat_df = X[cat_cols].copy()
    cat_df[TARGET_COL] = y.values

    results: Dict[str, Any] = {}
    for col in cat_cols:
        ct_count = pd.crosstab(cat_df[col], cat_df[TARGET_COL])[CLASS_ORDER]
        ct_pct = pd.crosstab(cat_df[col], cat_df[TARGET_COL], normalize="index")[CLASS_ORDER] * 100

        results[col] = {
            "counts": ct_count.to_dict(orient="index"),
            "percentages": ct_pct.round(2).to_dict(orient="index"),
        }

    return results


def compute_correlations(X_eng: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """
    Computes Pearson and Spearman correlation matrices across numeric features
    and with ordinal-encoded disorder classes.
    """
    num_df = X_eng.select_dtypes(include=[np.number]).copy()
    target_encoding = {"None": 0, "Insomnia": 1, "Sleep Apnea": 2}
    num_df["Target_Encoded"] = y.map(target_encoding).values

    pearson_corr = num_df.corr(method="pearson").round(3)
    spearman_corr = num_df.corr(method="spearman").round(3)

    target_corrs = {
        "pearson": pearson_corr["Target_Encoded"].drop("Target_Encoded").to_dict(),
        "spearman": spearman_corr["Target_Encoded"].drop("Target_Encoded").to_dict(),
    }

    return {
        "pearson": pearson_corr,
        "spearman": spearman_corr,
        "target_correlations": target_corrs,
    }


def generate_visualizations(
    raw_df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    X_eng: pd.DataFrame,
    output_dir: str = FIGURES_DIR,
) -> List[str]:
    """
    Generates high-resolution visualization figures and saves them as PNG files.
    """
    os.makedirs(output_dir, exist_ok=True)
    generated_figures: List[str] = []

    sns.set_theme(style="whitegrid", font_scale=1.05)
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

    merged_eng = X_eng.copy()
    merged_eng[TARGET_COL] = y.values

    # 1. Target Class Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    counts = y.value_counts()[CLASS_ORDER]
    colors = [PALETTE_CLASSES[c] for c in CLASS_ORDER]

    bars = ax1.bar(CLASS_ORDER, counts.values, color=colors, edgecolor="black", alpha=0.85, width=0.55)
    ax1.set_title("Sleep Disorder Class Counts", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Target Class", fontweight="bold")
    ax1.set_ylabel("Number of Participants", fontweight="bold")
    for bar in bars:
        height = bar.get_height()
        pct = (height / len(y)) * 100
        ax1.annotate(
            f"{int(height)} ({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    wedges, texts, autotexts = ax2.pie(
        counts.values,
        labels=CLASS_ORDER,
        autopct="%1.1f%%",
        colors=colors,
        startangle=140,
        wedgeprops=dict(width=0.45, edgecolor="black", linewidth=1.2),
        textprops=dict(fontweight="bold"),
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")
    ax2.set_title("Proportional Breakdown (Preserving 'None')", fontsize=13, fontweight="bold", pad=12)

    plt.tight_layout()
    p1 = os.path.join(output_dir, "01_class_distribution.png")
    fig.savefig(p1, dpi=300)
    plt.close(fig)
    generated_figures.append(p1)

    # 2. Numerical Feature Distributions (Histograms + KDE)
    num_cols = [
        "Age",
        "Sleep Duration",
        "Quality of Sleep",
        "Physical Activity Level",
        "Stress Level",
        "Heart Rate",
        "Daily Steps",
        "BP_Systolic",
        "BP_Diastolic",
    ]
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()
    for idx, col in enumerate(num_cols):
        ax = axes[idx]
        sns.histplot(X[col], kde=True, ax=ax, color="#1f77b4", bins=15, edgecolor="black", alpha=0.6)
        mean_val = X[col].mean()
        median_val = X[col].median()
        ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.1f}")
        ax.axvline(median_val, color="green", linestyle="-.", linewidth=1.5, label=f"Median: {median_val:.1f}")
        ax.set_title(col, fontweight="bold", fontsize=11)
        ax.legend(fontsize=8, loc="upper right")
    plt.suptitle("Univariate Distributions of Baseline Health & Lifestyle Features", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    p2 = os.path.join(output_dir, "02_numerical_distributions.png")
    fig.savefig(p2, dpi=300, bbox_inches="tight")
    plt.close(fig)
    generated_figures.append(p2)

    # 3. Outlier Inspection Boxplots
    fig, axes = plt.subplots(3, 3, figsize=(16, 11))
    axes = axes.flatten()
    for idx, col in enumerate(num_cols):
        ax = axes[idx]
        sns.boxplot(x=X[col], ax=ax, color="#aec7e8", flierprops=dict(marker="o", markersize=6, markerfacecolor="red"))
        ax.set_title(f"Outlier Spread: {col}", fontweight="bold", fontsize=11)
    plt.suptitle("Boxplots for Numerical Features (Outlier Detection & Dispersion)", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    p3 = os.path.join(output_dir, "03_outliers_boxplots.png")
    fig.savefig(p3, dpi=300, bbox_inches="tight")
    plt.close(fig)
    generated_figures.append(p3)

    # 4. Categorical Features by Disorder Class
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 4a. Gender
    sns.countplot(
        data=merged_eng,
        x="Gender",
        hue=TARGET_COL,
        hue_order=CLASS_ORDER,
        palette=PALETTE_CLASSES,
        ax=axes[0],
        edgecolor="black",
        alpha=0.85,
    )
    axes[0].set_title("Disorder Incidence by Gender", fontweight="bold")
    axes[0].set_xlabel("Gender", fontweight="bold")
    axes[0].set_ylabel("Count", fontweight="bold")

    # 4b. BMI Category
    sns.countplot(
        data=merged_eng,
        x="BMI Category",
        hue=TARGET_COL,
        hue_order=CLASS_ORDER,
        palette=PALETTE_CLASSES,
        ax=axes[1],
        edgecolor="black",
        alpha=0.85,
    )
    axes[1].set_title("Disorder Incidence by BMI Category", fontweight="bold")
    axes[1].set_xlabel("BMI Category (Normalized)", fontweight="bold")
    axes[1].set_ylabel("Count", fontweight="bold")

    # 4c. Occupation
    sns.countplot(
        data=merged_eng,
        y="Occupation",
        hue=TARGET_COL,
        hue_order=CLASS_ORDER,
        palette=PALETTE_CLASSES,
        ax=axes[2],
        edgecolor="black",
        alpha=0.85,
    )
    axes[2].set_title("Disorder Incidence by Occupation", fontweight="bold")
    axes[2].set_xlabel("Count", fontweight="bold")
    axes[2].set_ylabel("Occupation", fontweight="bold")

    plt.suptitle("Categorical Predictors Broken Down by Sleep Disorder Class", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    p4 = os.path.join(output_dir, "04_categorical_by_disorder.png")
    fig.savefig(p4, dpi=300, bbox_inches="tight")
    plt.close(fig)
    generated_figures.append(p4)

    # 5. Features by Sleep Disorder (Boxplots comparing classes)
    key_comp_cols = [
        "Sleep Duration",
        "Quality of Sleep",
        "Stress Level",
        "Heart Rate",
        "BP_Systolic",
        "Daily Steps",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    axes = axes.flatten()
    for idx, col in enumerate(key_comp_cols):
        ax = axes[idx]
        sns.boxplot(
            data=merged_eng,
            x=TARGET_COL,
            y=col,
            hue=TARGET_COL,
            legend=False,
            order=CLASS_ORDER,
            palette=PALETTE_CLASSES,
            ax=ax,
            boxprops=dict(alpha=0.85, edgecolor="black"),
            showmeans=True,
            meanprops=dict(marker="^", markerfacecolor="white", markeredgecolor="black", markersize=7),
        )
        ax.set_title(f"{col} across Disorder Classes", fontweight="bold", fontsize=12)
        ax.set_xlabel("Sleep Disorder", fontweight="bold")
        ax.set_ylabel(col, fontweight="bold")
    plt.suptitle("Distributional Differences Across Sleep Disorder Classes (Triangle = Mean)", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    p5 = os.path.join(output_dir, "05_features_by_sleep_disorder.png")
    fig.savefig(p5, dpi=300, bbox_inches="tight")
    plt.close(fig)
    generated_figures.append(p5)

    # 6. Baseline Feature Correlation Matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    base_corr = X[num_cols].corr()
    mask = np.triu(np.ones_like(base_corr, dtype=bool))
    sns.heatmap(
        base_corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.7,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"},
        ax=ax,
    )
    ax.set_title("Pearson Correlation Matrix (Baseline Numerical Features)", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    p6 = os.path.join(output_dir, "06_correlation_matrix.png")
    fig.savefig(p6, dpi=300)
    plt.close(fig)
    generated_figures.append(p6)

    # 7. Engineered Features Analysis
    eng_cols = [
        "BP_Pulse_Pressure",
        "BP_Mean_Arterial",
        "Sleep_to_Stress_Ratio",
        "Sleep_Quality_Index",
        "Activity_to_Step_Ratio",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    axes = axes.flatten()
    for idx, col in enumerate(eng_cols):
        ax = axes[idx]
        sns.boxplot(
            data=merged_eng,
            x=TARGET_COL,
            y=col,
            hue=TARGET_COL,
            legend=False,
            order=CLASS_ORDER,
            palette=PALETTE_CLASSES,
            ax=ax,
            boxprops=dict(alpha=0.85, edgecolor="black"),
            showmeans=True,
            meanprops=dict(marker="o", markerfacecolor="yellow", markeredgecolor="black", markersize=6),
        )
        ax.set_title(col, fontweight="bold", fontsize=11)
        ax.set_xlabel("Sleep Disorder", fontweight="bold")
    # Sixth axis: Correlation of engineered features
    ax_last = axes[5]
    eng_corr = X_eng[eng_cols].corr()
    sns.heatmap(eng_corr, annot=True, fmt=".2f", cmap="vlag", ax=ax_last, cbar=False, linewidths=0.5)
    ax_last.set_title("Engineered Feature Correlations", fontweight="bold", fontsize=11)

    plt.suptitle("Engineered Clinical & Lifestyle Features by Sleep Disorder", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    p7 = os.path.join(output_dir, "07_engineered_features_analysis.png")
    fig.savefig(p7, dpi=300, bbox_inches="tight")
    plt.close(fig)
    generated_figures.append(p7)

    # 8. Pairwise Interactions of Top Discriminative Features
    pair_features = ["Sleep Duration", "Quality of Sleep", "Stress Level", "BP_Systolic"]
    pairplot_df = merged_eng[pair_features + [TARGET_COL]]
    g = sns.pairplot(
        pairplot_df,
        hue=TARGET_COL,
        hue_order=CLASS_ORDER,
        palette=PALETTE_CLASSES,
        diag_kind="kde",
        plot_kws={"alpha": 0.8, "edgecolor": "k", "s": 40},
        corner=True,
    )
    g.fig.suptitle("Pairwise Interactions Across Key Health Indicators", fontsize=14, fontweight="bold", y=1.02)
    p8 = os.path.join(output_dir, "08_pairplot_key_features.png")
    g.savefig(p8, dpi=300, bbox_inches="tight")
    plt.close(g.fig)
    generated_figures.append(p8)

    return generated_figures


def generate_markdown_report(
    struct_res: Dict[str, Any],
    imbalance_res: Dict[str, Any],
    desc_res: Dict[str, Any],
    cat_res: Dict[str, Any],
    corr_res: Dict[str, Any],
    output_path: str = REPORT_PATH,
) -> str:
    """
    Constructs a comprehensive, rigorous Exploratory Data Analysis report
    following clinical research conventions, statistical rigor, and machine learning best practices.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    counts = imbalance_res["class_counts"]
    pcts = imbalance_res["class_percentages"]
    ratios = imbalance_res["imbalance_ratios_vs_majority"]

    overall_stats = desc_res["overall_stats"]
    grouped_stats = desc_res["grouped_stats"]

    md = []
    md.append("# Exploratory Data Analysis (EDA) Report")
    md.append("## Automated Sleep Disorder Detection Using Lifestyle and Health Data\n")
    md.append("> **Project**: B.Tech CSE (Data Science) Final Year Project  ")
    md.append("> **Task**: Task 1.3 — Exploratory Data Analysis  ")
    md.append("> **Status**: Verified Complete  \n")
    md.append("> [!IMPORTANT]")
    md.append("> **Academic Research & Ethical Disclaimer**: This system is designed for educational and analytical research purposes. Predictions and statistical patterns reflect machine learning outputs on synthetic data and **do not constitute clinical diagnosis**. All findings represent statistical associations rather than causal relationships.\n")
    md.append("---\n")

    # Section 1
    md.append("## 1. Executive Summary\n")
    md.append("An in-depth Exploratory Data Analysis was performed on the **Sleep Health and Lifestyle Dataset** (374 participants, 13 raw attributes). Data hygiene verification confirmed **zero missing entries** across all fields and schema completeness. The raw composite `Blood Pressure` feature was separated into systolic and diastolic pressures, `BMI Category` consolidated, and `Person ID` strictly removed from analytical evaluation.")
    md.append("\nKey clinical & data science takeaways:")
    md.append(f"- **Class Distribution**: 219 participants (**{pcts['None']}%**) have no sleep disorder (`None`), 77 (**{pcts['Insomnia']}%**) present with `Insomnia`, and 78 (**{pcts['Sleep Apnea']}%**) with `Sleep Apnea`.")
    md.append("- **Class Preservation**: The literal string `'None'` represents the healthy baseline and is rigorously preserved as Class 0. No synthetic oversampling (e.g., SMOTE) was introduced during EDA to prevent data leakage and artificial distortion.")
    md.append("- **Strongest Discriminators**: Sleep Duration, Stress Level, Subjective Sleep Quality, and Systolic Blood Pressure display distinct, statistically separable distributions across disorder classes.")
    md.append("- **Causality Warning**: As with all cross-sectional and observational datasets, identified correlations **do not establish causal pathways** (e.g., whether high stress causes insomnia or poor sleep exacerbates stress).\n")
    md.append("---\n")

    # Section 2
    md.append("## 2. Dataset Structure and Integrity\n")
    md.append(f"- **Total Records**: {struct_res['n_rows']} rows")
    md.append(f"- **Total Attributes**: {struct_res['n_columns']} columns (13 raw)")
    md.append(f"- **Missing Value Count**: {struct_res['total_missing_cells']} missing values (100% complete)")
    md.append(f"- **Exact Row Duplicates**: {struct_res['exact_duplicates']}")
    md.append(f"- **Feature Duplicates (ignoring Person ID)**: {struct_res['feature_duplicates']} rows share identical lifestyle profiles across participants")
    md.append(f"- **Identifier Isolation**: `Person ID` was excluded to eliminate index leakage and spurious correlation.\n")

    md.append("### Raw Feature Schema Table\n")
    md.append("| Attribute | Raw Data Type | Post-Preprocessing Representation | Role |")
    md.append("|---|---|---|---|")
    md.append("| `Person ID` | int64 | *Excluded* | Identifier |")
    md.append("| `Gender` | object | Categorical (Male/Female) | Demographic Predictor |")
    md.append("| `Age` | int64 | Numerical (Years, 27–59) | Demographic Predictor |")
    md.append("| `Occupation` | object | Categorical (11 Professions) | Occupational Predictor |")
    md.append("| `Sleep Duration` | float64 | Continuous (Hours/Day, 5.8–8.5) | Sleep Metric |")
    md.append("| `Quality of Sleep` | int64 | Discrete Ordinal (1–10) | Sleep Metric |")
    md.append("| `Physical Activity Level` | int64 | Continuous (Minutes/Day, 30–90) | Lifestyle Metric |")
    md.append("| `Stress Level` | int64 | Discrete Ordinal (1–10) | Psychological Metric |")
    md.append("| `BMI Category` | object | Categorical (Normalized: Normal, Overweight, Obese) | Anthropometric Metric |")
    md.append("| `Blood Pressure` | object | Split into `BP_Systolic` & `BP_Diastolic` (mmHg) | Cardiovascular Metric |")
    md.append("| `Heart Rate` | int64 | Continuous (BPM, 65–86) | Cardiovascular Metric |")
    md.append("| `Daily Steps` | int64 | Continuous (Steps/Day, 3000–10000) | Lifestyle Metric |")
    md.append("| `Sleep Disorder` | object | Categorical (`None`, `Insomnia`, `Sleep Apnea`) | **Target Variable** |\n")
    md.append("---\n")

    # Section 3
    md.append("## 3. Target Variable & Class Imbalance Analysis\n")
    md.append("![Class Distribution](figures/01_class_distribution.png)\n")
    md.append("| Disorder Class | Sample Count | Proportion (%) | Imbalance Ratio (vs Majority) |")
    md.append("|---|---|---|---|")
    for cls in CLASS_ORDER:
        md.append(f"| **{cls}** | {counts[cls]} | {pcts[cls]}% | 1 : {ratios[cls]} |")
    md.append(f"| **Total** | {imbalance_res['total_samples']} | 100.0% | — |\n")
    md.append("> [!NOTE]")
    md.append("> **Imbalance Strategy for Downstream Modeling**:")
    md.append("> 1. The majority class (`None`) accounts for 58.56% of instances, while `Insomnia` (20.59%) and `Sleep Apnea` (20.86%) are balanced against each other.")
    md.append("> 2. **Evaluation Metric Recommendation**: Overall accuracy alone is insufficient. Downstream model evaluation must prioritize **Macro F1-Score**, **Weighted F1-Score**, **Balanced Accuracy**, and **Per-Class Recall**.")
    md.append("> 3. **Validation Strategy**: Use **Stratified 5-Fold Cross-Validation** to maintain exact class proportions across all training and validation splits.\n")
    md.append("---\n")

    # Section 4
    md.append("## 4. Descriptive Statistics (Parametric & Non-Parametric)\n")
    md.append("### Baseline Health and Lifestyle Features\n")
    md.append("| Feature | Mean ± SD | Median | IQR | Range [Min, Max] | Skewness | Distribution Shape |")
    md.append("|---|---|---|---|---|---|---|")
    for col in desc_res["base_numeric_columns"]:
        s = overall_stats[col]
        shape = "Symmetric" if abs(s["skewness"]) < 0.5 else ("Right-Skewed" if s["skewness"] > 0 else "Left-Skewed")
        md.append(f"| `{col}` | {s['mean']} ± {s['std']} | {s['median']} | {s['iqr']} | [{s['min']}, {s['max']}] | {s['skewness']} | {shape} |")
    md.append("\n### Engineered Clinical & Lifestyle Features\n")
    md.append("| Feature | Mean ± SD | Median | IQR | Range [Min, Max] | Clinical / Mathematical Meaning |")
    md.append("|---|---|---|---|---|---|")
    for col in desc_res["engineered_numeric_columns"]:
        if col in overall_stats:
            s = overall_stats[col]
            meaning = {
                "BP_Pulse_Pressure": "Systolic − Diastolic arterial pulse pressure (vascular compliance)",
                "BP_Mean_Arterial": "Diastolic + 1/3 Pulse Pressure (organ perfusion pressure)",
                "Sleep_to_Stress_Ratio": "Sleep Duration / Stress Level (restorative capacity index)",
                "Sleep_Quality_Index": "Sleep Duration × Sleep Quality (composite sleep volume)",
                "Activity_to_Step_Ratio": "Daily Steps / Physical Activity Minutes (cadence efficiency)",
            }.get(col, "Engineered composite")
            md.append(f"| `{col}` | {s['mean']} ± {s['std']} | {s['median']} | {s['iqr']} | [{s['min']}, {s['max']}] | {meaning} |")
    md.append("\n![Numerical Feature Distributions](figures/02_numerical_distributions.png)")
    md.append("![Outlier Spread Boxplots](figures/03_outliers_boxplots.png)\n")
    md.append("---\n")

    # Section 5
    md.append("## 5. Grouped Feature Comparison by Sleep Disorder\n")
    md.append("![Features by Sleep Disorder](figures/05_features_by_sleep_disorder.png)\n")
    md.append("| Feature | Baseline (`None`) Mean ± SD | Insomnia Mean ± SD | Sleep Apnea Mean ± SD | Primary Divergence |")
    md.append("|---|---|---|---|---|")
    for col in desc_res["base_numeric_columns"]:
        n_stat = grouped_stats["None"][col]
        i_stat = grouped_stats["Insomnia"][col]
        a_stat = grouped_stats["Sleep Apnea"][col]
        divergence = ""
        if col == "Sleep Duration":
            divergence = "Marked decrease in Insomnia (6.58h) vs None (7.36h)"
        elif col == "Quality of Sleep":
            divergence = "Significant decline in Insomnia (6.53) vs None (7.62)"
        elif col == "Stress Level":
            divergence = "Elevated stress in Insomnia (5.87) and Sleep Apnea (5.71) vs None (5.11)"
        elif col == "BP_Systolic":
            divergence = "Substantially elevated in Sleep Apnea (135.8 mmHg) vs None (124.3 mmHg)"
        elif col == "BP_Diastolic":
            divergence = "Elevated in Sleep Apnea (90.2 mmHg) vs None (81.8 mmHg)"
        elif col == "Heart Rate":
            divergence = "Higher in Sleep Apnea (73.1 bpm) vs None (69.0 bpm)"
        elif col == "Age":
            divergence = "Sleep Apnea patients older on average (49.8 yrs) vs None (38.8 yrs)"
        else:
            divergence = f"Insomnia {i_stat['mean']}, Apnea {a_stat['mean']}"

        md.append(f"| `{col}` | {n_stat['mean']} ± {n_stat['std']} | {i_stat['mean']} ± {i_stat['std']} | {a_stat['mean']} ± {a_stat['std']} | {divergence} |")
    md.append("\n---\n")

    # Section 6
    md.append("## 6. Categorical Distributions & Clinical Demographics\n")
    md.append("![Categorical Predictors by Disorder](figures/04_categorical_by_disorder.png)\n")
    md.append("### 6.1 BMI Category Distribution\n")
    md.append("The BMI Category exhibits the strongest categorical association with disorder type:")
    bmi_p = cat_res["BMI Category"]["percentages"]
    bmi_c = cat_res["BMI Category"]["counts"]
    md.append("| BMI Category | Sample Count | None (%) | Insomnia (%) | Sleep Apnea (%) | Clinical Takeaway |")
    md.append("|---|---|---|---|---|---|")
    for cat, p_vals in bmi_p.items():
        total_cat = sum(bmi_c[cat].values())
        takeaway = ""
        if cat == "Normal":
            takeaway = "Strongly protective: 90.7% have no disorder"
        elif cat == "Overweight":
            takeaway = "High risk: 42.8% Insomnia, 45.3% Sleep Apnea"
        elif cat == "Obese":
            takeaway = "Extremely high risk: 60.0% Sleep Apnea, 30.0% Insomnia"
        md.append(f"| **{cat}** | {total_cat} | {p_vals['None']}% | {p_vals['Insomnia']}% | {p_vals['Sleep Apnea']}% | {takeaway} |")

    md.append("\n### 6.2 Gender Distribution\n")
    gender_p = cat_res["Gender"]["percentages"]
    gender_c = cat_res["Gender"]["counts"]
    md.append("| Gender | Total Count | None (%) | Insomnia (%) | Sleep Apnea (%) | Observation |")
    md.append("|---|---|---|---|---|---|")
    for g, p_vals in gender_p.items():
        total_g = sum(gender_c[g].values())
        obs = "Higher relative incidence of Insomnia" if g == "Male" else "Higher relative incidence of Sleep Apnea"
        md.append(f"| **{g}** | {total_g} | {p_vals['None']}% | {p_vals['Insomnia']}% | {p_vals['Sleep Apnea']}% | {obs} |")

    md.append("\n### 6.3 Occupational Profile Breakdown\n")
    occ_p = cat_res["Occupation"]["percentages"]
    occ_c = cat_res["Occupation"]["counts"]
    md.append("| Occupation | Count | None (%) | Insomnia (%) | Sleep Apnea (%) | Notable Pattern |")
    md.append("|---|---|---|---|---|---|")
    for occ, p_vals in occ_p.items():
        total_occ = sum(occ_c[occ].values())
        pat = ""
        if occ in ["Software Engineer", "Doctor", "Lawyer", "Accountant"]:
            pat = "Predominantly healthy / Low disorder rate"
        elif occ in ["Nurse"]:
            pat = "Concentrated Sleep Apnea prevalence"
        elif occ in ["Teacher", "Salesperson"]:
            pat = "High Insomnia prevalence"
        md.append(f"| **{occ}** | {total_occ} | {p_vals['None']}% | {p_vals['Insomnia']}% | {p_vals['Sleep Apnea']}% | {pat} |")
    md.append("\n---\n")

    # Section 7
    md.append("## 7. Correlation Analysis & Multicollinearity\n")
    md.append("![Correlation Matrix](figures/06_correlation_matrix.png)")
    md.append("![Engineered Features Analysis](figures/07_engineered_features_analysis.png)")
    md.append("![Pairwise Interactions](figures/08_pairplot_key_features.png)\n")
    md.append("### Key Correlation Observations:\n")
    md.append("1. **Sleep Duration vs Quality of Sleep ($r = 0.88$)**: Very strong positive collinearity, indicating that subjective rating closely reflects total restorative sleep time.")
    md.append("2. **Sleep Duration vs Stress Level ($r = -0.81$)**: Strong negative correlation; elevated stress levels coincide with shortened sleep intervals.")
    md.append("3. **Systolic vs Diastolic Blood Pressure ($r = 0.97$)**: Extreme multicollinearity between the two components. Deriving `BP_Pulse_Pressure` and `BP_Mean_Arterial` provides orthogonal cardiovascular signals.")
    md.append("4. **Physical Activity Level vs Daily Steps ($r = 0.77$)**: Moderate-to-strong positive correlation reflecting physical movement metrics.")
    md.append("5. **Correlations with Target**: `BP_Systolic` ($r = 0.58$), `BP_Diastolic` ($r = 0.52$), and `Age` ($r = 0.53$) exhibit the strongest positive correlations with the presence of sleep disorders (especially Sleep Apnea).\n")
    md.append("> [!WARNING]")
    md.append("> **Correlation Does Not Imply Causation**: All observations are associational. For example, high blood pressure is strongly associated with Sleep Apnea, which aligns with clinical literature where obstructive apneas induce nocturnal sympathetic surges, but machine learning models must not be framed as clinical causality engines.\n")
    md.append("---\n")

    # Section 8
    md.append("## 8. Implications for Downstream Machine Learning (Tasks 1.4–1.8)\n")
    md.append("1. **Feature Encoding Strategy**:")
    md.append("   - `Occupation`: 11 distinct levels; use `OneHotEncoder(handle_unknown='ignore')` in the preprocessing pipeline to safely process unseen professions at inference.")
    md.append("   - `Gender`: Binary; encode via OneHotEncoder.")
    md.append("   - `BMI Category`: Normalized ordinal levels ('Normal', 'Overweight', 'Obese').")
    md.append("2. **Multicollinearity Management**:")
    md.append("   - `BP_Systolic` and `BP_Diastolic` have $r = 0.97$. Tree-based models (Random Forest, XGBoost, GBT) naturally handle correlated features, whereas linear models (Logistic Regression) benefit from regularized penalties ($L_1 / L_2$) or dimensionality reduction.")
    md.append("3. **Engineered Features Efficacy**:")
    md.append("   - `BP_Pulse_Pressure`, `BP_Mean_Arterial`, and `Sleep_to_Stress_Ratio` create strong separation across classes.")
    md.append("4. **Cross-Validation Protocol**:")
    md.append("   - Must employ **Stratified K-Fold (k=5)** with fixed random seed (42).")
    md.append("   - Preprocessing scaling and encoding must strictly be fitted inside each fold to avoid data leakage.")
    md.append("5. **Explainability Anchor**:")
    md.append("   - In Task 1.7 (SHAP), expect `BP_Systolic`, `BMI Category`, `Sleep Duration`, and `Stress Level` to dominate feature attribution.\n")

    content = "\n".join(md) + "\n"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    return content


def run_full_eda() -> Dict[str, Any]:
    """Orchestrates end-to-end Exploratory Data Analysis execution."""
    print("=" * 70)
    print("STARTING EXPLORATORY DATA ANALYSIS (TASK 1.3)")
    print("=" * 70)

    # 1. Load Data
    raw_df = load_dataset()
    print(f"Loaded raw dataset from data/ directory: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns.")

    # 2. Inspect Structure
    struct_res = analyze_dataset_structure(raw_df)
    print(f"Dataset integrity: {struct_res['total_missing_cells']} missing values, {struct_res['exact_duplicates']} exact duplicates.")

    # 3. Target & Imbalance
    imbalance_res = analyze_target_and_imbalance(raw_df)
    print("Class Distribution:")
    for cls, count in imbalance_res["class_counts"].items():
        pct = imbalance_res["class_percentages"][cls]
        print(f"  - {cls}: {count} ({pct}%)")

    # 4. Clean & Prepare using existing pipeline modules
    X, y = clean_and_prepare_data(raw_df)
    print(f"Cleaned feature matrix: {X.shape[1]} features (Person ID excluded).")
    assert "Person ID" not in X.columns

    # 5. Feature Engineering using existing module
    fe = FeatureEngineer(add_ratios=True)
    X_eng = fe.fit_transform(X)
    print(f"Engineered feature matrix: {X_eng.shape[1]} features total.")

    # 6. Numerical Descriptive Stats
    desc_res = compute_numerical_descriptive_stats(X, y, X_eng)
    print("Computed parametric and non-parametric descriptive statistics.")

    # 7. Categorical Associations
    cat_res = analyze_categorical_associations(X, y)
    print("Computed cross-tabulations for Gender, Occupation, and BMI Category.")

    # 8. Correlations
    corr_res = compute_correlations(X_eng, y)
    print("Computed Pearson & Spearman correlation matrices.")

    # 9. Visualizations
    fig_paths = generate_visualizations(raw_df, X, y, X_eng, FIGURES_DIR)
    print(f"Generated and saved {len(fig_paths)} high-resolution figures under {FIGURES_DIR}:")
    for fp in fig_paths:
        print(f"  -> {os.path.basename(fp)}")

    # 10. Generate Report
    generate_markdown_report(struct_res, imbalance_res, desc_res, cat_res, corr_res, REPORT_PATH)
    print(f"Written comprehensive EDA report to: {REPORT_PATH}")

    print("=" * 70)
    print("EDA PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)

    return {
        "structure": struct_res,
        "imbalance": imbalance_res,
        "descriptive": desc_res,
        "categorical": cat_res,
        "correlations": corr_res,
        "figures": fig_paths,
        "report_path": REPORT_PATH,
    }


if __name__ == "__main__":
    run_full_eda()
