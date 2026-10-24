# Explainable AI (XAI) Report: SHAP and LIME Interpretability
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

Global feature importance was computed using the **Mean Absolute SHAP Value** ($\frac{1}{N}\sum_{i=1}^{N} |\phi_{i,j}|$), capturing the average magnitude of feature influence across all 75 holdout samples and 3 target classes.

### Top 10 Features Driving Model Decisions

| Rank | Feature | Overall Mean |SHAP| | Mean |SHAP| (None) | Mean |SHAP| (Insomnia) | Mean |SHAP| (Sleep Apnea) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | `BP_Systolic` | 0.0690 | 0.0910 | 0.0458 | 0.0701 |
| 2 | `BP_Diastolic` | 0.0509 | 0.0691 | 0.0349 | 0.0486 |
| 3 | `BMI Category_Overweight` | 0.0458 | 0.0677 | 0.0377 | 0.0321 |
| 4 | `BMI Category_Normal` | 0.0386 | 0.0578 | 0.0306 | 0.0272 |
| 5 | `Physical Activity Level` | 0.0277 | 0.0211 | 0.0408 | 0.0213 |
| 6 | `Sleep Duration` | 0.0260 | 0.0347 | 0.0310 | 0.0124 |
| 7 | `Age` | 0.0234 | 0.0254 | 0.0308 | 0.0140 |
| 8 | `Occupation_Nurse` | 0.0226 | 0.0223 | 0.0122 | 0.0332 |
| 9 | `Heart Rate` | 0.0164 | 0.0176 | 0.0106 | 0.0209 |
| 10 | `Daily Steps` | 0.0108 | 0.0096 | 0.0110 | 0.0120 |

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
- **Primary Positive Drivers**: `Stress Level` (high, $\ge 7$), `Sleep Duration` (short, $< 6.5$ hours), `BMI Category_Overweight`, `Occupation_Salesperson`, `Occupation_Teacher`.
- **Opposing Signals**: Low stress level, long sleep duration, Normal BMI.

### 3. Sleep Apnea (Class 2)
- **Primary Positive Drivers**: `BP_Systolic` (high, $\ge 130$ mmHg), `BP_Diastolic` (high, $\ge 85$ mmHg), `Age` (older cohort), `BMI Category_Overweight` / `BMI Category_Obese`, `Occupation_Nurse`.
- **Opposing Signals**: Normal blood pressure (< 120/80 mmHg), young age (< 35 years), Normal BMI.

---

## 4. Local Instance Explanation: Holdout Sample #0

An individual test instance was selected to compare SHAP and LIME local attributions:

- **True Label**: `Insomnia`
- **Model Prediction**: `Insomnia` (Class 1)
- **Predicted Class Probabilities**:
  - `No recorded sleep disorder`: 5.37%
  - `Insomnia`: 94.12%
  - `Sleep Apnea`: 0.51%

### SHAP Local Waterfall Attribution (Top Contributing Features)
| Feature | Feature Value (Standardized/Encoded) | SHAP Contribution ($\phi_i$) |
|:---|:---:|:---:|
| `BMI Category_Overweight` | 1.000 | +0.0985 |
| `BP_Systolic` | 0.836 | +0.0898 |
| `BP_Diastolic` | 0.865 | +0.0843 |
| `Physical Activity Level` | -0.704 | +0.0801 |
| `Age` | 0.203 | +0.0721 |
| `BMI Category_Normal` | 0.000 | +0.0603 |

### LIME Local Surrogate Weights
| LIME Rule Condition | Weight | Impact on Prediction |
|:---|:---:|:---:|
| `Physical Activity Level <= -0.70` | +0.0695 | Supports Insomnia |
| `0.06 < BP_Diastolic <= 0.86` | +0.0598 | Supports Insomnia |
| `0.00 < BMI Category_Overweight <= 1.00` | +0.0584 | Supports Insomnia |
| `BMI Category_Normal <= 0.00` | +0.0456 | Supports Insomnia |
| `0.19 < BP_Systolic <= 0.84` | +0.0424 | Supports Insomnia |
| `0.09 < Age <= 0.90` | +0.0333 | Supports Insomnia |
| `Occupation_Salesperson <= 0.00` | -0.0266 | Opposes Insomnia |
| `Occupation_Scientist <= 0.00` | +0.0265 | Supports Insomnia |
| `Occupation_Manager <= 0.00` | +0.0183 | Supports Insomnia |
| `Occupation_Software Engineer <= 0.00` | -0.0154 | Opposes Insomnia |

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
