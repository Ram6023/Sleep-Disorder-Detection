# Academic Project Report: ML-Based Intelligent System for Automated Sleep Disorder Detection Using Lifestyle and Health Data

> **Academic Degree**: Bachelor of Technology (B.Tech) in Computer Science and Engineering (Data Science)  
> **Course / Stage**: Capstone Project Report / Final Academic Submission  
> **Document Identifier**: `DOC-REPORT-1.12-FINAL`  
> **Project Repository**: `sleep-disorder-detection`  
> **Date**: October 2026  
> **Status**: Complete, Verified, and Submission-Ready  

---

> [!CAUTION]
> **Clinical & Research Ethics Disclaimer**:  
> This software system, machine learning model, and associated documentation are engineered strictly for academic research and educational evaluation. The models were trained and benchmarked on a synthetic dataset representing adult demographic, cardiovascular, and sleep attributes. **This system is not a certified medical device, does not possess clinical regulatory approval, and must not be used for clinical triage, diagnosis, prognosis, or medical treatment planning.** Model probability distributions represent mathematical posterior confidence values within the parameter space of the trained ensemble and do not reflect clinically calibrated epidemiological risk. SHAP values represent model feature attributions and do not imply physiological causation.

---

## Abstract

Sleep disorders such as Insomnia and Sleep Apnea affect hundreds of millions of individuals worldwide, contributing substantially to cardiovascular morbidity, workplace accidents, metabolic dysregulation, and diminished cognitive health. Traditional clinical diagnosis relies on Polysomnography (PSG), an overnight diagnostic procedure that is resource-intensive, expensive, and constrained by long waiting lists. This capstone project introduces an end-to-end, interpretable machine learning system for screening sleep disorders using non-invasive lifestyle indicators, physiological metrics, and demographic data. 

Using the Sleep Health and Lifestyle Dataset ($N=374$ records, 13 attributes), a machine learning workflow was implemented featuring leak-free preprocessing, 5-fold stratified cross-validation across eight candidate algorithms, hyperparameter optimization, and post-hoc model explainability via SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations). The optimized Random Forest classifier demonstrated superior discrimination, achieving a cross-validation Macro F1-score of $0.8693 \pm 0.0200$ and a holdout test accuracy of $96.00\%$ ($N=75$, Macro F1: $0.9352$, Weighted F1: $0.9599$, Macro ROC-AUC: $0.9957$). Explainability analysis identified systolic blood pressure, diastolic blood pressure, overweight BMI classification, physical activity level, and sleep duration as the primary features influencing predictions. The system is delivered via a production-grade FastAPI REST service paired with a modern React 19 and TypeScript web dashboard featuring real-time input validation, responsive visualizations, and automated integration testing.

**Keywords**: Machine Learning, Sleep Disorders, Insomnia, Sleep Apnea, Random Forest, Explainable AI (XAI), SHAP, FastAPI, React 19, TypeScript.

---

## 1. Introduction and Problem Statement

### 1.1 Background and Motivation
Sleep is an essential biological process necessary for cellular repair, neurocognitive consolidation, metabolic equilibrium, and immune vigilance. Despite its importance, sleep pathology has become an escalating global public health challenge. The two most prevalent sleep disorders in adult populations are:
1. **Insomnia**: Defined clinically by persistent difficulties initiating sleep, maintaining sleep continuity, or experiencing non-restorative sleep, often accompanied by heightened daytime stress and fatigue.
2. **Sleep Apnea (Obstructive Sleep Apnea)**: Characterized by repetitive episodes of complete or partial upper airway collapse during sleep, resulting in oxygen desaturation, autonomic arousal, elevated nocturnal blood pressure, and severe sleep fragmentation.

The gold standard for diagnosing sleep disorders remains clinical in-laboratory **Polysomnography (PSG)**. PSG requires patients to sleep in a specialized clinical facility while continuous multi-channel neurophysiological signals—including electroencephalography (EEG), electrooculography (EOG), electromyography (EMG), electrocardiography (ECG), nasal airflow, and pulse oximetry—are monitored by trained polysomnographic technologists. While PSG provides comprehensive diagnostic fidelity, it suffers from several critical bottlenecks:
- **Prohibitive Financial Costs**: A single diagnostic PSG session typically incurs high healthcare expenditures.
- **Limited Access and Extended Backlogs**: Specialized sleep laboratories and board-certified sleep medicine specialists are scarce, leading to backlogs often exceeding six to twelve months.
- **Inconvenience and First-Night Effect**: The unnatural clinical setting, coupled with extensive facial and scalp electrodes, frequently alters natural sleep architecture, causing false or unrepresentative nocturnal readings.
- **Lack of Proactive Screening**: PSG is reactive, reserved for individuals already experiencing advanced symptom severity.

### 1.2 Problem Formulation
Given the widespread prevalence of non-invasive health monitors, smart wearables, and routine clinical biometric data (e.g., blood pressure, heart rate, daily step counts, and validated sleep surveys), an urgent research need exists for automated, data-driven computational models capable of pre-screening individuals at risk of sleep disorders. 

Such a computational tool must address three primary engineering and scientific objectives:
1. **Multi-Class Discrimination**: Differentiating between individuals without sleep disorders (`None`), those experiencing `Insomnia`, and those exhibiting `Sleep Apnea`.
2. **Clinical Interpretability (Explainable AI)**: Black-box models are unacceptable in healthcare. Practitioners require transparent attributions detailing *why* an algorithm produced a specific classification.
3. **End-to-End Architectural Integration**: The system must bridge low-level algorithmic inference with accessible, verified, and secure web application endpoints suitable for research prototyping.

---

## 2. Objectives and Scope

### 2.1 Project Objectives
The core objectives of this capstone research project are:
- **Objective 1 — Rigorous Exploratory Data Analysis**: Profile the underlying statistical properties, multi-collinearities, and class distributions of lifestyle and biometric features without data leakage.
- **Objective 2 — Leak-Free Preprocessing Architecture**: Construct automated scikit-learn preprocessing pipelines isolating scaling, imputation, and one-hot encoding strictly within training folds.
- **Objective 3 — Multi-Model Benchmark & Optimization**: Systematically evaluate eight diverse classification algorithms using 5-Fold Stratified Cross-Validation, followed by systematic hyperparameter tuning of the top candidate.
- **Objective 4 — Model Explainability (SHAP & LIME)**: Quantify global and local feature attributions using game-theoretic Shapley values (`TreeExplainer`) and local surrogate modeling.
- **Objective 5 — Full-Stack Production Implementation**: Deploy the serialized model via a FastAPI REST service with dual blood-pressure parsing and construct an accessible, responsive React 19 + TypeScript frontend.
- **Objective 6 — Multi-Tiered Verification**: Subject all tiers to automated unit, integration, and Playwright end-to-end testing.

### 2.2 Project Scope and Boundaries
- **Academic Scope**: This research serves as a final year undergraduate engineering capstone. It operates on an open-access research dataset.
- **Diagnostic Non-Equivalence**: The system is framed as a screening and risk-stratification research prototype. It does not output medical diagnoses.
- **Target Classes**: Exactly three mutual exclusive states: Class 0 (`None`), Class 1 (`Insomnia`), and Class 2 (`Sleep Apnea`).

---

## 3. Dataset and Preprocessing Methodology

### 3.1 Dataset Characterization
The project utilizes the **Sleep Health and Lifestyle Dataset**, comprising $N=374$ participant records and 13 attributes capturing lifestyle, demographic, cardiovascular, and sleep metrics.

| Feature Name | Raw Data Type | Domain / Valid Range | Representation in Pipeline | Description |
|:---|:---:|:---:|:---:|:---|
| `Person ID` | Integer | $1 - 374$ | *Excluded* | Arbitrary record identifier (excluded to avoid index leakage) |
| `Gender` | String | `Male`, `Female` | One-Hot Encoded (1-bit) | Participant biological sex |
| `Age` | Integer | $27 - 59$ years | Standardized Float | Participant age |
| `Occupation` | String | 11 categories | One-Hot Encoded (11-bit) | Professional category (e.g., Nurse, Doctor, Engineer) |
| `Sleep Duration` | Float | $5.8 - 8.5$ hours | Standardized Float | Average sleep per night |
| `Quality of Sleep` | Integer | $1 - 10$ scale | Standardized Float | Subjective rating of sleep quality |
| `Physical Activity Level` | Integer | $30 - 90$ mins/day | Standardized Float | Daily physical activity duration |
| `Stress Level` | Integer | $1 - 10$ scale | Standardized Float | Subjective daily stress intensity |
| `BMI Category` | String | `Normal`, `Overweight`, `Obese` | One-Hot Encoded (4-bit) | Body mass index category (`Normal Weight` merged into `Normal`) |
| `Blood Pressure` | String | `"Systolic/Diastolic"` | 2 Continuous Features | Rest blood pressure in mmHg (split to `BP_Systolic` & `BP_Diastolic`) |
| `Heart Rate` | Integer | $65 - 86$ bpm | Standardized Float | Resting heart rate |
| `Daily Steps` | Integer | $3,000 - 10,000$ | Standardized Float | Daily step count |
| `Sleep Disorder` | String | `None`, `Insomnia`, `Sleep Apnea` | Integer Label $[0, 1, 2]$ | **Target Variable** |

### 3.2 Target Class Distribution
Analysis of the ground-truth target variable reveals moderate class imbalance:
- **Class 0 (`None` / "No recorded sleep disorder")**: $219$ instances ($58.56\%$)
- **Class 1 (`Insomnia`)**: $77$ instances ($20.59\%$)
- **Class 2 (`Sleep Apnea`)**: $78$ instances ($20.86\%$)
- **Total**: $374$ instances ($100.0\%$)

Because accuracy alone is misleading under class imbalance, all model evaluations prioritize Macro F1-score, Balanced Accuracy, and per-class Recall. To maintain exact class proportions, all dataset partitioning employs **Stratified K-Fold** splitting ($80\%$ train, $N=299$; $20\%$ holdout test, $N=75$).

### 3.3 Leak-Free Preprocessing Pipeline
To guarantee that evaluation metrics are free from data leakage, all data transformations are encapsulated inside a Scikit-Learn `ColumnTransformer`. Scaler parameters (mean, standard deviation) and categorical vocabularies are learned solely on the training partition ($X_{train}$):
1. **Numerical/Ordinal Stream ($9$ features)**:
   - Features: `Age`, `Sleep Duration`, `Quality of Sleep`, `Physical Activity Level`, `Stress Level`, `Heart Rate`, `Daily Steps`, `BP_Systolic`, `BP_Diastolic`.
   - Transformations: `SimpleImputer(strategy='median')` followed by `StandardScaler()`.
2. **Categorical Stream ($3$ features)**:
   - Features: `Gender`, `Occupation`, `BMI Category`.
   - Transformations: `SimpleImputer(strategy='most_frequent')` followed by `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
3. **Output Dimensionality**:
   - The preprocessing step maps the $12$ raw features into a dense vector of $25$ transformed continuous variables.
   - Setting `handle_unknown='ignore'` guarantees runtime robustness during API inference when encountering unseen occupations or categories.

---

## 4. Methodology and System Architecture

### 4.1 System Overview
The complete system is architected as an interconnected four-tier pipeline:
1. **Data and Machine Learning Core (`ml/`)**: Model training, preprocessing pipelines, hyperparameter search, and serialization.
2. **Model Persistence (`models/`)**: Serialized Joblib pipelines (`best_model.joblib`), target encoders, and evaluation metadata.
3. **Application Backend (`backend/`)**: Asynchronous FastAPI service exposing RESTful inference, metadata, and SHAP explainability.
4. **Client Presentation (`frontend/`)**: React 19 + TypeScript single-page application with responsive inputs, dynamic charts, and interactive presets.

### 4.2 Architecture Diagram

```mermaid
flowchart TD
    subgraph Client ["Client Presentation Tier (React 19 + TypeScript)"]
        UI["Healthcare Dashboard (App.tsx)"]
        Form["Assessment Form (PredictionForm.tsx)"]
        Result["Result Card & Posterior Probs (ResultCard.tsx)"]
        XAI_View["SHAP Visualizer (FeatureImportance.tsx)"]
        MetaModal["Model Architecture Modal (ModelInfoModal.tsx)"]
        UI --> Form
        UI --> Result
        UI --> XAI_View
        UI --> MetaModal
    end

    subgraph API ["Service Gateway Tier (FastAPI / Uvicorn)"]
        Router["FastAPI Application (backend/main.py)"]
        Contract["Payload Contract Validation (Pydantic v2 / ml/contract.py)"]
        BPParsing["Dual Blood Pressure Normalizer"]
        Health["GET /api/health"]
        ModelInfo["GET /api/model-info"]
        PredictEndpoint["POST /api/predict"]
        
        Router --> Health
        Router --> ModelInfo
        Router --> PredictEndpoint
        PredictEndpoint --> Contract
        Contract --> BPParsing
    end

    subgraph Engine ["Machine Learning & Explainability Engine (ml/)"]
        Mgr["ModelArtifactManager (ml/contract.py)"]
        Pipeline["Serialized Pipeline (models/best_model.joblib)"]
        Transform["ColumnTransformer (25 Features)"]
        RFC["RandomForestClassifier (100 Trees, max_depth=6)"]
        SHAP["SHAP TreeExplainer (Background N=100)"]
        
        Mgr --> Pipeline
        Pipeline --> Transform
        Transform --> RFC
        RFC --> SHAP
    end

    subgraph Persistence ["Model Artifact Store (models/)"]
        ModelFile["best_model.joblib"]
        EncoderFile["label_encoder.joblib"]
        MetaFile["model_metadata.json"]
        EvalFile["test_evaluation.json"]
        ShapMeta["shap_feature_importance.json"]
    end

    Form -->|POST /api/predict (JSON Payload)| Router
    PredictEndpoint -->|Validate & Ingest| Mgr
    Mgr -.->|Deserializes on Startup| Persistence
    RFC -->|Posterior Probabilities & Class| Result
    SHAP -->|Exact Shapley Attributions| XAI_View
    ModelInfo -->|Model Spec & CV Metrics| MetaModal
```

---

## 5. Model Comparison and Selection

### 5.1 Candidate Model Benchmarking
Eight representative machine learning algorithms were benchmarked using 5-Fold Stratified Cross-Validation on the training partition ($N=299$). Across all folds, models were evaluated on Balanced Accuracy, Macro Precision, Macro Recall, Macro F1, and Overall Accuracy.

| Model Algorithm | Mean CV Macro F1 | CV Macro F1 Std | Mean CV Balanced Acc | Mean CV Accuracy | Notes / Characteristics |
|:---|:---:|:---:|:---:|:---:|:---|
| **Random Forest (Tuned)** | **0.8693** | **± 0.0200** | **0.8673** | **0.8962** | **Selected**: Best balance, highest stability, native TreeExplainer compatibility |
| **Random Forest (Default)** | 0.8670 | ± 0.0215 | 0.8654 | 0.8929 | Strong baseline, mild overfitting on training leaves |
| **XGBoost Classifier** | 0.8621 | ± 0.0248 | 0.8598 | 0.8896 | High performance; slightly higher fold variance |
| **Gradient Boosting Classifier** | 0.8584 | ± 0.0231 | 0.8562 | 0.8862 | Solid gradient boosting performance |
| **Support Vector Classifier (RBF)** | 0.8492 | ± 0.0270 | 0.8480 | 0.8762 | High margin performance, non-linear kernel |
| **Logistic Regression (L2)** | 0.8350 | ± 0.0312 | 0.8310 | 0.8628 | Linear baseline; struggles with non-linear feature interactions |
| **K-Nearest Neighbors (KNN)** | 0.8245 | ± 0.0298 | 0.8204 | 0.8528 | Sensitive to distance metrics in 25-dimensional space |
| **Decision Tree (CART)** | 0.8120 | ± 0.0345 | 0.8090 | 0.8394 | High variance, prone to overfitting individual leaf nodes |

### 5.2 Selected Model Hyperparameters
The **Tuned Random Forest** was selected as the optimal production classifier. A systematic grid search over `max_depth`, `min_samples_split`, `min_samples_leaf`, and `n_estimators` yielded the following optimal configuration:
```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    min_samples_split=4,
    min_samples_leaf=1,
    random_state=42,
    criterion='gini'
)
```
Constraining `max_depth=6` regularized the ensemble against memorizing the small sample size while maintaining sufficient capacity to resolve complex cardiovascular and lifestyle thresholds.

---

## 6. Evaluation Results and Holdout Verification

### 6.1 Holdout Performance on Untouched Test Set ($N=75$)
Following cross-validation, the selected model was evaluated on the independent $20\%$ holdout test partition ($N=75$ records: $44$ None, $15$ Insomnia, $16$ Sleep Apnea).

| Metric | Holdout Evaluation Score | Baseline (Stratified Dummy) | Evaluation Notes |
|:---|:---:|:---:|:---|
| **Overall Accuracy** | **96.00%** ($72 / 75$) | 45.33% | Proportion of correct multi-class classifications |
| **Balanced Accuracy** | **93.47%** | 33.33% | Unweighted average of per-class recalls |
| **Macro F1-Score** | **0.9352** | 0.3120 | Equal-weight harmonic mean of precision and recall |
| **Weighted F1-Score** | **0.9599** | 0.4480 | Sample-weighted harmonic mean |
| **Macro ROC-AUC** | **0.9957** | 0.5000 | One-vs-Rest area under the ROC curve |

### 6.2 Per-Class Confusion Matrix and Detailed Metrics

$$\text{Confusion Matrix} = \begin{bmatrix} 44 & 0 & 0 \\ 0 & 13 & 2 \\ 0 & 1 & 15 \end{bmatrix}$$

- **Class 0 (`None` / "No recorded sleep disorder")**:
  - True Positives: $44 / 44$
  - Precision: $1.0000$ ($100.0\%$)
  - Recall: $1.0000$ ($100.0\%$)
  - F1-Score: $1.0000$
  - *Finding*: Zero healthy individuals were misclassified as experiencing a disorder, demonstrating zero false-positive disorder diagnoses.
- **Class 1 (`Insomnia`)**:
  - True Positives: $13 / 15$ ($2$ misclassified as Sleep Apnea)
  - False Positives: $1$ (from Sleep Apnea)
  - Precision: $0.9286$ ($13 / 14$)
  - Recall: $0.8667$ ($13 / 15$)
  - F1-Score: $0.8966$
- **Class 2 (`Sleep Apnea`)**:
  - True Positives: $15 / 16$ ($1$ misclassified as Insomnia)
  - False Positives: $2$ (from Insomnia)
  - Precision: $0.8824$ ($15 / 17$)
  - Recall: $0.9375$ ($15 / 16$)
  - F1-Score: $0.9091$

*Diagnostic Distinction*: All three misclassifications occurred strictly between Insomnia and Sleep Apnea (the two pathological states); **not a single disorder case was misclassified as healthy**, confirming high clinical screening safety on this test partition.

---

## 7. Explainable AI (SHAP and LIME)

### 7.1 Mathematical Foundations
To satisfy the ethical requirement for algorithmic transparency, post-hoc explainability was integrated using **SHAP (SHapley Additive exPlanations)**. Derived from cooperative game theory, Shapley values allocate credit to individual input features such that the sum of attributions equals the difference between the model's prediction $f(x)$ and the expected baseline outcome $E[f(x)]$:

$$f(x) - E[f(x)] = \sum_{j=1}^{M} \phi_j(x)$$

For the Random Forest model, exact Shapley values are calculated in polynomial time via `shap.TreeExplainer`, satisfying the foundational axioms of **Efficiency**, **Symmetry**, **Dummy Player**, and **Additivity**.

### 7.2 Global Feature Drivers
Global feature importance was evaluated by computing the Mean Absolute SHAP Value across the holdout cohort:

$$\text{Mean Absolute SHAP} = \frac{1}{N} \sum_{i=1}^{N} |\phi_{i,j}|$$

| Rank | Feature Name | Mean Absolute SHAP | Primary Target Association | Physiological Interpretation in Dataset |
|:---:|:---|:---:|:---:|:---|
| 1 | `BP_Systolic` | 0.0690 | Sleep Apnea | Systolic hypertension $\ge 130$ mmHg heavily increases Sleep Apnea probability |
| 2 | `BP_Diastolic` | 0.0509 | Sleep Apnea | Elevated diastolic pressure strongly co-occurs with Sleep Apnea |
| 3 | `BMI Category_Overweight` | 0.0458 | Insomnia & Sleep Apnea | Overweight status elevates risk of both sleep disorders |
| 4 | `BMI Category_Normal` | 0.0386 | None (Healthy) | Normal BMI strongly pushes predictions toward healthy baseline |
| 5 | `Physical Activity Level` | 0.0277 | Insomnia | Low activity correlates with Insomnia; moderate/high activity supports healthy sleep |
| 6 | `Sleep Duration` | 0.0260 | Healthy vs Insomnia | Short duration ($<6.0$ hrs) signals Insomnia; $>7.5$ hrs indicates healthy sleep |
| 7 | `Age` | 0.0234 | Sleep Apnea | Older cohorts in the dataset show higher incidence of Sleep Apnea |
| 8 | `Occupation_Nurse` | 0.0226 | Sleep Apnea | Occupational clustering due to demographic structure in dataset |
| 9 | `Heart Rate` | 0.0164 | Healthy vs Disorder | Elevated resting heart rate elevates disorder probabilities |
| 10 | `Daily Steps` | 0.0108 | General Health | High step volume ($>8,000$) supports healthy baseline |

### 7.3 Local Attribution and Surrogate Validation (LIME)
Local explanations provide patient-specific attributions. For example, for an individual presenting with high systolic blood pressure ($140$ mmHg), overweight BMI, and age $52$, local SHAP waterfall analysis displays positive contributions of $+0.0985$ and $+0.0898$ directly elevating the predicted probability of Sleep Apnea. LIME tabular explanations independently confirm concordance, weighting identical threshold rules as positive drivers.

---

## 8. Software Architecture and Full-Stack Implementation

### 8.1 Backend Implementation (`backend/main.py`)
The backend is implemented using **FastAPI** running on the ASGI server **Uvicorn**:
- **Pydantic v2 Schema Enforcement**: Strong runtime validation against physical limits ($1 \le \text{age} \le 120$, $30 \le \text{HR} \le 220$, $0.1 \le \text{Sleep} \le 24.0$).
- **Dual Blood Pressure Ingestion**: Ingests compound clinical strings (`"120/80"`) or split numeric floats (`bp_systolic`, `bp_diastolic`), validating that $\text{Systolic} > \text{Diastolic}$.
- **Lifecycle Management**: Deserializes the 524 KB Joblib pipeline on startup via FastAPI `lifespan` handler, guaranteeing zero per-request disk I/O.
- **REST Endpoints**:
  - `GET /api/health`: Service health, uptime, and model loading state.
  - `GET /api/model-info`: Supported classes, hyperparameter specifications, and validation bounds.
  - `POST /api/predict`: End-to-end inference returning class probabilities, display labels, and on-demand SHAP contributions.

### 8.2 Frontend Dashboard (`frontend/src/`)
Built with **React 19**, **TypeScript**, and **Vite**:
- **Design System**: A responsive glassmorphic aesthetic using modern CSS variables, semantic card layouts, and accessible color tokens.
- **Form Controls & Presets**: Includes one-click clinically representative presets (*Healthy Baseline*, *Insomnia Case*, *Sleep Apnea Case*) allowing rapid evaluation.
- **Interactive Visualizations**:
  - Multi-class probability bars with color coding (`green` for healthy, `amber` for insomnia, `purple` for sleep apnea).
  - Dynamic horizontal attribution bar chart rendering SHAP positive (supporting) and negative (opposing) feature impacts.
  - Model Information modal displaying verified metrics and architectural metadata.

---

## 9. Verification, Testing, and Quality Assurance

Testing was structured across three automated tiers, confirming 100% test pass rates across the stack:

| Test Suite | Framework / Tool | Test Count | Scope and Scenarios Verified | Status |
|:---|:---:|:---:|:---|:---:|
| **Backend Unit & API** | Pytest + FastAPI TestClient | 61 Tests | Input validation, physical boundaries, dual BP parsing, pipeline deserialization, probability sum integrity, SHAP explanation accuracy, HTTP 422/503 handlers | ✅ **PASS** |
| **Frontend Unit** | Vitest + Testing Library | 26 Tests | Component rendering, form state management, preset loading, blood pressure tab switching, result card rendering, SHAP bar rendering, error boundaries | ✅ **PASS** |
| **End-to-End (E2E)** | Playwright (Chromium) | 12 Tests | Live browser execution: full assessment lifecycle, healthy preset prediction, insomnia preset prediction, sleep apnea preset prediction, custom input flow, SHAP waterfall rendering, backend error alert display, modal inspection | ✅ **PASS** |
| **Production Build** | TypeScript (`tsc -b`) + Vite | 1 Bundle | Zero TypeScript compiler errors, asset optimization, strict type-checking | ✅ **PASS** |

---

## 10. Ethical Considerations and Limitations

### 10.1 Dataset Synthesis and Demographic Representation
- **Synthetic Nature**: The dataset contains synthetic records designed to model adult demographics. While statistical distributions align with clinical observations, the data lacks real-world observational noise, multi-morbidities, and unmeasured confounders.
- **Sample Size Constraint**: $N=374$ is small for clinical research. Findings should be viewed as algorithmic proof-of-concept demonstrations rather than clinical population studies.
- **Occupational Clustering**: Strong associations between specific occupations (e.g., Nurses and Sleep Apnea) reflect synthetic dataset generation artifacts rather than established occupational hazards.

### 10.2 Mathematical vs. Causal Attribution
- **Non-Causality of SHAP**: High SHAP attributions indicate that an attribute strongly shifted tree decision paths; they do **not** imply that modifying that attribute (e.g., changing professions) will alleviate a sleep disorder.
- **Non-Calibrated Probabilities**: Ensemble tree voting frequencies yield mathematical confidence estimates that must not be interpreted as epidemiological disease probabilities.

### 10.3 Privacy and Data Governance
- The system processes data entirely in-memory during inference without storing personal identifiable information (PII) or logging sensitive inputs to persistent databases.

---

## 11. Conclusion and Future Enhancements

### 11.1 Conclusion
This project successfully designed, implemented, and rigorously validated an automated, interpretable machine learning system for detecting sleep disorders from non-invasive lifestyle and biometric indicators. By combining a leak-free Scikit-Learn preprocessing pipeline, a regularized Random Forest classifier ($96.00\%$ holdout accuracy, $0.9352$ Macro F1), and game-theoretic SHAP explainability, the system delivers both high predictive performance and actionable diagnostic transparency. The integration of a FastAPI backend and a responsive React 19 dashboard demonstrates the feasibility of deploying machine learning decision-support prototypes for academic and research evaluation.

### 11.2 Future Enhancements
1. **Real-World Clinical Validation**: Partnering with sleep clinical centers to benchmark the model against patient cohorts undergoing simultaneous PSG testing.
2. **Wearable Sensor Stream Integration**: Expanding feature ingestion from static tabular metrics to dynamic time-series data captured via photoplethysmography (PPG) and tri-axial accelerometry.
3. **Deep Learning Sequence Modeling**: Investigating recurrent neural networks (LSTM) and Transformer architectures for multi-day longitudinal sleep pattern modeling.
4. **Calibrated Uncertainty Estimation**: Incorporating Conformal Prediction or Platt scaling to generate mathematically guaranteed confidence intervals for model predictions.

---

## 12. References and Documentation Citations

1. Lundberg, S. M., & Lee, S.-I. (2017). *A unified approach to interpreting model predictions*. Advances in Neural Information Processing Systems (NeurIPS 2017), 30, 4765–4774.
2. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). *"Why should I trust you?": Explaining the predictions of any classifier*. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16), 1135–1144.
3. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.
4. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.
5. Sateia, M. J. (2014). *International Classification of Sleep Disorders-Third Edition: Highlights and Modifications*. Chest, 146(5), 1387–1394.
6. American Academy of Sleep Medicine. (2018). *Clinical Practice Guideline for Diagnostic Testing for Adult Obstructive Sleep Apnea*. Journal of Clinical Sleep Medicine, 13(3), 479–504.
7. Project Internal Specifications:
   - [`README.md`](file:///c:/Users/Asus/Documents/sleep-disorder-detection/README.md): Master Academic & Technical System Documentation.
   - [`docs/api_contract.md`](file:///c:/Users/Asus/Documents/sleep-disorder-detection/docs/api_contract.md): Formal REST API Payload and Schema Specification.
   - [`reports/eda_report.md`](file:///c:/Users/Asus/Documents/sleep-disorder-detection/reports/eda_report.md): Exploratory Data Analysis & Class Distribution Report.
   - [`reports/shap_report.md`](file:///c:/Users/Asus/Documents/sleep-disorder-detection/reports/shap_report.md): Explainable AI (XAI) Attribution & Surrogate Report.
   - [`reports/model_serialization_audit.md`](file:///c:/Users/Asus/Documents/sleep-disorder-detection/reports/model_serialization_audit.md): Artifact Deserialization and Integrity Audit.
