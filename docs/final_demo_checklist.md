# Final Demonstration & Viva Checklist

## ML-Based Intelligent System for Automated Sleep Disorder Detection Using Lifestyle and Health Data

> **Purpose**: Step-by-step procedural guide for project viva, examiner demonstration, and technical defense.  
> **Target Audience**: Examiners, Evaluators, Project Guides, and Presenters.  
> **Status**: Verified and Production-Ready.  

---

> [!CAUTION]
> **Mandatory Verbal Disclaimer for Presenters**:  
> *"This project is an academic research and engineering prototype built on a synthetic dataset ($N=374$). It illustrates machine learning pipeline architecture, multi-class classification, and game-theoretic explainability. It is not a certified medical device and does not produce clinical diagnoses. Output probabilities are model posterior confidences, not epidemiological clinical risks."*

---

## 1. Prerequisites & Environment Verification

Before beginning the demonstration, confirm the terminal environment and installed tools:

1. **Python Virtual Environment**:
   - Verify Python version $\ge 3.10$:
     ```powershell
     .\.venv\Scripts\python --version
     ```
2. **Node.js & npm**:
   - Verify Node version $\ge 18$:
     ```powershell
     node -v
     npm -v
     ```
3. **Artifact Integrity**:
   - Confirm all 5 serialized model artifacts exist in `models/`:
     ```powershell
     Get-ChildItem models/
     ```
     - `best_model.joblib` (524 KB)
     - `label_encoder.joblib` (503 B)
     - `model_metadata.json` (1.6 KB)
     - `test_evaluation.json` (2.3 KB)
     - `shap_feature_importance.json` (10.7 KB)

---

## 2. Startup Sequence

Open two separate PowerShell terminal tabs at the project root:

### Terminal 1 — Backend Launch
```powershell
.\.venv\Scripts\python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
- **Expected Console Output**:
  ```
  INFO:     Started server process
  INFO:     Waiting for application startup.
  INFO:     ModelArtifactManager initialized with 3 target classes: ['None', 'Insomnia', 'Sleep Apnea']
  INFO:     Application startup complete.
  INFO:     Uvicorn running on http://127.0.0.1:8000
  ```

### Terminal 2 — Frontend Launch
```powershell
cd frontend
npm run dev
```
- **Expected Console Output**:
  ```
  VITE v6.x.x  ready in ~200 ms
  ➜  Local:   http://localhost:5173/
  ```

Open your browser and navigate to **`http://localhost:5173`**.

---

## 3. Demonstration Step-by-Step Script

### Stage 1: Health Check and API Documentation
1. **Service Health Verification**:
   - In browser or terminal, visit `http://127.0.0.1:8000/health`.
   - **Expected Result**:
     ```json
     {
       "status": "healthy",
       "model_loaded": true,
       "model_version": "1.0.0"
     }
     ```
2. **Interactive Swagger Docs**:
   - Navigate to `http://127.0.0.1:8000/docs`.
   - Show examiners the OpenAPI / Swagger documentation showing schemas for `SleepPredictionInput` and `SleepPredictionOutput`.

---

### Stage 2: Model Architecture & Evaluation Modal Inspection
1. On the web dashboard (`http://localhost:5173`), click the **"About Model"** button in the header.
2. Highlight for examiners:
   - **Tuned Model**: Random Forest (100 estimators, max depth 6).
   - **Feature Space**: 12 raw inputs transformed into 25 pipeline features via `ColumnTransformer`.
   - **Cross-Validation Benchmark**: 5-Fold Stratified Macro F1 = $0.8693 \pm 0.0200$.
   - **Holdout Test Metrics ($N=75$)**: Accuracy: $96.00\%$, Macro F1: $0.9352$, Macro ROC-AUC: $0.9957$.
3. Close the modal by clicking **"Close"** or the backdrop.

---

### Stage 3: Demonstration of Preset Clinical Scenarios

#### Scenario A — Healthy Baseline (No Recorded Sleep Disorder)
1. In the **Load Clinical Preset** selector, click the **"Healthy Baseline"** button.
   - Values automatically populate:
     - Biological Sex: `Male`, Age: `32`, Occupation: `Software Engineer`
     - Sleep Duration: `7.8` hrs, Sleep Quality: `8/10`, Physical Activity: `65` min
     - Stress Level: `4/10`, BMI Category: `Normal`, Heart Rate: `70` bpm, Steps: `8,000`
     - Blood Pressure: `118/76` mmHg (Standard Compound input)
2. Click **"Run Intelligent Assessment"**.
3. **Examine Output**:
   - Result Badge: **"No recorded sleep disorder"** (Green badge, Class 0).
   - Class Probabilities: `No recorded sleep disorder` $\approx 85-92\%$, `Insomnia` $< 10\%$, `Sleep Apnea` $< 5\%$.
   - Interpretation banner explains low cardiovascular risk, optimal sleep duration, and healthy BMI.

#### Scenario B — Insomnia Case
1. Click the **"Insomnia Case"** preset button.
   - Values populate:
     - Biological Sex: `Female`, Age: `44`, Occupation: `Teacher`
     - Sleep Duration: `6.0` hrs, Sleep Quality: `5/10`, Physical Activity: `40` min
     - Stress Level: `8/10`, BMI Category: `Overweight`, Heart Rate: `76` bpm, Steps: `5,000`
     - Blood Pressure: `125/82` mmHg
2. Click **"Run Intelligent Assessment"**.
3. **Examine Output**:
   - Result Badge: **"Insomnia"** (Amber badge, Class 1).
   - Posterior Probability: `Insomnia` $\ge 70\%$.
   - Interpretation notes elevated daily stress ($8/10$) paired with restricted sleep duration ($6.0$ hrs).

#### Scenario C — Sleep Apnea Case with Split Blood Pressure
1. Click the **"Sleep Apnea Case"** preset button.
   - Values populate:
     - Biological Sex: `Female`, Age: `52`, Occupation: `Nurse`
     - Sleep Duration: `6.4` hrs, Sleep Quality: `6/10`, Physical Activity: `35` min
     - Stress Level: `7/10`, BMI Category: `Overweight`, Heart Rate: `80` bpm, Steps: `4,000`
2. **Demonstrate Dual Blood Pressure Parsing**:
   - Switch the Blood Pressure format tab from **"Compound (120/80)"** to **"Separate Values"**.
   - Show how the systolic ($140$ mmHg) and diastolic ($95$ mmHg) inputs accept independent numeric entries.
3. Click **"Run Intelligent Assessment"**.
4. **Examine Output**:
   - Result Badge: **"Sleep Apnea"** (Purple badge, Class 2).
   - Posterior Probability: `Sleep Apnea` $\ge 75\%$.

---

### Stage 4: Explainable AI (SHAP Waterfall) Demonstration
1. Scroll down to the **"AI Decision Attribution (SHAP Waterfall)"** card.
2. Explain to examiners:
   - The chart displays exact Shapley attributions ($\phi_i(x)$) computed on the 25-feature vector.
   - **Supporting Features (Green / Right)**: Features increasing confidence in the predicted class.
   - **Opposing Features (Red / Left)**: Features pushing the model toward alternative classes.
3. Highlight that in the Sleep Apnea case:
   - `BP_Systolic` ($+0.09$) and `BP_Diastolic` ($+0.06$) are the strongest positive contributors.
   - `BMI Category_Overweight` ($+0.07$) further reinforces the classification.

---

### Stage 5: Input Validation and Error Handling Demonstration
Show that the system is guarded against biologically invalid inputs:

1. **Inverted Blood Pressure Test**:
   - Enter Systolic: `110`, Diastolic: `125`.
   - Click assessment.
   - **Expected Result**: Immediate inline validation alert: *"Systolic pressure (110) must be strictly greater than diastolic pressure (125)"*.
2. **Heart Rate Out of Range**:
   - Enter Heart Rate: `15` or `300` bpm.
   - **Expected Result**: Validation prevents submission with boundary error.
3. **Backend Outage Graceful Degradation**:
   - Show that if backend is unavailable, the UI presents an accessible red alert with an explicit recovery instruction without crashing.

---

### Stage 6: Responsive UI & Accessibility
1. Press `F12` to open Chrome Developer Tools.
2. Switch to Mobile Device Emulation (e.g., iPhone 14 or Pixel 7).
3. Demonstrate:
   - Responsive grid collapsing from 2 columns to a smooth 1-column mobile view.
   - Touch-friendly preset buttons.
   - High contrast ratios meeting WCAG AA standards.
   - Non-diagnostic disclaimer permanently visible at the top and bottom of the view.

---

## 4. Automated Test Verification Summary

Demonstrate the automated test suites live to confirm reproducibility:

### 1. Backend Pytest Suite
```powershell
.\.venv\Scripts\pytest -v
```
- **Expected Outcome**: **61 passed** in $< 15$ seconds.
- Validates:
  - `tests/test_preprocess.py`: Column transformations, imputer and scaler behavior.
  - `tests/test_model.py`: Model architecture, predictions, probability sums.
  - `tests/test_explain.py`: SHAP TreeExplainer, background sample caching, local waterfall vectors.
  - `tests/test_serialization_and_contract.py`: Dual BP schema parsing, boundary assertions, HTTP 422 handlers.
  - `tests/test_api.py`: FastAPI endpoints `/api/health`, `/api/model-info`, `/api/predict`.

### 2. Frontend Unit Tests (Vitest)
```powershell
cd frontend
npm run test
```
- **Expected Outcome**: **26 passed** in $< 2$ seconds.
- Validates form rendering, input updates, preset populating, API service calls, and SHAP visualizer components.

### 3. Frontend Production Build
```powershell
cd frontend
npm run build
```
- **Expected Outcome**: Clean build (`dist/`) with zero TypeScript errors in $< 500$ ms.

### 4. Playwright End-to-End Tests
Ensure backend is running on `127.0.0.1:8000`, then in `frontend/`:
```powershell
npm run test:e2e
```
- **Expected Outcome**: **12 passed** in $< 25$ seconds.
- Covers complete live browser flows across all presets, validation errors, and SHAP visualizations.

---

## 5. Troubleshooting Reference

| Symptom | Probable Cause | Immediate Remediation |
|:---|:---|:---|
| `Failed to connect to backend` alert in UI | Backend is not running on port 8000 | Run `.\.venv\Scripts\python -m uvicorn backend.main:app --port 8000` |
| `TypeError: fetch failed` during frontend tests | Backend mock not active or dev server collision | Ensure `npm run test` is executed (Vitest uses synthetic mocks) |
| Port 8000 already in use | Stale Python or Uvicorn process | Run `Get-Process python \| Stop-Process -Force` in PowerShell |
| Port 5173 already in use | Stale Vite dev server | Run `Get-Process node \| Stop-Process -Force` in PowerShell |
| Playwright browser error | Chromium binary not installed | Run `npx playwright install chromium` in `frontend/` directory |
| Inverted BP error | Systolic input $\le$ Diastolic input | Correct systolic pressure to be higher than diastolic (e.g. 120 vs 80) |
