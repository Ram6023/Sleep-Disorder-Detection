# API Payload Contract Specification (v1.0.0)
## ML-Based Intelligent System for Automated Sleep Disorder Detection

> **Version**: 1.0.0  
> **Target Framework**: FastAPI / Pydantic v2  
> **Status**: Completed & Verified (Task 1.8 Audit)  
> **Module Implementation**: [`ml/contract.py`](file:///c:/Users/Asus/Documents/sleep-disorder-detection/ml/contract.py)  

> [!CAUTION]
> **Clinical Non-Diagnostic Disclaimer**:  
> Predictions and explanations provided by this system are experimental analytical outputs for educational and research evaluation only. They do **not** constitute medical diagnoses, clinical guidance, or therapeutic recommendations.

---

## 1. Overview and Design Principles

This specification defines the formal payload contracts for the Sleep Disorder Detection inference service. It bridges raw user lifestyle inputs with the underlying 25-feature Scikit-Learn Random Forest pipeline (`models/best_model.joblib`).

### Key Design Tenets
1. **Zero Data Leakage**: Preprocessing transformations (scaling, one-hot encoding) are locked inside the serialized pipeline and fitted exclusively on training data.
2. **Dual Blood Pressure Ingestion**: Accepts both traditional clinical compound strings (`"120/80"`) and separate numeric parameters (`BP_Systolic`, `BP_Diastolic`).
3. **Robust Handling of Unknowns**: Unseen occupations and categories are gracefully ignored (`handle_unknown='ignore'`) without causing runtime 500 exceptions.
4. **Physiological Guardrails**: Input validation prevents biologically impossible values (e.g. negative heart rate, systolic $\le$ diastolic pressure).
5. **Class Order Determinism**: Model probabilities strictly correspond to `[0: None, 1: Insomnia, 2: Sleep Apnea]`.
6. **Built-in Explainability**: Local SHAP feature contributions are computed on demand using exact Shapley values.

---

## 2. Input Contract: `SleepPredictionInput`

### Schema Definition

| Field Name | Dataset Alias | Type | Required | Range / Constraints | Description |
|:---|:---|:---:|:---:|:---|:---|
| `gender` | `Gender` | `string` | **Yes** | `"Male"`, `"Female"` | Biological sex (case-insensitive) |
| `age` | `Age` | `integer` | **Yes** | $1 \le \text{age} \le 120$ | Patient age in years |
| `occupation` | `Occupation` | `string` | **Yes** | $1 \le \text{length} \le 100$ | Occupation title |
| `sleep_duration` | `Sleep Duration` | `float` | **Yes** | $0.1 \le \text{duration} \le 24.0$ | Average sleep duration (hours/day) |
| `quality_of_sleep` | `Quality of Sleep` | `integer` | **Yes** | $1 \le \text{quality} \le 10$ | Subjective sleep quality scale |
| `physical_activity_level` | `Physical Activity Level` | `integer` | **Yes** | $0 \le \text{activity} \le 1440$ | Daily physical activity in minutes |
| `stress_level` | `Stress Level` | `integer` | **Yes** | $1 \le \text{stress} \le 10$ | Subjective stress scale |
| `bmi_category` | `BMI Category` | `string` | **Yes** | `"Normal"`, `"Overweight"`, `"Obese"` | BMI classification (`"Normal Weight"` auto-normalized to `"Normal"`) |
| `blood_pressure` | `Blood Pressure` | `string` | *Optional\** | Pattern: `^\d{2,3}/\d{2,3}$` | Compound blood pressure (e.g. `"120/80"`) |
| `bp_systolic` | `BP_Systolic` | `float` | *Optional\** | $50.0 \le \text{systolic} \le 260.0$ | Systolic pressure in mmHg |
| `bp_diastolic` | `BP_Diastolic` | `float` | *Optional\** | $30.0 \le \text{diastolic} \le 160.0$ | Diastolic pressure in mmHg |
| `heart_rate` | `Heart Rate` | `integer` | **Yes** | $30 \le \text{hr} \le 220$ | Resting heart rate in BPM |
| `daily_steps` | `Daily Steps` | `integer` | **Yes** | $0 \le \text{steps} \le 100000$ | Average daily step count |
| `person_id` | `Person ID` | `integer` | *Optional* | Integer identifier | Client tracking ID (ignored by model) |

*\*Either `blood_pressure` string OR both `bp_systolic` and `bp_diastolic` must be supplied. In all cases, systolic must be strictly greater than diastolic.*

### Example Request Payloads

#### Option A: Canonical Title Case with Compound Blood Pressure
```json
{
  "Gender": "Male",
  "Age": 42,
  "Occupation": "Software Engineer",
  "Sleep Duration": 6.8,
  "Quality of Sleep": 7,
  "Physical Activity Level": 60,
  "Stress Level": 5,
  "BMI Category": "Normal",
  "Blood Pressure": "120/80",
  "Heart Rate": 70,
  "Daily Steps": 7500
}
```

#### Option B: Snake_Case with Pre-Split Blood Pressure
```json
{
  "gender": "Female",
  "age": 52,
  "occupation": "Nurse",
  "sleep_duration": 5.8,
  "quality_of_sleep": 4,
  "physical_activity_level": 35,
  "stress_level": 8,
  "bmi_category": "Overweight",
  "bp_systolic": 140.0,
  "bp_diastolic": 95.0,
  "heart_rate": 82,
  "daily_steps": 4000
}
```

---

## 3. Output Contract: `SleepPredictionOutput`

### Schema Definition

| Field Name | Type | Description |
|:---|:---:|:---|
| `prediction_id` | `string (UUID)` | Unique tracking identifier for this prediction query |
| `model_version` | `string` | Version of the serialized model pipeline (`1.0.0`) |
| `contract_version` | `string` | Version of this API contract (`1.0.0`) |
| `predicted_class` | `integer` | Encoded prediction index (`0`, `1`, or `2`) |
| `predicted_class_name` | `string` | Human-readable disorder label: `"No recorded sleep disorder"`, `"Insomnia"`, or `"Sleep Apnea"` |
| `probabilities` | `object` | Dictionary of class probabilities summing to 1.0 |
| `probabilities["No recorded sleep disorder"]` | `float` | Estimated probability for healthy baseline (Class 0) |
| `probabilities["Insomnia"]` | `float` | Estimated probability for Insomnia (Class 1) |
| `probabilities["Sleep Apnea"]` | `float` | Estimated probability for Sleep Apnea (Class 2) |
| `explanation` | `object (optional)` | Local explainability attribution details |
| `explanation.method` | `string` | Method utilized (`"SHAP TreeExplainer"`) |
| `explanation.base_value` | `float` | Baseline expected value ($E[f(x)]$) |
| `explanation.top_factors` | `array` | Top positive and negative feature attributions |
| `disclaimer` | `string` | Mandatory non-diagnostic clinical disclaimer |
| `timestamp_utc` | `string (ISO-8601)` | Timestamp when prediction was calculated |

### Example Response Payload (with Local SHAP Explanation)
```json
{
  "prediction_id": "c9f595c3-e278-456b-a028-42ca0935c56a",
  "model_version": "1.0.0",
  "contract_version": "1.0.0",
  "predicted_class": 0,
  "predicted_class_name": "No recorded sleep disorder",
  "probabilities": {
    "No recorded sleep disorder": 0.8905,
    "Insomnia": 0.0744,
    "Sleep Apnea": 0.0351
  },
  "explanation": {
    "method": "SHAP TreeExplainer",
    "base_value": 0.5794,
    "top_factors": [
      {
        "feature": "BP_Systolic",
        "value": -1.11,
        "shap_value": 0.0934,
        "direction": "supports_prediction"
      },
      {
        "feature": "BMI Category_Overweight",
        "value": 0.0,
        "shap_value": 0.0704,
        "direction": "supports_prediction"
      },
      {
        "feature": "BP_Diastolic",
        "value": -0.7545,
        "shap_value": 0.0568,
        "direction": "supports_prediction"
      },
      {
        "feature": "BMI Category_Normal",
        "value": 1.0,
        "shap_value": 0.0482,
        "direction": "supports_prediction"
      },
      {
        "feature": "Sleep Duration",
        "value": -0.4304,
        "shap_value": -0.0281,
        "direction": "opposes_prediction"
      }
    ]
  },
  "disclaimer": "Predictions and explanations are experimental analytical outputs for educational and research purposes only, not clinical diagnoses or medical advice. Consult a healthcare professional for sleep and health concerns.",
  "timestamp_utc": "2026-10-23T18:25:01.788860+00:00"
}
```

---

## 4. Error Handling and Validation Codes

When validation fails, the contract returns structured HTTP 422 Unprocessable Entity error payloads:

| Error Condition | Error Type | Message Format |
|:---|:---|:---|
| Missing required field | `missing` | `"Field required"` |
| Value out of range | `less_than_equal` / `greater_than_equal` | `"Input should be between {min} and {max}"` |
| Malformed Blood Pressure | `value_error` | `"Invalid Blood Pressure string '...'. Must follow 'Systolic/Diastolic' format (e.g. '120/80')."` |
| Inverted Blood Pressure | `value_error` | `"Systolic pressure ({sys} mmHg) must be strictly greater than diastolic pressure ({dia} mmHg)."` |
| Missing Blood Pressure Info | `value_error` | `"Blood pressure must be provided either as a 'Blood Pressure' string (e.g. '120/80') or as separate 'BP_Systolic' and 'BP_Diastolic' numeric values."` |

---

## 5. Metadata Endpoint Contract: `GET /api/model-info`

Returns model versioning, feature schemas, training performance metrics, and validation limits:

```json
{
  "model_name": "Random Forest (Base Features Only)",
  "model_version": "1.0.0",
  "contract_version": "1.0.0",
  "target_classes": ["None", "Insomnia", "Sleep Apnea"],
  "display_class_names": ["No recorded sleep disorder", "Insomnia", "Sleep Apnea"],
  "input_feature_count": 12,
  "transformed_feature_count": 25,
  "validation_bounds": {
    "age": {"min": 1, "max": 120},
    "sleep_duration": {"min": 0.1, "max": 24.0},
    "quality_of_sleep": {"min": 1, "max": 10},
    "physical_activity_level": {"min": 0, "max": 1440},
    "stress_level": {"min": 1, "max": 10},
    "heart_rate": {"min": 30, "max": 220},
    "daily_steps": {"min": 0, "max": 100000},
    "bp_systolic": {"min": 50.0, "max": 260.0},
    "bp_diastolic": {"min": 30.0, "max": 160.0}
  },
  "disclaimer": "Predictions and explanations are experimental analytical outputs for educational and research purposes only, not clinical diagnoses or medical advice. Consult a healthcare professional for sleep and health concerns."
}
```
