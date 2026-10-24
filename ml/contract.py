"""
API Payload Contract and Model Artifact Loader Module
for 'ML-Based Intelligent System for Automated Sleep Disorder Detection Using Lifestyle and Health Data'.

Provides:
- Pydantic v2 input and output schemas with strict physiological validation.
- Dual-format Blood Pressure parsing (compound 'Systolic/Diastolic' string or separate numbers).
- Dynamic aliasing (snake_case and dataset Title Case).
- Verified artifact manager for loading serialized models, encoders, and XAI metadata.
- Prediction engine with probability verification and SHAP explanation generation.
- Clinical non-diagnostic disclaimers and error handling.
"""

import os
import sys
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd
import joblib
from pydantic import BaseModel, Field, model_validator, ConfigDict

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.preprocess import clean_and_prepare_data, normalize_bmi_category, parse_blood_pressure, TARGET_CLASSES
from ml.explain import CLASS_DISPLAY_NAMES, extract_pipeline_components

CONTRACT_VERSION = "1.0.0"
MODEL_VERSION = "1.0.0"

DISCLAIMER_TEXT = (
    "Predictions and explanations are experimental analytical outputs for educational and research purposes only, "
    "not clinical diagnoses or medical advice. Consult a healthcare professional for sleep and health concerns."
)

KNOWN_GENDERS = ["Male", "Female"]
KNOWN_BMI_CATEGORIES = ["Normal", "Normal Weight", "Overweight", "Obese"]
KNOWN_OCCUPATIONS = [
    "Accountant",
    "Doctor",
    "Engineer",
    "Lawyer",
    "Manager",
    "Nurse",
    "Sales Representative",
    "Salesperson",
    "Scientist",
    "Software Engineer",
    "Teacher",
]

TARGET_CLASS_MAPPING = {
    0: {"internal": "None", "display": "No recorded sleep disorder"},
    1: {"internal": "Insomnia", "display": "Insomnia"},
    2: {"internal": "Sleep Apnea", "display": "Sleep Apnea"},
}


class SleepPredictionInput(BaseModel):
    """
    Input payload schema for single-instance sleep disorder prediction.
    Accepts both canonical dataset field names ('Sleep Duration') and snake_case aliases ('sleep_duration').
    Supports both composite blood pressure string ('120/80') and separate numeric readings.
    """

    gender: str = Field(
        ...,
        alias="Gender",
        description="Biological sex: 'Male' or 'Female'.",
        examples=["Male", "Female"],
    )
    age: int = Field(
        ...,
        alias="Age",
        ge=1,
        le=120,
        description="Patient age in years (1 - 120).",
        examples=[45],
    )
    occupation: str = Field(
        ...,
        alias="Occupation",
        min_length=1,
        max_length=100,
        description="Occupation title. Unseen occupations are safely handled by categorical encoding.",
        examples=["Software Engineer", "Nurse", "Doctor"],
    )
    sleep_duration: float = Field(
        ...,
        alias="Sleep Duration",
        gt=0.0,
        le=24.0,
        description="Average daily sleep duration in hours (0.1 - 24.0).",
        examples=[7.2],
    )
    quality_of_sleep: int = Field(
        ...,
        alias="Quality of Sleep",
        ge=1,
        le=10,
        description="Subjective rating of sleep quality on a 1-10 scale.",
        examples=[8],
    )
    physical_activity_level: int = Field(
        ...,
        alias="Physical Activity Level",
        ge=0,
        le=1440,
        description="Daily physical activity in minutes (0 - 1440).",
        examples=[60],
    )
    stress_level: int = Field(
        ...,
        alias="Stress Level",
        ge=1,
        le=10,
        description="Subjective stress rating on a 1-10 scale.",
        examples=[4],
    )
    bmi_category: str = Field(
        ...,
        alias="BMI Category",
        description="Body Mass Index category: 'Normal', 'Normal Weight', 'Overweight', or 'Obese'.",
        examples=["Normal", "Overweight", "Obese"],
    )
    blood_pressure: Optional[str] = Field(
        default=None,
        alias="Blood Pressure",
        description="Composite blood pressure formatted as 'Systolic/Diastolic' (e.g. '120/80').",
        examples=["120/80"],
    )
    bp_systolic: Optional[float] = Field(
        default=None,
        alias="BP_Systolic",
        ge=50.0,
        le=260.0,
        description="Systolic blood pressure in mmHg (50.0 - 260.0). Required if 'Blood Pressure' string omitted.",
        examples=[120.0],
    )
    bp_diastolic: Optional[float] = Field(
        default=None,
        alias="BP_Diastolic",
        ge=30.0,
        le=160.0,
        description="Diastolic blood pressure in mmHg (30.0 - 160.0). Required if 'Blood Pressure' string omitted.",
        examples=[80.0],
    )
    heart_rate: int = Field(
        ...,
        alias="Heart Rate",
        ge=30,
        le=220,
        description="Resting heart rate in beats per minute (30 - 220).",
        examples=[70],
    )
    daily_steps: int = Field(
        ...,
        alias="Daily Steps",
        ge=0,
        le=100000,
        description="Average daily step count (0 - 100000).",
        examples=[8000],
    )
    person_id: Optional[int] = Field(
        default=None,
        alias="Person ID",
        description="Optional client-side tracking identifier. Ignored during inference.",
        examples=[1],
    )

    model_config = ConfigDict(
        populate_by_name=True,
        extra="ignore",
    )

    @model_validator(mode="after")
    def validate_and_normalize_fields(self) -> "SleepPredictionInput":
        # 1. Normalize Gender
        gender_clean = self.gender.strip().capitalize()
        if gender_clean in ["Male", "Female"]:
            self.gender = gender_clean
        else:
            # Leave string for one-hot encoder ignore policy, but standardize spacing
            self.gender = self.gender.strip()

        # 2. Normalize BMI Category ('Normal Weight' -> 'Normal')
        bmi_clean = self.bmi_category.strip()
        if bmi_clean == "Normal Weight":
            bmi_clean = "Normal"
        self.bmi_category = bmi_clean

        # 3. Resolve Blood Pressure
        if self.blood_pressure is not None and str(self.blood_pressure).strip():
            raw_bp = str(self.blood_pressure).strip()
            parts = raw_bp.split("/")
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid Blood Pressure string '{raw_bp}'. Must follow 'Systolic/Diastolic' format (e.g. '120/80')."
                )
            try:
                sys_val = float(parts[0].strip())
                dia_val = float(parts[1].strip())
            except ValueError:
                raise ValueError(f"Blood pressure values must be numeric, got: '{raw_bp}'.")

            if sys_val < 50.0 or sys_val > 260.0:
                raise ValueError(f"Systolic blood pressure {sys_val} out of plausible bounds (50-260 mmHg).")
            if dia_val < 30.0 or dia_val > 160.0:
                raise ValueError(f"Diastolic blood pressure {dia_val} out of plausible bounds (30-160 mmHg).")

            self.bp_systolic = sys_val
            self.bp_diastolic = dia_val
        else:
            if self.bp_systolic is None or self.bp_diastolic is None:
                raise ValueError(
                    "Blood pressure must be provided either as a 'Blood Pressure' string (e.g. '120/80') "
                    "or as separate 'BP_Systolic' and 'BP_Diastolic' numeric values."
                )

        # 4. Physiological consistency check: Systolic must be strictly greater than Diastolic
        if self.bp_systolic <= self.bp_diastolic:
            raise ValueError(
                f"Systolic pressure ({self.bp_systolic} mmHg) must be strictly greater than "
                f"diastolic pressure ({self.bp_diastolic} mmHg)."
            )

        return self

    def to_feature_dict(self) -> Dict[str, Any]:
        """Converts validated input to exact 12-column dictionary expected by pipeline ColumnTransformer."""
        return {
            "Gender": self.gender,
            "Age": self.age,
            "Occupation": self.occupation,
            "Sleep Duration": self.sleep_duration,
            "Quality of Sleep": self.quality_of_sleep,
            "Physical Activity Level": self.physical_activity_level,
            "Stress Level": self.stress_level,
            "BMI Category": self.bmi_category,
            "Heart Rate": self.heart_rate,
            "Daily Steps": self.daily_steps,
            "BP_Systolic": self.bp_systolic,
            "BP_Diastolic": self.bp_diastolic,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Converts instance to a single-row pandas DataFrame matching pipeline schema."""
        return pd.DataFrame([self.to_feature_dict()])


class FeatureAttribution(BaseModel):
    """Local attribution of an individual feature toward the model's prediction."""
    feature: str = Field(..., description="Feature name.")
    value: Union[float, int, str] = Field(..., description="Feature value for this instance.")
    shap_value: float = Field(..., description="Shapley value attribution for the predicted class.")
    direction: str = Field(..., description="'supports_prediction' if positive, 'opposes_prediction' if negative.")


class PredictionExplanation(BaseModel):
    """Explainability container for individual prediction."""
    method: str = Field(default="SHAP TreeExplainer", description="XAI method used.")
    base_value: float = Field(..., description="Expected baseline probability for predicted class.")
    top_factors: List[FeatureAttribution] = Field(default_factory=list, description="Top positive and negative contributors.")


class SleepPredictionOutput(BaseModel):
    """
    Standardized, versioned prediction output schema.
    """
    prediction_id: str = Field(..., description="Unique UUID identifier for this prediction.")
    model_version: str = Field(default=MODEL_VERSION, description="Model release version.")
    contract_version: str = Field(default=CONTRACT_VERSION, description="API contract version.")
    predicted_class: int = Field(..., description="Predicted class index (0, 1, or 2).")
    predicted_class_name: str = Field(..., description="Human-readable disorder label.")
    probabilities: Dict[str, float] = Field(
        ...,
        description="Probability distribution across classes in standard class order.",
    )
    explanation: Optional[PredictionExplanation] = Field(
        default=None,
        description="Optional local SHAP feature attributions.",
    )
    disclaimer: str = Field(
        default=DISCLAIMER_TEXT,
        description="Mandatory non-diagnostic medical disclaimer.",
    )
    timestamp_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC ISO-8601 timestamp of inference.",
    )


class ModelContractInfo(BaseModel):
    """Information payload describing model capabilities, versions, and validation bounds."""
    model_name: str
    model_version: str
    contract_version: str
    target_classes: List[str]
    display_class_names: List[str]
    input_feature_count: int
    transformed_feature_count: int
    transformed_features: List[str]
    validation_bounds: Dict[str, Any]
    disclaimer: str


class ArtifactConsistencyError(Exception):
    """Raised when serialized model artifacts exhibit schema, class, or version discrepancies."""
    pass


class ModelArtifactManager:
    """
    Production artifact manager responsible for:
    - Loading serialized models, encoders, and XAI metadata in fresh processes.
    - Validating schema and version consistency across artifacts.
    - Executing leakage-free predictions and verifying probability distributions.
    - Generating fast, reliable single-instance SHAP local explanations.
    """

    def __init__(self, models_dir: Optional[str] = None):
        self.models_dir = models_dir or os.path.join(PROJECT_ROOT, "models")
        self.model_path = os.path.join(self.models_dir, "best_model.joblib")
        self.encoder_path = os.path.join(self.models_dir, "label_encoder.joblib")
        self.metadata_path = os.path.join(self.models_dir, "model_metadata.json")
        self.test_eval_path = os.path.join(self.models_dir, "test_evaluation.json")
        self.shap_json_path = os.path.join(self.models_dir, "shap_feature_importance.json")

        self.pipeline: Any = None
        self.label_encoder: Any = None
        self.metadata: Dict[str, Any] = {}
        self.test_evaluation: Dict[str, Any] = {}
        self.shap_metadata: Dict[str, Any] = {}

        self.preprocessor: Any = None
        self.classifier: Any = None
        self.feature_names: List[str] = []
        self._shap_explainer: Any = None

        self._load_and_validate_artifacts()

    def _load_and_validate_artifacts(self) -> None:
        """Loads and cross-verifies all 5 model artifacts."""
        # 1. Existence Check
        for path, name in [
            (self.model_path, "best_model.joblib"),
            (self.encoder_path, "label_encoder.joblib"),
            (self.metadata_path, "model_metadata.json"),
            (self.test_eval_path, "test_evaluation.json"),
            (self.shap_json_path, "shap_feature_importance.json"),
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required artifact '{name}' not found at: {path}")

        # 2. Deserialization
        self.pipeline = joblib.load(self.model_path)
        self.label_encoder = joblib.load(self.encoder_path)

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        with open(self.test_eval_path, "r", encoding="utf-8") as f:
            self.test_evaluation = json.load(f)
        with open(self.shap_json_path, "r", encoding="utf-8") as f:
            self.shap_metadata = json.load(f)

        # 3. Pipeline Structure Verification
        self.preprocessor, self.classifier, self.feature_names = extract_pipeline_components(self.pipeline)

        if len(self.feature_names) != 25:
            raise ArtifactConsistencyError(
                f"Expected 25 transformed features, got {len(self.feature_names)} from preprocessor."
            )

        # 4. Cross-Artifact Target Class Alignment
        encoder_classes = list(self.label_encoder.classes_)
        meta_classes = self.metadata.get("target_classes", [])
        if encoder_classes != TARGET_CLASSES or meta_classes != TARGET_CLASSES:
            raise ArtifactConsistencyError(
                f"Class mismatch between encoder ({encoder_classes}) and metadata ({meta_classes}). Expected {TARGET_CLASSES}"
            )

        # 5. Classifier Classes Alignment
        clf_classes = getattr(self.classifier, "classes_", None)
        if clf_classes is not None and list(clf_classes) != [0, 1, 2]:
            raise ArtifactConsistencyError(f"Classifier classes {clf_classes} do not match expected [0, 1, 2].")

        # 6. Transformed Features Consistency
        shap_features = self.shap_metadata.get("transformed_features", [])
        if shap_features != self.feature_names:
            raise ArtifactConsistencyError(
                "Mismatch between preprocessor get_feature_names_out() and shap_feature_importance.json."
            )

    @property
    def shap_explainer(self) -> Any:
        """Lazy-loaded, reusable SHAP TreeExplainer instance."""
        if self._shap_explainer is None:
            import shap
            # Fast, exact tree-path explainer on the fitted random forest
            self._shap_explainer = shap.TreeExplainer(self.classifier)
        return self._shap_explainer

    def get_contract_info(self) -> ModelContractInfo:
        """Returns metadata describing contract version and schema constraints."""
        return ModelContractInfo(
            model_name=self.metadata.get("full_model_name", "Random Forest"),
            model_version=MODEL_VERSION,
            contract_version=CONTRACT_VERSION,
            target_classes=TARGET_CLASSES,
            display_class_names=CLASS_DISPLAY_NAMES,
            input_feature_count=12,
            transformed_feature_count=len(self.feature_names),
            transformed_features=self.feature_names,
            validation_bounds={
                "age": {"min": 1, "max": 120},
                "sleep_duration": {"min": 0.1, "max": 24.0},
                "quality_of_sleep": {"min": 1, "max": 10},
                "physical_activity_level": {"min": 0, "max": 1440},
                "stress_level": {"min": 1, "max": 10},
                "heart_rate": {"min": 30, "max": 220},
                "daily_steps": {"min": 0, "max": 100000},
                "bp_systolic": {"min": 50.0, "max": 260.0},
                "bp_diastolic": {"min": 30.0, "max": 160.0},
                "bmi_categories": KNOWN_BMI_CATEGORIES,
                "genders": KNOWN_GENDERS,
            },
            disclaimer=DISCLAIMER_TEXT,
        )

    def predict(
        self,
        input_data: Union[SleepPredictionInput, Dict[str, Any], pd.DataFrame],
        include_explanation: bool = False,
        top_n_features: int = 5,
    ) -> SleepPredictionOutput:
        """
        Executes verified inference on a single patient input.

        Args:
            input_data: Validated SleepPredictionInput, raw dictionary, or single-row DataFrame.
            include_explanation: Whether to compute local SHAP feature attributions.
            top_n_features: Number of top contributing factors to include in explanation.

        Returns:
            SleepPredictionOutput instance with predicted class, mapped probabilities, and metadata.
        """
        # 1. Parse and validate through contract schema
        if isinstance(input_data, SleepPredictionInput):
            validated_input = input_data
        elif isinstance(input_data, dict):
            validated_input = SleepPredictionInput(**input_data)
        elif isinstance(input_data, pd.DataFrame):
            if len(input_data) != 1:
                raise ValueError("predict() expects exactly 1 row in DataFrame input.")
            row_dict = input_data.iloc[0].to_dict()
            validated_input = SleepPredictionInput(**row_dict)
        else:
            raise TypeError(f"Unsupported input type for prediction: {type(input_data)}")

        df_features = validated_input.to_dataframe()

        # 2. Execute Inference
        pred_class = int(self.pipeline.predict(df_features)[0])
        probabilities = self.pipeline.predict_proba(df_features)[0]

        # Verify probabilities
        if np.isnan(probabilities).any() or np.isinf(probabilities).any():
            raise ValueError("Model returned NaN or infinite probability values.")
        prob_sum = float(np.sum(probabilities))
        if not np.isclose(prob_sum, 1.0, atol=1e-4):
            raise ValueError(f"Probabilities do not sum to 1.0 (sum={prob_sum}).")

        # Map probabilities to human-readable class names in strict order [0, 1, 2]
        prob_dict = {
            CLASS_DISPLAY_NAMES[i]: round(float(probabilities[i]), 4)
            for i in range(len(CLASS_DISPLAY_NAMES))
        }

        # 3. Local Explanation (if requested)
        explanation_obj: Optional[PredictionExplanation] = None
        if include_explanation:
            X_trans = self.preprocessor.transform(df_features)
            if hasattr(X_trans, "toarray"):
                X_trans = X_trans.toarray()
            X_trans = np.asarray(X_trans, dtype=float)

            shap_vals = self.shap_explainer.shap_values(X_trans, check_additivity=False)
            # Shape is (1, n_features, n_classes) or list of (1, n_features)
            if isinstance(shap_vals, list):
                class_shap = shap_vals[pred_class][0]
            elif shap_vals.ndim == 3:
                class_shap = shap_vals[0, :, pred_class]
            else:
                class_shap = shap_vals[0]

            expected_val = self.shap_explainer.expected_value
            base_val = float(expected_val[pred_class]) if hasattr(expected_val, "__len__") else float(expected_val)

            # Sort top features by magnitude
            top_indices = np.argsort(np.abs(class_shap))[::-1][:top_n_features]
            factors: List[FeatureAttribution] = []
            for idx in top_indices:
                feat_name = self.feature_names[idx]
                feat_val = float(X_trans[0, idx])
                feat_shap = float(class_shap[idx])
                factors.append(
                    FeatureAttribution(
                        feature=feat_name,
                        value=round(feat_val, 4),
                        shap_value=round(feat_shap, 4),
                        direction="supports_prediction" if feat_shap >= 0 else "opposes_prediction",
                    )
                )

            explanation_obj = PredictionExplanation(
                method="SHAP TreeExplainer",
                base_value=round(base_val, 4),
                top_factors=factors,
            )

        return SleepPredictionOutput(
            prediction_id=str(uuid.uuid4()),
            model_version=MODEL_VERSION,
            contract_version=CONTRACT_VERSION,
            predicted_class=pred_class,
            predicted_class_name=CLASS_DISPLAY_NAMES[pred_class],
            probabilities=prob_dict,
            explanation=explanation_obj,
            disclaimer=DISCLAIMER_TEXT,
        )


if __name__ == "__main__":
    print("Testing ModelArtifactManager initialization and contract verification...")
    manager = ModelArtifactManager()
    info = manager.get_contract_info()
    print(f"Loaded model: {info.model_name} (v{info.model_version})")
    print(f"Features: {info.input_feature_count} raw, {info.transformed_feature_count} transformed.")

    sample_req = {
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
        "Daily Steps": 7500,
    }
    res = manager.predict(sample_req, include_explanation=True)
    print("\nSample Prediction Output:")
    print(res.model_dump_json(indent=2))
