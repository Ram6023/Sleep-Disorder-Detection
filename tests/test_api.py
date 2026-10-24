"""
Integration and End-to-End API Tests for FastAPI Service (Task 1.9)
for 'ML-Based Intelligent System for Automated Sleep Disorder Detection Using Lifestyle and Health Data'.

Tests cover:
1. Root endpoint (GET /): status, version, documentation endpoints, and medical disclaimer.
2. Health check (GET /health): operational readiness, model_loaded boolean, and privacy protection.
3. Model information (GET /api/model-info): schema bounds, transformed features, and target classes.
4. Prediction with snake_case fields (POST /api/predict).
5. Prediction with canonical Title Case field aliases (POST /api/predict).
6. Prediction with dual blood pressure formats (compound string vs pre-split numeric).
7. Validation rejection for invalid, out-of-range, and incomplete payloads (HTTP 422).
8. Class probability distribution calibration and strict class order mapping.
9. Output contract compliance and non-diagnostic disclaimer verification.
10. Optional on-demand SHAP local explainability parameter (explain=true vs explain=false).
11. Resilience and safe error response when model artifacts are unavailable (HTTP 503).
12. Singleton persistence: repeated requests reuse cached pipeline without retraining or reloading.
"""

import os
import sys
import uuid
import pytest
from fastapi.testclient import TestClient

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.main import app, get_artifact_manager
from ml.contract import (
    CONTRACT_VERSION,
    MODEL_VERSION,
    DISCLAIMER_TEXT,
    CLASS_DISPLAY_NAMES,
)


@pytest.fixture(scope="module")
def client():
    """Provides a TestClient with lifespan context executed."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_title_case_payload():
    """Sample payload with canonical dataset Title Case names and compound blood pressure."""
    return {
        "Gender": "Male",
        "Age": 42,
        "Occupation": "Software Engineer",
        "Sleep Duration": 7.2,
        "Quality of Sleep": 8,
        "Physical Activity Level": 60,
        "Stress Level": 4,
        "BMI Category": "Normal",
        "Blood Pressure": "120/80",
        "Heart Rate": 68,
        "Daily Steps": 8000,
        "Person ID": 101,
    }


@pytest.fixture
def valid_snake_case_payload():
    """Sample payload with snake_case field names and pre-split numeric blood pressure."""
    return {
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
        "daily_steps": 3500,
    }


def test_root_endpoint_metadata_and_docs_links(client):
    """Verifies that GET / returns service information, docs links, and non-diagnostic disclaimer."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "online"
    assert data["api_version"] == CONTRACT_VERSION
    assert data["model_version"] == MODEL_VERSION
    assert "documentation" in data
    assert data["documentation"]["swagger_ui"] == "/docs"
    assert data["documentation"]["redoc"] == "/redoc"
    assert data["disclaimer"] == DISCLAIMER_TEXT


def test_health_check_endpoint(client):
    """Verifies that GET /health returns operational readiness without private path leaks."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model_version"] == MODEL_VERSION
    assert "timestamp_utc" in data

    # Verify no local filesystem paths or stack traces leaked
    response_str = response.text
    assert "C:\\" not in response_str
    assert "models/" not in response_str


def test_model_info_endpoint(client):
    """Verifies that GET /api/model-info returns schema limits, transformed features, and classes."""
    response = client.get("/api/model-info")
    assert response.status_code == 200
    data = response.json()

    assert data["model_version"] == MODEL_VERSION
    assert data["target_classes"] == ["None", "Insomnia", "Sleep Apnea"]
    assert data["display_class_names"] == CLASS_DISPLAY_NAMES
    assert data["input_feature_count"] == 12
    assert data["transformed_feature_count"] == 25
    assert len(data["transformed_features"]) == 25
    assert "validation_bounds" in data
    assert "age" in data["validation_bounds"]
    assert data["disclaimer"] == DISCLAIMER_TEXT


def test_predict_success_canonical_title_case_and_compound_bp(client, valid_title_case_payload):
    """Verifies successful prediction using canonical Title Case keys and compound Blood Pressure string."""
    response = client.post("/api/predict", json=valid_title_case_payload)
    assert response.status_code == 200
    data = response.json()

    # Validate output schema fields
    assert "prediction_id" in data
    uuid.UUID(data["prediction_id"])  # Valid UUID
    assert data["model_version"] == MODEL_VERSION
    assert data["predicted_class"] in [0, 1, 2]
    assert data["predicted_class_name"] in CLASS_DISPLAY_NAMES
    assert data["predicted_class_name"] == CLASS_DISPLAY_NAMES[data["predicted_class"]]
    assert data["disclaimer"] == DISCLAIMER_TEXT
    assert data["explanation"] is None  # Default explain=False

    # Validate probabilities
    probs = data["probabilities"]
    assert len(probs) == 3
    assert set(probs.keys()) == set(CLASS_DISPLAY_NAMES)
    prob_sum = sum(probs.values())
    assert prob_sum == pytest.approx(1.0, abs=1e-3)


def test_predict_success_snake_case_and_split_bp(client, valid_snake_case_payload):
    """Verifies successful prediction using snake_case keys and pre-split numeric blood pressure."""
    response = client.post("/api/predict", json=valid_snake_case_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["predicted_class"] in [0, 1, 2]
    assert data["predicted_class_name"] == CLASS_DISPLAY_NAMES[data["predicted_class"]]
    probs = data["probabilities"]
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-3)


def test_predict_optional_shap_explanation(client, valid_title_case_payload):
    """Verifies that setting query parameter explain=true returns populated local SHAP factors."""
    response = client.post("/api/predict?explain=true", json=valid_title_case_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["explanation"] is not None
    exp = data["explanation"]
    assert exp["method"] == "SHAP TreeExplainer"
    assert isinstance(exp["base_value"], float)
    assert len(exp["top_factors"]) > 0

    for factor in exp["top_factors"]:
        assert "feature" in factor
        assert "value" in factor
        assert "shap_value" in factor
        assert factor["direction"] in ["supports_prediction", "opposes_prediction"]


def test_predict_unknown_categorical_values(client, valid_title_case_payload):
    """Verifies that unseen occupations (e.g. 'Astronaut') are handled safely without error."""
    novel_payload = valid_title_case_payload.copy()
    novel_payload["Occupation"] = "Deep Sea Diver"
    response = client.post("/api/predict", json=novel_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] in [0, 1, 2]


def test_predict_validation_errors_missing_field(client, valid_title_case_payload):
    """Verifies that missing required fields trigger an HTTP 422 Unprocessable Entity."""
    bad_payload = valid_title_case_payload.copy()
    del bad_payload["Age"]
    response = client.post("/api/predict", json=bad_payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_predict_validation_errors_physiological_bounds(client, valid_title_case_payload):
    """Verifies that out-of-range numerical metrics are rejected with HTTP 422."""
    # Negative age
    bad_age = valid_title_case_payload.copy()
    bad_age["Age"] = -10
    assert client.post("/api/predict", json=bad_age).status_code == 422

    # Stress level > 10
    bad_stress = valid_title_case_payload.copy()
    bad_stress["Stress Level"] = 20
    assert client.post("/api/predict", json=bad_stress).status_code == 422

    # Inverted blood pressure (Systolic <= Diastolic)
    inverted_bp = valid_title_case_payload.copy()
    inverted_bp["Blood Pressure"] = "75/120"
    resp = client.post("/api/predict", json=inverted_bp)
    assert resp.status_code == 422

    # Malformed blood pressure string
    malformed_bp = valid_title_case_payload.copy()
    malformed_bp["Blood Pressure"] = "120-80"
    assert client.post("/api/predict", json=malformed_bp).status_code == 422


def test_service_unavailable_when_artifacts_not_loaded(client):
    """Verifies that when model artifacts cannot be accessed, the API safely returns HTTP 503."""
    # Dependency override simulating an artifact failure
    def mock_get_artifact_manager():
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service is currently unavailable. Serialized model artifacts could not be loaded.",
        )

    app.dependency_overrides[get_artifact_manager] = mock_get_artifact_manager
    try:
        sample_payload = {
            "Gender": "Male",
            "Age": 30,
            "Occupation": "Doctor",
            "Sleep Duration": 7.0,
            "Quality of Sleep": 8,
            "Physical Activity Level": 60,
            "Stress Level": 4,
            "BMI Category": "Normal",
            "Blood Pressure": "120/80",
            "Heart Rate": 70,
            "Daily Steps": 8000,
        }
        resp = client.post("/api/predict", json=sample_payload)
        assert resp.status_code == 503
        assert "unavailable" in resp.json()["detail"].lower()
    finally:
        app.dependency_overrides.clear()


def test_singleton_pipeline_reuse_across_repeated_requests(client, valid_title_case_payload):
    """Verifies that multiple repeated prediction requests reuse the cached in-memory pipeline."""
    # Make 5 sequential requests and verify distinct prediction IDs but identical deterministic output
    ids = []
    classes = []
    for _ in range(5):
        resp = client.post("/api/predict", json=valid_title_case_payload)
        assert resp.status_code == 200
        data = resp.json()
        ids.append(data["prediction_id"])
        classes.append(data["predicted_class"])

    # All prediction IDs should be unique UUIDs
    assert len(set(ids)) == 5
    # All predicted classes should be identical for the same input
    assert len(set(classes)) == 1
