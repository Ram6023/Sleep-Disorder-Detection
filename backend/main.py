"""
FastAPI Prediction REST Service
for 'ML-Based Intelligent System for Automated Sleep Disorder Detection Using Lifestyle and Health Data'.

Provides:
- GET  /              : Service index, documentation links, and non-diagnostic disclaimer.
- GET  /health        : System health check and model loading readiness.
- GET  /api/model-info: Model metadata, contract version, feature schema, and bounds.
- POST /api/predict   : Multi-class prediction endpoint with mapped probabilities and optional SHAP XAI.
"""

import os
import sys
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Depends, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.contract import (
    CONTRACT_VERSION,
    MODEL_VERSION,
    DISCLAIMER_TEXT,
    SleepPredictionInput,
    SleepPredictionOutput,
    ModelContractInfo,
    ModelArtifactManager,
    ArtifactConsistencyError,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sleep_disorder_api")

# CORS Configuration
# Restrict to explicit local frontend origins by default, customizable via environment variable
raw_cors = os.environ.get(
    "CORS_ALLOWED_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000",
)
ALLOWED_ORIGINS = [origin.strip() for origin in raw_cors.split(",") if origin.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application lifecycle: loads model artifacts once at startup and cleans up at shutdown.
    Avoids retraining or reloading model pipeline on individual requests.
    """
    logger.info("Initializing Sleep Disorder Detection inference engine...")
    try:
        artifact_manager = ModelArtifactManager()
        app.state.artifact_manager = artifact_manager
        app.state.model_loaded = True
        app.state.load_error = None
        logger.info(
            "Successfully loaded model: %s (v%s, %d transformed features)",
            artifact_manager.metadata.get("full_model_name", "Random Forest"),
            MODEL_VERSION,
            len(artifact_manager.feature_names),
        )
    except Exception as exc:
        logger.exception("Failed to load serialized model artifacts at startup: %s", exc)
        app.state.artifact_manager = None
        app.state.model_loaded = False
        app.state.load_error = str(exc)

    yield

    logger.info("Shutting down Sleep Disorder Detection service...")
    app.state.artifact_manager = None
    app.state.model_loaded = False


# Create FastAPI application
app = FastAPI(
    title="Sleep Disorder Detection API",
    description=(
        "REST API for automated multi-class sleep disorder detection using lifestyle and health metrics. "
        "Provides validated inferences, calibrated probability distributions, and local SHAP explainability. "
        "Intended for research and educational purposes only; not a medical diagnostic tool."
    ),
    version=CONTRACT_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS Middleware with explicit origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def get_artifact_manager(request: Request) -> ModelArtifactManager:
    """
    Dependency injection provider returning the verified ModelArtifactManager singleton.
    Raises HTTP 503 if artifacts failed to load.
    """
    manager = getattr(request.app.state, "artifact_manager", None)
    if manager is None or not getattr(request.app.state, "model_loaded", False):
        error_msg = getattr(request.app.state, "load_error", "Model artifacts not loaded.")
        logger.error("Inference requested but model is unavailable: %s", error_msg)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service is currently unavailable. Serialized model artifacts could not be loaded.",
        )
    return manager


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """
    Catches unexpected internal errors and returns safe sanitized responses
    without leaking server-side stack traces or filesystem paths to clients.
    """
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    logger.exception("Unhandled server exception processing request %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred while processing the request."},
    )


@app.get(
    "/",
    summary="Service Status and Index",
    tags=["General"],
)
async def root() -> Dict[str, Any]:
    """Returns general service status, version, documentation endpoints, and medical disclaimer."""
    return {
        "service": "Sleep Disorder Detection Prediction API",
        "status": "online",
        "api_version": CONTRACT_VERSION,
        "model_version": MODEL_VERSION,
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "openapi_json": "/openapi.json",
        },
        "disclaimer": DISCLAIMER_TEXT,
    }


@app.get(
    "/health",
    summary="Health and Readiness Check",
    tags=["Monitoring"],
)
async def health(request: Request) -> Dict[str, Any]:
    """
    Returns system operational health, model loading readiness, and timestamp.
    Does not expose sensitive filesystem paths or environment secrets.
    """
    is_loaded = bool(getattr(request.app.state, "model_loaded", False))
    return {
        "status": "healthy" if is_loaded else "degraded",
        "model_loaded": is_loaded,
        "model_version": MODEL_VERSION,
        "contract_version": CONTRACT_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


@app.get(
    "/api/model-info",
    response_model=ModelContractInfo,
    summary="Model Metadata and Input Constraints",
    tags=["Model Information"],
)
async def get_model_info(
    manager: ModelArtifactManager = Depends(get_artifact_manager),
) -> ModelContractInfo:
    """
    Returns comprehensive model metadata, target class definitions,
    transformed feature names, input validation ranges, and non-diagnostic disclaimers.
    """
    return manager.get_contract_info()


@app.post(
    "/api/predict",
    response_model=SleepPredictionOutput,
    status_code=status.HTTP_200_OK,
    summary="Predict Sleep Disorder and Probability Distribution",
    tags=["Prediction"],
)
async def predict_sleep_disorder(
    payload: SleepPredictionInput,
    explain: bool = Query(
        default=False,
        description="Set to true to compute on-demand local SHAP feature attributions.",
    ),
    manager: ModelArtifactManager = Depends(get_artifact_manager),
) -> SleepPredictionOutput:
    """
    Executes multi-class classification on patient lifestyle and health metrics.

    - **Input**: Validated `SleepPredictionInput` (accepts both `"Blood Pressure": "120/80"` or pre-split `BP_Systolic`/`BP_Diastolic`).
    - **Output**: Calibrated class probabilities, predicted category (`"No recorded sleep disorder"`, `"Insomnia"`, `"Sleep Apnea"`),
      and optional local SHAP attributions.
    """
    try:
        result = manager.predict(payload, include_explanation=explain)
        return result
    except ValueError as val_err:
        logger.warning("Prediction validation failure: %s", val_err)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(val_err),
        )
    except Exception as exc:
        logger.exception("Unexpected error during inference: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference calculation failed due to an unexpected internal error.",
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "127.0.0.1")
    logger.info("Starting local development server on http://%s:%d", host, port)
    uvicorn.run("backend.main:app", host=host, port=port, reload=True)
