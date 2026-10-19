import logging
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.assistant.service import explain
from app.config import get_settings
from app.errors import AppError, register_error_handlers
from app.forecast.inference import get_forecast_service
from app.forecast.training import metrics_summary
from app.schemas import AssistantRequest, AssistantResponse, ForecastRequest, ForecastResponse, InspectionResult
from app.vision.inference import get_classifier, vision_status
from app.vision.preprocess import decode_image, validate_upload

logger = logging.getLogger("solarsight")
router = APIRouter(prefix="/api")


@router.get("/health")
def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "forecast_model_file_present": settings.forecast_model_path.exists(),
        "vision": vision_status(),
        "gemini_configured": bool(settings.gemini_api_key),
    }


@router.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest) -> ForecastResponse:
    return get_forecast_service().predict_day(request)


@router.get("/model/metrics")
def model_metrics() -> dict:
    return {"forecast": metrics_summary(get_forecast_service().bundle), "vision": vision_status()}


@router.get("/inspect/status")
def inspection_status() -> dict:
    return vision_status()


@router.post("/inspect", response_model=InspectionResult)
def inspect_panel(file: UploadFile = File(...)) -> InspectionResult:
    settings = get_settings()
    # Read one byte past the limit so oversized uploads are rejected without loading them fully.
    data = file.file.read(settings.max_upload_bytes + 1)
    validate_upload(file.filename or "", data, settings.max_upload_bytes)
    classifier = get_classifier()
    return classifier.classify(decode_image(data), file.filename or "upload", settings.low_confidence_threshold)


@router.post("/assistant/explain", response_model=AssistantResponse)
def assistant_explain(request: AssistantRequest) -> AssistantResponse:
    return explain(request)


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    if settings.auto_train_forecast or settings.forecast_model_path.exists():
        try:
            get_forecast_service()  # load (or train once) before the first request
        except AppError as error:
            logger.warning("Forecast model not ready at startup: %s", error.message)
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="SolarSight AI", version="1.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_settings().cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    register_error_handlers(app)
    app.include_router(router)
    return app


app = create_app()
