"""Runtime configuration, read from environment variables (and an optional .env file)."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[1]

# Later calls do not override variables that are already set in the real environment.
load_dotenv(BACKEND_DIR / ".env")
load_dotenv(BACKEND_DIR.parent / ".env")


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    forecast_model_path: Path
    vision_model_path: Path
    vision_labels_path: Path
    gemini_api_key: str
    gemini_model: str
    gemini_timeout_seconds: float
    max_upload_bytes: int
    cors_origins: list[str]
    auto_train_forecast: bool
    synthetic_days: int
    default_latitude: float
    reference_capacity_kw: float
    low_confidence_threshold: float


def get_settings() -> Settings:
    """Build settings on every call so tests can change the environment."""
    model_dir = Path(os.getenv("MODEL_DIR", BACKEND_DIR / "models"))
    return Settings(
        forecast_model_path=Path(os.getenv("FORECAST_MODEL_PATH", model_dir / "forecast_bundle.joblib")),
        vision_model_path=Path(os.getenv("VISION_MODEL_PATH", model_dir / "panel_classifier.keras")),
        vision_labels_path=Path(os.getenv("VISION_LABELS_PATH", model_dir / "panel_classifier_labels.json")),
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        gemini_timeout_seconds=float(os.getenv("GEMINI_TIMEOUT_SECONDS", "20")),
        max_upload_bytes=int(float(os.getenv("MAX_UPLOAD_MB", "5")) * 1024 * 1024),
        cors_origins=[o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip()],
        auto_train_forecast=_as_bool(os.getenv("AUTO_TRAIN_FORECAST", "true")),
        synthetic_days=int(os.getenv("SYNTHETIC_DAYS", "730")),
        default_latitude=float(os.getenv("DEFAULT_LATITUDE", "30.9")),
        reference_capacity_kw=float(os.getenv("REFERENCE_CAPACITY_KW", "100")),
        low_confidence_threshold=float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.6")),
    )
