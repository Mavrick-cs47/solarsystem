import os
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture(scope="session", autouse=True)
def isolated_environment(tmp_path_factory):
    """Point the app at a temp model dir and a small synthetic dataset so tests stay fast."""
    model_dir = tmp_path_factory.mktemp("models")
    patch = pytest.MonkeyPatch()
    patch.setenv("MODEL_DIR", str(model_dir))
    patch.setenv("SYNTHETIC_DAYS", "120")
    patch.setenv("AUTO_TRAIN_FORECAST", "true")
    patch.setenv("GEMINI_API_KEY", "")
    patch.delenv("FORECAST_MODEL_PATH", raising=False)
    patch.delenv("VISION_MODEL_PATH", raising=False)
    patch.delenv("VISION_LABELS_PATH", raising=False)
    yield
    patch.undo()


@pytest.fixture(scope="session")
def client(isolated_environment):
    from app.main import create_app

    with TestClient(create_app()) as test_client:  # runs lifespan: trains the demo model once
        yield test_client


@pytest.fixture
def valid_forecast_payload():
    return {
        "date": "2025-06-15",
        "capacity_kw": 100,
        "temperature_c": 30,
        "cloud_cover_pct": 20,
        "humidity_pct": 45,
        "wind_speed_ms": 3,
    }


def make_image_bytes(image_format="PNG", size=(64, 64), color=(120, 130, 140)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, color).save(buffer, format=image_format)
    return buffer.getvalue()


@pytest.fixture
def png_bytes():
    return make_image_bytes("PNG")
