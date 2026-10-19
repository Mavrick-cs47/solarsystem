"""Upload validation and image preprocessing (OpenCV). No TensorFlow needed here."""
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from app.errors import AppError

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_PIXELS = 25_000_000
IMAGE_SIZE = 224


def _looks_like(data: bytes, extension: str) -> bool:
    if extension in {".jpg", ".jpeg"}:
        return data.startswith(b"\xff\xd8\xff")
    if extension == ".png":
        return data.startswith(b"\x89PNG\r\n\x1a\n")
    if extension == ".webp":
        return data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    return False


def validate_upload(filename: str, data: bytes, max_bytes: int) -> None:
    """Check extension, size, file signature and pixel count before any decoding."""
    extension = Path(filename or "").suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise AppError(415, "unsupported_file_type", "Only JPG, PNG or WEBP images are accepted.")
    if not data:
        raise AppError(400, "empty_file", "The uploaded file is empty.")
    if len(data) > max_bytes:
        raise AppError(413, "file_too_large", f"Image exceeds the {max_bytes // (1024 * 1024)} MB limit.")
    if not _looks_like(data, extension):
        raise AppError(415, "file_content_mismatch", "File contents do not match its extension.")
    try:
        width, height = Image.open(BytesIO(data)).size
    except (UnidentifiedImageError, OSError):
        raise AppError(400, "unreadable_image", "The image could not be read.")
    if width * height > MAX_PIXELS:
        raise AppError(413, "image_too_large", "Image resolution is too large.")


def decode_image(data: bytes) -> np.ndarray:
    """Decode bytes into a BGR image array."""
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise AppError(400, "unreadable_image", "The image could not be decoded.")
    return image


def read_image_file(path: str | Path) -> np.ndarray | None:
    """Read from disk via numpy so non-ASCII Windows paths work (cv2.imread fails on them)."""
    try:
        return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    except OSError:
        return None


def to_model_input(bgr_image: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Resize and convert to RGB float32 in 0..255. Model-specific scaling lives inside the model."""
    resized = cv2.resize(bgr_image, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
