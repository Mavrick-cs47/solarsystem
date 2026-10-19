"""Load the trained classifier lazily and classify one uploaded image."""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.config import get_settings
from app.errors import AppError
from app.schemas import InspectionResult
from app.vision.preprocess import to_model_input

REVIEW_NOTE = (
    "Classifier output is a screening aid trained on a limited dataset. "
    "A qualified technician must confirm any finding before maintenance action."
)
TRAINING_HINT = "Train one with: python -m app.vision.train --dataset data/panel_images"


class PanelClassifier:
    def __init__(self, model, labels: dict):
        self.model = model
        self.labels = labels
        self.classes: list[str] = labels["classes"]

    def classify(self, bgr_image: np.ndarray, filename: str, low_confidence_threshold: float) -> InspectionResult:
        batch = to_model_input(bgr_image, self.labels["image_size"])[np.newaxis, ...]
        probabilities = self.model.predict(batch, verbose=0)[0]
        best = int(np.argmax(probabilities))
        confidence = float(probabilities[best])
        return InspectionResult(
            filename=filename,
            predicted_class=self.classes[best],
            confidence=round(confidence, 4),
            probabilities={name: round(float(p), 4) for name, p in zip(self.classes, probabilities)},
            low_confidence=confidence < low_confidence_threshold,
            model_test_accuracy=self.labels.get("test_metrics", {}).get("accuracy"),
            note=REVIEW_NOTE,
        )


def vision_status() -> dict:
    """Describe whether image classification can run, without loading TensorFlow."""
    settings = get_settings()
    if not settings.vision_model_path.exists() or not settings.vision_labels_path.exists():
        return {"available": False, "reason": f"No trained model found. {TRAINING_HINT}"}
    labels = json.loads(settings.vision_labels_path.read_text())
    return {
        "available": True,
        "classes": labels["classes"],
        "trained_at": labels.get("trained_at"),
        "test_metrics": labels.get("test_metrics"),
        "images_per_split": labels.get("images_per_split"),
    }


@lru_cache(maxsize=2)
def _load_classifier(model_path: str, labels_path: str) -> PanelClassifier:
    try:
        import tensorflow as tf
    except ImportError:
        raise AppError(
            503,
            "tensorflow_not_installed",
            "TensorFlow is not installed. Run: pip install -r requirements-vision.txt",
        )
    labels = json.loads(Path(labels_path).read_text())
    return PanelClassifier(tf.keras.models.load_model(model_path), labels)


def get_classifier() -> PanelClassifier:
    settings = get_settings()
    if not settings.vision_model_path.exists() or not settings.vision_labels_path.exists():
        raise AppError(503, "vision_model_missing", f"No trained image model is available. {TRAINING_HINT}")
    return _load_classifier(str(settings.vision_model_path), str(settings.vision_labels_path))
