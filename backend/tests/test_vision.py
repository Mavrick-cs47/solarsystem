import numpy as np
import pytest
from PIL import Image

from app.errors import AppError
from app.vision.dataset import DatasetError, list_labelled_images, split_dataset
from app.vision.inference import PanelClassifier
from app.vision.preprocess import decode_image, to_model_input, validate_upload
from tests.conftest import make_image_bytes


def make_dataset(root, counts):
    for class_name, count in counts.items():
        (root / class_name).mkdir(parents=True)
        for i in range(count):
            Image.new("RGB", (32, 32), (i * 3 % 255, 50, 90)).save(root / class_name / f"{i}.png")


def test_dataset_split_is_disjoint_and_stratified(tmp_path):
    make_dataset(tmp_path, {"clean": 40, "dusty": 20})
    paths, labels, classes = list_labelled_images(tmp_path)
    splits = split_dataset(paths, labels, seed=1)

    assert classes == ["clean", "dusty"]
    seen = [set(items[0]) for items in splits.values()]
    assert not (seen[0] & seen[1]) and not (seen[0] & seen[2]) and not (seen[1] & seen[2])
    assert sum(len(s) for s in seen) == 60
    test_labels = splits["test"][1]
    assert 0 in test_labels and 1 in test_labels  # both classes present in the test split


def test_dataset_errors_are_explicit(tmp_path):
    with pytest.raises(DatasetError, match="not found"):
        list_labelled_images(tmp_path / "missing")
    make_dataset(tmp_path, {"clean": 40})
    with pytest.raises(DatasetError, match="two class"):
        list_labelled_images(tmp_path)
    make_dataset(tmp_path, {"dusty": 3})
    with pytest.raises(DatasetError, match="at least 10"):
        list_labelled_images(tmp_path)


def test_unreadable_images_are_excluded(tmp_path):
    make_dataset(tmp_path, {"a": 12, "b": 12})
    (tmp_path / "a" / "broken.png").write_bytes(b"not an image")
    paths, _, _ = list_labelled_images(tmp_path)
    assert len(paths) == 24


@pytest.mark.parametrize(
    "filename,data,status,code",
    [
        ("scan.gif", b"GIF89a", 415, "unsupported_file_type"),
        ("empty.png", b"", 400, "empty_file"),
        ("fake.png", b"this is plain text, not a png", 415, "file_content_mismatch"),
        ("renamed.jpg", make_image_bytes("PNG"), 415, "file_content_mismatch"),
    ],
)
def test_upload_validation_rejects_bad_files(filename, data, status, code):
    with pytest.raises(AppError) as caught:
        validate_upload(filename, data, max_bytes=1_000_000)
    assert (caught.value.status_code, caught.value.code) == (status, code)


def test_upload_size_limit_and_accepted_formats():
    big = make_image_bytes("PNG", size=(400, 400))
    with pytest.raises(AppError) as caught:
        validate_upload("big.png", big, max_bytes=100)
    assert caught.value.status_code == 413
    for image_format, name in [("PNG", "a.png"), ("JPEG", "a.jpg"), ("WEBP", "a.webp")]:
        validate_upload(name, make_image_bytes(image_format), max_bytes=1_000_000)


def test_preprocessing_output_shape_and_channel_order():
    red_png = make_image_bytes("PNG", color=(255, 0, 0))
    model_input = to_model_input(decode_image(red_png))
    assert model_input.shape == (224, 224, 3) and model_input.dtype == np.float32
    assert model_input[0, 0, 0] == 255 and model_input[0, 0, 2] == 0  # RGB order after BGR decode


class FakeModel:
    def predict(self, batch, verbose=0):
        assert batch.shape == (1, 224, 224, 3)
        return np.array([[0.15, 0.7, 0.15]])


def test_classifier_returns_probabilities_and_flags_low_confidence():
    labels = {"classes": ["clean", "dusty", "cracked"], "image_size": 224, "test_metrics": {"accuracy": 0.9}}
    classifier = PanelClassifier(FakeModel(), labels)
    image = decode_image(make_image_bytes("PNG"))

    result = classifier.classify(image, "panel.png", low_confidence_threshold=0.6)
    assert result.predicted_class == "dusty" and result.confidence == 0.7
    assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-3)
    assert not result.low_confidence and result.model_test_accuracy == 0.9

    assert classifier.classify(image, "panel.png", low_confidence_threshold=0.9).low_confidence
