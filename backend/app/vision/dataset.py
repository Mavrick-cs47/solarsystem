"""Folder-per-class dataset discovery and a stratified train/validation/test split.

Expected layout (class names come from the folder names):

    data/panel_images/
        clean/        *.jpg
        dusty/        *.jpg
        bird_drop/    *.jpg
"""
from pathlib import Path

from sklearn.model_selection import train_test_split

from app.vision.preprocess import ALLOWED_EXTENSIONS, read_image_file

MIN_IMAGES_PER_CLASS = 10


class DatasetError(RuntimeError):
    """Raised when the dataset folder is missing or unusable."""


def list_labelled_images(dataset_dir: Path, check_readable: bool = True) -> tuple[list[Path], list[int], list[str]]:
    if not dataset_dir.is_dir():
        raise DatasetError(
            f"Dataset folder not found: {dataset_dir}. Create one sub-folder per class and put images in it."
        )
    class_dirs = sorted(p for p in dataset_dir.iterdir() if p.is_dir())
    class_names = [p.name for p in class_dirs]
    if len(class_names) < 2:
        raise DatasetError("Need at least two class sub-folders to train a classifier.")

    paths, labels = [], []
    for label, class_dir in enumerate(class_dirs):
        images = sorted(p for p in class_dir.rglob("*") if p.suffix.lower() in ALLOWED_EXTENSIONS)
        if check_readable:
            images = [p for p in images if read_image_file(p) is not None]
        if len(images) < MIN_IMAGES_PER_CLASS:
            raise DatasetError(
                f"Class '{class_dir.name}' has {len(images)} readable images; need at least {MIN_IMAGES_PER_CLASS}."
            )
        paths.extend(images)
        labels.extend([label] * len(images))
    return paths, labels, class_names


def split_dataset(paths, labels, seed: int = 42, validation_fraction: float = 0.15, test_fraction: float = 0.15) -> dict:
    """Stratified split. Every image lands in exactly one of train / validation / test."""
    train_paths, holdout_paths, train_labels, holdout_labels = train_test_split(
        paths, labels, test_size=validation_fraction + test_fraction, stratify=labels, random_state=seed
    )
    test_share = test_fraction / (validation_fraction + test_fraction)
    validation_paths, test_paths, validation_labels, test_labels = train_test_split(
        holdout_paths, holdout_labels, test_size=test_share, stratify=holdout_labels, random_state=seed
    )
    return {
        "train": (train_paths, train_labels),
        "validation": (validation_paths, validation_labels),
        "test": (test_paths, test_labels),
    }
