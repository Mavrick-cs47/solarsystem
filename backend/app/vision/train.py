"""Train the panel-condition classifier.

    python -m app.vision.train --dataset data/panel_images --epochs 10

Uses MobileNetV2 transfer learning (ImageNet weights are downloaded by Keras on first use;
pass --no-pretrained to train from scratch offline). TensorFlow is imported inside the
functions so the rest of the backend works without it installed.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from app.config import get_settings
from app.vision.dataset import DatasetError, list_labelled_images, split_dataset
from app.vision.preprocess import IMAGE_SIZE, read_image_file, to_model_input


def make_tf_dataset(paths, labels, batch_size: int, shuffle: bool, seed: int):
    import tensorflow as tf

    shuffle_rng = np.random.default_rng(seed)

    def generate():
        order = np.arange(len(paths))
        if shuffle:
            shuffle_rng.shuffle(order)
        for index in order:
            yield to_model_input(read_image_file(paths[index])), labels[index]

    signature = (
        tf.TensorSpec(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
    dataset = tf.data.Dataset.from_generator(generate, output_signature=signature)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_model(class_count: int, pretrained: bool):
    import tensorflow as tf
    from tensorflow.keras import layers

    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet" if pretrained else None,
    )
    base.trainable = not pretrained  # frozen feature extractor when pretrained

    inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x = layers.RandomFlip("horizontal_and_vertical")(inputs)  # augmentation, active only in training
    x = layers.RandomRotation(0.08)(x)
    x = layers.RandomContrast(0.15)(x)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)  # 0..255 -> -1..1, baked into the model
    x = base(x, training=False if pretrained else None)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(class_count, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def evaluate(model, test_paths, test_labels, class_names, batch_size: int) -> dict:
    dataset = make_tf_dataset(test_paths, test_labels, batch_size, shuffle=False, seed=0)
    predicted = np.argmax(model.predict(dataset, verbose=0), axis=1)
    label_ids = list(range(len(class_names)))
    report = classification_report(
        test_labels, predicted, labels=label_ids, target_names=class_names, output_dict=True, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(test_labels, predicted)),
        "per_class": {name: report[name] for name in class_names},
        "macro_avg": report["macro avg"],
        "confusion_matrix": confusion_matrix(test_labels, predicted, labels=label_ids).tolist(),
        "test_images": len(test_labels),
    }


def train(dataset_dir: Path, epochs: int, batch_size: int, pretrained: bool, seed: int) -> dict:
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    paths, labels, class_names = list_labelled_images(dataset_dir)
    splits = split_dataset(paths, labels, seed=seed)

    train_paths, train_labels = splits["train"]
    class_weights = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
    model = build_model(len(class_names), pretrained)
    model.fit(
        make_tf_dataset(train_paths, train_labels, batch_size, shuffle=True, seed=seed),
        validation_data=make_tf_dataset(*splits["validation"], batch_size, shuffle=False, seed=seed),
        epochs=epochs,
        class_weight=dict(enumerate(class_weights)),
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
    )

    metrics = evaluate(model, *splits["test"], class_names, batch_size)
    return {
        "model": model,
        "labels": {
            "classes": class_names,
            "image_size": IMAGE_SIZE,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dataset_dir": str(dataset_dir),
            "images_per_split": {name: len(items[0]) for name, items in splits.items()},
            "pretrained_imagenet": pretrained,
            "test_metrics": metrics,
        },
    }


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Train the SolarSight panel classifier.")
    parser.add_argument("--dataset", type=Path, required=True, help="Folder with one sub-folder per class")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-pretrained", action="store_true", help="Train from scratch (no ImageNet download)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        result = train(args.dataset, args.epochs, args.batch_size, not args.no_pretrained, args.seed)
    except DatasetError as error:
        raise SystemExit(f"Dataset problem: {error}")
    except Exception as error:
        if "URL fetch failure" in str(error):
            raise SystemExit(
                "Could not download the ImageNet weights (no internet access?). "
                "Connect to the internet and retry, or pass --no-pretrained to train from scratch."
            )
        raise

    settings.vision_model_path.parent.mkdir(parents=True, exist_ok=True)
    result["model"].save(settings.vision_model_path)
    settings.vision_labels_path.write_text(json.dumps(result["labels"], indent=2))
    print(json.dumps(result["labels"]["test_metrics"], indent=2))
    print(f"\nSaved model to {settings.vision_model_path}")


if __name__ == "__main__":
    main()
