"""Forecast model training. Run as: python -m app.forecast.training [--csv FILE]

Two regressors are trained on the same weather/time features:
  * irradiance_wm2           -> W/m²
  * capacity factor (0..1)   -> power_kw / capacity_kw, reported as % of capacity

For each target a baseline (linear regression) is compared with Random Forest and
Gradient Boosting. Rows are split chronologically (70/15/15) so the model is never
tested on the past. The winner is picked on the validation block only, refit on
train+validation, and scored once on the untouched test block.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.config import get_settings
from app.data.prepare import prepare_training_data
from app.data.synthetic import generate_synthetic_dataset
from app.forecast.features import MODEL_FEATURES, add_time_features

BASELINE_NAME = "linear_regression_baseline"


def build_candidates(seed: int) -> dict:
    return {
        BASELINE_NAME: make_pipeline(StandardScaler(), LinearRegression()),
        "random_forest": RandomForestRegressor(
            n_estimators=120, min_samples_leaf=5, n_jobs=-1, random_state=seed
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.1, random_state=seed
        ),
    }


def chronological_split(frame: pd.DataFrame, train_fraction=0.70, validation_fraction=0.15):
    ordered = frame.sort_values("timestamp").reset_index(drop=True)
    train_end = int(len(ordered) * train_fraction)
    validation_end = int(len(ordered) * (train_fraction + validation_fraction))
    return ordered.iloc[:train_end], ordered.iloc[train_end:validation_end], ordered.iloc[validation_end:]


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_regressor(frame: pd.DataFrame, target: str, seed: int, scale: float = 1.0) -> dict:
    """Select, refit and evaluate a model for one target column.

    `scale` converts metrics into display units (e.g. 100 turns a 0..1 capacity factor into %).
    """
    train, validation, test = chronological_split(frame)
    candidates = build_candidates(seed)

    validation_metrics = {}
    for name, model in candidates.items():
        model.fit(train[MODEL_FEATURES], train[target])
        predictions = model.predict(validation[MODEL_FEATURES])
        validation_metrics[name] = regression_metrics(validation[target] * scale, predictions * scale)

    best_name = min(validation_metrics, key=lambda name: validation_metrics[name]["rmse"])
    final_model = build_candidates(seed)[best_name]
    train_and_validation = pd.concat([train, validation])
    final_model.fit(train_and_validation[MODEL_FEATURES], train_and_validation[target])

    test_predictions = final_model.predict(test[MODEL_FEATURES])
    daytime = (test["solar_elevation_deg"] > 0).to_numpy()
    residuals = test[target].to_numpy() - test_predictions

    # Baseline scored on test as well, so the comparison in the README is honest.
    baseline = build_candidates(seed)[BASELINE_NAME].fit(train_and_validation[MODEL_FEATURES], train_and_validation[target])
    baseline_test = regression_metrics(test[target] * scale, baseline.predict(test[MODEL_FEATURES]) * scale)

    return {
        "model": final_model,
        "selected_model": best_name,
        "validation_metrics": validation_metrics,
        "test_metrics": regression_metrics(test[target] * scale, test_predictions * scale),
        "test_metrics_daytime_only": regression_metrics(
            test[target].to_numpy()[daytime] * scale, test_predictions[daytime] * scale
        ),
        "baseline_test_metrics": baseline_test,
        # 5th/95th percentile of test residuals: an empirical ~90% band, not a guarantee.
        "residual_quantiles": {
            "q05": float(np.quantile(residuals, 0.05)),
            "q95": float(np.quantile(residuals, 0.95)),
        },
        "split_sizes": {"train": len(train), "validation": len(validation), "test": len(test)},
        "test_period": [str(test["timestamp"].iloc[0]), str(test["timestamp"].iloc[-1])],
    }


def train_forecast_bundle(
    raw: pd.DataFrame, data_source: str, latitude: float, default_capacity_kw: float, seed: int = 42
) -> dict:
    clean, cleaning_report = prepare_training_data(raw, default_capacity_kw)
    frame = add_time_features(clean, latitude)
    frame["capacity_factor"] = (frame["power_kw"] / frame["capacity_kw"]).clip(0, 1.2)

    irradiance = train_regressor(frame, "irradiance_wm2", seed)
    power = train_regressor(frame, "capacity_factor", seed, scale=100)

    return {
        "irradiance": irradiance,
        "power": power,
        "metadata": {
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "data_source": data_source,  # "synthetic" or "uploaded"
            "latitude": latitude,
            "features": MODEL_FEATURES,
            "seed": seed,
            "sklearn_version": sklearn.__version__,
            "rows_used": len(frame),
            "data_range": [str(frame["timestamp"].min()), str(frame["timestamp"].max())],
            "cleaning_report": cleaning_report,
            "units": {"irradiance": "W/m²", "power": "% of installed capacity"},
        },
    }


def metrics_summary(bundle: dict) -> dict:
    """JSON-safe view of a bundle (everything except the fitted models)."""
    summary = {"metadata": bundle["metadata"]}
    for key in ("irradiance", "power"):
        summary[key] = {name: value for name, value in bundle[key].items() if name != "model"}
    return summary


def save_bundle(bundle: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    path.with_suffix(".metrics.json").write_text(json.dumps(metrics_summary(bundle), indent=2))


def train_and_save(csv_path: str | None = None, output_path: Path | None = None) -> dict:
    settings = get_settings()
    if csv_path:
        raw, data_source = pd.read_csv(csv_path), "uploaded"
    else:
        raw = generate_synthetic_dataset(
            days=settings.synthetic_days,
            latitude=settings.default_latitude,
            capacity_kw=settings.reference_capacity_kw,
        )
        data_source = "synthetic"
    bundle = train_forecast_bundle(
        raw, data_source, settings.default_latitude, settings.reference_capacity_kw
    )
    save_bundle(bundle, output_path or settings.forecast_model_path)
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the SolarSight forecast models.")
    parser.add_argument("--csv", help="Historical CSV (default: deterministic synthetic data)")
    parser.add_argument("--output", type=Path, help="Where to write the model bundle")
    args = parser.parse_args()

    bundle = train_and_save(args.csv, args.output)
    print(json.dumps(metrics_summary(bundle), indent=2))
    print(f"\nData source: {bundle['metadata']['data_source']}")
    if bundle["metadata"]["data_source"] == "synthetic":
        print("NOTE: metrics are on synthetic data and do not reflect real-world accuracy.")


if __name__ == "__main__":
    main()
