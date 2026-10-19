"""Load the trained bundle once and turn a ForecastRequest into hourly predictions."""
import logging
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.config import get_settings
from app.errors import AppError
from app.forecast.features import MODEL_FEATURES, add_time_features, diurnal_temperature
from app.schemas import ForecastPoint, ForecastRequest, ForecastResponse, ForecastSummary, ModelInfo

logger = logging.getLogger("solarsight")

INTERVAL_NOTE = (
    "Band = empirical 5th–95th percentile of test-set residuals. It describes past error "
    "on the evaluation data, not a guaranteed range for tomorrow."
)


class ForecastService:
    def __init__(self, bundle: dict):
        self.bundle = bundle
        self.metadata = bundle["metadata"]

    def predict_day(self, request: ForecastRequest) -> ForecastResponse:
        weather = self._hourly_weather_frame(request)
        features = add_time_features(weather, self.metadata["latitude"])
        x = features[MODEL_FEATURES]
        is_daytime = (features["solar_elevation_deg"] > 0).to_numpy()

        irradiance = np.clip(self.bundle["irradiance"]["model"].predict(x), 0, None)
        irradiance[~is_daytime] = 0

        capacity_factor = np.clip(self.bundle["power"]["model"].predict(x), 0, 1)
        quantiles = self.bundle["power"]["residual_quantiles"]  # capacity-factor units (0..1)
        lower = np.clip(capacity_factor + quantiles["q05"], 0, 1)
        upper = np.clip(capacity_factor + quantiles["q95"], 0, 1)
        for series in (capacity_factor, lower, upper):
            series[~is_daytime] = 0

        capacity = request.capacity_kw
        power_kw, lower_kw, upper_kw = capacity_factor * capacity, lower * capacity, upper * capacity

        points = [
            ForecastPoint(
                timestamp=str(features["timestamp"].iloc[i]),
                hour=int(features["timestamp"].iloc[i].hour),
                temperature_c=round(float(features["temperature_c"].iloc[i]), 1),
                cloud_cover_pct=round(float(features["cloud_cover_pct"].iloc[i]), 1),
                irradiance_wm2=round(float(irradiance[i]), 1),
                power_kw=round(float(power_kw[i]), 2),
                power_lower_kw=round(float(lower_kw[i]), 2),
                power_upper_kw=round(float(upper_kw[i]), 2),
            )
            for i in range(len(features))
        ]
        peak_index = int(np.argmax(power_kw))
        summary = ForecastSummary(
            total_energy_kwh=round(float(power_kw.sum()), 1),
            total_energy_lower_kwh=round(float(lower_kw.sum()), 1),
            total_energy_upper_kwh=round(float(upper_kw.sum()), 1),
            peak_power_kw=round(float(power_kw[peak_index]), 2),
            peak_hour=int(points[peak_index].hour),
            mean_daytime_irradiance_wm2=round(float(irradiance[is_daytime].mean()) if is_daytime.any() else 0.0, 1),
            daily_insolation_kwh_m2=round(float(irradiance.sum() / 1000), 2),
        )
        return ForecastResponse(
            request=request,
            summary=summary,
            points=points,
            model=self._model_info(),
            warnings=self._warnings(),
        )

    def _hourly_weather_frame(self, request: ForecastRequest) -> pd.DataFrame:
        hours = np.arange(24)
        timestamps = pd.Timestamp(request.date) + pd.to_timedelta(hours, unit="h")
        if request.hourly_weather:
            hourly = request.hourly_weather
            return pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "temperature_c": [h.temperature_c for h in hourly],
                    "cloud_cover_pct": [h.cloud_cover_pct for h in hourly],
                    "humidity_pct": [h.humidity_pct for h in hourly],
                    "wind_speed_ms": [h.wind_speed_ms for h in hourly],
                }
            )
        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "temperature_c": diurnal_temperature(request.temperature_c, hours),
                "cloud_cover_pct": request.cloud_cover_pct,
                "humidity_pct": request.humidity_pct,
                "wind_speed_ms": request.wind_speed_ms,
            }
        )

    def _model_info(self) -> ModelInfo:
        power = self.bundle["power"]
        return ModelInfo(
            irradiance_model=self.bundle["irradiance"]["selected_model"],
            power_model=power["selected_model"],
            data_source=self.metadata["data_source"],
            trained_at=self.metadata["trained_at"],
            power_test_metrics={
                "all_hours": power["test_metrics"],
                "daytime_only": power["test_metrics_daytime_only"],
                "baseline_all_hours": power["baseline_test_metrics"],
                "unit": "% of installed capacity (MAE/RMSE)",
                "test_period": power["test_period"],
            },
            interval_note=INTERVAL_NOTE,
        )

    def _warnings(self) -> list[str]:
        warnings = []
        if self.metadata["data_source"] == "synthetic":
            warnings.append(
                "The model was trained on SYNTHETIC demonstration data. "
                "Outputs are illustrative and not validated against a real plant."
            )
        warnings.append("Inputs are treated as a single-day weather scenario, not a numerical weather forecast.")
        return warnings


def _load_or_train(path: Path) -> ForecastService:
    settings = get_settings()
    if not path.exists():
        if not settings.auto_train_forecast:
            raise AppError(
                503,
                "forecast_model_missing",
                "No trained forecast model found. Run: python -m app.forecast.training",
            )
        logger.info("No forecast bundle at %s; training on synthetic data once.", path)
        from app.forecast.training import train_and_save

        bundle = train_and_save(output_path=path)
    else:
        bundle = joblib.load(path)
    return ForecastService(bundle)


@lru_cache(maxsize=4)
def _cached_service(path_text: str) -> ForecastService:
    return _load_or_train(Path(path_text))


def get_forecast_service() -> ForecastService:
    return _cached_service(str(get_settings().forecast_model_path))
