"""Features shared by the synthetic data generator, training and inference."""
import numpy as np
import pandas as pd

WEATHER_FEATURES = ["temperature_c", "cloud_cover_pct", "humidity_pct", "wind_speed_ms"]
TIME_FEATURES = ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "solar_elevation_deg"]
MODEL_FEATURES = TIME_FEATURES + WEATHER_FEATURES

DIURNAL_TEMPERATURE_SWING_C = 5.0


def solar_elevation_deg(timestamps: pd.Series, latitude: float) -> np.ndarray:
    """Approximate solar elevation using the standard declination / hour-angle formula.

    Timestamps are treated as local solar time, which is good enough for a demo
    forecaster and avoids a timezone dependency.
    """
    day_of_year = timestamps.dt.dayofyear.to_numpy()
    hour = timestamps.dt.hour.to_numpy() + timestamps.dt.minute.to_numpy() / 60
    declination = np.radians(23.45) * np.sin(2 * np.pi * (284 + day_of_year) / 365)
    hour_angle = np.radians(15 * (hour - 12))
    latitude_rad = np.radians(latitude)
    sin_elevation = (
        np.sin(latitude_rad) * np.sin(declination)
        + np.cos(latitude_rad) * np.cos(declination) * np.cos(hour_angle)
    )
    return np.degrees(np.arcsin(np.clip(sin_elevation, -1, 1)))


def diurnal_temperature(daily_mean_c, hours: np.ndarray) -> np.ndarray:
    """Daily mean temperature (scalar or per-hour array) with a swing peaking at 15:00."""
    return daily_mean_c + DIURNAL_TEMPERATURE_SWING_C * np.sin(2 * np.pi * (hours - 9) / 24)


def add_time_features(frame: pd.DataFrame, latitude: float) -> pd.DataFrame:
    """Return a copy of `frame` with cyclical time features and solar elevation."""
    result = frame.copy()
    timestamps = pd.to_datetime(result["timestamp"])
    hour = timestamps.dt.hour + timestamps.dt.minute / 60
    day_of_year = timestamps.dt.dayofyear
    result["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    result["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    result["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365)
    result["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365)
    result["solar_elevation_deg"] = solar_elevation_deg(timestamps, latitude)
    return result
