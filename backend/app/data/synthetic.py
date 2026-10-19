"""Deterministic synthetic solar dataset.

This is physically *plausible* demo data (clear-sky irradiance attenuated by cloud,
temperature-derated PV output). It is NOT a real observation record, and metrics
computed on it say nothing about real-world accuracy.
"""
import numpy as np
import pandas as pd

from app.forecast.features import add_time_features, diurnal_temperature


def generate_synthetic_dataset(
    days: int = 730,
    seed: int = 42,
    latitude: float = 30.9,
    capacity_kw: float = 100.0,
    start: str = "2023-01-01",
    inject_issues: bool = False,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    row_count = days * 24
    frame = pd.DataFrame({"timestamp": pd.date_range(start, periods=row_count, freq="h")})
    frame = add_time_features(frame, latitude)
    day_of_year = frame["timestamp"].dt.dayofyear.to_numpy()
    hour = frame["timestamp"].dt.hour.to_numpy()

    # Cloud cover: seasonal mean (cloudier around the monsoon) plus an AR(1) process
    # so that consecutive hours are correlated like real weather.
    seasonal_cloud = 35 + 25 * np.sin(2 * np.pi * (day_of_year - 110) / 365)
    cloud = np.empty(row_count)
    deviation = 0.0
    shocks = rng.normal(0, 9, row_count)
    for i in range(row_count):
        deviation = 0.93 * deviation + shocks[i]
        cloud[i] = seasonal_cloud[i] + deviation
    cloud = np.clip(cloud, 0, 100)

    daily_mean_temperature = 24 + 9 * np.sin(2 * np.pi * (day_of_year - 105) / 365)
    temperature = diurnal_temperature(daily_mean_temperature, hour)
    temperature +=-0.03 * (cloud - 40) + rng.normal(0, 0.8, row_count)

    humidity = np.clip(55 + 0.3 * (cloud - 40) - 0.8 * (temperature - 24) + rng.normal(0, 5, row_count), 15, 100)
    wind = rng.gamma(shape=2.0, scale=1.4, size=row_count)

    elevation_rad = np.radians(frame["solar_elevation_deg"].to_numpy())
    clear_sky = 1000 * np.clip(np.sin(elevation_rad), 0, None) ** 1.15
    cloud_transmission = 1 - 0.75 * (cloud / 100) ** 3.4
    irradiance = np.clip(clear_sky * cloud_transmission * (1 + rng.normal(0, 0.04, row_count)), 0, None)
    irradiance[frame["solar_elevation_deg"].to_numpy() <= 0] = 0

    cell_temperature = temperature + 0.025 * irradiance
    capacity_factor = (irradiance / 1000) * 0.82 * (1 - 0.004 * (cell_temperature - 25))
    capacity_factor *= 1 + rng.normal(0, 0.015, row_count)
    power_kw = np.clip(capacity_factor, 0, 1) * capacity_kw

    dataset = pd.DataFrame(
        {
            "timestamp": frame["timestamp"],
            "temperature_c": temperature.round(2),
            "cloud_cover_pct": cloud.round(1),
            "humidity_pct": humidity.round(1),
            "wind_speed_ms": wind.round(2),
            "irradiance_wm2": irradiance.round(1),
            "power_kw": power_kw.round(3),
            "capacity_kw": capacity_kw,
        }
    )
    return _inject_data_issues(dataset, rng) if inject_issues else dataset


def _inject_data_issues(dataset: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add the kinds of problems real logger exports have, to exercise data cleaning."""
    messy = dataset.copy()
    row_count = len(messy)
    missing_rows = rng.choice(row_count, size=max(1, row_count // 100), replace=False)
    messy.loc[missing_rows, "temperature_c"] = np.nan
    impossible_rows = rng.choice(row_count, size=max(1, row_count // 500), replace=False)
    messy.loc[impossible_rows, "cloud_cover_pct"] = 140
    negative_rows = rng.choice(row_count, size=max(1, row_count // 500), replace=False)
    messy.loc[negative_rows, "power_kw"] = -5
    return pd.concat([messy, messy.iloc[:5]], ignore_index=True)  # duplicate timestamps
