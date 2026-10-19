import numpy as np
import pandas as pd
import pytest

from app.data.prepare import DataValidationError, prepare_training_data
from app.data.synthetic import generate_synthetic_dataset


def test_synthetic_generator_is_deterministic():
    first = generate_synthetic_dataset(days=20, seed=7)
    second = generate_synthetic_dataset(days=20, seed=7)
    pd.testing.assert_frame_equal(first, second)
    assert not first.equals(generate_synthetic_dataset(days=20, seed=8))


def test_synthetic_data_has_no_night_generation():
    data = generate_synthetic_dataset(days=10)
    midnight = data[data["timestamp"].dt.hour == 0]
    assert (midnight["power_kw"] == 0).all()
    assert data["power_kw"].max() <= data["capacity_kw"].iloc[0]


def test_missing_columns_are_reported():
    data = generate_synthetic_dataset(days=30).drop(columns=["humidity_pct", "power_kw"])
    with pytest.raises(DataValidationError, match="humidity_pct"):
        prepare_training_data(data)


def test_messy_data_is_cleaned_and_reported():
    messy = generate_synthetic_dataset(days=60, inject_issues=True)
    clean, report = prepare_training_data(messy)

    assert report["duplicate_timestamps_dropped"] == 5
    assert report["out_of_range_values_blanked"] > 0
    assert clean["timestamp"].is_unique and clean["timestamp"].is_monotonic_increasing
    assert clean["cloud_cover_pct"].between(0, 100).all()
    assert (clean["power_kw"] >= 0).all()
    assert not clean.isna().any().any()


def test_short_weather_gap_is_interpolated_but_target_gap_is_dropped():
    data = generate_synthetic_dataset(days=30)
    data.loc[100:101, "temperature_c"] = np.nan
    data.loc[200, "power_kw"] = np.nan
    clean, report = prepare_training_data(data)
    assert report["incomplete_rows_dropped"] == 1
    assert len(clean) == len(data) - 1


def test_too_little_data_is_rejected():
    with pytest.raises(DataValidationError, match="usable rows"):
        prepare_training_data(generate_synthetic_dataset(days=5))


def test_non_numeric_values_become_missing_not_crashes():
    data = generate_synthetic_dataset(days=30).astype({"temperature_c": object})
    data.loc[10, "temperature_c"] = "n/a"
    clean, _ = prepare_training_data(data)
    assert len(clean) <= len(data)
