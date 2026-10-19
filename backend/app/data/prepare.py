"""Validation and cleaning of historical weather + generation data."""
import pandas as pd

REQUIRED_COLUMNS = [
    "timestamp",
    "temperature_c",
    "cloud_cover_pct",
    "humidity_pct",
    "wind_speed_ms",
    "irradiance_wm2",
    "power_kw",
]
WEATHER_COLUMNS = ["temperature_c", "cloud_cover_pct", "humidity_pct", "wind_speed_ms"]

# Values outside these ranges are treated as sensor/logging errors, not as data.
VALID_RANGES = {
    "temperature_c": (-40, 70),
    "cloud_cover_pct": (0, 100),
    "humidity_pct": (0, 100),
    "wind_speed_ms": (0, 60),
    "irradiance_wm2": (0, 1500),
    "power_kw": (0, None),
    "capacity_kw": (0.001, None),
}
MAX_INTERPOLATION_GAP_HOURS = 3


class DataValidationError(ValueError):
    """Raised when the supplied data cannot be used for training."""


def prepare_training_data(
    raw: pd.DataFrame, default_capacity_kw: float = 100.0, min_rows: int = 500
) -> tuple[pd.DataFrame, dict]:
    """Return (clean_frame, report). The report counts every row/value that was changed."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in raw.columns]
    if missing_columns:
        raise DataValidationError(f"Missing required columns: {', '.join(missing_columns)}")

    frame = raw.copy()
    report = {"input_rows": len(frame)}

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    report["unparseable_timestamps_dropped"] = int(frame["timestamp"].isna().sum())
    frame = frame.dropna(subset=["timestamp"])

    if "capacity_kw" not in frame.columns:
        frame["capacity_kw"] = default_capacity_kw

    numeric_columns = [column for column in VALID_RANGES if column in frame.columns]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    report["out_of_range_values_blanked"] = 0
    for column in numeric_columns:
        low, high = VALID_RANGES[column]
        invalid = frame[column].notna() & (
            (frame[column] < low) if high is None else ((frame[column] < low) | (frame[column] > high))
        )
        report["out_of_range_values_blanked"] += int(invalid.sum())
        frame.loc[invalid, column] = float("nan")

    before = len(frame)
    frame = frame.sort_values("timestamp").drop_duplicates(subset="timestamp", keep="first")
    report["duplicate_timestamps_dropped"] = before - len(frame)

    # Only short weather gaps are interpolated; target and capacity are never invented.
    frame = frame.set_index("timestamp")
    frame[WEATHER_COLUMNS] = frame[WEATHER_COLUMNS].interpolate(
        method="time", limit=MAX_INTERPOLATION_GAP_HOURS, limit_area="inside"
    )
    frame = frame.reset_index()

    required_values = WEATHER_COLUMNS + ["irradiance_wm2", "power_kw", "capacity_kw"]
    incomplete = frame[required_values].isna().any(axis=1)
    report["incomplete_rows_dropped"] = int(incomplete.sum())
    frame = frame.loc[~incomplete].reset_index(drop=True)

    report["output_rows"] = len(frame)
    if len(frame) < min_rows:
        raise DataValidationError(
            f"Only {len(frame)} usable rows after cleaning; at least {min_rows} are required."
        )
    return frame, report
