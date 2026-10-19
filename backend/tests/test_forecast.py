import pytest

from app.data.synthetic import generate_synthetic_dataset
from app.forecast.inference import ForecastService
from app.forecast.training import chronological_split, regression_metrics, train_forecast_bundle
from app.schemas import ForecastRequest


@pytest.fixture(scope="module")
def service():
    raw = generate_synthetic_dataset(days=365)  # full year: heavy-cloud hours must be in training data
    bundle = train_forecast_bundle(raw, "synthetic", latitude=30.9, default_capacity_kw=100)
    return ForecastService(bundle)


def make_request(**overrides):
    values = dict(date="2025-06-15", capacity_kw=100, temperature_c=30, cloud_cover_pct=20, humidity_pct=45, wind_speed_ms=3)
    return ForecastRequest(**{**values, **overrides})


def test_chronological_split_never_leaks_future_into_training():
    frame = generate_synthetic_dataset(days=30)
    train, validation, test = chronological_split(frame)
    assert train["timestamp"].max() < validation["timestamp"].min()
    assert validation["timestamp"].max() < test["timestamp"].min()
    assert len(train) + len(validation) + len(test) == len(frame)


def test_metrics_match_hand_calculation():
    metrics = regression_metrics([1, 2, 3], [1, 2, 5])
    assert metrics["mae"] == pytest.approx(2 / 3)
    assert metrics["rmse"] == pytest.approx((4 / 3) ** 0.5)
    assert metrics["r2"] == pytest.approx(1 - 4 / 2)


def test_bundle_records_baseline_and_chosen_model(service):
    power = service.bundle["power"]
    assert set(power["validation_metrics"]) == {"linear_regression_baseline", "random_forest", "gradient_boosting"}
    assert power["selected_model"] in power["validation_metrics"]
    assert service.metadata["data_source"] == "synthetic"


def test_forecast_shape_and_night_output(service):
    result = service.predict_day(make_request())
    assert len(result.points) == 24
    by_hour = {p.hour: p for p in result.points}
    assert by_hour[0].power_kw == 0 and by_hour[23].power_kw == 0
    assert 10 <= result.summary.peak_hour <= 14
    assert result.summary.peak_power_kw > 0
    assert any("SYNTHETIC" in warning for warning in result.warnings)


def test_interval_brackets_prediction_and_respects_capacity(service):
    result = service.predict_day(make_request(capacity_kw=250))
    for point in result.points:
        assert 0 <= point.power_lower_kw <= point.power_kw <= point.power_upper_kw <= 250
    assert result.summary.total_energy_lower_kwh <= result.summary.total_energy_kwh <= result.summary.total_energy_upper_kwh


def test_cloudier_day_produces_less_energy(service):
    clear = service.predict_day(make_request(cloud_cover_pct=5)).summary.total_energy_kwh
    overcast = service.predict_day(make_request(cloud_cover_pct=95)).summary.total_energy_kwh
    assert overcast < clear * 0.6


def test_energy_scales_with_capacity(service):
    small = service.predict_day(make_request(capacity_kw=100)).summary.total_energy_kwh
    large = service.predict_day(make_request(capacity_kw=200)).summary.total_energy_kwh
    assert large == pytest.approx(2 * small, rel=0.01)


def test_hourly_weather_must_have_24_entries():
    entry = dict(temperature_c=20, cloud_cover_pct=10, humidity_pct=40, wind_speed_ms=2)
    with pytest.raises(ValueError, match="24"):
        make_request(hourly_weather=[entry] * 5)
    assert len(make_request(hourly_weather=[entry] * 24).hourly_weather) == 24
