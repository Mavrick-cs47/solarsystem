"""Request/response contracts shared by the routes, the assistant and the frontend."""
from datetime import date as Date
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class HourlyWeather(BaseModel):
    temperature_c: float = Field(ge=-40, le=60)
    cloud_cover_pct: float = Field(ge=0, le=100)
    humidity_pct: float = Field(ge=0, le=100)
    wind_speed_ms: float = Field(ge=0, le=60)


class ForecastRequest(BaseModel):
    date: Date
    capacity_kw: float = Field(gt=0, le=1_000_000)
    temperature_c: float = Field(description="Daily mean temperature", ge=-40, le=60)
    cloud_cover_pct: float = Field(ge=0, le=100)
    humidity_pct: float = Field(ge=0, le=100)
    wind_speed_ms: float = Field(ge=0, le=60)
    hourly_weather: Optional[list[HourlyWeather]] = Field(
        default=None, description="Optional 24 hourly overrides; replaces the daily values above."
    )

    @model_validator(mode="after")
    def check_hourly_length(self):
        if self.hourly_weather is not None and len(self.hourly_weather) != 24:
            raise ValueError("hourly_weather must contain exactly 24 entries")
        return self


class ForecastPoint(BaseModel):
    timestamp: str
    hour: int
    temperature_c: float
    cloud_cover_pct: float
    irradiance_wm2: float
    power_kw: float
    power_lower_kw: float
    power_upper_kw: float


class ForecastSummary(BaseModel):
    total_energy_kwh: float
    total_energy_lower_kwh: float
    total_energy_upper_kwh: float
    peak_power_kw: float
    peak_hour: int
    mean_daytime_irradiance_wm2: float
    daily_insolation_kwh_m2: float


class ModelInfo(BaseModel):
    irradiance_model: str
    power_model: str
    data_source: Literal["synthetic", "uploaded"]
    trained_at: str
    power_test_metrics: dict
    interval_note: str


class ForecastResponse(BaseModel):
    request: ForecastRequest
    summary: ForecastSummary
    points: list[ForecastPoint]
    model: ModelInfo
    warnings: list[str]


class InspectionResult(BaseModel):
    filename: str
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]
    low_confidence: bool
    model_test_accuracy: Optional[float] = None
    note: str


class AssistantRequest(BaseModel):
    forecast: Optional[ForecastResponse] = None
    inspection: Optional[InspectionResult] = None
    question: Optional[str] = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def need_some_evidence(self):
        if self.forecast is None and self.inspection is None:
            raise ValueError("Provide a forecast, an inspection result, or both.")
        return self


class RetrievedSource(BaseModel):
    id: str
    title: str
    excerpt: str


class AssistantResponse(BaseModel):
    summary: str
    observed_facts: list[str]
    model_predictions: list[str]
    uncertainty: list[str]
    suggested_next_steps: list[str]
    retrieved_sources: list[RetrievedSource]
    requires_human_review: Literal[True] = True
    generated_by: Literal["gemini", "fallback"]
    fallback_reason: Optional[str] = None
