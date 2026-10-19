"""Investigation assistant: Gemini when configured, a deterministic explainer otherwise.

Gemini is only ever given the evidence in the request (forecast, model metrics, image result)
plus retrieved guidance notes. Its answer must parse into the AssistantResponse schema; if it
does not, or the call fails, the deterministic fallback is returned instead.
"""
import json
import logging

import httpx
from pydantic import BaseModel, ValidationError

from app.assistant.knowledge import retrieve_guidance
from app.config import get_settings
from app.schemas import AssistantRequest, AssistantResponse, RetrievedSource

logger = logging.getLogger("solarsight")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

SYSTEM_RULES = (
    "You explain solar-farm model outputs to an operations engineer. Use ONLY the facts in CONTEXT. "
    "Never invent sensor readings, defect classes, metrics or evidence. Keep four things separate: "
    "observed_facts (inputs the user supplied), model_predictions (what the models output), "
    "uncertainty (limits, error, low confidence, synthetic data), suggested_next_steps (actions for a human "
    "to consider). Every maintenance suggestion must say it needs human review. If the evidence is "
    "insufficient, say so. Reply with JSON only, using keys: summary (string), observed_facts, "
    "model_predictions, uncertainty, suggested_next_steps (each a list of short strings)."
)


class _GeminiAnswer(BaseModel):
    summary: str
    observed_facts: list[str]
    model_predictions: list[str]
    uncertainty: list[str]
    suggested_next_steps: list[str]


def explain(request: AssistantRequest) -> AssistantResponse:
    priority = [request.inspection.predicted_class.replace("_", " ")] if request.inspection else []
    sources = retrieve_guidance(_search_terms(request), priority)
    settings = get_settings()
    if not settings.gemini_api_key:
        return build_fallback(request, sources, "GEMINI_API_KEY is not configured.")
    try:
        answer = _ask_gemini(request, sources, settings)
    except _GeminiUnavailable as error:
        return build_fallback(request, sources, str(error))
    return AssistantResponse(
        **answer.model_dump(),
        retrieved_sources=sources,
        generated_by="gemini",
    )


class _GeminiUnavailable(Exception):
    pass


def _ask_gemini(request: AssistantRequest, sources: list[RetrievedSource], settings) -> _GeminiAnswer:
    context = {
        "evidence": _evidence_snapshot(request),
        "retrieved_guidance": [{"id": s.id, "text": s.excerpt} for s in sources],
        "user_question": request.question or "Explain these results and what to check next.",
    }
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM_RULES}]},
        "contents": [{"role": "user", "parts": [{"text": "CONTEXT:\n" + json.dumps(context)}]}],
        "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"},
    }
    try:
        response = httpx.post(
            GEMINI_URL.format(model=settings.gemini_model),
            headers={"x-goog-api-key": settings.gemini_api_key},  # header keeps the key out of URLs/logs
            json=body,
            timeout=settings.gemini_timeout_seconds,
        )
        response.raise_for_status()
    except httpx.TimeoutException:
        raise _GeminiUnavailable("Gemini request timed out.")
    except httpx.HTTPStatusError as error:
        status = error.response.status_code
        reason = "rate limit reached" if status == 429 else f"HTTP {status}"
        raise _GeminiUnavailable(f"Gemini unavailable ({reason}).")
    except httpx.HTTPError:
        raise _GeminiUnavailable("Gemini could not be reached.")

    try:
        text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        return _GeminiAnswer.model_validate_json(text)
    except (KeyError, IndexError, TypeError, ValueError, ValidationError):
        raise _GeminiUnavailable("Gemini returned a response that did not match the expected format.")


def _search_terms(request: AssistantRequest) -> list[str]:
    terms = [request.question or ""]
    if request.inspection:
        terms += [request.inspection.predicted_class.replace("_", " ")]
    if request.forecast:
        terms += ["forecast output predicted", f"cloud {request.forecast.request.cloud_cover_pct}"]
        if request.forecast.model.data_source == "synthetic":
            terms.append("synthetic")
    return terms


def _evidence_snapshot(request: AssistantRequest) -> dict:
    evidence: dict = {}
    if request.forecast:
        forecast = request.forecast
        evidence["forecast"] = {
            "inputs": forecast.request.model_dump(mode="json", exclude={"hourly_weather"}),
            "summary": forecast.summary.model_dump(),
            "model": forecast.model.model_dump(),
            "warnings": forecast.warnings,
        }
    if request.inspection:
        evidence["image_inspection"] = request.inspection.model_dump()
    return evidence


def build_fallback(request: AssistantRequest, sources: list[RetrievedSource], reason: str) -> AssistantResponse:
    """Template-based explanation built only from the supplied evidence."""
    facts, predictions, uncertainty, steps = [], [], [], []

    if request.forecast:
        forecast = request.forecast
        inputs, summary, model = forecast.request, forecast.summary, forecast.model
        facts.append(
            f"Scenario for {inputs.date}: {inputs.capacity_kw:g} kW capacity, mean {inputs.temperature_c:g} °C, "
            f"{inputs.cloud_cover_pct:g}% cloud cover, {inputs.humidity_pct:g}% humidity, {inputs.wind_speed_ms:g} m/s wind."
        )
        predictions.append(
            f"Predicted energy {summary.total_energy_kwh:g} kWh (band {summary.total_energy_lower_kwh:g}–"
            f"{summary.total_energy_upper_kwh:g} kWh), peak {summary.peak_power_kw:g} kW at {summary.peak_hour}:00."
        )
        test_metrics = model.power_test_metrics["all_hours"]
        uncertainty.append(
            f"Held-out test error of the power model: MAE {test_metrics['mae']:.2f}% and RMSE "
            f"{test_metrics['rmse']:.2f}% of capacity (R² {test_metrics['r2']:.3f})."
        )
        if model.data_source == "synthetic":
            uncertainty.append("The model was trained on synthetic data, so it is not validated for a real plant.")
        steps.append("If actual output differs from the band, check meter data and weather inputs before suspecting a fault.")

    if request.inspection:
        inspection = request.inspection
        predictions.append(
            f"Image classifier predicts '{inspection.predicted_class}' with {inspection.confidence:.0%} confidence for "
            f"{inspection.filename}."
        )
        if inspection.low_confidence:
            uncertainty.append("Classifier confidence is low; treat this result as unreliable and re-photograph the panel.")
        if inspection.model_test_accuracy is not None:
            uncertainty.append(f"Classifier test accuracy on its own dataset was {inspection.model_test_accuracy:.1%}.")
        steps.append("Have a technician inspect the panel in person to confirm or reject the classifier's result.")

    if not facts:
        facts.append("No forecast inputs were supplied.")
    steps.append("No maintenance action should be taken without review by qualified personnel.")

    return AssistantResponse(
        summary="Automated summary of the supplied model outputs (no language model was used).",
        observed_facts=facts,
        model_predictions=predictions or ["No model predictions were supplied."],
        uncertainty=uncertainty,
        suggested_next_steps=steps,
        retrieved_sources=sources,
        generated_by="fallback",
        fallback_reason=reason,
    )
