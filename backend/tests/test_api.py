import httpx

from tests.conftest import make_image_bytes


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["vision"]["available"] is False
    assert body["gemini_configured"] is False


def test_forecast_valid_request(client, valid_forecast_payload):
    response = client.post("/api/forecast", json=valid_forecast_payload)
    assert response.status_code == 200
    body = response.json()
    assert len(body["points"]) == 24
    assert body["model"]["data_source"] == "synthetic"
    assert body["summary"]["total_energy_kwh"] > 0


def test_forecast_rejects_invalid_inputs(client, valid_forecast_payload):
    for field, bad_value in [("cloud_cover_pct", 150), ("capacity_kw", 0), ("humidity_pct", -1), ("date", "not-a-date")]:
        response = client.post("/api/forecast", json={**valid_forecast_payload, field: bad_value})
        assert response.status_code == 422, field
    missing = {k: v for k, v in valid_forecast_payload.items() if k != "temperature_c"}
    assert client.post("/api/forecast", json=missing).status_code == 422


def test_model_metrics_expose_real_evaluation(client):
    body = client.get("/api/model/metrics").json()
    power = body["forecast"]["power"]
    assert {"mae", "rmse", "r2"} <= set(power["test_metrics"])
    assert power["baseline_test_metrics"]["rmse"] >= power["test_metrics"]["rmse"]


def test_inspect_without_trained_model_reports_clearly(client, png_bytes):
    response = client.post("/api/inspect", files={"file": ("panel.png", png_bytes, "image/png")})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "vision_model_missing"
    assert "app.vision.train" in response.json()["error"]["message"]


def test_inspect_upload_errors(client, monkeypatch, png_bytes):
    assert client.post("/api/inspect").status_code == 422
    wrong_type = client.post("/api/inspect", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert wrong_type.status_code == 415
    spoofed = client.post("/api/inspect", files={"file": ("panel.png", b"hello world", "image/png")})
    assert spoofed.status_code == 415
    monkeypatch.setenv("MAX_UPLOAD_MB", "0.0001")
    too_big = client.post("/api/inspect", files={"file": ("panel.png", make_image_bytes("PNG", (300, 300)), "image/png")})
    assert too_big.status_code == 413


def forecast_result(client, payload):
    return client.post("/api/forecast", json=payload).json()


def test_assistant_fallback_without_api_key(client, valid_forecast_payload):
    forecast = forecast_result(client, valid_forecast_payload)
    response = client.post("/api/assistant/explain", json={"forecast": forecast})
    assert response.status_code == 200
    body = response.json()
    assert body["generated_by"] == "fallback" and "GEMINI_API_KEY" in body["fallback_reason"]
    assert body["requires_human_review"] is True
    assert any("SYNTHETIC" not in fact and "kW capacity" in fact for fact in body["observed_facts"])
    assert any("synthetic" in item.lower() for item in body["uncertainty"])
    assert body["retrieved_sources"]
    assert any("qualified personnel" in step for step in body["suggested_next_steps"])


def test_assistant_uses_inspection_evidence_without_inventing(client):
    inspection = {
        "filename": "p1.png", "predicted_class": "dusty", "confidence": 0.41,
        "probabilities": {"clean": 0.3, "dusty": 0.41, "cracked": 0.29}, "low_confidence": True,
        "model_test_accuracy": 0.88, "note": "n",
    }
    body = client.post("/api/assistant/explain", json={"inspection": inspection}).json()
    assert any("dusty" in item for item in body["model_predictions"])
    assert any("low" in item.lower() for item in body["uncertainty"])
    assert any(source["id"] == "om-soiling" for source in body["retrieved_sources"])


def test_assistant_requires_evidence(client):
    assert client.post("/api/assistant/explain", json={"question": "why?"}).status_code == 422


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code, self._payload = status_code, payload or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=httpx.Request("POST", "http://x"), response=self)

    def json(self):
        return self._payload


def gemini_payload(text):
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


def test_assistant_uses_gemini_when_configured_and_valid(client, monkeypatch, valid_forecast_payload):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    sent = {}

    def fake_post(url, headers, json, timeout):
        sent.update(headers=headers, body=json, url=url)
        answer = '{"summary":"s","observed_facts":["f"],"model_predictions":["p"],"uncertainty":["u"],"suggested_next_steps":["n"]}'
        return FakeResponse(payload=gemini_payload(answer))

    monkeypatch.setattr(httpx, "post", fake_post)
    body = client.post("/api/assistant/explain", json={"forecast": forecast_result(client, valid_forecast_payload)}).json()
    assert body["generated_by"] == "gemini" and body["requires_human_review"] is True
    assert body["retrieved_sources"]
    assert "test-key" not in sent["url"] and sent["headers"]["x-goog-api-key"] == "test-key"
    assert "total_energy_kwh" in sent["body"]["contents"][0]["parts"][0]["text"]  # real evidence supplied


def test_assistant_falls_back_on_rate_limit_timeout_and_bad_output(client, monkeypatch, valid_forecast_payload):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    request_body = {"forecast": forecast_result(client, valid_forecast_payload)}

    scenarios = {
        "rate limit": lambda *a, **k: FakeResponse(429),
        "invalid JSON": lambda *a, **k: FakeResponse(payload=gemini_payload("not json at all")),
        "wrong schema": lambda *a, **k: FakeResponse(payload=gemini_payload('{"summary": "only"}')),
        "empty candidates": lambda *a, **k: FakeResponse(payload={"candidates": []}),
    }
    for name, fake in scenarios.items():
        monkeypatch.setattr(httpx, "post", fake)
        body = client.post("/api/assistant/explain", json=request_body).json()
        assert body["generated_by"] == "fallback", name

    def raise_timeout(*args, **kwargs):
        raise httpx.ReadTimeout("slow")

    monkeypatch.setattr(httpx, "post", raise_timeout)
    body = client.post("/api/assistant/explain", json=request_body).json()
    assert body["generated_by"] == "fallback" and "timed out" in body["fallback_reason"]


def test_retrieval_prioritises_the_predicted_condition(client):
    for predicted, expected_note in [("cracked", "om-physical"), ("bird_drop", "om-bird-drop"), ("snow_covered", "om-snow")]:
        inspection = {
            "filename": "p.png", "predicted_class": predicted, "confidence": 0.9,
            "probabilities": {predicted: 0.9, "other": 0.1}, "low_confidence": False, "note": "n",
        }
        body = client.post("/api/assistant/explain", json={"inspection": inspection}).json()
        assert body["retrieved_sources"][0]["id"] == expected_note
        assert any(s["id"] == "om-safety" for s in body["retrieved_sources"])
