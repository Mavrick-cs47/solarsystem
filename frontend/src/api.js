const API_BASE = import.meta.env.VITE_API_BASE ?? "";

// Turns FastAPI's two error shapes ({error:{message}} and 422 {detail:[...]}) into one message.
export function describeError(status, body) {
  if (body?.error?.message) return body.error.message;
  if (Array.isArray(body?.detail)) {
    return body.detail
      .map((item) => `${(item.loc ?? []).slice(1).join(".") || "input"}: ${item.msg}`)
      .join("; ");
  }
  return `Request failed (HTTP ${status}).`;
}

async function request(path, options) {
  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, options);
  } catch {
    throw new Error("Cannot reach the SolarSight API. Is the backend running on port 8000?");
  }
  const body = await response.json().catch(() => null);
  if (!response.ok) throw new Error(describeError(response.status, body));
  return body;
}

const postJson = (path, payload) =>
  request(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

export const fetchHealth = () => request("/api/health");
export const fetchModelMetrics = () => request("/api/model/metrics");
export const runForecast = (inputs) => postJson("/api/forecast", inputs);
export const explain = (payload) => postJson("/api/assistant/explain", payload);

export function inspectImage(file) {
  const form = new FormData();
  form.append("file", file);
  return request("/api/inspect", { method: "POST", body: form });
}
