import { describe, expect, it } from "vitest";
import { buildCsv } from "./report.js";
import { describeError } from "./api.js";

const forecast = {
  model: { data_source: "synthetic" },
  points: [
    { timestamp: "2025-06-15 12:00:00", hour: 12, temperature_c: 31, cloud_cover_pct: 20, irradiance_wm2: 900, power_kw: 70, power_lower_kw: 65, power_upper_kw: 75 },
    { timestamp: "2025-06-15 13:00:00", hour: 13, temperature_c: 32, cloud_cover_pct: 20, irradiance_wm2: 880, power_kw: 68, power_lower_kw: 63, power_upper_kw: 73 },
  ],
};

describe("buildCsv", () => {
  it("writes a header, one row per hour, and labels the data source", () => {
    const lines = buildCsv(forecast).split("\n");
    expect(lines).toHaveLength(3);
    expect(lines[0]).toBe("timestamp,hour,temperature_c,cloud_cover_pct,irradiance_wm2,power_kw,power_lower_kw,power_upper_kw,data_source");
    expect(lines[1].endsWith(",70,65,75,synthetic")).toBe(true);
  });
});

describe("describeError", () => {
  it("prefers the API error message", () => {
    expect(describeError(503, { error: { message: "No trained model" } })).toBe("No trained model");
  });
  it("formats validation errors with the field name", () => {
    const body = { detail: [{ loc: ["body", "cloud_cover_pct"], msg: "must be <= 100" }] };
    expect(describeError(422, body)).toBe("cloud_cover_pct: must be <= 100");
  });
  it("falls back to the status code", () => {
    expect(describeError(500, null)).toBe("Request failed (HTTP 500).");
  });
});
