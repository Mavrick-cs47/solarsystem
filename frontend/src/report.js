const CSV_COLUMNS = [
  "timestamp",
  "hour",
  "temperature_c",
  "cloud_cover_pct",
  "irradiance_wm2",
  "power_kw",
  "power_lower_kw",
  "power_upper_kw",
];

// data_source is repeated on every row so a CSV passed around later still says it is synthetic.
export function buildCsv(forecast) {
  const header = [...CSV_COLUMNS, "data_source"].join(",");
  const rows = forecast.points.map((point) =>
    [...CSV_COLUMNS.map((column) => point[column]), forecast.model.data_source].join(",")
  );
  return [header, ...rows].join("\n");
}

export function downloadCsv(forecast) {
  const blob = new Blob([buildCsv(forecast)], { type: "text/csv;charset=utf-8" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `solarsight-forecast-${forecast.request.date}.csv`;
  link.click();
  URL.revokeObjectURL(link.href);
}
