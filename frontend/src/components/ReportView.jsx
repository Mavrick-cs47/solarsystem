import { downloadCsv } from "../report.js";
import { BulletList, EmptyState, WarningList } from "./ui.jsx";

export default function ReportView({ forecast, inspection, answer }) {
  if (!forecast) return <EmptyState>Run a forecast to build a report.</EmptyState>;
  const { request, summary, model } = forecast;
  const metrics = model.power_test_metrics.all_hours;

  return (
    <div className="report">
      <div className="no-print report-actions">
        <button className="button" onClick={() => downloadCsv(forecast)}>Download CSV</button>
        <button className="button secondary" onClick={() => window.print()}>Print / save as PDF</button>
      </div>

      <h2>SolarSight forecast report — {request.date}</h2>
      <WarningList warnings={forecast.warnings} />

      <h3>Inputs</h3>
      <p>
        {request.capacity_kw} kW capacity · {request.temperature_c} °C mean · {request.cloud_cover_pct}% cloud ·{" "}
        {request.humidity_pct}% humidity · {request.wind_speed_ms} m/s wind
      </p>

      <h3>Forecast</h3>
      <p>
        Energy {summary.total_energy_kwh} kWh ({summary.total_energy_lower_kwh}–{summary.total_energy_upper_kwh} kWh band); peak{" "}
        {summary.peak_power_kw} kW at {summary.peak_hour}:00; mean daytime irradiance {summary.mean_daytime_irradiance_wm2} W/m².
      </p>

      <h3>Model evaluation ({model.data_source} data)</h3>
      <p>
        {model.power_model}: MAE {metrics.mae.toFixed(2)}%, RMSE {metrics.rmse.toFixed(2)}%, R² {metrics.r2.toFixed(3)} (held-out test set).
      </p>

      {inspection && (
        <>
          <h3>Image inspection</h3>
          <p>
            {inspection.filename}: {inspection.predicted_class} ({(inspection.confidence * 100).toFixed(1)}% confidence).
            {inspection.low_confidence && " Low confidence."} {inspection.note}
          </p>
        </>
      )}

      {answer && (
        <>
          <h3>Assistant notes ({answer.generated_by})</h3>
          <BulletList items={[...answer.uncertainty, ...answer.suggested_next_steps]} />
        </>
      )}

      <h3>Hourly values</h3>
      <div className="table-scroll">
        <table>
          <thead>
            <tr><th>Hour</th><th>Power (kW)</th><th>Low–high (kW)</th><th>Irradiance (W/m²)</th><th>Cloud (%)</th></tr>
          </thead>
          <tbody>
            {forecast.points.map((point) => (
              <tr key={point.timestamp}>
                <td>{point.hour}:00</td>
                <td>{point.power_kw}</td>
                <td>{point.power_lower_kw}–{point.power_upper_kw}</td>
                <td>{point.irradiance_wm2}</td>
                <td>{point.cloud_cover_pct}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="muted small">Requires human review before any maintenance action.</p>
    </div>
  );
}
