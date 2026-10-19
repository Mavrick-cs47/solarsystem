import { EmptyState } from "./ui.jsx";

const fmt = (value, digits = 3) => Number(value).toFixed(digits);

function MetricsRow({ label, metrics, unit }) {
  return (
    <tr>
      <td>{label}</td>
      <td>{fmt(metrics.mae, 2)}{unit}</td>
      <td>{fmt(metrics.rmse, 2)}{unit}</td>
      <td>{fmt(metrics.r2)}</td>
    </tr>
  );
}

export default function ModelMetrics({ metrics }) {
  if (!metrics) return <EmptyState>Model metrics are not loaded yet.</EmptyState>;
  const { power, irradiance, metadata } = metrics.forecast;
  return (
    <>
      <p className="muted">
        Trained {metadata.trained_at.slice(0, 10)} on <strong>{metadata.data_source}</strong> data ({metadata.rows_used.toLocaleString()} rows).
        Chronological split: {power.split_sizes.train.toLocaleString()} train / {power.split_sizes.validation.toLocaleString()} validation / {power.split_sizes.test.toLocaleString()} test.
        Test period: {power.test_period[0].slice(0, 10)} to {power.test_period[1].slice(0, 10)}.
      </p>
      <div className="table-scroll">
        <table>
          <thead>
            <tr><th>Model (power, test set)</th><th>MAE</th><th>RMSE</th><th>R²</th></tr>
          </thead>
          <tbody>
            <MetricsRow label="Linear baseline" metrics={power.baseline_test_metrics} unit="%" />
            <MetricsRow label={`${power.selected_model} (selected), all hours`} metrics={power.test_metrics} unit="%" />
            <MetricsRow label={`${power.selected_model}, daytime only`} metrics={power.test_metrics_daytime_only} unit="%" />
            <MetricsRow label={`Irradiance: ${irradiance.selected_model}, all hours`} metrics={irradiance.test_metrics} unit=" W/m²" />
          </tbody>
        </table>
      </div>
      <p className="muted small">
        Power errors are in % of installed capacity. Night hours (zero output) inflate R², so daytime-only figures are shown too.
        The shaded band on the chart is the empirical 5–95% range of past test residuals; it is not a guarantee.
        {metadata.data_source === "synthetic" && " These metrics come from synthetic data and do not show real-world accuracy."}
      </p>
    </>
  );
}
