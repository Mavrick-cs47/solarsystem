import { Stat } from "./ui.jsx";

export default function Overview({ forecast }) {
  const { summary, model } = forecast;
  const metrics = model.power_test_metrics.all_hours;
  return (
    <div className="stat-grid">
      <Stat
        label="Predicted energy"
        value={`${summary.total_energy_kwh.toLocaleString()} kWh`}
        hint={`${summary.total_energy_lower_kwh.toLocaleString()}–${summary.total_energy_upper_kwh.toLocaleString()} kWh band`}
      />
      <Stat label="Peak output" value={`${summary.peak_power_kw} kW`} hint={`at ${summary.peak_hour}:00`} />
      <Stat
        label="Irradiance"
        value={`${summary.mean_daytime_irradiance_wm2} W/m²`}
        hint={`daytime mean · ${summary.daily_insolation_kwh_m2} kWh/m² per day`}
      />
      <Stat
        label="Model error (test)"
        value={`MAE ${metrics.mae.toFixed(2)}%`}
        hint={`RMSE ${metrics.rmse.toFixed(2)}% · R² ${metrics.r2.toFixed(3)} · ${model.data_source} data`}
      />
    </div>
  );
}
