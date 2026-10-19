import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const POWER_COLOR = "#d9730d";
const BAND_COLOR = "#d9730d";
const IRRADIANCE_COLOR = "#2563a8";

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  const point = payload[0].payload;
  return (
    <div className="chart-tooltip">
      <strong>{String(label).padStart(2, "0")}:00</strong>
      <div>Power: {point.power_kw} kW</div>
      <div>
        Band: {point.power_lower_kw}–{point.power_upper_kw} kW
      </div>
      <div>Irradiance: {point.irradiance_wm2} W/m²</div>
      <div>Cloud cover: {point.cloud_cover_pct}%</div>
    </div>
  );
}

export default function ForecastChart({ points }) {
  const data = points.map((p) => ({ ...p, band: [p.power_lower_kw, p.power_upper_kw] }));
  return (
    <div className="chart" role="img" aria-label="Hourly power forecast with uncertainty band and irradiance">
      <ResponsiveContainer width="100%" height={320}>
        <ComposedChart data={data} margin={{ top: 8, right: 8, bottom: 8, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--grid)" />
          <XAxis dataKey="hour" tickFormatter={(h) => `${h}h`} interval={2} stroke="var(--muted)" />
          <YAxis
            yAxisId="power"
            stroke="var(--muted)"
            label={{ value: "Power (kW)", angle: -90, position: "insideLeft", fill: "var(--muted)" }}
          />
          <YAxis
            yAxisId="irradiance"
            orientation="right"
            stroke="var(--muted)"
            label={{ value: "Irradiance (W/m²)", angle: 90, position: "insideRight", fill: "var(--muted)" }}
          />
          <Tooltip content={<ChartTooltip />} />
          <Legend />
          <Area yAxisId="power" dataKey="band" name="Empirical 5–95% band" stroke="none" fill={BAND_COLOR} fillOpacity={0.18} />
          <Line yAxisId="power" dataKey="power_kw" name="Power (kW)" stroke={POWER_COLOR} strokeWidth={2.5} dot={false} />
          <Line yAxisId="irradiance" dataKey="irradiance_wm2" name="Irradiance (W/m²)" stroke={IRRADIANCE_COLOR} strokeWidth={1.5} strokeDasharray="5 4" dot={false} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
