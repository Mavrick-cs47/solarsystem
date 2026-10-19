import { useState } from "react";

const FIELDS = [
  { name: "capacity_kw", label: "Capacity (kW)", min: 1, max: 1000000, step: 1 },
  { name: "temperature_c", label: "Mean temperature (°C)", min: -40, max: 60, step: 0.5 },
  { name: "cloud_cover_pct", label: "Cloud cover (%)", min: 0, max: 100, step: 1 },
  { name: "humidity_pct", label: "Humidity (%)", min: 0, max: 100, step: 1 },
  { name: "wind_speed_ms", label: "Wind speed (m/s)", min: 0, max: 60, step: 0.5 },
];

export function validateInputs(values) {
  const errors = {};
  if (!values.date) errors.date = "Choose a date.";
  for (const field of FIELDS) {
    const raw = values[field.name];
    const number = Number(raw);
    if (raw === "" || Number.isNaN(number)) errors[field.name] = "Enter a number.";
    else if (number < field.min || number > field.max) errors[field.name] = `Use ${field.min}–${field.max}.`;
  }
  return errors;
}

export default function ForecastForm({ initialValues, onSubmit, loading }) {
  const [values, setValues] = useState(initialValues);
  const [errors, setErrors] = useState({});

  const update = (name) => (event) => setValues({ ...values, [name]: event.target.value });

  const submit = (event) => {
    event.preventDefault();
    const found = validateInputs(values);
    setErrors(found);
    if (Object.keys(found).length) return;
    onSubmit({
      date: values.date,
      ...Object.fromEntries(FIELDS.map((f) => [f.name, Number(values[f.name])])),
    });
  };

  return (
    <form className="form-grid" onSubmit={submit} noValidate>
      <label>
        Date
        <input type="date" value={values.date} onChange={update("date")} />
        {errors.date && <span className="field-error">{errors.date}</span>}
      </label>
      {FIELDS.map((field) => (
        <label key={field.name}>
          {field.label}
          <input
            type="number"
            inputMode="decimal"
            min={field.min}
            max={field.max}
            step={field.step}
            value={values[field.name]}
            onChange={update(field.name)}
          />
          {errors[field.name] && <span className="field-error">{errors[field.name]}</span>}
        </label>
      ))}
      <button type="submit" className="button" disabled={loading}>
        {loading ? "Running…" : "Run forecast"}
      </button>
    </form>
  );
}
