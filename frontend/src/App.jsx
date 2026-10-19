import { useCallback, useEffect, useState } from "react";
import { explain, fetchHealth, fetchModelMetrics, inspectImage, runForecast } from "./api.js";
import AssistantPanel from "./components/AssistantPanel.jsx";
import ForecastChart from "./components/ForecastChart.jsx";
import ForecastForm from "./components/ForecastForm.jsx";
import InspectionPanel from "./components/InspectionPanel.jsx";
import ModelMetrics from "./components/ModelMetrics.jsx";
import Overview from "./components/Overview.jsx";
import ReportView from "./components/ReportView.jsx";
import { Card, EmptyState, ErrorBanner, WarningList } from "./components/ui.jsx";

const TABS = [
  ["forecast", "Forecast"],
  ["inspection", "Panel inspection"],
  ["assistant", "Assistant"],
  ["report", "Report"],
];

const DEFAULT_INPUTS = {
  date: new Date().toISOString().slice(0, 10),
  capacity_kw: 100,
  temperature_c: 28,
  cloud_cover_pct: 25,
  humidity_pct: 45,
  wind_speed_ms: 3,
};

// Each async action tracks its own loading + error so one failure never blanks the page.
function useAction(action) {
  const [state, setState] = useState({ loading: false, error: null });
  const run = useCallback(
    async (...args) => {
      setState({ loading: true, error: null });
      try {
        const result = await action(...args);
        setState({ loading: false, error: null });
        return result;
      } catch (error) {
        setState({ loading: false, error: error.message });
        return null;
      }
    },
    [action]
  );
  return [run, state];
}

export default function App() {
  const [tab, setTab] = useState("forecast");
  const [forecast, setForecast] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [visionStatus, setVisionStatus] = useState(null);
  const [inspection, setInspection] = useState(null);
  const [answer, setAnswer] = useState(null);

  const [doForecast, forecastState] = useAction(runForecast);
  const [doInspect, inspectState] = useAction(inspectImage);
  const [doExplain, explainState] = useAction(explain);

  const submitForecast = useCallback(
    async (inputs) => {
      const result = await doForecast(inputs);
      if (result) {
        setForecast(result);
        setAnswer(null);
        fetchModelMetrics().then(setMetrics).catch(() => {});
      }
    },
    [doForecast]
  );

  useEffect(() => {
    fetchHealth().then((health) => setVisionStatus(health.vision)).catch(() => {});
    submitForecast({ ...DEFAULT_INPUTS });
  }, [submitForecast]);

  const submitInspection = async (file) => {
    const result = await doInspect(file);
    if (result) {
      setInspection(result);
      setAnswer(null);
    }
  };

  const ask = async (question) => {
    const result = await doExplain({ forecast, inspection, question: question || null });
    if (result) setAnswer(result);
  };

  return (
    <div className="app">
      <header className="topbar no-print">
        <h1>SolarSight AI</h1>
        <nav aria-label="Sections">
          {TABS.map(([id, label]) => (
            <button key={id} className={tab === id ? "tab active" : "tab"} onClick={() => setTab(id)}>
              {label}
            </button>
          ))}
        </nav>
      </header>

      <main>
        {tab === "forecast" && (
          <>
            <Card title="Weather scenario">
              <ForecastForm initialValues={DEFAULT_INPUTS} onSubmit={submitForecast} loading={forecastState.loading} />
              <ErrorBanner message={forecastState.error} />
            </Card>
            {forecast ? (
              <>
                <WarningList warnings={forecast.warnings} />
                <Overview forecast={forecast} />
                <Card title="Hourly forecast">
                  <ForecastChart points={forecast.points} />
                </Card>
                <Card title="Model evaluation">
                  <ModelMetrics metrics={metrics} />
                </Card>
              </>
            ) : (
              !forecastState.error && <EmptyState>{forecastState.loading ? "Running forecast…" : "No forecast yet."}</EmptyState>
            )}
          </>
        )}

        {tab === "inspection" && (
          <Card title="Panel image inspection">
            <InspectionPanel
              status={visionStatus}
              result={inspection}
              loading={inspectState.loading}
              error={inspectState.error}
              onInspect={submitInspection}
            />
          </Card>
        )}

        {tab === "assistant" && (
          <Card title="Investigation assistant">
            <AssistantPanel
              answer={answer}
              loading={explainState.loading}
              error={explainState.error}
              hasEvidence={Boolean(forecast || inspection)}
              onAsk={ask}
            />
          </Card>
        )}

        {tab === "report" && (
          <Card>
            <ReportView forecast={forecast} inspection={inspection} answer={answer} />
          </Card>
        )}
      </main>
    </div>
  );
}
