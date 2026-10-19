import { useEffect, useState } from "react";
import { EmptyState, ErrorBanner } from "./ui.jsx";

const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp"];
const MAX_MB = 5;

export function checkFile(file) {
  if (!ACCEPTED_TYPES.includes(file.type)) return "Choose a JPG, PNG or WEBP image.";
  if (file.size > MAX_MB * 1024 * 1024) return `Image is larger than ${MAX_MB} MB.`;
  return null;
}

export default function InspectionPanel({ status, result, loading, error, onInspect }) {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [localError, setLocalError] = useState(null);

  useEffect(() => {
    if (!file) return undefined;
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const choose = (event) => {
    const chosen = event.target.files?.[0];
    if (!chosen) return;
    const problem = checkFile(chosen);
    setLocalError(problem);
    setFile(problem ? null : chosen);
  };

  const modelMissing = status && !status.available;

  return (
    <div className="inspection">
      {modelMissing && (
        <p className="banner banner-warning">
          Image classifier unavailable: {status.reason}
        </p>
      )}
      <label className="upload">
        <span>Panel photo (JPG, PNG or WEBP, up to {MAX_MB} MB)</span>
        <input type="file" accept=".jpg,.jpeg,.png,.webp" onChange={choose} />
      </label>
      <ErrorBanner message={localError || error} />
      {previewUrl && file && <img className="preview" src={previewUrl} alt="Selected panel" />}
      <button className="button" disabled={!file || loading || modelMissing} onClick={() => onInspect(file)}>
        {loading ? "Classifying…" : "Classify image"}
      </button>

      {!result && !file && <EmptyState>No image selected yet.</EmptyState>}
      {result && (
        <div className="result">
          <h3>
            {result.predicted_class.replace(/_/g, " ")}{" "}
            <span className="muted">({(result.confidence * 100).toFixed(1)}% confidence)</span>
          </h3>
          {result.low_confidence && (
            <p className="banner banner-warning">Low confidence: treat this result as unreliable.</p>
          )}
          <div className="bars">
            {Object.entries(result.probabilities)
              .sort((a, b) => b[1] - a[1])
              .map(([name, probability]) => (
                <div key={name} className="bar-row">
                  <span>{name.replace(/_/g, " ")}</span>
                  <div className="bar"><div style={{ width: `${probability * 100}%` }} /></div>
                  <span>{(probability * 100).toFixed(1)}%</span>
                </div>
              ))}
          </div>
          {result.model_test_accuracy != null && (
            <p className="muted small">Classifier accuracy on its held-out test split: {(result.model_test_accuracy * 100).toFixed(1)}%.</p>
          )}
          <p className="muted small">{result.note}</p>
        </div>
      )}
    </div>
  );
}
