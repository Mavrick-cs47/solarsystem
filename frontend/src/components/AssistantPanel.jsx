import { useState } from "react";
import { BulletList, EmptyState, ErrorBanner } from "./ui.jsx";

export default function AssistantPanel({ answer, loading, error, hasEvidence, onAsk }) {
  const [question, setQuestion] = useState("");

  const submit = (event) => {
    event.preventDefault();
    onAsk(question.trim());
  };

  return (
    <div className="assistant">
      <form onSubmit={submit} className="ask-row">
        <input
          type="text"
          maxLength={500}
          placeholder="Optional question, e.g. why might output be lower than predicted?"
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
        />
        <button className="button" disabled={!hasEvidence || loading}>
          {loading ? "Thinking…" : "Explain results"}
        </button>
      </form>
      {!hasEvidence && <EmptyState>Run a forecast or classify an image first; the assistant only explains real results.</EmptyState>}
      <ErrorBanner message={error} />
      {answer && (
        <div className="answer">
          <p className="muted small">
            {answer.generated_by === "gemini"
              ? "Written by Gemini from the evidence shown here."
              : `Template summary (no language model used): ${answer.fallback_reason}`}
          </p>
          <p><strong>{answer.summary}</strong></p>
          <h3>Observed facts (your inputs)</h3>
          <BulletList items={answer.observed_facts} />
          <h3>Model predictions</h3>
          <BulletList items={answer.model_predictions} />
          <h3>Uncertainty</h3>
          <BulletList items={answer.uncertainty} />
          <h3>Suggested next steps</h3>
          <BulletList items={answer.suggested_next_steps} />
          <p className="banner banner-warning">Human review is required before any maintenance action.</p>
          <h3>Retrieved guidance (not model-generated)</h3>
          {answer.retrieved_sources.map((source) => (
            <details key={source.id}>
              <summary>{source.title}</summary>
              <p>{source.excerpt}</p>
            </details>
          ))}
        </div>
      )}
    </div>
  );
}
