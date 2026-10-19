// Small shared building blocks so every panel looks and behaves the same.

export function Card({ title, children, actions }) {
  return (
    <section className="card">
      {(title || actions) && (
        <header className="card-header">
          <h2>{title}</h2>
          {actions}
        </header>
      )}
      {children}
    </section>
  );
}

export function ErrorBanner({ message }) {
  return message ? (
    <p className="banner banner-error" role="alert">
      {message}
    </p>
  ) : null;
}

export function WarningList({ warnings = [] }) {
  return warnings.map((warning) => (
    <p key={warning} className="banner banner-warning">
      {warning}
    </p>
  ));
}

export function EmptyState({ children }) {
  return <p className="empty">{children}</p>;
}

export function Stat({ label, value, hint }) {
  return (
    <div className="stat">
      <span className="stat-label">{label}</span>
      <strong className="stat-value">{value}</strong>
      {hint && <span className="stat-hint">{hint}</span>}
    </div>
  );
}

export function BulletList({ items }) {
  if (!items?.length) return <p className="muted">None.</p>;
  return (
    <ul className="bullets">
      {items.map((item) => (
        <li key={item}>{item}</li>
      ))}
    </ul>
  );
}
