import React from "react";
import { useNavigate } from "react-router-dom";

export default function Landing() {
  const nav = useNavigate();

  return (
    <div>
      {/* Hero */}
      <section style={{ padding: "64px 0" }}>
        <h1 style={{ fontSize: 36, marginBottom: 12 }}>
          Portfolio Analytics. Clear, fast, actionable.
        </h1>
        <p style={{ maxWidth: 720, lineHeight: 1.6, marginBottom: 24 }}>
          Run risk & return metrics, optimize weights, simulate stress scenarios, and
          save multiple portfolios to revisit anytime.
        </p>
        <button onClick={() => nav("/analyze")} style={{ padding: "10px 16px" }}>
          Analyze portfolio
        </button>
      </section>

      {/* Feature bullets */}
      <section style={{ display: "grid", gap: 16, gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))" }}>
        <Feature title="One-click analytics" body="Sharpe, Sortino, drawdown, and efficient frontier." />
        <Feature title="Stress testing" body="Custom shocks, correlations, and Monte Carlo scenarios." />
        <Feature title="Saved portfolios" body="Log in to save & reload holdings instantly." />
        <Feature title="Fast data" body="Live price history under the hood with caching for speed." />
      </section>

      {/* How it works */}
      <section style={{ padding: "48px 0" }}>
        <h2>How it works</h2>
        <ol style={{ lineHeight: 1.9 }}>
          <li>Sign up or log in with your email.</li>
          <li>Fill the <i>Portfolio Form</i> with tickers, quantities, cost.</li>
          <li>See analytics, simulate stress, and save for later.</li>
        </ol>
        <button onClick={() => nav("/analyze")} style={{ padding: "10px 16px" }}>
          Analyze portfolio
        </button>
      </section>
    </div>
  );
}

function Feature({ title, body }) {
  return (
    <div style={{ border: "1px solid #ddd", borderRadius: 12, padding: 16 }}>
      <h3 style={{ marginTop: 0 }}>{title}</h3>
      <p style={{ marginBottom: 0 }}>{body}</p>
    </div>
  );
}
