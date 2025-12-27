// src/components/dashboard/RiskTab.jsx
import React from "react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
} from "chart.js";
import { Line, Bar } from "react-chartjs-2";
import DrawdownShockBuilder from "../DrawdownShockBuilder";

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, BarElement, Title, Tooltip, Legend);

export default function RiskTab({ data, request }) {
  // ---------- Basics ----------
  const tickers = React.useMemo(() => {
    const raw = request?.holdings?.map(h => (h.ticker || "").toUpperCase()).filter(Boolean) || [];
    return Array.from(new Set(raw));
  }, [request]);
  const canStress = Boolean(request?.holdings?.length);

  // ---------- Stress Testing ----------
  const [shocks, setShocks] = React.useState([]);
  const [stLoading, setSTLoading] = React.useState(false);
  const [stError, setSTError] = React.useState(null);
  const [stResult, setSTResult] = React.useState(null);

  async function runStressTest(e) {
    e?.preventDefault?.();
    setSTLoading(true); setSTError(null); setSTResult(null);
    try {
      const scenario = {
        name: "User Scenario",
        shocks,
        day1_shock_pct: {},
        mu_scale: 1.0,
        vol_scale: 1.0,
        corr_toward: null,
        corr_alpha: 0,
        horizon_days: 126,
        n_sims: 200,
      };
      const payload = {
                        name: request?.name || data?.portfolio || "Portfolio",
                        base_currency: request?.base_currency || data?.base_currency || "USD",   // ← add this
                        holdings: request?.holdings || [],
                        scenario,
                      };
      const res = await fetch("/api/v1/portfolio/stress", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const j = await res.json();
      if (!res.ok) throw new Error(j.detail || res.statusText);
      setSTResult(j);
    } catch (err) {
      setSTError(err.message);
    } finally {
      setSTLoading(false);
    }
  }

  const moneyTick = React.useCallback(
    (v) => {
      const ccy = data?.base_currency || "USD";
      try {
        return new Intl.NumberFormat(undefined, { style: "currency", currency: ccy }).format(v);
      } catch {
        return String(v);
      }
    },
    [data?.base_currency]
  );
  const stressChart = React.useMemo(() => {
  const pathsVal = stResult?.paths_value || [];
  const pathsIdx = stResult?.paths_index || stResult?.paths || [];
  const paths = pathsVal.length ? pathsVal : pathsIdx;

  const maxLen = Math.max(0, ...(paths.map(p => p.length)));
  return {
    labels: Array.from({ length: maxLen }, (_, i) => i),
    datasets: (paths.slice(0, 25) || []).map((path, i) => ({
      label: i === 0 ? (pathsVal.length ? "Stress Paths (Value)" : "Stress Paths (Index)") : undefined,
      data: path,
      borderColor: "rgba(99,102,241,0.35)",
      borderWidth: 1,
      pointRadius: 0,
    })),
  };
}, [stResult]);

  // ---------- Historical Risk Metrics (auto-load) ----------
  const [rkLoading, setRKLoading] = React.useState(false);
  const [rkError, setRKError] = React.useState(null);
  const [rk, setRK] = React.useState(null);

  React.useEffect(() => {
    let ignore = false;
    async function go() {
      if (!request?.holdings?.length) return;
      setRKLoading(true); setRKError(null); setRK(null);
      try {
        const res = await fetch("/api/v1/portfolio/risk", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(request) // expects { name, base_currency, holdings }
        });
        const j = await res.json();
        if (!res.ok) throw new Error(j.detail || res.statusText);
        if (!ignore) setRK(j);
      } catch (e) {
        if (!ignore) setRKError(e.message);
      } finally {
        if (!ignore) setRKLoading(false);
      }
    }
    go();
    return () => { ignore = true; };
  }, [request]);

  // Histogram data from daily returns
  const hist = React.useMemo(() => {
    if (!rk?.returns?.length) return null;
    return makeHistogram(rk.returns, 0.005, 0.1); // 0.5% bins, ±10% range
  }, [rk]);

  // Charts for risk metrics
  const ddChart = React.useMemo(() => {
    if (!rk?.dates?.length) return null;
    return {
      labels: rk.dates,
      datasets: [{
        label: "Drawdown",
        data: rk.drawdown.map(x => x == null ? null : x * 100),
        borderColor: "rgba(220,53,69,1)",
        backgroundColor: "rgba(220,53,69,0.1)",
        fill: true,
        pointRadius: 0,
        borderWidth: 2
      }]
    };
  }, [rk]);

  const volChart = React.useMemo(() => {
    if (!rk?.dates?.length) return null;
    return {
      labels: rk.dates,
      datasets: [{
        label: "Rolling Volatility (annual)",
        data: rk.rolling_vol.map(x => x == null ? null : x * 100),
        borderColor: "rgba(70,130,180,1)",
        pointRadius: 0,
        borderWidth: 2
      }]
    };
  }, [rk]);

  const sharpeChart = React.useMemo(() => {
    if (!rk?.dates?.length) return null;
    return {
      labels: rk.dates,
      datasets: [{
        label: "Rolling Sharpe",
        data: rk.rolling_sharpe,
        borderColor: "rgba(16,185,129,1)",
        pointRadius: 0,
        borderWidth: 2
      }]
    };
  }, [rk]);

  const histChart = React.useMemo(() => {
    if (!hist) return null;
    return {
      labels: hist.labels,
      datasets: [{
        label: "Daily Return Frequency",
        data: hist.counts,
        backgroundColor: "rgba(99,102,241,0.85)"
      }]
    };
  }, [hist]);

  return (
    <div>
      <h3 style={{ marginTop: 0 }}>Risk</h3>

      {/* ================= Historical Risk (auto) ================= */}
      {rkLoading && <p>Computing risk metrics…</p>}
      {rkError && <p style={{ color: "crimson" }}>Error: {rkError}</p>}

      {rk && (
        <>
          <div style={{ display: "flex", gap: 12, marginBottom: 8 }}>
            <Card title="VaR (95%) (daily)" value={fmtPct(rk.var_95)} />
            <Card title="CVaR (95%) (daily)" value={fmtPct(rk.cvar_95)} />
          </div>

          <h4 style={{ marginTop: 10 }}>Historical Drawdown</h4>
          <div style={{ height: 320 }}>
            <Line
              data={ddChart}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false }, title: { display: false } },
                scales: {
                  y: { title: { display: true, text: "Drawdown (%)" }, ticks: { callback: v => `${v}%` } },
                  x: { ticks: { maxRotation: 0, autoSkip: true } }
                }
              }}
            />
          </div>

          <h4 style={{ marginTop: 18 }}>Rolling Volatility (Annualized)</h4>
          <div style={{ height: 300 }}>
            <Line
              data={volChart}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                  y: { title: { display: true, text: "Volatility (%)" }, ticks: { callback: v => `${v}%` } },
                  x: { ticks: { maxRotation: 0, autoSkip: true } }
                }
              }}
            />
          </div>

          <h4 style={{ marginTop: 18 }}>Rolling Sharpe Ratio</h4>
          <div style={{ height: 300 }}>
            <Line
              data={sharpeChart}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: { y: { title: { display: true, text: "Sharpe" } } }
              }}
            />
          </div>

          <h4 style={{ marginTop: 18 }}>Histogram of Daily Returns</h4>
          <div style={{ height: 280 }}>
            <Bar
              data={histChart}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                  x: { title: { display: true, text: "Daily Return (%)" } },
                  y: { title: { display: true, text: "Frequency" } }
                }
              }}
            />
          </div>
        </>
      )}

      {/* ================= Stress Tester ================= */}
      <h4 style={{ marginTop: 22 }}>Stress Tester</h4>
      {!canStress && <p style={{ color: "orangered" }}>Original holdings not available. Re-run analytics to enable.</p>}

      <div style={{ marginTop: 8 }}>
        <DrawdownShockBuilder tickers={tickers} value={shocks} onChange={setShocks} />
      </div>

      <button type="button" disabled={!canStress || stLoading} onClick={runStressTest} style={{ marginTop: 10 }}>
        {stLoading ? "Running…" : "Run Stress Test"}
      </button>
      {stError && <p style={{ color: "crimson" }}>Error: {stError}</p>}

      {stResult && (
        <>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(220px,1fr))", gap: 12, marginTop: 12 }}>
            <Card title="Final Return (Median)" value={fmtPct(stResult.summary.final_return.median)} />
            <Card title="Max DD (Median)" value={fmtPct(stResult.summary.max_drawdown.median)} />
            <Card title="P(LT -20%)" value={fmtPct(stResult.summary.prob_breach["lt_-20"])} />
          </div>

          <div style={{ height: 360, marginTop: 16 }}>
            <Line
              data={stressChart}
              options={{
                responsive: true,
                plugins: {
                  legend: { display: false },
                  title: {
                    display: true,
                    text: stResult?.paths_value ? "Stress Paths (Portfolio Value)" : "Stress Paths (Index)"
                  }
                },
                scales: {
                  x: { title: { display: true, text: "Day" } },
                  y: stResult?.paths_value
                    ? { title: { display: true, text: `Value (${data?.base_currency || "USD"})` }, ticks: { callback: moneyTick } }
                    : { title: { display: true, text: "Index" } }
                },
                maintainAspectRatio: false
              }}
            />
          </div>
        </>
      )}
    </div>
  );
}

/* ---------------- Helpers ---------------- */
function Card({ title, value }) {
  return (
    <div style={{ border: "1px solid #eee", borderRadius: 12, padding: 12, background: "#fff" }}>
      <div style={{ fontSize: 12, color: "#6b7280" }}>{title}</div>
      <div style={{ fontSize: 20, fontWeight: 700 }}>{value}</div>
    </div>
  );
}

const fmtPct = (x) => (x == null ? "—" : (Number(x) * 100).toFixed(2) + "%");

/**
 * Build a simple histogram for daily returns.
 * @param {number[]} returns - daily returns (decimals)
 * @param {number} binWidth - width (decimal). 0.005 = 0.5%
 * @param {number} range - +/- range around zero. 0.1 = 10%
 * @returns {{labels: string[], counts: number[]}}
 */
function makeHistogram(returns, binWidth = 0.005, range = 0.1) {
  const edges = [];
  for (let x = -range; x <= range + 1e-9; x += binWidth) edges.push(x);
  const counts = Array(edges.length - 1).fill(0);

  for (const rRaw of returns) {
    const r = Number(rRaw);
    if (!Number.isFinite(r)) continue;
    if (r < -range || r > range) continue; // clip tails for display
    const idx = Math.min(Math.max(Math.floor((r + range) / binWidth), 0), counts.length - 1);
    counts[idx] += 1;
  }

  const labels = [];
  for (let i = 0; i < edges.length - 1; i++) {
    const a = (edges[i] * 100).toFixed(1);
    const b = (edges[i + 1] * 100).toFixed(1);
    labels.push(`${a}–${b}%`);
  }
  return { labels, counts };
}
