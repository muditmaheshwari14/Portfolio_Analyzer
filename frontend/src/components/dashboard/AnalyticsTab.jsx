// src/components/dashboard/AnalyticsTab.jsx
import React, { useMemo, useState } from "react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  ArcElement,
  BarElement,
  Title,
  Tooltip,
  Legend
} from "chart.js";
import { Line, Scatter } from "react-chartjs-2";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  ArcElement,
  BarElement,
  Title,
  Tooltip,
  Legend
);

const th = { textAlign: "left", fontWeight: 600, padding: "8px 10px", whiteSpace: "nowrap" };
const td = { padding: "8px 10px", verticalAlign: "top", whiteSpace: "nowrap" };
const tr = { borderBottom: "1px solid #f1f5f9" };

const fmt = (x) => (x == null ? "—" : Number(x).toFixed(3));
const pct = (x) => (x == null ? "—" : (Number(x) * 100).toFixed(2) + "%");
const fmtMoney = (x, ccy = "USD") =>
  x == null ? "—" : new Intl.NumberFormat(undefined, { style: "currency", currency: ccy }).format(Number(x));
const numPct = (x) => (x == null ? "—" : (x * 100).toFixed(2) + "%");

function SumRow({ label, vals, fmtFn }) {
  return (
    <tr style={tr}>
      <td style={td}>{label}</td>
      {vals.map((v, i) => (
        <td key={i} style={td}>{fmtFn(v)}</td>
      ))}
    </tr>
  );
}
function Row({ label, m }) {
  return (
    <tr style={tr}>
      <td style={td}><b>{label}</b></td>
      <td style={td}>{fmt(m?.sharpe)}</td>
      <td style={td}>{fmt(m?.sortino)}</td>
      <td style={td}>{pct(m?.max_drawdown)}</td>
    </tr>
  );
}

/** Switch (checkbox) with simple pill styling */
function Switch({ checked, onChange, label }) {
  return (
    <label style={{ display: "flex", alignItems: "center", gap: 10, userSelect: "none" }}>
      <span
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        style={{
          width: 42, height: 24, borderRadius: 999,
          background: checked ? "#2563eb" : "#cbd5e1",
          position: "relative", cursor: "pointer", transition: "background .2s",
        }}
      >
        <span
          style={{
            position: "absolute", top: 3, left: checked ? 22 : 3,
            width: 18, height: 18, borderRadius: "50%", background: "white",
            boxShadow: "0 1px 2px rgba(0,0,0,.2)", transition: "left .2s",
          }}
        />
      </span>
      <span style={{ fontSize: 14, color: "#0f172a" }}>{label}</span>
    </label>
  );
}

/** MONTE CARLO controls (now with button in same row and toggle for inflation) */
function MonteCarloControls({ assumptions, defaultLookback = 10, onRun }) {
  const [lookbackYears, setLookbackYears] = useState(defaultLookback);
  const [mcYears, setMcYears] = useState(assumptions?.horizon_years || 15);
  const [nSims, setNSims] = useState(assumptions?.n_sims || 3000);
  const [feesBps, setFeesBps] = useState(assumptions?.fees_bps ?? 20);
  // keep blockLen internally (not editable anymore)
  const [blockLen] = useState(assumptions?.block_len || 21);
  // NEW: editable inflation input (decimal)
  const [inflation, setInflation] = useState(
    assumptions?.inflation ?? 0.025 // default 2.5%
  );

  const canRun = typeof onRun === "function";

  const inputStyle = {
    height: 40,
    padding: "8px 12px",
    border: "1px solid #cbd5e1",
    borderRadius: 10,
    fontSize: 14,
    width: "100%",
    background: "#f8fafc",
    outline: "none",
    boxSizing: "border-box",
  };
  const labelStyle = {
    display: "block",
    fontSize: 13,
    color: "#334155",
    marginBottom: 6,
    fontWeight: 600,
    whiteSpace: "nowrap",
  };
  const Field = ({ label, children }) => (
    <div style={{ minWidth: 0, display: "flex", flexDirection: "column" }}>
      <label style={labelStyle}>{label}</label>
      {children}
    </div>
  );

  return (
    <div
      style={{
        background: "#fff",
        border: "1px solid #e2e8f0",
        borderRadius: 12,
        boxShadow: "0 1px 2px rgba(0,0,0,0.05)",
        padding: 16,
        marginBottom: 12,
      }}
    >
      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (!canRun) return;
          onRun({
            lookbackYears: Number(lookbackYears),
            mcYears: Number(mcYears),
            nSims: Number(nSims),
            inflation: Number(inflation), // <-- send user-entered inflation
            feesBps: Number(feesBps),
            blockLen: Number(blockLen),   // still passed, but no longer editable
          });
        }}
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(5, minmax(0, 1fr)) 190px", // 5 fields + button
          columnGap: 16,
          rowGap: 12,
          alignItems: "end",
        }}
      >
        <Field label="Lookback (years)">
          <input
            type="number"
            min={1}
            max={30}
            value={lookbackYears}
            onChange={(e) => setLookbackYears(e.target.value)}
            style={inputStyle}
          />
        </Field>

        <Field label="MC Horizon (years)">
          <input
            type="number"
            min={1}
            max={50}
            value={mcYears}
            onChange={(e) => setMcYears(e.target.value)}
            style={inputStyle}
          />
        </Field>

        <Field label="Simulations">
          <input
            type="number"
            min={200}
            max={20000}
            step={100}
            value={nSims}
            onChange={(e) => setNSims(e.target.value)}
            style={inputStyle}
          />
        </Field>

        <Field label="Fees (bps/yr)">
          <input
            type="number"
            min={0}
            max={500}
            step={1}
            value={feesBps}
            onChange={(e) => setFeesBps(e.target.value)}
            style={inputStyle}
          />
        </Field>

        {/* REPLACED: Inflation input instead of Block len */}
        <Field label="Inflation (decimal)">
          <input
            type="number"
            min={0}
            max={0.2}
            step="0.001"
            value={inflation}
            onChange={(e) => setInflation(e.target.value)}
            style={inputStyle}
            placeholder="e.g., 0.025 for 2.5%"
          />
        </Field>

        {/* Button column (fixed width) */}
        <div style={{ display: "flex", justifyContent: "flex-end" }}>
          <button
            type="submit"
            disabled={!canRun}
            style={{
              background: canRun ? "#2563eb" : "#94a3b8",
              color: "white",
              padding: "10px 16px",
              border: "none",
              borderRadius: 10,
              fontWeight: 600,
              fontSize: 14,
              cursor: canRun ? "pointer" : "not-allowed",
              transition: "background 0.2s",
              boxShadow: "0 1px 2px rgba(0,0,0,0.08)",
              height: 40,
              width: "100%",
            }}
          >
            ▶ Run Monte Carlo
          </button>
        </div>
      </form>
    </div>
  );
}



export default function AnalyticsTab({ data, request, onRerun }) {
  const {
    portfolio,
    metrics,
    weights,
    mc_summary,
    efficient_frontier,
    date_window,
    tickers
  } = data || {};

  const baseCcy = data?.base_currency || "USD";

  // ---------- MONTE CARLO (toggle Nominal vs Real) ----------
  const mcx = mc_summary; // rich object from backend
  const [showReal, setShowReal] = useState(false);

  // Chart paths (nominal vs real)
  const pctPaths = useMemo(() => {
    if (!mcx) return null;
    if (showReal && mcx.percentile_paths_real) return mcx.percentile_paths_real;
    return mcx.percentile_paths;
  }, [mcx, showReal]);

  // Tables (nominal vs real)
  const perfSummary = useMemo(() => {
    if (!mcx) return null;
    return showReal && mcx.performance_summary_real
      ? mcx.performance_summary_real
      : mcx.performance_summary;
  }, [mcx, showReal]);

  const expAnn = useMemo(() => {
    if (!mcx) return null;
    return showReal && mcx.expected_annual_return_real
      ? mcx.expected_annual_return_real
      : mcx.expected_annual_return;
  }, [mcx, showReal]);

  const annProb = useMemo(() => {
    if (!mcx) return null;
    return showReal && mcx.annual_return_probabilities_real
      ? mcx.annual_return_probabilities_real
      : mcx.annual_return_probabilities;
  }, [mcx, showReal]);

  const yearLabels = pctPaths?.years || [];
  const moneyTickMC = (v) => {
    try { return new Intl.NumberFormat(undefined, { style: "currency", currency: baseCcy }).format(v); }
    catch { return String(v); }
  };

  const mChart = pctPaths ? {
    labels: yearLabels,
    datasets: [
      { label: "10th Percentile", data: pctPaths.p10, borderColor: "rgba(59,130,246,1)", borderWidth: 2, pointRadius: 0 },
      { label: "25th Percentile", data: pctPaths.p25, borderColor: "rgba(99,102,241,1)", borderWidth: 2, pointRadius: 0 },
      { label: "50th Percentile", data: pctPaths.p50, borderColor: "rgba(107,114,128,1)", borderWidth: 2, pointRadius: 0 },
      { label: "75th Percentile", data: pctPaths.p75, borderColor: "rgba(16,185,129,1)", borderWidth: 2, pointRadius: 0 },
      { label: "90th Percentile", data: pctPaths.p90, borderColor: "rgba(37,99,235,1)", borderWidth: 2, pointRadius: 0 },
    ]
  } : null;

  const mOpts = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { position: "bottom" },
      title: { display: true, text: showReal ? "Simulated Portfolio Balances (Real, inflation-adjusted)" : "Simulated Portfolio Balances (Nominal)" }
    },
    scales: {
      x: { title: { display: true, text: "Year" } },
      y: { title: { display: true, text: `Portfolio Balance (${baseCcy})` }, ticks: { callback: moneyTickMC } }
    }
  };

  // ---------- Efficient Frontier ----------
  const vols = efficient_frontier?.volatilities || [];
  const rets = efficient_frontier?.returns || [];
  const sharpes = efficient_frontier?.sharpes || [];
  const points = vols.map((v, i) => ({ x: v, y: rets[i], sharpe: sharpes?.[i] }));
  const maxSharpeIdx = sharpes?.length ? sharpes.indexOf(Math.max(...sharpes)) : -1;
  const minRiskIdx = vols?.length ? vols.indexOf(Math.min(...vols)) : -1;
  const maxSharpePoint = maxSharpeIdx >= 0 ? [{ x: vols[maxSharpeIdx], y: rets[maxSharpeIdx], sharpe: sharpes[maxSharpeIdx] }] : [];
  const minRiskPoint = minRiskIdx >= 0 ? [{ x: vols[minRiskIdx], y: rets[minRiskIdx], sharpe: sharpes[minRiskIdx] }] : [];
  const frontierChart = {
    datasets: [
      { label: "Simulated Portfolios", data: points, backgroundColor: "rgba(54,162,235,0.35)", pointRadius: 2, pointHoverRadius: 4, pointHitRadius: 3 },
      ...(maxSharpePoint.length ? [{ label: "Max Sharpe", data: maxSharpePoint, pointRadius: 7, pointHoverRadius: 9, pointBackgroundColor: "rgba(16,185,129,1)", pointBorderColor: "rgba(16,185,129,1)" }] : []),
      ...(minRiskPoint.length ? [{ label: "Min Risk", data: minRiskPoint, pointRadius: 7, pointHoverRadius: 9, pointBackgroundColor: "rgba(239,68,68,1)", pointBorderColor: "rgba(239,68,68,1)" }] : [])
    ]
  };
  const frontierOpts = {
    responsive: true,
    interaction: { mode: "nearest", intersect: true },
    scales: {
      x: { title: { display: true, text: "Volatility (Risk)" }, ticks: { callback: (v) => (v * 100).toFixed(1) + "%" } },
      y: { title: { display: true, text: "Expected Annual Return" }, ticks: { callback: (v) => (v * 100).toFixed(1) + "%" } }
    },
    plugins: {
      legend: { position: "top" },
      tooltip: {
        callbacks: {
          label: (ctx) => {
            const p = ctx.parsed || {};
            const r = (p.y * 100).toFixed(2) + "%";
            const x = (p.x * 100).toFixed(2) + "%";
            const sh = ctx.raw?.sharpe != null && isFinite(ctx.raw.sharpe) ? ctx.raw.sharpe.toFixed(2) : "—";
            return `Return: ${r}, Risk: ${x}, Sharpe: ${sh}`;
          }
        }
      }
    }
  };

  // ---------- weight labels ----------
  const nameByTicker = new Map((request?.holdings || []).map((h) => [String(h.ticker || "").toUpperCase(), h.name || h.ticker]));
  const orderedAssets = (tickers || []).map((t) => ({ ticker: t, name: nameByTicker.get(String(t || "").toUpperCase()) || t }));

  return (
    <div>
      <h2 style={{ marginTop: 0 }}>{portfolio}</h2>
      <p>Window: <b>{date_window?.start}</b> → <b>{date_window?.end}</b></p>

      {/* === MC Controls === */}
      <h3 style={{ margin: "6px 0 8px" }}>Monte Carlo Settings</h3>
      <MonteCarloControls
        assumptions={mcx ? { ...mcx.assumptions, horizon_years: mcx.horizon_years } : null}
        defaultLookback={10}
        onRun={onRerun}
        showReal={showReal}
        onToggleReal={(v) => setShowReal(!!v)}
      />

      {/* Assumptions summary (inflation removed per your request) */}
      {mcx?.assumptions && (
        <div style={{ color: "#64748b", marginBottom: 12 }}>
          Horizon <b>{mcx.horizon_years}</b> yrs · Sims <b>{Number(mcx.assumptions.n_sims ?? 0).toLocaleString()}</b> ·
          Fees <b>{mcx.assumptions.fees_bps ?? 0}</b> bps/yr ·
          Block <b>{mcx.assumptions.block_len || 0}d</b>
        </div>
      )}

      {/* === Portfolio Metrics === */}
      <h3>Portfolio Metrics</h3>
      <div style={{ overflowX: "auto" }}>
        <table cellPadding="8" style={{ borderCollapse: "collapse", minWidth: 560 }}>
          <thead>
            <tr style={{ borderBottom: "1px solid #eee" }}>
              <th style={th}>Variant</th>
              <th style={th}>Sharpe</th>
              <th style={th}>Sortino</th>
              <th style={th}>Max Drawdown</th>
            </tr>
          </thead>
          <tbody>
            <Row label="User-weighted" m={metrics?.user_weighted} />
            <Row label="Equal-weighted" m={metrics?.equal_weighted} />
            <Row label="Optimized" m={metrics?.optimized} />
          </tbody>
        </table>
      </div>

      {/* === Monte Carlo Chart === */}
      <h3 style={{ marginTop: 18 }}>{showReal ? "Monte Carlo (Real, Inflation-adjusted)" : "Monte Carlo (Nominal)"}</h3>
      <div style={{ height: 380, marginBottom: 20 }}>
        {mChart ? <Line data={mChart} options={mOpts} /> : <p>No Monte Carlo data.</p>}
      </div>

      {/* Toggle placed under the chart */}
      <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 18 }}>
        <Switch
          checked={showReal}
          onChange={(v) => setShowReal(!!v)}
          label="Inflation-adjusted (Real)"
        />
      </div>

      {/* === MC Tables (toggle-aware) === */}
      {mcx && perfSummary && expAnn && annProb && (
        <>
          <h3>Performance Summary</h3>
          <div style={{ overflowX: "auto" }}>
            <table cellPadding="8" style={{ borderCollapse: "collapse", minWidth: 720 }}>
              <thead>
                <tr style={{ borderBottom: "1px solid #eee" }}>
                  <th style={th}>Metric</th>
                  {perfSummary.percentiles.map((p) => <th key={p} style={th}>{p}th Percentile</th>)}
                </tr>
              </thead>
              <tbody>
                <SumRow label={`Time Weighted Rate of Return (${showReal ? "real" : "nominal"})`} vals={perfSummary.twrr} fmtFn={pct} />
                <SumRow label="Annualized Volatility"                 vals={perfSummary.vol_annual} fmtFn={pct} />
                <SumRow label="Sharpe Ratio"                          vals={perfSummary.sharpe} fmtFn={fmt} />
                <SumRow label="Sortino Ratio"                         vals={perfSummary.sortino} fmtFn={fmt} />
                <SumRow label="Maximum Drawdown"                      vals={perfSummary.max_drawdown} fmtFn={pct} />
                <SumRow label={`Portfolio End Balance (${showReal ? "real" : "nominal"})`} vals={perfSummary.end_balance} fmtFn={(x)=>fmtMoney(x, baseCcy)} />
              </tbody>
            </table>
          </div>

          <h3 style={{ marginTop: 18 }}>Expected Annual Return</h3>
          <div style={{ overflowX: "auto" }}>
            <table cellPadding="8" style={{ borderCollapse: "collapse", minWidth: 720 }}>
              <thead>
                <tr style={{ borderBottom: "1px solid #eee" }}>
                  <th style={th}>Percentile</th>
                  {expAnn.horizons.map((h) => <th key={h} style={th}>{h} Years</th>)}
                </tr>
              </thead>
              <tbody>
                {["p10","p25","p50","p75","p90"].map((k) => (
                  <tr key={k} style={tr}>
                    <td style={td}>{k.replace("p","") + "th Percentile"}</td>
                    {expAnn[k].map((v,i)=><td key={i} style={td}>{pct(v)}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <h3 style={{ marginTop: 18 }}>Annual Return Probabilities</h3>
          <div style={{ overflowX: "auto" }}>
            <table cellPadding="8" style={{ borderCollapse: "collapse", minWidth: 720 }}>
              <thead>
                <tr style={{ borderBottom: "1px solid #eee" }}>
                  <th style={th}>Return</th>
                  {annProb.horizons.map((h) => <th key={h} style={th}>{h} Years</th>)}
                </tr>
              </thead>
              <tbody>
                {annProb.thresholds.map((thr, r) => (
                  <tr key={r} style={tr}>
                    <td style={td}>&ge; {Number(thr).toFixed(2)}%</td>
                    {annProb.matrix[r].map((p, c) => (
                      <td key={c} style={td}>{pct(p)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {/* === Efficient Frontier === */}
      <h3>Efficient Frontier</h3>
      <div style={{ height: 420 }}>
        {points.length ? <Scatter data={frontierChart} options={frontierOpts} /> : <p>No Efficient Frontier data.</p>}
      </div>

      {/* === Weights === */}
      <h3 style={{ marginTop: 18 }}>Portfolio Weights</h3>
      <div style={{ overflowX: "auto" }}>
        <table cellPadding="8" style={{ borderCollapse: "collapse", minWidth: 560 }}>
          <thead>
            <tr style={{ borderBottom: "1px solid #eee" }}>
              <th style={th}>Asset</th>
              <th style={th}>Ticker</th>
              <th style={th}>User</th>
              <th style={th}>Equal</th>
              <th style={th}>Optimized</th>
            </tr>
          </thead>
          <tbody>
            {(tickers || []).map((t, i) => (
              <tr key={t} style={tr}>
                <td style={td}>{(new Map((request?.holdings || []).map(h => [String(h.ticker || "").toUpperCase(), h.name || h.ticker]))).get(String(t || "").toUpperCase()) || t}</td>
                <td style={td}><code>{t}</code></td>
                <td style={td}>{numPct(weights?.user?.[i])}</td>
                <td style={td}>{numPct(weights?.equal?.[i])}</td>
                <td style={td}>{numPct(weights?.optimized?.[i])}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
