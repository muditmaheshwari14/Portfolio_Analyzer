import React from "react";
import {
  Chart as ChartJS, ArcElement, Tooltip, Legend, CategoryScale, LinearScale,
  PointElement, LineElement, BarElement
} from "chart.js";
import { Pie, Line, Bar } from "react-chartjs-2";

ChartJS.register(
  ArcElement, Tooltip, Legend, CategoryScale, LinearScale,
  PointElement, LineElement, BarElement
);

export default function MetricsTab({ data, request }) {
  const base = data?.base_currency || "USD";
  const holdings = request?.holdings || [];
  const perAsset = data?.valuation?.per_asset || [];
  const [lookbackYears, setLookbackYears] = React.useState(10);
  // ======= DISTRIBUTIONS (by weight) =======
  // asset class & region are from the 'request' holdings; we join weights from perAsset
  const weightBySymbol = new Map(perAsset.map(a => [a.symbol, a.weight || 0]));
  const withWeights = holdings.map(h => ({
    ...h,
    weight: weightBySymbol.get(h.ticker) ?? 0,
    region: h.country || "Any",
    asset_class: h.asset_class || "Unknown",
  }));

  const clsDist = sumByLabel(withWeights, x => x.asset_class, x => x.weight);
  const regDist = sumByLabel(withWeights, x => x.region, x => x.weight);

  // sector comes from backend enrichment (per_asset.sector) + weights
  const sectorRows = perAsset.map(a => ({ label: a.sector || "Unknown", weight: a.weight || 0 }));
  const sectorDist = aggregate(sectorRows);

  const pieOptions = (title) => ({
    plugins: {
      legend: { position: "bottom" },
      tooltip: {
        callbacks: {
          label: (ctx) => {
            const label = ctx.label || "";
            const value = ctx.parsed || 0;
            const total = ctx.dataset.data.reduce((s, v) => s + v, 0) || 1;
            const pct = (value / total) * 100;
            return `${label}: ${pct.toFixed(1)}%`;
          }
        }
      },
      title: { display: !!title, text: title }
    }
  });

  const pieAssetClass = toPie(clsDist, "Asset Class");
  const pieRegion = toPie(regDist, "Region");
  const pieSector = toPie(sectorDist, "Sector");

  // ======= PORTFOLIO vs BENCHMARK =======
  const [benchmark, setBenchmark] = React.useState("SPY");
  const [bench, setBench] = React.useState(null);
  const start = data?.date_window?.start;
  const end = data?.date_window?.end;
  const baseCcy = data?.base_currency || "USD";
  const userIndex = data?.user_index || [];
  const userDates = data?.user_index_dates || [];

  const INVEST_AMOUNT = 10000; // <-- fixed common amount

  

  // Benchmark summary (from API)
  const bsum = bench?.summary || {};
  const benchFinal = bench?.final_value ?? null;


  // Build portfolio 10k path from user_index (normalized index ~1)
  const portfolio10kPath = React.useMemo(() => {
    if (!userIndex?.length) return [];
    const first = Number(userIndex[0]);
    if (!Number.isFinite(first) || first === 0) return [];
    const scale = INVEST_AMOUNT / first;
    return userIndex.map(v => Number(v) * scale); // starts at exactly 10,000
  }, [userIndex]);


  // Portfolio summary
  const portFinal = portfolio10kPath?.length ? portfolio10kPath[portfolio10kPath.length - 1] : null;
  const portDaily = seriesDailyRets(portfolio10kPath);
  const portVolAnn = annualizedVolFromDaily(portDaily);
  const portMdd = maxDrawdownFromPath(portfolio10kPath);
  const portCagr = data?.metrics?.cagr ?? null;
  const portSharpe = data?.metrics?.user_weighted?.sharpe ?? null;
  const portSortino = data?.metrics?.user_weighted?.sortino ?? null;
  // Portfolio annual returns from the 10k path (based on user_index)
  const portAnnual = React.useMemo(() => yearEndReturns(userDates, portfolio10kPath), [userDates, portfolio10kPath]);

  // Benchmark annual returns from API (already computed server-side)
  const benchAnnual = bench?.annual ? {
    years: bench.annual.years || [],
    rets: (bench.annual.benchmark || [])
  } : { years: [], rets: [] };

  // Union of years (in case series differ)
  const allYears = Array.from(new Set([...(portAnnual.years||[]), ...(benchAnnual.years||[])])).sort((a,b)=>a-b);

  // Map to year->ret for easy alignment
  const portYrMap = new Map((portAnnual.years||[]).map((y,i)=>[y, portAnnual.rets[i]]));
  const benchYrMap = new Map((benchAnnual.years||[]).map((y,i)=>[y, benchAnnual.rets[i]]));

  // Build aligned arrays for bar chart
  const barsPortfolio = allYears.map(y => portYrMap.has(y) ? portYrMap.get(y) : 0);
  const barsBenchmark = allYears.map(y => benchYrMap.has(y) ? benchYrMap.get(y) : 0);

  const annualChart = {
  labels: allYears,
  datasets: [
    { label: "Portfolio", data: barsPortfolio.map(v => v * 100), backgroundColor: "rgba(17,24,39,0.85)" },
    { label: benchmark,   data: barsBenchmark.map(v => v * 100), backgroundColor: "rgba(99,102,241,0.85)" }
  ]
};
  const annualOpts = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { position: "bottom" } },
    scales: {
      y: { title: { display: true, text: "Annual Return (%)" }, ticks: { callback: v => `${v}%` } },
      x: { title: { display: true, text: "Year" } }
    }
  };


  React.useEffect(() => {
  if (!baseCcy) return;
  const params = new URLSearchParams({
    symbol: benchmark,
    base: baseCcy,
    years: String(lookbackYears),
    amount: String(INVEST_AMOUNT)
  });
  fetch(`/api/v1/benchmark?${params.toString()}`)
    .then(r => r.json())
    .then(setBench)
    .catch(() => setBench(null));
}, [benchmark, baseCcy, lookbackYears]);

// --- Align & downsample to monthly points (first trading day of each month) ---
  function toMapByDate(dates, values) {
  const m = new Map();
  for (let i = 0; i < Math.min(dates.length, values.length); i++) {
      m.set(String(dates[i]), Number(values[i]));
    }
  return m;
  }
  const pMap = toMapByDate(userDates || [], portfolio10kPath || []);
  const bMap = toMapByDate(bench?.dates || [], bench?.value_path || []);
  const commonDates = (userDates || []).filter(d => bMap.has(String(d)));
  let monthlyDates = [];
  if (commonDates.length) {
    let lastYm = "";
    for (const d of commonDates) {
      const ym = d.slice(0, 7);
      if (ym !== lastYm) { monthlyDates.push(d); lastYm = ym; }
    }
    const lastCommon = commonDates[commonDates.length - 1];
    if (monthlyDates[monthlyDates.length - 1] !== lastCommon) monthlyDates.push(lastCommon);
  } else {
    // fall back to raw aligned dates (or just portfolio dates)
    monthlyDates = userDates || [];
  }


  
  // 4) Build monthly series
  const monthlyPortfolioValues = monthlyDates.map(d => pMap.get(d));
  const monthlyBenchmarkValues = monthlyDates.map(d => bMap.get(d));

  // Chart.js dataset (money on Y axis)
  const compareChart = {
  labels: monthlyDates,
  datasets: [
    {
      label: "Portfolio (value)",
      data: monthlyPortfolioValues,
      borderColor: "rgba(17,24,39,0.9)",
      borderWidth: 2,
      pointRadius: 0
    },
    {
      label: `${benchmark} (value)`,
      data: monthlyBenchmarkValues,
      borderColor: "rgba(99,102,241,0.9)",
      borderWidth: 2,
      pointRadius: 0
    }
  ]
};

  const moneyTick = (v) => {
  try { return new Intl.NumberFormat(undefined, { style: "currency", currency: baseCcy }).format(v); }
  catch { return String(v); }
};
  const compareOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: { legend: { position: "top" } },
  scales: {
    y: { title: { display: true, text: `Value (${baseCcy})` }, ticks: { callback: moneyTick } },
    x: {
      ticks: {
        maxRotation: 0,
        autoSkip: false,
        callback: (val, idx) => {
          // Show the year only at the first tick of each year; blank otherwise
          const thisLabel = (monthlyDates[idx] || "");
          const prevLabel = monthlyDates[idx - 1] || "";
          const y  = thisLabel.slice(0, 4);
          const py = prevLabel.slice(0, 4);
          return (idx === 0 || y !== py) ? y : "";
        },
      },
    },
  },
};


  // A little summary text for the “same amount in index”
  const indexSummary = bench ? {
  amount: INVEST_AMOUNT,
  finalValue: Number(bench.final_value || 0),
  profitAbs: Number(bench.final_profit_abs || 0),
  profitPct: Number(bench.final_profit_pct || 0)
} : null;

  const portfolioSummary = React.useMemo(() => {
    if (!portfolio10kPath.length) return null;
    const final = portfolio10kPath[portfolio10kPath.length - 1];
    const profitAbs = final - INVEST_AMOUNT;
    const profitPct = INVEST_AMOUNT > 0 ? (final / INVEST_AMOUNT - 1.0) : 0;
    return { amount: INVEST_AMOUNT, finalValue: final, profitAbs, profitPct };
  }, [portfolio10kPath]);

  // ======= P/L TABLE =======
  const totalPL = data?.valuation?.total_pl_abs ?? 0;
  const totalPLPct = data?.valuation?.total_pl_pct ?? 0;
  

  return (
    <div>
      <h3 style={{ marginTop: 0 }}>Metrics</h3>

      {/* pies row 1 */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(260px,1fr))", gap: 16 }}>
        <div style={card}><h4>Asset Class Distribution</h4><Pie data={pieAssetClass} options={pieOptions()} /></div>
        <div style={card}><h4>Region Distribution</h4><Pie data={pieRegion} options={pieOptions()} /></div>
        <div style={card}><h4>Sector Distribution</h4><Pie data={pieSector} options={pieOptions()} /></div>
      </div>

      {/* portfolio vs benchmark */}
      <h4 style={{ marginTop: 18 }}>Portfolio vs Benchmark</h4>
      <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 8 }}>
        <span>Benchmark:</span>
        <select value={benchmark} onChange={(e)=>setBenchmark(e.target.value)} style={{ padding: 6, borderRadius: 8 }}>
          <option value="SPY">S&P 500 (SPY)</option>
          <option value="NIFTYBEES.NS">NIFTY 50 (NIFTYBEES.NS)</option>
          <option value="VUKE.L">FTSE 100 (VUKE.L)</option>
          <option value="1306.T">TOPIX ETF (1306.T)</option>
          <option value="BTC-USD">Bitcoin</option>
        </select>
        <span style={{ color:"#6b7280", fontSize:12 }}>Base: {baseCcy}</span>
      </div>

      <div style={{ height: 360 }}>
        <Line data={compareChart} options={compareOptions} />
      </div>

      {indexSummary && (
        <div style={{ marginTop: 10, fontSize: 14 }}>
          If you invested <b>{fmtMoney(indexSummary.amount, baseCcy)}</b> in <b>{benchmark}</b> on {start},
          it would be worth <b>{fmtMoney(indexSummary.finalValue, baseCcy)}</b> today
          ({fmtSigned(indexSummary.profitAbs, baseCcy)}, {fmtPct(indexSummary.profitPct)}).
          <br />
          {portfolioSummary && (
            <>Whereas <b>your portfolio</b> is worth <b>{fmtMoney(portfolioSummary.finalValue, baseCcy)}</b> today
            ({fmtSigned(portfolioSummary.profitAbs, baseCcy)}, {fmtPct(portfolioSummary.profitPct)}).</>
          )}
        </div>
      )}

      <h4 style={{ marginTop: 28 }}>Performance Summary</h4>
      <table border="1" cellPadding="6" style={{ borderCollapse: "collapse", width: "100%", marginTop: 6 }}>
        <thead>
          <tr>
            <th>Metric</th>
            <th>Portfolio</th>
            <th>{benchmark}</th>
          </tr>
        </thead>
        <tbody>
          <tr><td>Start Balance</td>
            <td>{fmtMoney(INVEST_AMOUNT, baseCcy)}</td>
            <td>{fmtMoney(INVEST_AMOUNT, baseCcy)}</td></tr>

          <tr><td>End Balance</td>
            <td>{portFinal != null ? fmtMoney(portFinal, baseCcy) : "—"}</td>
            <td>{benchFinal != null ? fmtMoney(benchFinal, baseCcy) : "—"}</td></tr>

          <tr><td>Annualized Return (CAGR)</td>
            <td>{fmtPct(portCagr)}</td>
            <td>{fmtPct(bsum.cagr)}</td></tr>

          <tr><td>Standard Deviation (annual)</td>
            <td>{fmtPct(portVolAnn)}</td>
            <td>{fmtPct(bsum.stdev_annual)}</td></tr>

          <tr><td>Maximum Drawdown</td>
            <td>{fmtPct(portMdd)}</td>
            <td>{fmtPct(bsum.max_drawdown)}</td></tr>

          <tr><td>Best Year</td>
            <td>{fmtPct(Math.max(...(portAnnual.rets||[0])))}</td>
            <td>{fmtPct(Math.max(...(benchAnnual.rets||[0])))}</td></tr>

          <tr><td>Worst Year</td>
            <td>{fmtPct(Math.min(...(portAnnual.rets||[0])))}</td>
            <td>{fmtPct(Math.min(...(benchAnnual.rets||[0])))}</td></tr>
        </tbody>
      </table>

      <h4 style={{ marginTop: 28 }}>Annual Returns</h4>
      <div style={{ height: 320 }}>
        <Bar data={annualChart} options={annualOpts} />
      </div>

      {/* P/L table */}
      <h4 style={{ marginTop: 18 }}>Profit / Loss (Per Asset)</h4>
      <table border="1" cellPadding="6" style={{ borderCollapse: "collapse", width: "100%", marginTop: 6 }}>
        <thead>
          <tr>
            <th>Symbol</th>
            <th>Sector</th>
            <th>Weight</th>
            <th>Invested ({base})</th>
            <th>Current ({base})</th>
            <th>P/L ({base})</th>
            <th>P/L %</th>
          </tr>
        </thead>
        <tbody>
          {perAsset.map(a => (
            <tr key={a.symbol}>
              <td>{a.symbol}</td>
              <td>{a.sector || "—"}</td>
              <td>{fmtPct(a.weight)}</td>
              <td>{fmtMoney(a.invested_base, base)}</td>
              <td>{fmtMoney(a.current_value_base, base)}</td>
              <td style={{ color: a.pl_abs >= 0 ? "#166534" : "crimson" }}>{fmtMoney(a.pl_abs, base, true)}</td>
              <td style={{ color: (a.pl_pct ?? 0) >= 0 ? "#166534" : "crimson" }}>{fmtPct(a.pl_pct)}</td>
            </tr>
          ))}
        </tbody>
        <tfoot>
          <tr>
            <th colSpan={3} style={{ textAlign: "right" }}>Total</th>
            <th>{fmtMoney(sum(perAsset.map(a => a.invested_base || 0)), base)}</th>
            <th>{fmtMoney(sum(perAsset.map(a => a.current_value_base || 0)), base)}</th>
            <th style={{ color: totalPL >= 0 ? "#166534" : "crimson" }}>{fmtMoney(totalPL, base, true)}</th>
            <th style={{ color: (totalPLPct ?? 0) >= 0 ? "#166534" : "crimson" }}>{fmtPct(totalPLPct)}</th>
          </tr>
        </tfoot>
      </table>
    </div>
  );
}

const card = { border: "1px solid #eee", borderRadius: 12, padding: 12, background: "#fff" };

function sum(arr) { return arr.reduce((s, v) => s + (Number.isFinite(v) ? v : 0), 0); }
function fmtMoney(x, ccy, sign=false) {
  if (x == null) return "—";
  const v = Number(x);
  const f = new Intl.NumberFormat(undefined, { style: "currency", currency: ccy });
  return sign ? (v >= 0 ? `+${f.format(v)}` : f.format(v)) : f.format(v);
}
function fmtPct(x) { return x == null ? "—" : (Number(x) * 100).toFixed(2) + "%"; }
function fmtSigned(x, ccy) {
  if (x == null) return "—";
  return (x >= 0 ? "+" : "") + fmtMoney(x, ccy);
}

function yearEndReturns(dates, values) {
  // returns { years: [YYYY...], rets: [decimals...] } using LAST trading day per calendar year
  if (!dates?.length || !values?.length) return { years: [], rets: [] };
  const byYear = new Map(); // year -> last value
  for (let i = 0; i < Math.min(dates.length, values.length); i++) {
    const y = String(dates[i]).slice(0, 4);
    byYear.set(y, Number(values[i])); // overwrites until the last day of that year
  }
  const years = Array.from(byYear.keys()).sort();
  const vals = years.map(y => byYear.get(y));
  const rets = [];
  for (let i = 1; i < vals.length; i++) rets.push(vals[i] / vals[i - 1] - 1);
  return { years: years.slice(1).map(x => Number(x)), rets };
}

function seriesDailyRets(values) {
  const out = [];
  for (let i = 1; i < values.length; i++) {
    const prev = Number(values[i - 1]), cur = Number(values[i]);
    if (Number.isFinite(prev) && prev !== 0 && Number.isFinite(cur)) {
      out.push(cur / prev - 1);
    }
  }
  return out;
}
function annualizedVolFromDaily(dailyRets) {
  if (!dailyRets?.length) return null;
  // population stdev like backend (ddof=0)
  const m = dailyRets.reduce((s, v) => s + v, 0) / dailyRets.length;
  const varp = dailyRets.reduce((s, v) => s + (v - m) * (v - m), 0) / dailyRets.length;
  return Math.sqrt(varp) * Math.sqrt(252);
}
function maxDrawdownFromPath(values) {
  if (!values?.length) return null;
  let peak = values[0], mdd = 0;
  for (const v of values) {
    peak = Math.max(peak, v);
    mdd = Math.min(mdd, v / peak - 1);
  }
  return mdd;
}

function sumByLabel(rows, keyFn, weightFn) {
  const map = new Map();
  for (const r of rows) {
    const k = keyFn(r) || "Unknown";
    const w = Number(weightFn(r) || 0);
    map.set(k, (map.get(k) || 0) + w);
  }
  return Array.from(map.entries()).map(([label, weight]) => ({ label, weight }));
}
function aggregate(rows) {
  const map = new Map();
  for (const r of rows) {
    const k = r.label || "Unknown";
    const w = Number(r.weight || 0);
    map.set(k, (map.get(k) || 0) + w);
  }
  return Array.from(map.entries()).map(([label, weight]) => ({ label, weight }));
}
function toPie(items /* {label, weight}[] */) {
  const labels = items.map(i => i.label);
  const data = items.map(i => i.weight);
  return {
    labels,
    datasets: [{
      label: "Share of portfolio",
      data,
      backgroundColor: [
        "rgba(17,24,39,0.9)","rgba(99,102,241,0.9)","rgba(16,185,129,0.9)",
        "rgba(245,158,11,0.9)","rgba(239,68,68,0.9)","rgba(59,130,246,0.9)",
        "rgba(250,204,21,0.9)","rgba(251,113,133,0.9)","rgba(20,184,166,0.9)"
      ]
    }]
  };
}
