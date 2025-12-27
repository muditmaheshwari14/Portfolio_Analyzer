import React from "react";

export default function DrawdownShockBuilder({ tickers = [], value = [], onChange }) {
  const [ticker, setTicker] = React.useState(tickers[0] || "");
  const [direction, setDirection] = React.useState("Fall");
  const [percent, setPercent] = React.useState(10);
  const [mode, setMode] = React.useState("random"); // day1 | random | first_k | last_k
  const [kDays, setKDays] = React.useState(5);

  React.useEffect(() => {
    if (!tickers.includes(ticker)) setTicker(tickers[0] || "");
  }, [tickers]); // eslint-disable-line

  const needsK = mode === "first_k" || mode === "last_k";
  const valid =
    ticker &&
    Number.isFinite(Number(percent)) &&
    Number(percent) >= 0 && Number(percent) <= 99 &&
    (!needsK || (Number.isFinite(Number(kDays)) && Number(kDays) > 0));

  function addShock(e) {
    e.preventDefault();
    if (!valid) return;
    const sign = direction === "Fall" ? -1 : 1;
    let pct = sign * (Number(percent) / 100);
    if (pct < -0.99) pct = -0.99;        // clamp here
    const entry = { ticker, pct, mode, ...(needsK ? { k_days: Number(kDays) } : {}) };
    onChange([...(value || []), entry]);
  }
  function removeShock(idx) {
    const next = [...value];
    next.splice(idx, 1);
    onChange(next);
  }
  function clearAll() {
    onChange([]);
  }

  return (
    <div style={S.card}>
      <div style={S.header}>
        <h4 style={{ margin: 0 }}>Drawdown Shocks</h4>
        <p style={S.sub}>Add one or more shocks to assets in this portfolio.</p>
      </div>

      {/* SINGLE ROW GRID — Add button sits in an auto-sized column on the right */}
      <div
        style={{
          ...S.grid,
          gridTemplateColumns: needsK
            ? "1fr 1fr 1fr 1fr 1fr auto"  // + k days
            : "1fr 1fr 1fr 1fr auto",     // no k days
        }}
      >
        {/* Asset */}
        <label style={S.field}>
          <span style={S.label}>Asset</span>
          <select value={ticker} onChange={(e) => setTicker(e.target.value)} style={S.input}>
            {tickers.map((t) => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
        </label>

        {/* Direction */}
        <label style={S.field}>
          <span style={S.label}>Direction</span>
          <div style={S.segmented}>
            <button
              type="button"
              onClick={() => setDirection("Fall")}
              style={{ ...S.segment, ...(direction === "Fall" ? S.segmentActive : {}) }}
            >
              Fall
            </button>
            <button
              type="button"
              onClick={() => setDirection("Rise")}
              style={{ ...S.segment, ...(direction === "Rise" ? S.segmentActive : {}) }}
            >
              Rise
            </button>
          </div>
        </label>

        {/* % move */}
        <label style={S.field}>
          <span style={S.label}>% move</span>
          <div style={S.inputWrap}>
            <input
              type="number"
              min="0"
              max="99"
              step="1"
              value={percent}
              onChange={(e) => {
                const v = Number(e.target.value);
                if (Number.isFinite(v)) setPercent(Math.max(0, Math.min(99, v)));
              }}
              style={S.input}
            />
            <span style={{ marginLeft: 6, alignSelf: "center" }}>%</span>
          </div>
        </label>

        {/* When */}
        <label style={S.field}>
          <span style={S.label}>When</span>
          <select value={mode} onChange={(e) => setMode(e.target.value)} style={S.input}>
            <option value="day1">Day-1 (gap)</option>
            <option value="random">Random day</option>
            <option value="first_k">First k days</option>
            <option value="last_k">Last k days</option>
          </select>
        </label>

        {/* k days */}
        {needsK && (
          <label style={S.field}>
            <span style={S.label}>k days</span>
            <input
              type="number"
              min="1"
              step="1"
              value={kDays}
              onChange={(e) => setKDays(e.target.value)}
              style={S.input}
            />
          </label>
        )}

        {/* Add (auto column; small, stays inside the grid) */}
        <div style={{ display: "flex", alignItems: "end", justifyContent: "end" }}>
          <button
            type="button"
            onClick={addShock}
            disabled={!valid}
            style={{ ...S.add, ...(valid ? {} : S.addDisabled) }}
          >
            Add
          </button>
        </div>
      </div>

      {/* List matches the same width as the grid */}
      <div style={S.listShell}>
        {value?.length ? (
          <>
            <div style={S.listHeader}>
              <span style={{ fontSize: 13, color: "#374151" }}>
                {value.length} shock{value.length > 1 ? "s" : ""} added
              </span>
              <button type="button" onClick={clearAll} style={S.clearBtn}>Clear all</button>
            </div>
            <div style={S.listBox}>
              {value.map((s, i) => (
                <div key={i} style={S.item}>
                  <div style={S.badges}>
                    <span style={{ ...S.tag, background: "#eef2ff", color: "#3730a3" }}>{s.ticker}</span>
                    <span style={{ ...S.tag, background: "#ecfeff", color: "#155e75" }}>
                      {(s.pct * 100).toFixed(1)}%
                    </span>
                    <span style={{ ...S.tag, background: "#f0fdf4", color: "#166534" }}>
                      {labelForMode(s)}
                    </span>
                    {"k_days" in s && s.k_days && (
                      <span style={{ ...S.tag, background: "#fff7ed", color: "#9a3412" }}>{s.k_days}d</span>
                    )}
                  </div>
                  <button onClick={() => removeShock(i)} style={S.remove}>×</button>
                </div>
              ))}
            </div>
          </>
        ) : (
          <div style={{ color: "#666", fontSize: 13 }}>No shocks added yet.</div>
        )}
      </div>
    </div>
  );
}

function labelForMode(s) {
  if (s.mode === "day1") return "day-1";
  if (s.mode === "random") return "random";
  if (s.mode === "first_k") return "first k";
  if (s.mode === "last_k") return "last k";
  return s.mode;
}

const CTRL_H = 34;

const S = {
  card: { border: "1px solid #e5e7eb", borderRadius: 12, padding: 16, background: "#fff", width: "100%" },
  header: { marginBottom: 8 },
  sub: { fontSize: 13, color: "#6b7280", margin: "2px 0 0 0" },

  grid: {
    display: "grid",
    gap: 12,
    alignItems: "end",
    marginBottom: 12,
  },

  field: { display: "flex", flexDirection: "column", gap: 4 },
  label: { fontSize: 12, color: "#374151" },

  input: {
    height: CTRL_H,
    width: "100%",
    borderRadius: 8,
    border: "1px solid #d1d5db",
    padding: "6px 10px",
    fontSize: 14,
    boxSizing: "border-box",
    background: "#fff",
  },
  inputWrap: { position: "relative", display: "flex", alignItems: "center" },
  suffix: { position: "absolute", right: 10, fontSize: 12, color: "#6b7280" },

  segmented: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    border: "1px solid #d1d5db",
    borderRadius: 8,
    overflow: "hidden",
    height: CTRL_H,
  },
  segment: { background: "#fff", border: "none", cursor: "pointer", fontSize: 14 },
  segmentActive: { background: "#111827", color: "#fff" },

  add: {
    height: CTRL_H,
    borderRadius: 8,
    border: "none",
    background: "#111827",
    color: "#fff",
    padding: "0 18px",
    cursor: "pointer",
    fontSize: 14,
    whiteSpace: "nowrap",
  },
  addDisabled: { background: "#9ca3af", cursor: "not-allowed" },

  listShell: { width: "100%" },
  listHeader: { display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 },
  clearBtn: { border: "1px solid #e5e7eb", background: "#fafafa", borderRadius: 6, padding: "4px 10px", cursor: "pointer", fontSize: 12 },
  listBox: { border: "1px solid #e5e7eb", borderRadius: 8, padding: 10, display: "flex", flexDirection: "column", gap: 6, background: "#fafafa" },
  item: { display: "flex", justifyContent: "space-between", alignItems: "center", background: "#fff", border: "1px solid #e5e7eb", borderRadius: 8, padding: "6px 10px" },
  badges: { display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" },
  tag: { padding: "3px 8px", borderRadius: 999, fontSize: 12, fontWeight: 600 },
  remove: { background: "transparent", border: "none", fontSize: 16, color: "#6b7280", cursor: "pointer" },
};
