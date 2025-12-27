// src/PortfolioForm.jsx
import React, { useEffect, useRef, useState } from "react";

// Keep your existing visible choices
const REGION_REQUIRED_FOR = new Set(["Equity", "ETF", "Fund", "Bond"]);
const ASSET_CLASSES = ["Equity", "ETF", "Fund", "Commodity", "Crypto"];
const REGIONS = ["Any", "USA", "India", "UK", "Europe", "Canada", "Japan","Australia","Hong Kong","China", "Global"];

// Internal threshold for typing before we fetch
const MIN_CHARS = 1; // ← was 4 in your file; 1 makes catalog search feel snappy

// Map UI labels → backend catalog values (no change to what the user sees)
const REGION_MAP = { "United States": "USA", "United Kingdom": "UK", "Global": "Any","Europe": "Any" };
const CLASS_MAP  = { "Fund": "MutualFund" };

const normRegion = (r) => REGION_MAP[r] || r || "Any";
const normClass  = (c) => CLASS_MAP[c]  || c;

function useDebounced(value, ms = 250) {
  const [v, setV] = useState(value);
  useEffect(() => { const id = setTimeout(() => setV(value), ms); return () => clearTimeout(id); }, [value, ms]);
  return v;
}

function initialRow() {
  return { asset_class: "Equity", region: "Any", name: "", symbol: "", currency: "", quantity: 0, buy_price: 0 };
}

export default function PortfolioForm({ onSubmit }) {
  const [name, setName] = useState("Global Portfolio");
  const [baseCurrency, setBaseCurrency] = useState("USD");
  const [rows, setRows] = useState([initialRow()]);

  // per-row query and results
  const [query, setQuery] = useState({});     // rowIndex -> typed text
  const [res, setRes] = useState({});         // rowIndex -> results[]
  const [loading, setLoading] = useState({}); // rowIndex -> boolean
  const abortRefs = useRef({});               // rowIndex -> AbortController
  const refs = useRef({});                    // rowIndex -> input container

  const deb = useDebounced(query, 250);

  const setRow = (i, patch) =>
    setRows(prev =>
      prev.map((r, idx) => {
        if (idx !== i) return r;
        const next = { ...r, ...patch };
        // if class changes → reset region for non-regional classes, clear selection
        if (patch.asset_class !== undefined) {
          next.region = REGION_REQUIRED_FOR.has(patch.asset_class) ? (r.region || "Any") : "Any";
          next.name = ""; next.symbol = ""; next.currency = "";
        }
        // if region changes → clear selection
        if (patch.region !== undefined) {
          next.name = ""; next.symbol = ""; next.currency = "";
        }
        return next;
      })
    );

  const addRow = () => setRows(prev => [...prev, initialRow()]);
  const removeRow = (i) => setRows(prev => prev.filter((_, idx) => idx !== i));

  // ----------------------
  // Remote search (catalog DB only)
  // ----------------------
  useEffect(() => {
    const run = async () => {
      for (const [iStr, raw] of Object.entries(deb)) {
        const i = Number(iStr);
        const row = rows[i];
        if (!row) continue;

        const text = (raw || "").trim();

        // Close list if empty or too short
        if (text.length < MIN_CHARS) {
          setRes(prev => ({ ...prev, [i]: [] }));
          continue;
        }

        // cancel in-flight fetch for this row
        try { abortRefs.current[i]?.abort(); } catch {}
        const controller = new AbortController();
        abortRefs.current[i] = controller;

        setLoading(prev => ({ ...prev, [i]: true }));
        try {
          const url = new URL("/api/v1/assets/search", window.location.origin);
          url.searchParams.set("q", text);
          url.searchParams.set("limit", "100");
          url.searchParams.set("asset_class", normClass(row.asset_class));

          // Region only where it matters
          if (REGION_REQUIRED_FOR.has(row.asset_class) && row.region && row.region !== "Any") {
            url.searchParams.set("region", normRegion(row.region));
          }

          const r = await fetch(url.toString(), { signal: controller.signal });
          const data = await r.json().catch(() => ({}));
          const list = r.ok ? (data.results || []) : [];
          setRes(prev => ({ ...prev, [i]: list }));
        } catch (e) {
          if (e?.name !== "AbortError") {
            console.error("asset search failed", e);
            setRes(prev => ({ ...prev, [i]: [] }));
          }
        } finally {
          setLoading(prev => ({ ...prev, [i]: false }));
          abortRefs.current[i] = null;
        }
      }
    };
    run();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [deb, rows.map(r => `${r.asset_class}|${r.region}`).join("|")]);

  // Close popovers when clicking outside
  useEffect(() => {
    function onDocClick(e) {
      const inside = Object.values(refs.current || {}).some(n => n && n.contains && n.contains(e.target));
      if (!inside) setQuery({});
    }
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, []);

  function pick(i, item) {
    setRow(i, { name: item.name || item.symbol, symbol: item.symbol, currency: item.currency || "" });
    setQuery(q => ({ ...q, [i]: "" }));
    setRes(m => ({ ...m, [i]: [] }));
  }

  async function handleSubmit(e) {
    e.preventDefault();

    // validation (keep your existing UX)
    for (let i = 0; i < rows.length; i++) {
      const r = rows[i];
      const hasAnyInput = (r.name?.trim() || "") !== "" || Number(r.quantity) > 0 || Number(r.buy_price) > 0;
      if (!hasAnyInput) continue;

      if (!r.symbol) { alert(`Row ${i + 1}: please select an asset from the dropdown.`); return; }
      if (!(Number(r.quantity) > 0)) { alert(`Row ${i + 1}: quantity must be > 0.`); return; }
      if (!(Number(r.buy_price) >= 0)) { alert(`Row ${i + 1}: buy price must be ≥ 0.`); return; }
    }

    const holdings = rows
      .filter(r => r.symbol && Number(r.quantity) > 0)
      .map(r => ({
        asset_class: normClass(r.asset_class),
        country: (REGION_REQUIRED_FOR.has(r.asset_class) && r.region !== "Any") ? normRegion(r.region) : null,
        name: r.name?.trim() || r.symbol,
        ticker: r.symbol,
        currency: r.currency || "",
        quantity: Number(r.quantity),
        buy_price: Number(r.buy_price),
      }));

    if (holdings.length === 0) {
      alert("Add at least one holding before submitting.");
      return;
    }

    await onSubmit({ name, base_currency: baseCurrency, holdings });
  }

  // ---- UI (unchanged look) ----
  const ctrlH = 36;
  const S = {
    label: { fontSize: 12, color: "#6b7280" },
    input: { height: ctrlH, width: "100%", borderRadius: 8, border: "1px solid #d1d5db", padding: "6px 10px", fontSize: 14, boxSizing: "border-box" },
    button: { height: ctrlH, borderRadius: 8, border: "1px solid #d1d5db", background: "#fafafa", cursor: "pointer", padding: "0 12px" },
    primary: { height: ctrlH, borderRadius: 8, border: "none", background: "#111827", color: "#fff", padding: "0 14px", cursor: "pointer" },
    headerGrid: { display: "grid", gridTemplateColumns: "2fr 1fr", gap: 12 },
    row: { border: "1px solid #e5e7eb", borderRadius: 10, padding: 12, background: "#fff" },
    grid: { display: "grid", gridTemplateColumns: "1fr 1fr 2fr 1fr 1fr auto", gap: 10, alignItems: "end" },
    pop: { position: "absolute", zIndex: 20, top: ctrlH + 6, left: 0, right: 0, maxHeight: 260, overflowY: "auto", background: "#fff", border: "1px solid #eee", borderRadius: 8, boxShadow: "0 12px 24px rgba(0,0,0,0.08)" },
    item: { padding: "8px 10px", cursor: "pointer" },
    itemSub: { fontSize: 12, color: "#6b7280" },
  };

  return (
    <form onSubmit={handleSubmit} style={{ display: "grid", gap: 12 }}>
      <div style={S.headerGrid}>
        <label>
          <div style={S.label}>Portfolio name</div>
          <input style={S.input} value={name} onChange={(e) => setName(e.target.value)} />
        </label>
        <label>
          <div style={S.label}>Base currency</div>
          <select style={S.input} value={baseCurrency} onChange={(e) => setBaseCurrency(e.target.value)}>
            <option>USD</option><option>INR</option><option>GBP</option><option>EUR</option><option>JPY</option>
          </select>
        </label>
      </div>

      {rows.map((r, i) => {
        const items = res[i] || [];
        const showRegion = REGION_REQUIRED_FOR.has(r.asset_class);

        return (
          <div key={i} style={S.row}>
            <div style={{ ...S.grid, gridTemplateColumns: showRegion ? "1fr 1fr 2fr 1fr 1fr auto" : "1fr 2fr 1fr 1fr auto" }}>
              {/* Asset Class */}
              <label>
                <div style={S.label}>Asset Class</div>
                <select style={S.input} value={r.asset_class} onChange={(e) => setRow(i, { asset_class: e.target.value })}>
                  {ASSET_CLASSES.map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
              </label>

              {/* Region (only when needed) */}
              {showRegion && (
                <label>
                  <div style={S.label}>Region/Country</div>
                  <select style={S.input} value={r.region} onChange={(e) => setRow(i, { region: e.target.value })}>
                    {REGIONS.map((c) => <option key={c} value={c}>{c}</option>)}
                  </select>
                </label>
              )}

              {/* Asset dropdown */}
              <label ref={(el) => (refs.current[i] = el)}>
                <div style={S.label}>Asset</div>
                <div style={{ position: "relative" }}>
                  <input
                    style={S.input}
                    placeholder="Type to search (AAPL, RELIANCE.NS, BTC-USD…) "
                    value={r.name}
                    onChange={(e) => {
                      setRow(i, { name: e.target.value });
                      setQuery((q) => ({ ...q, [i]: e.target.value }));
                    }}
                  />
                  {(query[i] ?? "") !== "" && (
                    <div style={S.pop}>
                      {loading[i] && <div style={S.item}>Searching…</div>}
                      {!loading[i] && items.length === 0 && <div style={S.item}>No matches</div>}
                      {!loading[i] && items.map((it) => (
                        <div key={it.symbol} style={S.item} onMouseDown={() => pick(i, it)}>
                          <div><strong>{it.name}</strong></div>
                          <div style={S.itemSub}>
                            {it.symbol} • {it.exchange || "—"}{it.currency ? ` • ${it.currency}` : ""}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
                {r.symbol && (
                  <div style={{ fontSize: 12, color: "#6b7280", marginTop: 4 }}>
                    Symbol: <code>{r.symbol}</code> • Currency: <b>{r.currency || "—"}</b>
                  </div>
                )}
              </label>

              {/* Quantity */}
              <label>
                <div style={S.label}>Quantity</div>
                <input
                  type="number"
                  style={S.input}
                  value={r.quantity}
                  onChange={(e) => setRow(i, { quantity: e.target.value })}
                />
              </label>

              {/* Buy price (local) */}
              <label>
                <div style={S.label}>Buy price {r.currency ? `(${r.currency})` : ""}</div>
                <input
                  type="number"
                  style={S.input}
                  value={r.buy_price}
                  onChange={(e) => setRow(i, { buy_price: e.target.value })}
                />
              </label>

              <div style={{ display: "flex", gap: 8, alignItems: "end" }}>
                <button type="button" style={S.button} onClick={() => removeRow(i)}>Remove</button>
              </div>
            </div>
          </div>
        );
      })}

      <div style={{ display: "flex", gap: 8 }}>
        <button type="button" style={S.button} onClick={addRow}>Add holding</button>
        <button type="submit" style={S.primary}>Get analytics</button>
      </div>

      <div style={{ color: "#6b7280", fontSize: 12 }}>
        Enter buy price in the asset’s local currency. We’ll convert everything to your base currency ({baseCurrency}) automatically.
      </div>
    </form>
  );
}
