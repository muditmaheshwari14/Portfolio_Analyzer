// src/components/Dashboard.jsx
import React, { useState } from "react";
import AnalyticsTab from "./dashboard/AnalyticsTab";
import RiskTab from "./dashboard/RiskTab";
import MetricsTab from "./dashboard/MetricsTab";
import ReportTab from "./dashboard/ReportTab"; 
import { useNavigate } from "react-router-dom";
import { useAuth } from "../AuthContext"; // uses your existing AuthContext (JWT). :contentReference[oaicite:0]{index=0}

export default function Dashboard({ data, request, onRerun }) {
  const navigate = useNavigate();
  const [active, setActive] = useState("analytics");
  const { token } = useAuth();

  const [saving, setSaving] = useState(false);
  const [saveMsg, setSaveMsg] = useState(null);

  if (!data) {
    return (
      <div>
        <button onClick={() => navigate("/app/form")} style={{ marginBottom: 16 }}>
          ← Back
        </button>
        <p>No analytics loaded.</p>
      </div>
    );
  }

  const tabs = [
    { id: "analytics", label: "Analytics" },
    { id: "risk", label: "Risk" },
    { id: "metrics", label: "Metrics" },
    { id: "report", label: "Report" }, 
  ];

  

  async function handleSave() {
    setSaveMsg(null);
    if (!token) {
      navigate("/login");
      return;
    }
    try {
      setSaving(true);
      // Shape matches your backend Portfolio model: { name, base_currency, holdings }. :contentReference[oaicite:1]{index=1}
      const payload = {
        name: request?.name || data?.portfolio || "Portfolio",
        base_currency: request?.base_currency || data?.base_currency || "USD",
        holdings: request?.holdings || [],
      };
      const res = await fetch("/api/v1/portfolios", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(payload),
      });
      const j = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(j.detail || res.statusText);
      setSaveMsg("Saved ✓");
    } catch (e) {
      setSaveMsg(e.message || "Save failed");
    } finally {
      setSaving(false);
    }
  }

  const hasHoldings = Boolean(request?.holdings?.length);

  return (
    <div>
      {/* header actions */}
      <div
        style={{
          display: "flex",
          gap: 8,
          alignItems: "center",
          marginBottom: 12,
          flexWrap: "wrap",
        }}
      >
        <button onClick={() => navigate("/app/form")}>← Back to Form</button>

        <button
          onClick={handleSave}
          disabled={saving || !hasHoldings}
          title={!token ? "Log in to save" : ""}
          style={{
            padding: "8px 12px",
            borderRadius: 8,
            border: "1px solid #e5e7eb",
            background: "#111827",
            color: "#fff",
            opacity: saving ? 0.7 : 1,
            cursor: saving || !hasHoldings ? "not-allowed" : "pointer",
          }}
        >
          {saving ? "Saving…" : "Save portfolio"}
        </button>

        {saveMsg && (
          <span
            style={{
              color: saveMsg.includes("✓") ? "#16a34a" : "#b91c1c",
              fontWeight: 600,
            }}
          >
            {saveMsg}
          </span>
        )}
      </div>

      {/* tabs */}
      <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
        {tabs.map((t) => (
          <button
            key={t.id}
            onClick={() => setActive(t.id)}
            style={{
              padding: "8px 14px",
              borderRadius: 10,
              border: "1px solid #e5e7eb",
              background: active === t.id ? "#111827" : "#fff",
              color: active === t.id ? "#fff" : "#111",
              cursor: "pointer",
            }}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* tab content */}
      {active === "analytics" && (
        <AnalyticsTab data={data} request={request} onRerun={onRerun} />
      )}
      {active === "risk" && <RiskTab data={data} request={request} />}
      {active === "metrics" && <MetricsTab data={data} request={request} />}
      {active === "report" && (<ReportTab data={data} request={request} />)}
    </div>
  );
}
