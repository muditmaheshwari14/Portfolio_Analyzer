import React, { useState } from "react";
import { Routes, Route } from "react-router-dom";
import Landing from "./pages/Landing";
import Analyze from "./pages/Analyze";
import Login from "./pages/Login";
import Signup from "./pages/Signup";
import MyPortfolios from "./pages/MyPortfolios";
import Account from "./pages/Account";
import AuthedLayout from "./layouts/AuthedLayout";
import RequireAuth from "./routes/RequireAuth";

import PortfolioForm from "./components/PortfolioForm";
import Dashboard from "./components/Dashboard";

export default function App() {
  const [analytics, setAnalytics] = useState(null);
  const [lastRequest, setLastRequest] = useState(null);

  // Sensible defaults (you can surface these in UI later; AnalyticsTab form will call rerun with overrides)
  const DEFAULTS = {
    lookbackYears: 10,  // history used to estimate returns
    mcYears: 15,        // forward horizon
    nSims: 3000,
    inflation: 0.025,   // 2.5% per year
    feesBps: 20,        // 0.20% per year total fee drag
    blockLen: 21,       // ~1 trading month blocks for vol clustering (0 = off)
  };

  // Primary submit (from PortfolioForm/MyPortfolios)
  async function handleSubmit(payload, opts = {}) {
    setLastRequest(payload);

    const {
      lookbackYears = DEFAULTS.lookbackYears,
      mcYears = DEFAULTS.mcYears,
      nSims = DEFAULTS.nSims,
      inflation = DEFAULTS.inflation,
      feesBps = DEFAULTS.feesBps,
      blockLen = DEFAULTS.blockLen,
    } = opts;

    const url = `/api/v1/portfolio/analytics`
      + `?years=${encodeURIComponent(lookbackYears)}`
      + `&mc_years=${encodeURIComponent(mcYears)}`
      + `&n_sims=${encodeURIComponent(nSims)}`
      + `&inflation=${encodeURIComponent(inflation)}`
      + `&fees_bps=${encodeURIComponent(feesBps)}`
      + `&block_len=${encodeURIComponent(blockLen)}`;

    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Analytics failed");
    setAnalytics(data);

    // Navigate to dashboard
    window.history.pushState({}, "", "/app/dashboard");
    window.dispatchEvent(new PopStateEvent("popstate"));
  }

  // Re-run analytics with same last payload but new knobs (used by AnalyticsTab form)
  async function rerunAnalytics(opts = {}) {
    if (!lastRequest) return;
    return handleSubmit(lastRequest, opts);
  }

  return (
    <div style={{ padding: 24, fontFamily: "system-ui, sans-serif" }}>
      <Routes>
        {/* Public */}
        <Route path="/" element={<Landing />} />
        <Route path="/analyze" element={<Analyze />} />
        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<Signup />} />

        {/* Authed workspace with tabs */}
        <Route
          path="/app"
          element={
            <RequireAuth>
              <AuthedLayout />
            </RequireAuth>
          }
        >
          <Route path="account" element={<Account />} />
          <Route path="portfolios" element={<MyPortfolios onLoadPortfolio={handleSubmit} />} />
          <Route path="form" element={<PortfolioForm onSubmit={handleSubmit} />} />
          {/* Pass analytics data + lastRequest + rerun function to Dashboard */}
          <Route path="dashboard" element={<Dashboard data={analytics} request={lastRequest} onRerun={rerunAnalytics} />} />
        </Route>

        {/* Fallback */}
        <Route path="*" element={<Landing />} />
      </Routes>
    </div>
  );
}
