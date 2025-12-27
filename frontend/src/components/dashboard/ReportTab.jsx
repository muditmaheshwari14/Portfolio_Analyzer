// src/components/dashboard/ReportTab.jsx
import React, { useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

// Lazy-load heavy libs for PDF export only when needed
let jsPDFRef = null;
let html2canvasRef = null;
async function ensurePdfLibs() {
  if (!jsPDFRef) {
    const { jsPDF } = await import("jspdf");
    jsPDFRef = jsPDF;
  }
  if (!html2canvasRef) {
    html2canvasRef = (await import("html2canvas")).default;
  }
}

export default function ReportTab({ data, request }) {
  const [loading, setLoading] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [err, setErr] = useState(null);
  const [markdown, setMarkdown] = useState("");
  const [notes, setNotes] = useState("");

  const reportRef = useRef(null);

  const hasHoldings = Boolean(request?.holdings?.length);
  const portfolioName = request?.name || data?.portfolio || "Portfolio";
  const baseCcy = data?.base_currency || request?.base_currency || "USD";

  // You can wire this to whatever benchmark symbol you allow in MetricsTab
  const defaultBenchmark = "SPY";

  async function fetchRiskSnapshot() {
    if (!hasHoldings) return null;
    const r = await fetch("/api/v1/portfolio/risk", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
    const jr = await r.json();
    if (!r.ok) throw new Error(jr?.detail || "Risk fetch failed");
    return jr;
  }

  async function fetchBenchmarkSnapshot() {
    // Use the same window your analytics ran on (if present)
    const start = data?.date_window?.start;
    const end = data?.date_window?.end;

    const url = new URL("/api/v1/benchmark", window.location.origin);
    url.searchParams.set("symbol", defaultBenchmark);
    if (start) url.searchParams.set("start", start);
    if (end) url.searchParams.set("end", end);
    url.searchParams.set("base", baseCcy);
    url.searchParams.set("amount", "10000");

    const rb = await fetch(url.toString(), { method: "GET" });
    const jb = await rb.json();
    if (!rb.ok) throw new Error(jb?.detail || "Benchmark fetch failed");

    // Compact slice the backend will use
    return {
      symbol: jb.symbol,
      base: jb.base,
      summary: jb.summary, // { cagr, stdev_annual, max_drawdown, best_year, worst_year }
      final_value: jb.final_value,
      final_profit_abs: jb.final_profit_abs,
      final_profit_pct: jb.final_profit_pct,
      years: jb.annual?.years || [],
      returns: jb.annual?.benchmark || [],
    };
  }

  async function handleGenerate() {
    try {
      setErr(null);
      setLoading(true);
      setMarkdown("");

      const [risk, benchmark] = await Promise.allSettled([
        fetchRiskSnapshot(),
        fetchBenchmarkSnapshot(),
      ]);

      const payload = {
        request,
        analytics: data,
        risk: risk.status === "fulfilled" ? risk.value : null,
        benchmark: benchmark.status === "fulfilled" ? benchmark.value : null,
        notes,
      };

      const res = await fetch("/api/v1/report", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const j = await res.json();
      if (!res.ok || !j?.ok) throw new Error(j?.detail || "Report generation failed");

      setMarkdown(j.markdown || "");
    } catch (e) {
      setErr(e.message || "Report generation failed");
    } finally {
      setLoading(false);
    }
  }

  async function handleDownloadPdf() {
    if (!markdown) return;
    try {
      setDownloading(true);
      await ensurePdfLibs();

      const node = reportRef.current;
      // Render to canvas at higher scale for crisp PDF
      const canvas = await html2canvasRef(node, {
        scale: 2,
        useCORS: true,
        backgroundColor: "#ffffff",
      });
      const imgData = canvas.toDataURL("image/png");

      const pdf = new jsPDFRef({ unit: "pt", format: "a4", orientation: "portrait" });
      const pageWidth = pdf.internal.pageSize.getWidth();
      const pageHeight = pdf.internal.pageSize.getHeight();
      const margin = 36; // 0.5"
      const usableWidth = pageWidth - margin * 2;

      const imgWidth = usableWidth;
      const imgHeight = (canvas.height * imgWidth) / canvas.width;
      const sliceHeight = pageHeight - margin * 2;

      // If the rendered content is taller than one page, add multiple pages by shifting the same image upward
      if (imgHeight > sliceHeight) {
        const pages = Math.ceil(imgHeight / sliceHeight);
        const tmp = new jsPDFRef({ unit: "pt", format: "a4", orientation: "portrait" });
        for (let p = 0; p < pages; p++) {
          if (p > 0) tmp.addPage();
          const offsetY = margin - p * sliceHeight;
          tmp.addImage(imgData, "PNG", margin, offsetY, imgWidth, imgHeight, undefined, "FAST");
        }
        const blob = tmp.output("blob");
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `${portfolioName.replace(/\s+/g, "_")}_report.pdf`;
        a.click();
        URL.revokeObjectURL(url);
      } else {
        pdf.addImage(imgData, "PNG", margin, margin, imgWidth, imgHeight, undefined, "FAST");
        const blob = pdf.output("blob");
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `${portfolioName.replace(/\s+/g, "_")}_report.pdf`;
        a.click();
        URL.revokeObjectURL(url);
      }
    } catch (e) {
      alert(e.message || "Failed to create PDF");
    } finally {
      setDownloading(false);
    }
  }

  function downloadMarkdown() {
    if (!markdown) return;
    const blob = new Blob([markdown], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${portfolioName.replace(/\s+/g, "_")}_report.md`;
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div>
      {/* Controls */}
      <div
        style={{
          display: "flex",
          gap: 8,
          alignItems: "center",
          marginBottom: 12,
          flexWrap: "wrap",
        }}
      >
        <button
          onClick={handleGenerate}
          disabled={loading || !hasHoldings}
          style={{
            padding: "8px 12px",
            borderRadius: 8,
            border: "1px solid #e5e7eb",
            background: "#111827",
            color: "#fff",
            cursor: loading || !hasHoldings ? "not-allowed" : "pointer",
            opacity: loading ? 0.75 : 1,
          }}
        >
          {loading ? "Generating…" : "Generate report"}
        </button>

        <button
          onClick={handleDownloadPdf}
          disabled={!markdown || downloading}
          style={{
            padding: "8px 12px",
            borderRadius: 8,
            border: "1px solid #e5e7eb",
            background: "#fff",
            color: "#111",
            cursor: !markdown || downloading ? "not-allowed" : "pointer",
          }}
        >
          {downloading ? "Preparing PDF…" : "Download PDF"}
        </button>

        <button
          onClick={downloadMarkdown}
          disabled={!markdown}
          style={{
            padding: "8px 12px",
            borderRadius: 8,
            border: "1px solid #e5e7eb",
            background: "#fff",
            color: "#111",
            cursor: !markdown ? "not-allowed" : "pointer",
          }}
        >
          Download .md
        </button>

        <div style={{ marginLeft: "auto", display: "flex", gap: 6 }}>
          <input
            type="text"
            placeholder="(Optional) notes to include"
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            style={{
              padding: "8px 10px",
              border: "1px solid #e5e7eb",
              borderRadius: 8,
              minWidth: 260,
            }}
          />
        </div>
      </div>

      {err && (
        <div style={{ color: "#b91c1c", fontWeight: 600, marginBottom: 8 }}>
          {err}
        </div>
      )}

      {/* Report preview (what you see is what the PDF captures) */}
      <div
        id="report-preview"
        ref={reportRef}
        style={{
          border: "1px solid #e5e7eb",
          borderRadius: 12,
          padding: 16,
          background: "#ffffff",
          maxWidth: 900,
        }}
      >
        {!markdown ? (
          <div style={{ color: "#6b7280" }}>
            Generate the report to see it here. The PDF will mirror what you see.
          </div>
        ) : (
          <article style={{ lineHeight: 1.55, fontSize: 15 }}>
            <h1 style={{ fontSize: 20, margin: 0, marginBottom: 8 }}>
              {portfolioName} — Analysis Report
            </h1>
            <div style={{ color: "#6b7280", marginBottom: 16 }}>
              Generated from current Analytics & Risk results (neutral, explanatory). Base currency: {baseCcy}
            </div>
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{markdown}</ReactMarkdown>
          </article>
        )}
      </div>
    </div>
  );
}
