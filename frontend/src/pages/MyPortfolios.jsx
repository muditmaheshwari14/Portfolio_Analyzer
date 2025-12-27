import React from "react";
import { useAuth } from "../AuthContext";

export default function MyPortfolios({ onLoadPortfolio }) {
  const { token } = useAuth();
  const [items, setItems] = React.useState([]);
  const [loading, setLoading] = React.useState(true);
  const [err, setErr] = React.useState(null);

  async function fetchList() {
    if (!token) return;
    setLoading(true); setErr(null);
    try {
      const res = await fetch("/api/v1/portfolios", {
        headers: { Authorization: `Bearer ${token}` },
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.detail || res.statusText);
      setItems(data || []);
    } catch (e) {
      setErr(e.message || "Failed to load portfolios");
    } finally {
      setLoading(false);
    }
  }

  React.useEffect(() => { fetchList(); /* eslint-disable-next-line */ }, [token]);

  async function loadOne(id) {
    try {
      const res = await fetch(`/api/v1/portfolios/${id}`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      const detail = await res.json();
      if (!res.ok) throw new Error(detail?.detail || res.statusText);
      // Hand off to analytics flow
      onLoadPortfolio?.({ name: detail.name, holdings: detail.holdings });
    } catch (e) {
      alert(e.message || "Failed to load portfolio");
    }
  }

  async function removeOne(id) {
    if (!window.confirm("Delete this portfolio?")) return;
    try {
      const res = await fetch(`/api/v1/portfolios/${id}`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || res.statusText);
      }
      setItems((s) => s.filter((r) => r.id !== id));
    } catch (e) {
      alert(e.message || "Failed to delete");
    }
  }

  return (
    <div>
      <h3>My Portfolios</h3>
      {loading && <div>Loading…</div>}
      {err && <div style={{ color:"crimson" }}>{err}</div>}
      {!loading && !items.length && <div>No saved portfolios yet.</div>}
      <div style={{ display:"grid", gap:8 }}>
        {items.map((p) => (
          <div key={p.id}
               style={{ border:"1px solid #eee", borderRadius:8, padding:12, display:"flex",
                        justifyContent:"space-between", alignItems:"center" }}>
            <div>
              <div style={{ fontWeight:600 }}>{p.name}</div>
              <div style={{ fontSize:12, color:"#666" }}>
                Created: {new Date(p.created_at).toLocaleString()}
              </div>
            </div>
            <div style={{ display:"flex", gap:8 }}>
              <button onClick={() => loadOne(p.id)}>Load</button>
              <button onClick={() => removeOne(p.id)} style={{ color:"#b91c1c" }}>Delete</button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
