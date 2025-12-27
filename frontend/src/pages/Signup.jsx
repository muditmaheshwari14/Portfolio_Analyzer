import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";

export default function Signup() {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [err, setErr] = useState(null);
  const [loading, setLoading] = useState(false);

async function onSubmit(e) {
  e.preventDefault();
  setErr(null);

  if (password !== confirm) { setErr("Passwords do not match."); return; }
  if (password.length < 6)   { setErr("Password must be at least 6 characters."); return; }
  if (password.length > 256)  { setErr("Password must be at most 256 characters."); return; }

  setLoading(true);
  try {
    const res = await fetch("/api/v1/auth/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email: email.trim(), password }),
    });

    // Read as text first, then try to parse JSON
    const text = await res.text();
    let data;
    try { data = JSON.parse(text); } catch { /* not json */ }

    if (!res.ok) {
      // Prefer JSON detail; otherwise show raw text; fallback generic
      const msg = (data && data.detail) || text || `HTTP ${res.status}`;
      throw new Error(msg);
    }

    // Success -> go to login
    // (optionally show toast from `data` if you return something)
    navigate("/login", { replace: true });
  } catch (e) {
    setErr(e.message || "Sign up failed");
  } finally {
    setLoading(false);
  }
}

  return (
    <div style={{ maxWidth: 420, margin: "40px auto" }}>
      <h2 style={{ marginBottom: 12 }}>Create your account</h2>
      <p style={{ color: "#666", marginTop: 0 }}>
        Sign up with your email to save portfolios and access analytics anytime.
      </p>

      <form onSubmit={onSubmit} style={{ display: "grid", gap: 12 }}>
        <label>
          Email
          <input
            type="email"
            value={email}
            onChange={e=>setEmail(e.target.value)}
            required
            style={{ width: "100%", padding: 8, marginTop: 6 }}
          />
        </label>

        <label>
          Password
          <input
            type="password"
            value={password}
            onChange={e=>setPassword(e.target.value)}
            required
            style={{ width: "100%", padding: 8, marginTop: 6 }}
          />
        </label>

        <label>
          Confirm password
          <input
            type="password"
            value={confirm}
            onChange={e=>setConfirm(e.target.value)}
            required
            style={{ width: "100%", padding: 8, marginTop: 6 }}
          />
        </label>

        <button type="submit" disabled={loading} style={{ padding: "10px 14px" }}>
          {loading ? "Creating account..." : "Sign up"}
        </button>

        {err && <div style={{ color: "crimson" }}>{err}</div>}

        <div style={{ marginTop: 8 }}>
          Already have an account? <Link to="/login">Log in</Link>
        </div>
      </form>
    </div>
  );
}
