import React, { useState } from "react";
import { useAuth } from "../AuthContext";
import { useNavigate, Link, useLocation } from "react-router-dom";

export default function Login() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const from = (location.state && location.state.from && location.state.from.pathname) || "/app/form";

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [remember, setRemember] = useState(false);
  const [err, setErr] = useState(null);
  const [loading, setLoading] = useState(false);

  async function onSubmit(e) {
    e.preventDefault();
    setErr(null);
    setLoading(true);

    try {
      const form = new URLSearchParams();
      form.append("username", email.trim());
      form.append("password", password);

      const res = await fetch("/api/v1/auth/login", { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Login failed");

      // ✅ Pass the remember flag here
      login(data.access_token, remember);
      navigate(from, { replace: true });
    } catch (e) {
      setErr(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ maxWidth: 420, margin: "40px auto" }}>
      <h2 style={{ marginBottom: 12 }}>Log in</h2>
      <p style={{ color: "#666", marginTop: 0 }}>
        Use the email/password you registered with to access your workspace.
      </p>

      <form onSubmit={onSubmit} style={{ display: "grid", gap: 12 }}>
        <label>
          Email
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
            style={{ width: "100%", padding: 8, marginTop: 6 }}
          />
        </label>

        <label>
          Password
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            style={{ width: "100%", padding: 8, marginTop: 6 }}
          />
        </label>

        {/* ✅ Remember Me checkbox */}
        <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <input
            type="checkbox"
            checked={remember}
            onChange={(e) => setRemember(e.target.checked)}
          />
          <span>Remember me on this device</span>
        </label>

        <button type="submit" disabled={loading} style={{ padding: "10px 14px" }}>
          {loading ? "Logging in..." : "Log in"}
        </button>

        {err && <div style={{ color: "crimson" }}>{err}</div>}

        <div style={{ marginTop: 8 }}>
          No account? <Link to="/signup">Create one</Link>
        </div>
      </form>
    </div>
  );
}
