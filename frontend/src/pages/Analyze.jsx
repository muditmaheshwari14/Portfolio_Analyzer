import React, { useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../AuthContext";

export default function Analyze() {
  const { token } = useAuth();
  const nav = useNavigate();

  useEffect(() => {
    if (token) nav("/app/form");
  }, [token, nav]);

  return (
    <div style={{ maxWidth: 520 }}>
      <h2>Ready to analyze?</h2>
      <p>Log in or create an account to access the Portfolio Form and save portfolios.</p>
      <div style={{ display: "flex", gap: 12 }}>
        <Link to="/login"><button>Log in</button></Link>
        <Link to="/signup"><button>Create account</button></Link>
      </div>
    </div>
  );
}
