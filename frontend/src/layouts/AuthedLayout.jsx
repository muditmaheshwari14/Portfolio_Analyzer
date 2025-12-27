import React from "react";
import { NavLink, Outlet } from "react-router-dom";
import { useAuth } from "../AuthContext";

export default function AuthedLayout() {
  const { me, logout } = useAuth();

  return (
    <div>
      {/* Top bar */}
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 16 }}>
        <strong>Portfolio Analytics</strong>
        <div>
          {me && <span style={{ marginRight: 8 }}>{me.email}</span>}
          <button onClick={logout}>Logout</button>
        </div>
      </div>

      {/* Tabs */}
      <div style={tabsWrap}>
        <NavLink to="/app/account" style={tabLink} className={({isActive}) => isActive ? "active" : ""}>Account</NavLink>
        <NavLink to="/app/portfolios" style={tabLink} className={({isActive}) => isActive ? "active" : ""}>My Portfolios</NavLink>
        <NavLink to="/app/form" style={tabLink} className={({isActive}) => isActive ? "active" : ""}>Portfolio Form</NavLink>
      </div>

      <div style={{ border: "1px solid #eee", borderTop: "none", padding: 16, borderRadius: "0 0 12px 12px" }}>
        <Outlet />
      </div>
    </div>
  );
}

const tabsWrap = {
  display: "flex",
  gap: 8,
  border: "1px solid #eee",
  borderRadius: 12,
  padding: 6,
  marginBottom: 12
};
const tabLink = ({ isActive }) => ({
  padding: "8px 12px",
  textDecoration: "none",
  borderRadius: 8,
  color: isActive ? "#000" : "#333"
});
