import React, { createContext, useContext, useEffect, useState } from "react";

const AuthCtx = createContext(null);
export function useAuth() {
  return useContext(AuthCtx);
}

export default function AuthProvider({ children }) {
  // Load from either session or local storage
  function getInitialToken() {
    return sessionStorage.getItem("token") || localStorage.getItem("token") || null;
  }

  const [token, setToken] = useState(getInitialToken());
  const [me, setMe] = useState(null);

  // Fetch user info when token changes
  useEffect(() => {
    if (!token) {
      setMe(null);
      return;
    }
    fetch("/api/v1/me", { headers: { Authorization: `Bearer ${token}` } })
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => setMe(data))
      .catch(() => setMe(null));
  }, [token]);

  // Login and store token in chosen storage
  function login(t, remember = false) {
    // Clear previous tokens
    sessionStorage.removeItem("token");
    localStorage.removeItem("token");

    if (remember) {
      localStorage.setItem("token", t); // persist
    } else {
      sessionStorage.setItem("token", t); // temporary
    }
    setToken(t);
  }

  function logout() {
    sessionStorage.removeItem("token");
    localStorage.removeItem("token");
    setToken(null);
  }

  return (
    <AuthCtx.Provider value={{ token, me, login, logout }}>
      {children}
    </AuthCtx.Provider>
  );
}
