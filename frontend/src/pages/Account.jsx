import React from "react";
import { useAuth } from "../AuthContext";

export default function Account() {
  const { me } = useAuth();
  return (
    <div>
      <h3>Account</h3>
      <p>Email: <b>{me?.email}</b></p>
      <p>JWT session active. You can save portfolios and view them in “My Portfolios”.</p>
    </div>
  );
}
