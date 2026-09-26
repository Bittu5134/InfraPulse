import { createContext, useContext, useEffect, useState } from "react";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [auth, setAuth] = useState(() => {
    const token = localStorage.getItem("infrapulse_token");
    const role = localStorage.getItem("infrapulse_role");
    const category = localStorage.getItem("infrapulse_category");
    return token ? { token, role, category: category || null } : null;
  });

  useEffect(() => {
    if (auth) {
      localStorage.setItem("infrapulse_token", auth.token);
      localStorage.setItem("infrapulse_role", auth.role);
      if (auth.category) localStorage.setItem("infrapulse_category", auth.category);
    } else {
      localStorage.removeItem("infrapulse_token");
      localStorage.removeItem("infrapulse_role");
      localStorage.removeItem("infrapulse_category");
    }
  }, [auth]);

  function loginWithToken(tokenResponse) {
    setAuth({
      token: tokenResponse.access_token,
      role: tokenResponse.role,
      category: tokenResponse.category || null,
    });
  }

  function logout() {
    setAuth(null);
  }

  return (
    <AuthContext.Provider value={{ auth, loginWithToken, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}
