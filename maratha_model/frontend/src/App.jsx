import { Navigate, Route, Routes } from "react-router-dom";
import UserLogin from "./pages/UserLogin";
import StaffLogin from "./pages/StaffLogin";
import UserDashboard from "./pages/UserDashboard";
import StaffDashboard from "./pages/StaffDashboard";
import { useAuth } from "./context/AuthContext";
import "./App.css";

function Protected({ role, children }) {
  const { auth } = useAuth();
  if (!auth || auth.role !== role) {
    return <Navigate to={role === "staff" ? "/staff/login" : "/"} replace />;
  }
  return children;
}

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<UserLogin />} />
      <Route path="/staff/login" element={<StaffLogin />} />
      <Route
        path="/user/dashboard"
        element={
          <Protected role="user">
            <UserDashboard />
          </Protected>
        }
      />
      <Route
        path="/staff/dashboard"
        element={
          <Protected role="staff">
            <StaffDashboard />
          </Protected>
        }
      />
    </Routes>
  );
}
