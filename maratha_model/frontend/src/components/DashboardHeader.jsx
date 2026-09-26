import React from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { 
  Building2, 
  LogOut, 
  Shield, 
  Plus,
  Layers
} from "lucide-react";

export default function DashboardHeader({ 
  onNewComplaint,
  isStaff = false
}) {
  const { auth, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate(isStaff ? "/staff/login" : "/");
  };

  return (
    <header className="clean-header">
      <div className="clean-header-inner">
        {/* Brand */}
        <div className="brand-group">
          <Link to={isStaff ? "/staff/dashboard" : "/user/dashboard"} className="clean-brand-link">
            <div className="clean-brand-icon">
              <Building2 size={18} strokeWidth={2.4} />
            </div>
            <div className="clean-brand-text">
              <span className="brand-title">InfraPulse</span>
              <span className="brand-subtitle">
                {isStaff ? "Operations Portal" : "Infrastructure Management"}
              </span>
            </div>
          </Link>
        </div>

        {/* Right Section: Primary Action & Profile */}
        <div className="clean-header-actions">
          {!isStaff && onNewComplaint && (
            <button 
              className="btn-primary-action"
              onClick={onNewComplaint}
            >
              <Plus size={16} strokeWidth={2.5} />
              <span>Submit Complaint</span>
            </button>
          )}

          <div className="user-profile-menu">
            <div className="user-avatar-circle">
              {isStaff ? (
                <Shield size={14} />
              ) : (
                <span>{(auth?.name || "U")[0].toUpperCase()}</span>
              )}
            </div>
            <div className="user-info-text">
              <span className="user-name">
                {isStaff ? `${auth?.category} Dept` : (auth?.name || "Citizen")}
              </span>
              <span className="user-role-label">
                {isStaff ? "Maintenance Crew" : "Signed In"}
              </span>
            </div>
          </div>

          <button 
            className="btn-ghost-logout" 
            onClick={handleLogout}
            title="Sign out"
            aria-label="Sign out"
          >
            <LogOut size={15} />
          </button>
        </div>
      </div>
    </header>
  );
}
