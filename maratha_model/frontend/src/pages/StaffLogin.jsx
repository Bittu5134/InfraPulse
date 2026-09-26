import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { login, signup } from "../api/client";
import { useAuth } from "../context/AuthContext";
import { 
  Compass, 
  Mail, 
  Lock, 
  User, 
  Shield, 
  ArrowRight, 
  Loader2, 
  AlertCircle,
  Wrench,
  CheckCircle2,
  FileSpreadsheet
} from "lucide-react";

const CATEGORIES = ["Structural", "Functional", "Performance"];

export default function StaffLogin() {
  const [mode, setMode] = useState("login");
  const [form, setForm] = useState({
    name: "",
    email: "",
    password: "",
    category: CATEGORIES[0],
  });
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const { loginWithToken } = useAuth();
  const navigate = useNavigate();

  async function submit(e) {
    e.preventDefault();
    setError("");
    setSubmitting(true);
    try {
      const res =
        mode === "login" ? await login(form) : await signup("staff", form);
      loginWithToken(res);
      navigate("/staff/dashboard");
    } catch (err) {
      setError(err.message || "Failed to authenticate staff credentials");
    } finally {
      setSubmitting(false);
    }
  }

  const fillStaffAccount = (categoryKey, email) => {
    setForm({
      name: `Staff (${categoryKey})`,
      email: email,
      password: "demo1234",
      category: categoryKey
    });
    setError("");
  };

  return (
    <div className="editorial-split-viewport staff-viewport-theme">
      {/* Left Column: High-End Editorial Architectural & Infrastructure Showcase */}
      <div className="split-editorial-hero">
        <div className="editorial-bg-image-layer">
          <img 
            src="/infra_hero.jpg" 
            alt="Civil engineering urban infrastructure operations" 
            className="editorial-hero-img ops-filter"
          />
          <div className="editorial-image-overlay ops-overlay"></div>
          <div className="blueprint-grid-overlay"></div>
        </div>

        <div className="editorial-content-scaffold">
          {/* Top Brand Tagline */}
          <div className="editorial-top-brand">
            <div className="brand-badge-symbol symbol-staff">
              <Shield size={20} strokeWidth={2.4} />
            </div>
            <div className="brand-meta-group">
              <span className="brand-name-bold">INFRAPULSE OPERATIONS</span>
              <span className="brand-division-tag">DISPATCH & REMEDIATION CONTROL</span>
            </div>
          </div>

          {/* Center Editorial Typography */}
          <div className="editorial-center-copy">
            <div className="editorial-kicker">MUNICIPAL OPERATIONS TERMINAL</div>
            <h1 className="editorial-headline">
              Deterministic priority dispatch for physical facility remediation.
            </h1>
            <p className="editorial-paragraph">
              Authorized municipal officers and engineering crews receive real-time streaming work orders sequenced strictly by hazard severity and material degradation scores.
            </p>

            <div className="editorial-stats-row">
              <div className="ed-stat">
                <span className="ed-stat-val">3 DIVISIONS</span>
                <span className="ed-stat-lbl">STRUCTURAL · FUNCTIONAL · PERFORMANCE</span>
              </div>
              <div className="ed-stat-divider"></div>
              <div className="ed-stat">
                <span className="ed-stat-val">0-100</span>
                <span className="ed-stat-lbl">CV SEVERITY RESOLUTION TIE-BREAKER</span>
              </div>
              <div className="ed-stat-divider"></div>
              <div className="ed-stat">
                <span className="ed-stat-val">WSS</span>
                <span className="ed-stat-lbl">AUTOMATIC QUEUE SYNCHRONIZATION</span>
              </div>
            </div>
          </div>

          {/* Bottom Specifications */}
          <div className="editorial-bottom-specs">
            <div className="specs-cadastral">
              <span className="spec-coord">AUTHORIZED OPERATIONS ONLY</span>
            </div>
          </div>
        </div>
      </div>

      {/* Right Column: Architectural Precision Authentication Panel */}
      <div className="split-form-pane">
        <div className="form-inner-wrapper">
          {/* Top Mobile Brand */}
          <div className="mobile-brand-row">
            <div className="brand-badge-symbol small symbol-staff">
              <Shield size={16} strokeWidth={2.4} />
            </div>
            <span className="brand-name-bold">INFRAPULSE STAFF</span>
          </div>

          {/* Form Header */}
          <div className="auth-header-block">
            <div className="auth-step-kicker">AUTHORIZED OPERATIONS ACCESS</div>
            <h2 className="auth-page-title">
              {mode === "login" ? "Staff Dispatch Sign In" : "Register Engineering Crew"}
            </h2>
            <p className="auth-page-sub">
              {mode === "login"
                ? "Enter officer credentials to access your division's prioritized repair queue."
                : "Register a department technician or engineering supervisor account."}
            </p>
          </div>

          {/* Mode Switch Tabs */}
          <div className="editorial-auth-tabs">
            <button
              type="button"
              className={`ed-tab ${mode === "login" ? "active" : ""}`}
              onClick={() => { setMode("login"); setError(""); }}
            >
              STAFF SIGN IN
            </button>
            <button
              type="button"
              className={`ed-tab ${mode === "signup" ? "active" : ""}`}
              onClick={() => { setMode("signup"); setError(""); }}
            >
              REGISTER STAFF
            </button>
          </div>

          {/* Quick Demo Pre-fill for Staff */}
          {mode === "login" && (
            <div className="quick-docket-bar staff-demo-bar">
              <span className="docket-fill-label">DIVISION DISPATCH:</span>
              <div className="docket-fill-buttons">
                <button 
                  type="button" 
                  onClick={() => fillStaffAccount("Structural", "staff.structural@infrapulse.demo")} 
                  className="btn-docket-fill"
                >
                  STRUCTURAL
                </button>
                <button 
                  type="button" 
                  onClick={() => fillStaffAccount("Functional", "staff.functional@infrapulse.demo")} 
                  className="btn-docket-fill"
                >
                  FUNCTIONAL
                </button>
                <button 
                  type="button" 
                  onClick={() => fillStaffAccount("Performance", "staff.performance1@infrapulse.demo")} 
                  className="btn-docket-fill"
                >
                  PERFORMANCE
                </button>
              </div>
            </div>
          )}

          {/* Main Auth Form */}
          <form onSubmit={submit} className="editorial-form-stack">
            {mode === "signup" && (
              <>
                <div className="input-group-editorial">
                  <label className="ed-label" htmlFor="staff-fullname">STAFF OFFICER NAME</label>
                  <div className="ed-input-frame">
                    <User size={15} className="ed-input-icon" />
                    <input
                      id="staff-fullname"
                      type="text"
                      className="ed-text-field"
                      placeholder="e.g. Officer Raman"
                      value={form.name}
                      onChange={(e) => setForm({ ...form, name: e.target.value })}
                      required
                    />
                  </div>
                </div>

                <div className="input-group-editorial">
                  <label className="ed-label" htmlFor="staff-dept">ASSIGNED DIVISION</label>
                  <div className="ed-input-frame">
                    <Wrench size={15} className="ed-input-icon" />
                    <select
                      id="staff-dept"
                      className="ed-select-field"
                      value={form.category}
                      onChange={(e) => setForm({ ...form, category: e.target.value })}
                    >
                      {CATEGORIES.map((c) => (
                        <option key={c} value={c}>
                          {c.toUpperCase()} DIVISION ({c === "Structural" ? "Spalling" : c === "Functional" ? "Stagnant Water" : "Cracked Tiles & Paint"})
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </>
            )}

            <div className="input-group-editorial">
              <label className="ed-label" htmlFor="staff-email">OFFICER IDENTIFIER / EMAIL</label>
              <div className="ed-input-frame">
                <Mail size={15} className="ed-input-icon" />
                <input
                  id="staff-email"
                  type="email"
                  className="ed-text-field"
                  placeholder="staff.structural@infrapulse.demo"
                  value={form.email}
                  onChange={(e) => setForm({ ...form, email: e.target.value })}
                  required
                />
              </div>
            </div>

            <div className="input-group-editorial">
              <label className="ed-label" htmlFor="staff-pass">SECURITY KEY / PASSWORD</label>
              <div className="ed-input-frame">
                <Lock size={15} className="ed-input-icon" />
                <input
                  id="staff-pass"
                  type="password"
                  className="ed-text-field"
                  placeholder="••••••••"
                  value={form.password}
                  onChange={(e) => setForm({ ...form, password: e.target.value })}
                  required
                />
              </div>
            </div>

            {error && (
              <div className="editorial-alert-banner">
                <AlertCircle size={14} />
                <span>{error}</span>
              </div>
            )}

            <button 
              type="submit" 
              className="ed-submit-btn btn-staff-submit" 
              disabled={submitting}
            >
              {submitting ? (
                <>
                  <Loader2 size={15} className="spin-icon" />
                  <span>VERIFYING DIVISION CLEARANCE...</span>
                </>
              ) : (
                <>
                  <span>{mode === "login" ? "OPEN DIVISION DISPATCH QUEUE" : "INITIALIZE STAFF PROFILE"}</span>
                  <ArrowRight size={14} />
                </>
              )}
            </button>
          </form>

          {/* Switch to Citizen Portal */}
          <div className="editorial-switch-row">
            <Link to="/" className="ed-switch-link">
              <span>← Back to <strong>Citizen Docket Portal</strong></span>
            </Link>
          </div>

          <div className="footer-system-stamp">
            <span>AUTHORIZED STAFF OPERATIONS PORTAL</span>
          </div>
        </div>
      </div>
    </div>
  );
}
