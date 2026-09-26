import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { login, signup } from "../api/client";
import { useAuth } from "../context/AuthContext";
import { 
  Compass, 
  Mail, 
  Lock, 
  User, 
  ArrowRight, 
  Loader2, 
  AlertCircle, 
  ShieldCheck, 
  FileCheck2,
  Building2,
  CheckCircle2
} from "lucide-react";

export default function UserLogin() {
  const [mode, setMode] = useState("login");
  const [form, setForm] = useState({ name: "", email: "", password: "" });
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
        mode === "login" ? await login(form) : await signup("user", form);
      loginWithToken(res);
      navigate("/user/dashboard");
    } catch (err) {
      setError(err.message || "Failed to authenticate session");
    } finally {
      setSubmitting(false);
    }
  }

  const fillDemoAccount = (num = 1) => {
    setForm({
      name: `Demo Citizen ${num}`,
      email: `user${num}@infrapulse.demo`,
      password: "demo1234"
    });
    setError("");
  };

  return (
    <div className="editorial-split-viewport">
      {/* Left Column: High-End Editorial Architectural & Infrastructure Showcase */}
      <div className="split-editorial-hero">
        <div className="editorial-bg-image-layer">
          <img 
            src="/infra_hero.jpg" 
            alt="Civil engineering urban infrastructure" 
            className="editorial-hero-img"
          />
          <div className="editorial-image-overlay"></div>
          <div className="blueprint-grid-overlay"></div>
        </div>

        <div className="editorial-content-scaffold">
          {/* Top Brand Tagline */}
          <div className="editorial-top-brand">
            <div className="brand-badge-symbol">
              <Compass size={20} strokeWidth={2.4} />
            </div>
            <div className="brand-meta-group">
              <span className="brand-name-bold">INFRAPULSE</span>
              <span className="brand-division-tag">CIVIL WORKS SURVEILLANCE · SYS-2.4</span>
            </div>
          </div>

          {/* Center Editorial Typography */}
          <div className="editorial-center-copy">
            <div className="editorial-kicker">INFRASTRUCTURE HEALTH INTELLIGENCE</div>
            <h1 className="editorial-headline">
              Engineering-grade defect surveillance for modern civil infrastructure.
            </h1>
            <p className="editorial-paragraph">
              Empowering citizens and municipal engineers to photograph physical defects, compute damage severity via computer vision, and execute priority remediation queues.
            </p>

            <div className="editorial-stats-row">
              <div className="ed-stat">
                <span className="ed-stat-val">81%</span>
                <span className="ed-stat-lbl">REAL-PHOTO PRECISION</span>
              </div>
              <div className="ed-stat-divider"></div>
              <div className="ed-stat">
                <span className="ed-stat-val">4 CLASSES</span>
                <span className="ed-stat-lbl">SPALLING · WATER · TILES · PAINT</span>
              </div>
              <div className="ed-stat-divider"></div>
              <div className="ed-stat">
                <span className="ed-stat-val">&lt; 100ms</span>
                <span className="ed-stat-lbl">IN-PROCESS ML LATENCY</span>
              </div>
            </div>
          </div>

          {/* Bottom Specifications */}
          <div className="editorial-bottom-specs">
            <div className="specs-cadastral">
              <span className="spec-coord">CIVIC INFRASTRUCTURE SURVEILLANCE PLATFORM</span>
            </div>
          </div>
        </div>
      </div>

      {/* Right Column: Architectural Precision Authentication Panel */}
      <div className="split-form-pane">
        <div className="form-inner-wrapper">
          {/* Top Mobile Brand */}
          <div className="mobile-brand-row">
            <div className="brand-badge-symbol small">
              <Compass size={16} strokeWidth={2.4} />
            </div>
            <span className="brand-name-bold">INFRAPULSE</span>
          </div>

          {/* Form Header */}
          <div className="auth-header-block">
            <div className="auth-step-kicker">CITIZEN OPERATOR ACCESS</div>
            <h2 className="auth-page-title">
              {mode === "login" ? "Sign In to Docket" : "Create Citizen ID"}
            </h2>
            <p className="auth-page-sub">
              {mode === "login" 
                ? "Enter your institutional or citizen credentials to file and track defect reports."
                : "Register an account to monitor campus and urban infrastructure integrity."}
            </p>
          </div>

          {/* Mode Switch Tabs */}
          <div className="editorial-auth-tabs">
            <button
              type="button"
              className={`ed-tab ${mode === "login" ? "active" : ""}`}
              onClick={() => { setMode("login"); setError(""); }}
            >
              OPERATOR SIGN IN
            </button>
            <button
              type="button"
              className={`ed-tab ${mode === "signup" ? "active" : ""}`}
              onClick={() => { setMode("signup"); setError(""); }}
            >
              REGISTER ID
            </button>
          </div>

          {/* Quick Demo Pre-fill */}
          {mode === "login" && (
            <div className="quick-docket-bar">
              <span className="docket-fill-label">QUICK VERIFICATION:</span>
              <div className="docket-fill-buttons">
                <button type="button" onClick={() => fillDemoAccount(1)} className="btn-docket-fill">
                  USER-01
                </button>
                <button type="button" onClick={() => fillDemoAccount(2)} className="btn-docket-fill">
                  USER-02
                </button>
                <button type="button" onClick={() => fillDemoAccount(3)} className="btn-docket-fill">
                  USER-03
                </button>
              </div>
            </div>
          )}

          {/* Main Auth Form */}
          <form onSubmit={submit} className="editorial-form-stack">
            {mode === "signup" && (
              <div className="input-group-editorial">
                <label className="ed-label" htmlFor="user-fullname">REPORTER FULL NAME</label>
                <div className="ed-input-frame">
                  <User size={15} className="ed-input-icon" />
                  <input
                    id="user-fullname"
                    type="text"
                    className="ed-text-field"
                    placeholder="e.g. Yash Sinha"
                    value={form.name}
                    onChange={(e) => setForm({ ...form, name: e.target.value })}
                    required
                  />
                </div>
              </div>
            )}

            <div className="input-group-editorial">
              <label className="ed-label" htmlFor="user-email">EMAIL ADDRESS / OPERATOR ID</label>
              <div className="ed-input-frame">
                <Mail size={15} className="ed-input-icon" />
                <input
                  id="user-email"
                  type="email"
                  className="ed-text-field"
                  placeholder="user1@infrapulse.demo"
                  value={form.email}
                  onChange={(e) => setForm({ ...form, email: e.target.value })}
                  required
                />
              </div>
            </div>

            <div className="input-group-editorial">
              <label className="ed-label" htmlFor="user-pass">SECURITY KEY / PASSWORD</label>
              <div className="ed-input-frame">
                <Lock size={15} className="ed-input-icon" />
                <input
                  id="user-pass"
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
              className="ed-submit-btn" 
              disabled={submitting}
            >
              {submitting ? (
                <>
                  <Loader2 size={15} className="spin-icon" />
                  <span>AUTHENTICATING DOCKET...</span>
                </>
              ) : (
                <>
                  <span>{mode === "login" ? "ENTER DOCKET CONTROL" : "INITIALIZE NEW ACCOUNT"}</span>
                  <ArrowRight size={14} />
                </>
              )}
            </button>
          </form>

          {/* Switch to Staff Portal */}
          <div className="editorial-switch-row">
            <Link to="/staff/login" className="ed-switch-link">
              <ShieldCheck size={14} />
              <span>Municipal staff or remediation crew? <strong>Operations Login →</strong></span>
            </Link>
          </div>

          <div className="footer-system-stamp">
            <span>INFRAPULSE PLATFORM</span>
          </div>
        </div>
      </div>
    </div>
  );
}
