import { useEffect, useRef, useState } from "react";
import { myComplaints, submitComplaint, WS_URL } from "../api/client";
import { useAuth } from "../context/AuthContext";
import DashboardHeader from "../components/DashboardHeader";
import ComplaintForm from "../components/ComplaintForm";
import ComplaintCard from "../components/ComplaintCard";
import { 
  Plus, 
  Search, 
  Inbox, 
  Clock, 
  CheckCircle2, 
  AlertCircle 
} from "lucide-react";

export default function UserDashboard() {
  const { auth } = useAuth();
  const [complaints, setComplaints] = useState([]);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [showForm, setShowForm] = useState(false);
  const [statusFilter, setStatusFilter] = useState("all");
  const [searchQuery, setSearchQuery] = useState("");
  const wsRef = useRef(null);

  async function refresh() {
    try {
      const data = await myComplaints();
      setComplaints(data);
      setError("");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh();
    const ws = new WebSocket(`${WS_URL}/ws/queue`);
    wsRef.current = ws;
    ws.onmessage = () => refresh();
    return () => ws.close();
  }, []);

  async function handleComplaintSubmit(formData) {
    setSubmitting(true);
    setError("");
    try {
      await submitComplaint(formData);
      await refresh();
      setShowForm(false);
      return true;
    } catch (err) {
      setError(err.message);
      return false;
    } finally {
      setSubmitting(false);
    }
  }

  // Key metrics (Clean, meaningful, 3 numbers only)
  const totalCount = complaints.length;
  const activeCount = complaints.filter(c => c.status !== "Resolved").length;
  const resolvedCount = complaints.filter(c => c.status === "Resolved").length;

  // Filter complaints
  const filteredComplaints = complaints.filter(c => {
    const matchesStatus = statusFilter === "all" ? true : c.status === statusFilter;
    const matchesSearch = !searchQuery.trim() ? true : (
      (c.defect_label && c.defect_label.toLowerCase().includes(searchQuery.toLowerCase())) ||
      (c.address && c.address.toLowerCase().includes(searchQuery.toLowerCase())) ||
      (c.description && c.description.toLowerCase().includes(searchQuery.toLowerCase())) ||
      (c.category && c.category.toLowerCase().includes(searchQuery.toLowerCase()))
    );
    return matchesStatus && matchesSearch;
  });

  return (
    <div className="clean-app-shell">
      {/* 1. Header with single primary action */}
      <DashboardHeader 
        onNewComplaint={() => setShowForm(true)}
        isStaff={false}
      />

      <main className="clean-main-container">
        {/* 2. Top Summary Bar (Spacious, 3 meaningful metrics) */}
        <section className="clean-hero-bar">
          <div className="hero-text-block">
            <h1 className="hero-greeting">Your Complaints</h1>
            <p className="hero-subtext">
              Track previously submitted infrastructure issues and live repair queue progress.
            </p>
          </div>

          <div className="hero-metrics-cluster">
            <div className="clean-stat-chip">
              <span className="stat-number">{totalCount}</span>
              <span className="stat-label">Total Filed</span>
            </div>
            <div className="clean-stat-chip stat-active">
              <span className="stat-number">{activeCount}</span>
              <span className="stat-label">In Progress</span>
            </div>
            <div className="clean-stat-chip stat-resolved">
              <span className="stat-number">{resolvedCount}</span>
              <span className="stat-label">Resolved</span>
            </div>
          </div>
        </section>

        {/* 3. New Complaint Modal / Expanded Card */}
        {showForm && (
          <section className="clean-form-container">
            <ComplaintForm 
              onSubmit={handleComplaintSubmit}
              submitting={submitting}
              error={error}
              onCancel={() => setShowForm(false)}
            />
          </section>
        )}

        {/* 4. Controls: Filter tabs & Search */}
        <section className="clean-feed-controls">
          <div className="filter-pill-bar">
            <button 
              className={`filter-btn ${statusFilter === "all" ? "active" : ""}`}
              onClick={() => setStatusFilter("all")}
            >
              All ({totalCount})
            </button>
            <button 
              className={`filter-btn ${statusFilter === "Submitted" ? "active" : ""}`}
              onClick={() => setStatusFilter("Submitted")}
            >
              Submitted
            </button>
            <button 
              className={`filter-btn ${statusFilter === "Assigned" ? "active" : ""}`}
              onClick={() => setStatusFilter("Assigned")}
            >
              Assigned
            </button>
            <button 
              className={`filter-btn ${statusFilter === "In Progress" ? "active" : ""}`}
              onClick={() => setStatusFilter("In Progress")}
            >
              In Progress
            </button>
            <button 
              className={`filter-btn ${statusFilter === "Resolved" ? "active" : ""}`}
              onClick={() => setStatusFilter("Resolved")}
            >
              Resolved
            </button>
          </div>

          <div className="search-box-clean">
            <Search size={14} className="search-glass" />
            <input 
              type="text"
              placeholder="Search by issue or location..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="clean-search-input"
            />
          </div>
        </section>

        {/* 5. Clean Complaints Feed */}
        <section className="clean-complaints-feed">
          {loading ? (
            <div className="clean-loading-state">
              <span>Loading complaints...</span>
            </div>
          ) : filteredComplaints.length === 0 ? (
            <div className="clean-empty-box">
              <div className="empty-icon-circle">
                <Inbox size={28} strokeWidth={1.6} />
              </div>
              <h3 className="empty-title">
                {searchQuery || statusFilter !== "all" 
                  ? "No matching complaints found" 
                  : "No complaints submitted yet"}
              </h3>
              <p className="empty-desc">
                {searchQuery || statusFilter !== "all" 
                  ? "Try resetting your search or selecting a different status filter."
                  : "Photograph damaged roads, leaking drains, spalling concrete, or peeling paint to submit a report."}
              </p>
              {!showForm && (
                <button 
                  className="btn-primary-action"
                  onClick={() => setShowForm(true)}
                >
                  <Plus size={16} />
                  <span>Submit a Complaint</span>
                </button>
              )}
            </div>
          ) : (
            <div className="clean-cards-stack">
              {filteredComplaints.map((c) => (
                <ComplaintCard key={c.id} complaint={c} />
              ))}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
