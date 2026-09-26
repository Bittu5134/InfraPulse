import { useEffect, useRef, useState } from "react";
import { imageUrl, staffQueue, updateStatus, WS_URL } from "../api/client";
import { useAuth } from "../context/AuthContext";
import DashboardHeader from "../components/DashboardHeader";
import StatusBadge from "../components/StatusBadge";
import ComplaintTimeline from "../components/ComplaintTimeline";
import { 
  Check, 
  ArrowRight, 
  Loader2, 
  MapPin, 
  Search, 
  Inbox, 
  Maximize2,
  ChevronDown,
  ChevronUp
} from "lucide-react";

const NEXT_STATUS = {
  Submitted: "Assigned",
  Assigned: "In Progress",
  "In Progress": "Resolved",
};

export default function StaffDashboard() {
  const { auth } = useAuth();
  const [queue, setQueue] = useState([]);
  const [error, setError] = useState("");
  const [busyId, setBusyId] = useState(null);
  const [expandedId, setExpandedId] = useState(null);
  const [selectedImage, setSelectedImage] = useState(null);
  const [searchQuery, setSearchQuery] = useState("");
  const wsRef = useRef(null);

  async function refresh() {
    try {
      const data = await staffQueue();
      setQueue(data);
      setError("");
    } catch (err) {
      setError(err.message);
    }
  }

  useEffect(() => {
    refresh();
    const ws = new WebSocket(`${WS_URL}/ws/queue`);
    wsRef.current = ws;
    ws.onmessage = () => refresh();
    return () => ws.close();
  }, []);

  async function advance(c) {
    const next = NEXT_STATUS[c.status];
    if (!next) return;
    setBusyId(c.id);
    try {
      await updateStatus(c.id, next);
      await refresh();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusyId(null);
    }
  }

  const formatDefectName = (raw) => {
    if (!raw) return "Unspecified Defect";
    return raw.replace(/_/g, " ");
  };

  const filteredQueue = queue.filter(c => {
    if (!searchQuery.trim()) return true;
    const q = searchQuery.toLowerCase();
    return (
      (c.defect_label && c.defect_label.toLowerCase().includes(q)) ||
      (c.address && c.address.toLowerCase().includes(q)) ||
      (c.name && c.name.toLowerCase().includes(q)) ||
      (c.description && c.description.toLowerCase().includes(q))
    );
  });

  return (
    <div className="clean-app-shell">
      <DashboardHeader isStaff={true} />

      <main className="clean-main-container">
        {/* Top Summary Bar */}
        <section className="clean-hero-bar">
          <div className="hero-text-block">
            <h1 className="hero-greeting">{auth?.category || "Maintenance"} Dispatch Queue</h1>
            <p className="hero-subtext">
              Active tickets sorted by defect priority and severity. Advance tickets through each stage of work.
            </p>
          </div>

          <div className="hero-metrics-cluster">
            <div className="clean-stat-chip">
              <span className="stat-number">{queue.length}</span>
              <span className="stat-label">In Queue</span>
            </div>
            <div className="clean-stat-chip stat-active">
              <span className="stat-number">{queue.filter(c => c.status === "In Progress").length}</span>
              <span className="stat-label">In Progress</span>
            </div>
          </div>
        </section>

        {/* Search Bar */}
        <section className="clean-feed-controls">
          <div className="search-box-clean search-box-wide">
            <Search size={14} className="search-glass" />
            <input 
              type="text"
              placeholder="Search queue by issue, location, or reporter..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="clean-search-input"
            />
          </div>
        </section>

        {/* Queue Items Feed */}
        <section className="clean-complaints-feed">
          {filteredQueue.length === 0 ? (
            <div className="clean-empty-box">
              <div className="empty-icon-circle">
                <Check size={28} className="text-emerald" strokeWidth={2.5} />
              </div>
              <h3 className="empty-title">Queue is all clear</h3>
              <p className="empty-desc">
                There are currently no active tickets assigned to the {auth?.category} Department.
              </p>
            </div>
          ) : (
            <div className="clean-cards-stack">
              {filteredQueue.map((c) => {
                const next = NEXT_STATUS[c.status];
                const isBusy = busyId === c.id;
                const isExpanded = expandedId === c.id;

                return (
                  <div key={c.id} className="clean-complaint-card staff-queue-card-clean">
                    <div className="card-main-content">
                      {/* Priority Rank */}
                      <div className="staff-rank-badge">
                        <span className="rank-hash">#</span>
                        <span className="rank-digit">{c.queue_position}</span>
                      </div>

                      {/* Photo Thumbnail */}
                      <div 
                        className="card-thumbnail-wrap" 
                        onClick={() => setSelectedImage(imageUrl(c.image_path))}
                        title="Click to view photo"
                      >
                        <img 
                          src={imageUrl(c.image_path)} 
                          alt={c.defect_label}
                          className="card-thumbnail-img" 
                          loading="lazy"
                        />
                        <div className="thumb-zoom-icon">
                          <Maximize2 size={12} />
                        </div>
                      </div>

                      {/* Details Column */}
                      <div className="card-info-pane">
                        <div className="card-headline-row">
                          <h3 className="card-title">{formatDefectName(c.defect_label)}</h3>
                          <StatusBadge status={c.status} />
                        </div>

                        <div className="card-location-row">
                          <MapPin size={13} className="loc-pin" />
                          <span className="loc-text">{c.address}</span>
                          {c.name && (
                            <>
                              <span className="meta-dot">·</span>
                              <span className="loc-reporter">Reported by {c.name}</span>
                            </>
                          )}
                        </div>

                        {c.description && (
                          <p className="card-brief-desc">{c.description}</p>
                        )}

                        <div className="card-timeline-mini">
                          <ComplaintTimeline status={c.status} />
                        </div>
                      </div>

                      {/* Action Button */}
                      <div className="staff-card-action-pane">
                        {next ? (
                          <button
                            type="button"
                            className="btn-advance-ticket"
                            onClick={() => advance(c)}
                            disabled={isBusy}
                          >
                            {isBusy ? (
                              <Loader2 size={14} className="spinning-icon" />
                            ) : (
                              <>
                                <span>Mark {next}</span>
                                <ArrowRight size={13} />
                              </>
                            )}
                          </button>
                        ) : (
                          <div className="ticket-resolved-chip">
                            <Check size={14} strokeWidth={2.5} />
                            <span>Resolved</span>
                          </div>
                        )}

                        <button 
                          type="button" 
                          className="btn-toggle-details"
                          onClick={() => setExpandedId(isExpanded ? null : c.id)}
                        >
                          <span>{isExpanded ? "Less" : "Details"}</span>
                          {isExpanded ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
                        </button>
                      </div>
                    </div>

                    {/* Expandable Technical Details */}
                    {isExpanded && (
                      <div className="card-expanded-drawer">
                        <div className="drawer-grid">
                          <div className="drawer-item">
                            <span className="drawer-label">AI Confidence</span>
                            <span className="drawer-value">
                              {c.confidence ? `${(c.confidence * 100).toFixed(0)}%` : "N/A"}
                            </span>
                          </div>
                          <div className="drawer-item">
                            <span className="drawer-label">Damage Extent</span>
                            <span className="drawer-value">{c.severity_score ? Math.round(c.severity_score) : 0} / 100</span>
                          </div>
                          <div className="drawer-item">
                            <span className="drawer-label">Defect Priority Tier</span>
                            <span className="drawer-value">Priority {c.type_priority}</span>
                          </div>
                          <div className="drawer-item">
                            <span className="drawer-label">Complaint ID</span>
                            <span className="drawer-value font-mono">#{c.id.slice(0, 8).toUpperCase()}</span>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </section>

        {/* Image Modal */}
        {selectedImage && (
          <div className="clean-modal-backdrop" onClick={() => setSelectedImage(null)}>
            <div className="clean-modal-content" onClick={(e) => e.stopPropagation()}>
              <div className="modal-header-simple">
                <span className="modal-defect-title">Inspection Photo</span>
                <button className="modal-close-simple" onClick={() => setSelectedImage(null)}>✕</button>
              </div>
              <div className="modal-img-body">
                <img src={selectedImage} alt="Defect specimen" className="modal-full-photo" />
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
