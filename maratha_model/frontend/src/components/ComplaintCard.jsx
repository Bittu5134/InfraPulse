import React, { useState } from "react";
import { 
  MapPin, 
  ChevronDown, 
  ChevronUp, 
  Check, 
  Clock,
  Maximize2
} from "lucide-react";
import StatusBadge from "./StatusBadge";
import ComplaintTimeline from "./ComplaintTimeline";
import { imageUrl } from "../api/client";

export default function ComplaintCard({ complaint }) {
  const [expanded, setExpanded] = useState(false);
  const [showFullImage, setShowFullImage] = useState(false);

  const formatDefectName = (raw) => {
    if (!raw) return "Unspecified Issue";
    return raw.replace(/_/g, " ");
  };

  const formatDate = (isoString) => {
    if (!isoString) return "";
    try {
      const date = new Date(isoString);
      return date.toLocaleDateString("en-US", {
        month: "short",
        day: "numeric"
      });
    } catch {
      return "";
    }
  };

  const isResolved = complaint.status === "Resolved";

  return (
    <div className={`clean-complaint-card ${isResolved ? "is-resolved" : ""}`}>
      {/* Primary Card Body — Scannable in 2-3 seconds */}
      <div className="card-main-content">
        {/* Left: Defect Photo Thumbnail */}
        <div 
          className="card-thumbnail-wrap" 
          onClick={() => setShowFullImage(true)}
          title="Click to view full photo"
        >
          <img 
            src={imageUrl(complaint.image_path)} 
            alt={complaint.defect_label || "Complaint photo"} 
            className="card-thumbnail-img"
            loading="lazy"
          />
          <div className="thumb-zoom-icon">
            <Maximize2 size={12} />
          </div>
        </div>

        {/* Center: Essential Level 1 & Level 2 Info */}
        <div className="card-info-pane">
          <div className="card-headline-row">
            <h3 className="card-title">{formatDefectName(complaint.defect_label)}</h3>
            <div className="card-badges-row">
              <span className="clean-category-badge" data-category={complaint.category}>
                {complaint.category || "General"}
              </span>
              <StatusBadge status={complaint.status} />
            </div>
          </div>

          <div className="card-location-row">
            <MapPin size={13} className="loc-pin" />
            <span className="loc-text">{complaint.address || "No location specified"}</span>
            {complaint.created_at && (
              <>
                <span className="meta-dot">·</span>
                <span className="loc-date">{formatDate(complaint.created_at)}</span>
              </>
            )}
          </div>

          {complaint.description && (
            <p className="card-brief-desc">{complaint.description}</p>
          )}

          {/* Level 1 Progress Line */}
          <div className="card-timeline-mini">
            <ComplaintTimeline status={complaint.status} />
          </div>
        </div>

        {/* Right: Queue Position Badge & Detail Expander */}
        <div className="card-queue-pane">
          {!isResolved && complaint.queue_position ? (
            <div className="queue-position-pill" title="Position in repair queue">
              <span className="qp-label">Queue</span>
              <span className="qp-num">#{complaint.queue_position}</span>
            </div>
          ) : (
            <div className="resolved-check-badge">
              <Check size={14} strokeWidth={2.5} />
              <span>Resolved</span>
            </div>
          )}

          <button 
            type="button" 
            className="btn-toggle-details"
            onClick={() => setExpanded(!expanded)}
            aria-expanded={expanded}
          >
            <span>{expanded ? "Less" : "Details"}</span>
            {expanded ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
          </button>
        </div>
      </div>

      {/* Level 3: Expandable Secondary Inspection Details */}
      {expanded && (
        <div className="card-expanded-drawer">
          <div className="drawer-grid">
            <div className="drawer-item">
              <span className="drawer-label">AI Classification Confidence</span>
              <span className="drawer-value">
                {complaint.confidence ? `${(complaint.confidence * 100).toFixed(0)}%` : "Verified"}
              </span>
            </div>

            <div className="drawer-item">
              <span className="drawer-label">Damage Extent Score</span>
              <span className="drawer-value">
                {Math.round(complaint.severity_score || 0)} / 100
              </span>
            </div>

            <div className="drawer-item">
              <span className="drawer-label">Assigned Department</span>
              <span className="drawer-value">
                {complaint.category ? `${complaint.category} Division` : "General Maintenance"}
              </span>
            </div>

            <div className="drawer-item">
              <span className="drawer-label">Tracking ID</span>
              <span className="drawer-value font-mono">
                #{complaint.id ? complaint.id.slice(0, 8).toUpperCase() : "N/A"}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Modal for full photo view */}
      {showFullImage && (
        <div className="clean-modal-backdrop" onClick={() => setShowFullImage(false)}>
          <div className="clean-modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header-simple">
              <span className="modal-defect-title">{formatDefectName(complaint.defect_label)}</span>
              <button className="modal-close-simple" onClick={() => setShowFullImage(false)}>✕</button>
            </div>
            <div className="modal-img-body">
              <img 
                src={imageUrl(complaint.image_path)} 
                alt="Defect detail" 
                className="modal-full-photo"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
