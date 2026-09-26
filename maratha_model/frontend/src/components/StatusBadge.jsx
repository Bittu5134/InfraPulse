import React from "react";

export default function StatusBadge({ status }) {
  const normalized = (status || "Submitted").toLowerCase().replace(/\s/g, "");

  const getStatusConfig = () => {
    switch (normalized) {
      case "submitted":
        return { label: "Submitted", className: "badge-submitted" };
      case "assigned":
        return { label: "Assigned", className: "badge-assigned" };
      case "inprogress":
        return { label: "In Progress", className: "badge-progress" };
      case "resolved":
        return { label: "Resolved", className: "badge-resolved" };
      default:
        return { label: status || "Submitted", className: "badge-submitted" };
    }
  };

  const { label, className } = getStatusConfig();

  return (
    <span className={`clean-status-pill ${className}`}>
      {label}
    </span>
  );
}
