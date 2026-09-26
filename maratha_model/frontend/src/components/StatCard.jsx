import React from "react";

export default function StatCard({ title, value, subtitle, code, trend, indicatorColor = "blue" }) {
  return (
    <div className={`infra-metric-card indicator-${indicatorColor}`}>
      <div className="metric-header">
        <span className="metric-title">{title}</span>
        {code && <span className="metric-code">{code}</span>}
      </div>
      <div className="metric-body">
        <span className="metric-val">{value}</span>
        {subtitle && <span className="metric-sub">{subtitle}</span>}
      </div>
      {trend && (
        <div className="metric-footer">
          <span className="metric-status-line">{trend}</span>
        </div>
      )}
    </div>
  );
}
