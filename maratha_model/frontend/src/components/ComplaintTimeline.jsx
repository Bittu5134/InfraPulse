import React from "react";
import { Check } from "lucide-react";

export default function ComplaintTimeline({ status }) {
  const steps = ["Submitted", "Assigned", "In Progress", "Resolved"];
  const currentIdx = steps.indexOf(status) !== -1 ? steps.indexOf(status) : 0;

  return (
    <div className="clean-stepper" aria-label={`Status: ${status}`}>
      <div className="stepper-track">
        {steps.map((step, idx) => {
          const isDone = idx < currentIdx;
          const isCurrent = idx === currentIdx;

          return (
            <div 
              key={step} 
              className={`stepper-node ${isDone ? "is-done" : ""} ${isCurrent ? "is-current" : ""}`}
            >
              <div className="node-marker">
                {isDone ? <Check size={10} strokeWidth={3} /> : null}
              </div>
              <span className="node-label">{step}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
