# Documentation Report - InfraPulse

**Problem Statement**: Photo-Based Defect Detection & Priority Maintenance System  
**System Name**: InfraPulse  
**Target Environment**: Python Web Application (FastAPI, SQLite, Jinja2, Tailwind CSS, DaisyUI)  

---

## 1. System Overview and Architecture

InfraPulse is an automated web system designed for maintenance defect reporting, priority queue ranking, domain-based staff routing, and ticket lifecycle tracking.

### System Components:
1. **User Registration and Submission**: Ingests user complaints containing defect photographs, requester contact details, location, and description.
2. **Defect Priority Engine**: Categorizes defects into one of three domains (Structural, Functional, Performance) and calculates priority scores.
3. **Department Priority Queues**: Routes tickets to dedicated queues (Structural, Functional, Performance) sorted by priority score.
4. **Staff Maintenance Interface**: Enables staff members to assign tickets, update status (`Submitted` $\to$ `Assigned` $\to$ `In Progress` $\to$ `Resolved`), and export queue records to CSV.
5. **Notification and Privacy Controls**: In-app notifications for status updates and masking of personal contact details on public ticket views.

---

## 2. Defect Detection and Classification Logic

Upon receiving a photograph, the system identifies the defect type and maps it to a category:

| Defect Type | Category | Category Weight ($W_{\text{cat}}$) | Defect Boost ($B_{\text{defect}}$) |
| :--- | :--- | :---: | :---: |
| **Spalling** | **Structural** | `1.5` | `2.0` |
| **Stagnant Water** | **Functional** | `1.2` | `1.5` |
| **Cracked Tiles** | **Performance** | `1.0` | `1.2` |
| **Paint Peeling** | **Performance** | `1.0` | `1.0` |

*Note*: Within the Performance category, Cracked Tiles is given a higher priority boost than Paint Peeling in accordance with the specification.

---

## 3. Priority Ranking Methodology

Complaints within each department queue are sorted by priority score:

$$\text{Priority Score} = \left( \text{Severity} \times 0.6 + \left(\frac{\text{Extent}}{100}\right) \times 4.0 + B_{\text{defect}} \right) \times W_{\text{cat}}$$

Where:
- $\text{Severity} \in [1.0, 10.0]$: Severity rating of the defect.
- $\text{Extent} \in [0\%, 100\%]$: Defect coverage percentage.
- $B_{\text{defect}}$: Priority boost based on defect type.
- $W_{\text{cat}}$: Category weight coefficient (1.5 for Structural, 1.2 for Functional, 1.0 for Performance).

### Lifecycle Transitions:
- **Active Queue**: Ordered by $\text{Priority Score} \downarrow$, then $\text{Created Date} \uparrow$.
- **Status Flow**: `Submitted` $\to$ `Assigned` $\to$ `In Progress` $\to$ `Resolved`.
- **Queue Removal**: Tickets marked `Resolved` are removed from the active queue standing calculation.

---

## 4. REST API Integration

External classification models can supply defect attributes via the REST API endpoint:

```http
POST /api/v1/complaints/{complaint_id}/classify
Content-Type: application/json

{
  "defect_name": "Spalling",
  "category": "Structural",
  "severity": 8.5,
  "extent": 45.0
}
```

---

## 5. Limitations and Improvement Opportunities

### Current Strengths:
- Deterministic priority scoring based on mathematical formulation.
- Domain-level permission enforcement preventing unauthorized cross-category edits.
- Automatic image format normalization to PNG format.

### Future Improvements:
1. **Multi-Defect Detection**: Handling images with multiple defects by evaluating combined risk factors.
2. **Surface Measurement**: Using dimensional calibration to compute physical defect surface area.
