# Technical Documentation Report - InfraPulse

**Problem Statement**: Photo-Based Defect Detection & Priority Maintenance System  
**System Name**: InfraPulse  
**Target Environment**: Python Web Application (FastAPI, SQLite, PyTorch, Jinja2, Tailwind CSS)  
**Version**: 3.0.0  

---

## 1. System Overview and Implementation Scope

InfraPulse is an automated web platform designed for facility maintenance defect reporting, objective priority queue ranking, domain-based squad dispatch, and full lifecycle tracking.

### Core Problem Statement Deliverables:
1. **User Defect Submission**: Multi-format photo intake with location and description metadata.
2. **Tri-Category Classification**: Routing into **Structural**, **Functional**, and **Performance** departments.
3. **Mathematical Priority Formulation**: Dynamic computation of priority scores using the weighted formula.
4. **Queue Dispatch & Status Lifecycle**: Step-by-step state transitions (`Submitted` $\to$ `Assigned` $\to$ `In Progress` $\to$ `Resolved`) with removal of resolved tickets from active queues.
5. **Role-Based Portals**: Dedicated interfaces for Users, Department Staff, and System Administrators.

---

## 2. Extra Features & Quality of Life (QoL) Enhancements

| Category | Feature | Technical Implementation | Benefit |
| :--- | :--- | :--- | :--- |
| **Computer Vision** | **PyTorch Vision Model** | EfficientNet-B0 transfer learning with custom head | True pixel-level defect analysis instead of text heuristics |
| **Explainability** | **GradCAM++ Localization** | Attention heatmap & Canny edge density extraction | Objective calculation of physical severity (0–100%) and extent (0–100%) |
| **Evaluation** | **Model Benchmark Center** | Paginated `/test` route with in-memory caching | Side-by-side evaluation against holdout datasets without CPU/RAM overload |
| **User Experience** | **WYSIWYG Markdown Editor** | Toolbar + live preview + `bleach` sanitizer | Rich multi-line formatted defect descriptions (lists, tables, code) |
| **Collaboration** | **Live Discussion Feed** | In-app polling with Web Audio API sound chime | Real-time communication between requesters and operators |
| **Notification** | **In-App Notification Center** | Navbar dropdown with live unread badge polling | Instant alerts on ticket assignments and status updates |
| **Security & RBAC** | **Domain Restriction** | `HTTP 403 Forbidden` checks on cross-category actions | Enforces departmental jurisdiction and prevents accidental reassignment |
| **Data Privacy** | **Contact Masking** | Conditional Jinja2 rendering based on session auth | Protects user email and phone numbers on public views |
| **Reporting** | **Enterprise CSV Export** | Streaming CSV generator across all category queues | Easy reporting and data extraction for facility managers |
| **Design** | **Ergonomic Theme** | Cloudflare-inspired palette with light/dark toggle | High eye comfort with low-glare neutral backgrounds |
| **Portability** | **100% Offline Assets** | Self-hosted Tailwind JS & FontAwesome webfonts | Operates without internet or external CDN dependencies |
| **DevOps** | **Docker & Compose** | Multi-stage Dockerfile with volume persistence | One-command production deployment |

---

## 3. Defect Detection and Priority Ranking Methodology

### Priority Scoring Formulation:
$$\text{Priority Score} = \left( \text{Severity} \times 0.6 + \left(\frac{\text{Extent}}{100}\right) \times 4.0 + B_{\text{defect}} \right) \times W_{\text{cat}}$$

### Parameter Hierarchy:
- **Category Weights ($W_{\text{cat}}$)**:
  - **Structural** = `1.5` (Critical structural safety)
  - **Functional** = `1.2` (Operational disruptions / health hazards)
  - **Performance** = `1.0` (Aesthetic / surface wear)
- **Defect Boosts ($B_{\text{defect}}$)**:
  - **Spalling** = `+2.0` (Highest priority)
  - **Stagnant Water** = `+1.5` (High priority)
  - **Cracked Tiles** = `+1.2` (Medium priority, ranked above paint peeling)
  - **Paint Peeling** = `+1.0` (Standard priority)

---

## 4. Machine Learning & Model Performance

### Test Set Benchmark (241 Holdout Images):
- **Overall Accuracy**: **88.8%**
- **Weighted F1-Score**: **0.89**
- **Macro F1-Score**: **0.814**

#### Confusion Matrix:
```text
Actual \ Predicted     Cracked Tiles   Paint Peeling   Spalling   Stagnant Water
--------------------------------------------------------------------------------
Cracked Tiles (83)          76               0             4            3
Paint Peeling (78)           6              65             3            4
Spalling (75)                1               5            68            1
Stagnant Water (5)           0               0             0            5
```

---

## 5. Verification and Quality Assurance

The system includes automated end-to-end unit and integration tests covering:
1. Priority mathematical scoring hierarchy compliance.
2. User account registration, authentication, and photo defect submission.
3. Staff domain authorization and ticket self-assignment.
4. Administrative staff provisioning and system governance.
5. Model benchmark `/test` route rendering and lazy-loaded evaluation.

All automated tests execute cleanly via `pytest` with 100% pass rate.
