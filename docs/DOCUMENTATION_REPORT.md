# Technical Documentation Report - InfraPulse

**Problem Statement**: Photo-Based Defect Detection & Priority Maintenance System  
**System Name**: InfraPulse  
**Target Environment**: Python Web Application (FastAPI, SQLite, PyTorch, Jinja2, Tailwind CSS, EasyMDE)  
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

## 2. Exhaustive List of Extra Features & Quality of Life (QoL) Enhancements

| # | Feature | Technical Architecture | Practical Value / Benefit |
| :-: | :--- | :--- | :--- |
| **1** | **PyTorch Deep Learning Model** | EfficientNet-B0 transfer learning with custom classification head | Replaces text heuristics with true pixel visual inference (**88.8% test accuracy**, **0.89 F1**) |
| **2** | **GradCAM++ Visual Explainability** | Layer `backbone.features[-1]` attention heatmaps + Canny edge density | Computes physical Severity (0–100%) and Extent (0–100%) from damage pixels |
| **3** | **Interactive Benchmark Center (`/test`)** | Paginated holdout test route (10/page) with in-memory caching | Live side-by-side evaluation against dataset images with zero CPU/RAM exhaustion |
| **4** | **Embedded EasyMDE WYSIWYG Editor** | Client-side EasyMDE toolbar with side-by-side preview & fullscreen | Rich text formatting (bold, italic, headers, quotes, lists, tables, code) for defect reporting |
| **5** | **Server-Side Safe Markdown Sanitizer** | Python `markdown` library with `bleach` whitelist tag sanitizer | Renders rich typography on ticket details while guaranteeing protection against XSS |
| **6** | **Real-Time Ticket Discussion Feed** | Threaded comment feed with asynchronous background polling | Direct bidirectional collaboration between residents and maintenance crews |
| **7** | **Web Audio API Feedback** | Client-side acoustic audio chime synthesis | Audible feedback when new ticket comments or updates arrive |
| **8** | **In-App Notification Center** | Global navbar notification bell with unread badge polling | Real-time alerts on ticket assignments and status transitions with direct links |
| **9** | **Departmental RBAC & Jurisdiction** | Server-side `HTTP 403 Forbidden` checks on cross-category actions | Enforces strict jurisdictional boundary between Structural, Functional, and Performance staff |
| **10** | **Contact Privacy Redaction** | Conditional Jinja2 rendering based on session auth | Automatically masks personal phone numbers and emails on public views |
| **11** | **Enterprise CSV Export** | Streaming CSV generator with custom domain and status filters | Seamless export for facility audits, external reporting, and compliance records |
| **12** | **Cloudflare-Inspired Ergonomic Theme** | Soft neutral palette (`#f8fafc` / `#0b0f19`) with Cloudflare orange accents | High eye-comfort interface with persistent dark/light theme switching |
| **13** | **Multi-Dimensional Search & Filtering** | Search by ID, address, defect description, status, and severity | Fast lookup and sorting across thousands of queue records |
| **14** | **100% Offline Static Assets** | Bundled Tailwind, FontAwesome webfonts, and EasyMDE in `app/static/vendor/` | Completely air-gapped intranet deployment capability (zero CDN reliance) |
| **15** | **Automatic Image Normalization** | Pillow pipeline validating headers and converting inputs to PNG | Eliminates malicious file extensions and standardizes storage formats |
| **16** | **Production Docker & Compose** | Multi-stage Dockerfile with volume mounting and auto-seeding | Single-command deployment (`docker compose up --build`) |

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

All automated tests execute cleanly via `pytest` with a 100% pass rate.
