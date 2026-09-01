# 🛡️ InfraPulse - Infrastructure Defect Detection & Priority Maintenance System

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![SQLite](https://img.shields.io/badge/Database-SQLite%20Async-003B57.svg)](https://www.sqlite.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-Offline%20Vendor-38B2AC.svg)](https://tailwindcss.com/)

**InfraPulse** is an automated photo-based infrastructure defect detection and priority maintenance platform built for **Takneek PS**. It eliminates manual bottleneck reporting by automatically classifying uploaded defect photographs into dedicated domain queues (**Structural**, **Functional**, **Performance**), mathematically scoring priority rank based on visible severity and coverage extent, and managing end-to-end ticket lifecycle dispatching for campus maintenance squads.

---

## 🏛️ System Architecture

```mermaid
graph TD
    subgraph "Client Tier (100% Self-Hosted Frontend)"
        U[Public User Portal]
        S[Domain Staff Portal]
        A[Admin Portal]
    end

    subgraph "Backend Tier (FastAPI + Async SQLAlchemy)"
        Router[FastAPI Routing Engine]
        Auth[Session Auth & PBKDF2 Hashing]
        Pillow[Pillow Image Converter & Validator]
        Priority[Priority Calculation Engine]
        Notifier[Notification Service]
    end

    subgraph "Data Tier"
        DB[(SQLite / PostgreSQL Database)]
        Storage[(Local File System /uploads)]
    end

    subgraph "External ML Pipeline (Optional Integration)"
        ML[PyTorch / YOLOv8 REST API Client]
    end

    U -->|Submit Defect Photo & Details| Router
    S -->|Claim & Resolve Domain Tickets| Router
    A -->|Staff & Ticket Governance| Router
    Router --> Auth
    Router --> Pillow
    Pillow --> Storage
    Router --> Priority
    Priority --> DB
    Router --> Notifier
    Notifier --> DB
    ML -.->|POST /api/v1/complaints/classify| Router
```

---

## 🔄 Defect Routing & Priority Queue State Machine

```mermaid
stateDiagram-v2
    [*] --> Upload: User Submits Photo & Location
    Upload --> ImageProcessing: Pillow Validates & Converts to PNG
    ImageProcessing --> DefectClassification: Automatic Defect Assessment

    state DefectClassification {
        [*] --> CategoryMapping
        CategoryMapping --> Structural: Spalling (Weight: 1.5, Boost: +2.0)
        CategoryMapping --> Functional: Stagnant Water (Weight: 1.2, Boost: +1.5)
        CategoryMapping --> Performance_Tiles: Cracked Tiles (Weight: 1.0, Boost: +1.2)
        CategoryMapping --> Performance_Paint: Paint Peeling (Weight: 1.0, Boost: +1.0)
    }

    DefectClassification --> PriorityFormula: Compute Mathematical Priority Score
    PriorityFormula --> ActiveQueue: Route to Category's Dedicated Queue

    state "Ticket Lifecycle" as Lifecycle {
        ActiveQueue --> Submitted: Placed in Live Ranked Queue
        Submitted --> Assigned: Staff Claims Ticket (Assign to Me)
        Assigned --> In_Progress: Work Commences
        In_Progress --> Resolved: Defect Fixed
    }

    Resolved --> Archive: Automatically Dropped from Active Queue
    Archive --> [*]
```

---

## 📐 Priority Queue Mathematical Formulation

As required by the specification, complaints are ranked dynamically within each domain queue using an objective mathematical formulation accounting for visible defect severity, surface extent coverage, defect hazard hierarchy, and domain criticality:

$$\text{Priority Score} = \left( \text{Severity} \times 0.6 + \left(\frac{\text{Extent}}{100}\right) \times 4.0 + B_{\text{defect}} \right) \times W_{\text{cat}}$$

### 1. Domain Category Criticality Weight ($W_{\text{cat}}$)
- **Structural**: `1.5` (Critical structural safety integrity)
- **Functional**: `1.2` (Plumbing, drainage, stagnant water issues)
- **Performance**: `1.0` (Aesthetics, surface finishes, floor tiles)

### 2. Defect Hierarchy Boost ($B_{\text{defect}}$)
| Defect Type | Domain Category | Defect Priority Boost ($B_{\text{defect}}$) | Description |
| :--- | :--- | :---: | :--- |
| **Spalling** | **Structural** | `+2.0` | Concrete surface peeling exposing internal rebar |
| **Stagnant Water** | **Functional** | `+1.5` | Flooding, drainage blockage, and hygiene risk |
| **Cracked Tiles** | **Performance** | `+1.2` | Tripping hazard and flooring breakage |
| **Paint Peeling** | **Performance** | `+1.0` | Wall coating degradation and aesthetic defect |

> **Hierarchy Guarantee**: $\text{Spalling (Structural)} > \text{Stagnant Water (Functional)} > \text{Cracked Tiles (Performance)} > \text{Paint Peeling (Performance)}$.

---

## 🗄️ Relational Database Schema Overview

InfraPulse utilizes an asynchronous SQLite database defined in [schema.sql](schema.sql):

```mermaid
erDiagram
    USERS ||--o{ COMPLAINTS : submits
    USERS ||--o{ NOTIFICATIONS : receives
    STAFF ||--o{ COMPLAINTS : claims
    STAFF ||--o{ NOTIFICATIONS : receives
    COMPLAINTS ||--o{ TICKET_COMMENTS : contains

    USERS {
        int id PK
        string name
        string email UK
        string phone
        string password_hash
        timestamp created_at
    }

    STAFF {
        int id PK
        string name
        string email UK
        string domain
        string password_hash
        timestamp created_at
    }

    COMPLAINTS {
        bigint id PK "10-Digit Random ID"
        int user_id FK
        string user_name
        string user_email
        string address
        string photo_path
        string category "Structural | Functional | Performance"
        string defect_name
        float severity "1.0 - 10.0"
        float extent "0.0% - 100.0%"
        float priority_score
        int assigned_staff_id FK
        string status "Submitted | Assigned | In Progress | Resolved"
        timestamp created_at
    }

    TICKET_COMMENTS {
        int id PK
        bigint ticket_id FK
        string sender_name
        string sender_role
        text message
        timestamp created_at
    }

    NOTIFICATIONS {
        int id PK
        int user_id FK
        int staff_id FK
        string title
        string message
        string link_url
        boolean is_read
        timestamp created_at
    }
```

---

## 🚀 Quick Start & Deployment

### Option A: Running with Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/Bittu5134/InfraPulse.git
cd InfraPulse

# Build and start containerized application
docker compose up --build -d
```
The application will be accessible at `http://localhost:8000`.

### Option B: Local Python Virtual Environment

```bash
# 1. Clone & create virtual environment
git clone https://github.com/Bittu5134/InfraPulse.git
cd InfraPulse
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Reset database and seed default demo accounts
python reset_db.py

# 4. Launch development server
PYTHONPATH=. uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🔑 Pre-Seeded Demo Accounts

| Portal | Email | Password | Role / Domain |
| :--- | :--- | :--- | :--- |
| **User Portal** | `user@infrapulse.org` | `user123` | Registered Public User |
| **Structural Staff** | `structural@infrapulse.org` | `staff123` | Structural Maintenance Squad |
| **Functional Staff** | `functional@infrapulse.org` | `staff123` | Functional Maintenance Squad |
| **Performance Staff** | `performance@infrapulse.org` | `staff123` | Performance Maintenance Squad |
| **Admin Portal** | `admin@infrapulse.org` | `admin123` | System Administrator |

---

## 🔌 External ML Inference REST API

External models (e.g. YOLOv8, Faster R-CNN, or ResNet) can push inference results directly into the priority queue via the REST API:

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

**Response**:
```json
{
  "status": "success",
  "complaint_id": 8492019482,
  "category": "Structural",
  "defect_name": "Spalling",
  "priority_score": 13.95,
  "message": "Complaint classified and priority score updated"
}
```

---

## 🧪 Automated Testing Suite

InfraPulse includes full end-to-end integration and unit tests covering priority math, image conversion, user registration, staff claiming, domain security restrictions, and admin oversight:

```bash
PYTHONPATH=. .venv/bin/pytest -v
```

---

## 📁 Repository Structure

```text
InfraPulse/
├── Dockerfile                  # Production container definition
├── docker-compose.yml          # Container orchestration configuration
├── schema.sql                  # Relational database DDL schema & indexes
├── reset_db.py                 # Clean database reset & seeding utility
├── requirements.txt            # Python dependencies
├── app/
│   ├── main.py                 # FastAPI application factory & routes
│   ├── config.py               # Weights, boosts, & environment configuration
│   ├── database.py             # Asynchronous SQLAlchemy engine & sessionmaker
│   ├── models.py               # SQLAlchemy ORM models
│   ├── auth.py                 # Password hashing & session protection
│   ├── priority_queue.py       # Mathematical priority engine & queue sorters
│   ├── routers/
│   │   ├── user.py             # User registration, submission, & ticket tracking
│   │   ├── staff.py            # Staff queue management, actions, & CSV export
│   │   ├── admin.py            # Administrative staff and ticket oversight
│   │   └── api.py              # REST API endpoints (comments, notifs, ML ingest)
│   ├── static/
│   │   ├── vendor/             # 100% self-hosted Tailwind, DaisyUI & FontAwesome
│   │   └── uploads/            # Standardized PNG defect photo storage
│   └── templates/              # Jinja2 responsive HTML templates
├── docs/
│   ├── InfraPulse.pdf          # Official problem statement specification
│   ├── DOCUMENTATION_REPORT.md # Technical documentation report deliverable
│   └── walkthrough.md          # Step-by-step feature walkthrough log
└── tests/
    └── test_app.py             # Pytest automated test suite
```

---

## ⚖️ License & Attribution
Developed for the **Takneek PS** Infrastructure Defect Priority Maintenance Web System competition. Built with 100% self-contained, open-source libraries.
