# System Design Document (HLD / LLD) - InfraPulse

**System Name**: InfraPulse - Defect Detection and Priority Maintenance System  
**Version**: 2.0  
**Stack**: Python (FastAPI), SQLite (Async SQLAlchemy), Jinja2, Tailwind CSS, DaisyUI  

---

## 1. System Overview

### 1.1 Problem Context
Campus and institutional facilities receive diverse maintenance requests daily. Without structured prioritization, urgent structural concerns (such as concrete spalling) can be treated with the same urgency as cosmetic defects (such as paint peeling), creating safety hazards and delayed resolutions.

### 1.2 Objectives
- Automate the routing of submitted defect reports to appropriate department queues: Structural, Functional, or Performance.
- Compute an objective priority score for each complaint based on visible severity, coverage extent, defect type, and domain criticality.
- Enforce predefined status transitions (`Submitted` $\to$ `Assigned` $\to$ `In Progress` $\to$ `Resolved`).
- Restrict staff actions to their assigned department domains.

---

## 2. High-Level Design (HLD)

### 2.1 Architecture Overview

```mermaid
graph TB
    subgraph Client Tier
        UI_User["User Portal (/user)"]
        UI_Staff["Staff Portal (/staff)"]
        UI_Admin["Admin Portal (/admin)"]
    end

    subgraph Application Tier
        Auth["Authentication & Session Layer"]
        UserRouter["User Router"]
        StaffRouter["Staff Router"]
        AdminRouter["Admin Router"]
        APIRouter["API Router"]
        PriorityEngine["Priority Scoring Engine"]
        ImageProcessing["Image Processor (Pillow)"]
    end

    subgraph Data Tier
        DB[("SQLite Database")]
        Storage[("File Storage (/uploads)")]
    end

    UI_User --> UserRouter
    UI_Staff --> StaffRouter
    UI_Admin --> AdminRouter

    UserRouter --> ImageProcessing --> Storage
    UserRouter --> PriorityEngine --> DB
    StaffRouter --> DB
    AdminRouter --> DB
    APIRouter --> DB
```

---

## 3. Low-Level Design (LLD)

### 3.1 Data Schema

```mermaid
erDiagram
    USERS ||--o{ COMPLAINTS : submits
    USERS ||--o{ NOTIFICATIONS : receives
    STAFF ||--o{ COMPLAINTS : assigned_to
    STAFF ||--o{ NOTIFICATIONS : receives
    COMPLAINTS ||--o{ TICKET_COMMENTS : contains

    USERS {
        int id PK
        string name
        string email UK
        string phone
        string password_hash
        datetime created_at
    }

    STAFF {
        int id PK
        string name
        string email UK
        string domain
        string password_hash
        datetime created_at
    }

    COMPLAINTS {
        bigint id PK
        int user_id FK
        string user_name
        string user_email
        string user_phone
        text address
        text description
        string photo_path
        string category
        string defect_name
        float severity
        float extent
        float priority_score
        int assigned_staff_id FK
        string status
        datetime created_at
    }

    TICKET_COMMENTS {
        int id PK
        bigint ticket_id FK
        string sender_name
        string sender_role
        text message
        datetime created_at
    }

    NOTIFICATIONS {
        int id PK
        int user_id FK
        int staff_id FK
        string title
        string message
        string link_url
        boolean is_read
        datetime created_at
    }
```

---

## 4. Priority Queue Mathematical Formulation

Complaints are ordered within department queues by computed priority scores:

$$\text{Priority Score} = \left( \text{Severity} \times 0.6 + \left(\frac{\text{Extent}}{100}\right) \times 4.0 + B_{\text{defect}} \right) \times W_{\text{cat}}$$

### Parameters
- **Severity** $\in [1.0, 10.0]$: Defect severity rating.
- **Extent** $\in [0\%, 100\%]$: Surface defect coverage area.
- **Category Weight ($W_{\text{cat}}$)**:
  - Structural: `1.5`
  - Functional: `1.2`
  - Performance: `1.0`
- **Defect Boost ($B_{\text{defect}}$)**:
  - Spalling: `+2.0`
  - Stagnant Water: `+1.5`
  - Cracked Tiles: `+1.2`
  - Paint Peeling: `+1.0`

---

## 5. Security and Access Control

| Role | Permissions |
| :--- | :--- |
| **Public / Guest** | Submit complaints; view complaints with contact information masked. |
| **Registered User** | Submit complaints, view personal ticket dashboard, participate in ticket comment timeline, and receive status notifications. |
| **Staff Member** | View assigned department queue; assign and update status for tickets within domain; export queue to CSV. Cannot modify tickets outside domain. |
| **Administrator** | Manage staff accounts; view cross-department ticket reports; remove records. |

---

## 6. Functional Capabilities

1. **In-Ticket Comments**: Provides a chronological messaging timeline on ticket detail pages with background polling.
2. **Notification Center**: Navigation dropdown that alerts users and staff to ticket assignment and status changes.
3. **Contact Data Privacy**: Masks personal contact information for visitors without authentication on ticket pages.
4. **Image Format Normalization**: Converts all incoming image uploads to PNG format via Pillow.
5. **CSV Export**: Allows staff and administrators to export queue data for reporting.
6. **Self-Hosted Assets**: Uses local copies of Tailwind, DaisyUI, and FontAwesome for reliable offline operation.
