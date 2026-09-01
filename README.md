# InfraPulse - Photo-Based Defect Priority Maintenance System

**InfraPulse** is a photo-based infrastructure defect detection and priority maintenance web system built for **Takneek PS**. It enables automated ML defect detection, domain category routing (**Structural**, **Functional**, **Performance**), real-time priority queue calculation, user/staff/admin control panels, and status tracking.

---

## 🚀 Quick Setup & Execution

### 1. Clone Repository & Install Dependencies
```bash
git clone https://github.com/Bittu5134/InfraPulse.git
cd InfraPulse

# Create virtual environment & install requirements
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Reset Database & Seed Demo Accounts
```bash
python reset_db.py
```

### 3. Launch Web Application
```bash
PYTHONPATH=. uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Open [http://localhost:8000](http://localhost:8000) in your browser.

---

## 🔑 Demo Account Credentials

| Portal | Email | Password | Role / Domain |
| :--- | :--- | :--- | :--- |
| **Demo User** | `user@infrapulse.org` | `user123` | Public User / Ticket Creator |
| **Structural Staff** | `structural@infrapulse.org` | `staff123` | Structural Maintenance Squad |
| **Functional Staff** | `functional@infrapulse.org` | `staff123` | Functional Maintenance Squad |
| **Performance Staff** | `performance@infrapulse.org` | `staff123` | Performance Maintenance Squad |
| **Administrator** | `admin@infrapulse.org` | `admin123` | System Administrator |

---

## 📐 Priority Queue Mathematical Formula

The priority queue engine ranks active maintenance tickets using the following mathematical formulation:

$$\text{Priority Score} = \left( \text{Severity} \times 0.6 + \left(\frac{\text{Extent}}{100}\right) \times 4.0 + B_{\text{defect}} \right) \times W_{\text{cat}}$$

Where:
- **Category Weights ($W_{\text{cat}}$)**: Structural = `1.5`, Functional = `1.2`, Performance = `1.0`
- **Defect Priority Boosts ($B_{\text{defect}}$)**: Spalling = `2.0`, Stagnant Water = `1.5`, Cracked Tiles = `1.2`, Paint Peeling = `1.0`

*Hierarchy Enforced*: `Spalling` (Structural) > `Stagnant Water` (Functional) > `Cracked Tiles` (Performance) > `Paint Peeling` (Performance).

---

## 🛠️ Key Features

- **Automatic Defect Classification**: Uploaded defect photos (JPG, PNG, WEBP, BMP, etc.) are processed, converted to `.png`, and assigned severity & priority scores.
- **Domain-Specific Staff Queues**: Staff logged into their respective domain (Structural, Functional, Performance) default to viewing their category's active queue.
- **Lifecycle Tracking**: `Submitted` $\to$ `Assigned` $\to$ `In Progress` $\to$ `Resolved`. (Resolved tickets automatically drop off active ranking queues).
- **Privacy Protection**: Personal requester details on single ticket pages (`/ticket/{ticket_id}`) are hidden for unauthorized public visitors.
- **CSV Data Export**: Staff can export active queue data to a `.csv` file directly from the control panel.
- **Real-Time Notification Center**: Bell notifications in the header for staff assignment and ticket status changes.
- **Ticket Activity & Live Chat**: Background polling with Web Audio API sound effects for ticket comments.

---

## 🧪 Running Automated Unit Tests

```bash
PYTHONPATH=. .venv/bin/pytest -v
```

---

## 📁 Technical Documentation Report

A full technical documentation report detailing detection logic, priority math, evaluation results, limitations, and model accuracy suggestions is located in [docs/DOCUMENTATION_REPORT.md](docs/DOCUMENTATION_REPORT.md).
