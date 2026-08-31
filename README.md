# InfraPulse - Photo-Based Defect Detection & Priority Maintenance System

InfraPulse is a fully self-hosted web application built for the **Takneek PS Challenge**. It provides automated complaint registration, photo-based defect classification, live priority queueing across three core domain categories (**Structural**, **Functional**, **Performance**), and separate portals for users and maintenance staff.

---

## Features

1. **User Portal**:
   - Register complaints with Name, Location, Description, and Defect Photograph.
   - Live complaint tracking showing status (`Submitted` → `Assigned` → `In Progress` → `Resolved`), detected defect label, category, and live position in queue.
   - Dynamic real-time sync powered by **HTMX**.

2. **Staff Portals**:
   - Category-specific live priority queues for **Structural**, **Functional**, and **Performance** maintenance teams.
   - Predefined state transitions: `Submitted` → `Assigned` → `In Progress` → `Resolved`.
   - Automatic queue removal when a complaint is marked `Resolved`.

3. **Defect Classification & Priority Rules**:
   - **Structural**: `Spalling` (Highest Category Weight: 3.0)
   - **Functional**: `Stagnant Water` (Category Weight: 2.0)
   - **Performance**: `Cracked Tiles` > `Paint Peeling` (Category Weight: 1.0, Priority: Cracked tiles > Paint peeling)
   - Priority ranking formula combining defect urgency boost, visible severity ($1-10$), and visible defect extent percentage ($0-100\%$).

4. **Self-Hosted & Zero Build Setup**:
   - Python-only stack (**FastAPI** + **Jinja2** + **SQLite** + **HTMX** + **DaisyUI / Tailwind CSS** via CDN). No Node.js or npm build step required.

---

## Setup & Running Locally

### 1. Prerequisites
- Python 3.10 or higher.

### 2. Installation
```bash
# Clone repository
git clone https://github.com/your-org/InfraPulse.git
cd InfraPulse

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Web Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
The website will be accessible at `http://localhost:8000`.

---

## ML Model Integration API

The website exposes REST API endpoints for external defect detection / classification models to post inference results.

### Endpoint:
`POST /api/v1/complaints/{complaint_id}/classify`

### Example Request (cURL):
```bash
curl -X POST "http://localhost:8000/api/v1/complaints/1/classify" \
     -H "Content-Type: application/json" \
     -d '{
           "defect_name": "Spalling",
           "category": "Structural",
           "severity": 8.5,
           "extent": 50.0
         }'
```

### Python Request Example:
```python
import requests

url = "http://localhost:8000/api/v1/complaints/1/classify"
payload = {
    "defect_name": "Cracked Tiles",
    "category": "Performance",
    "severity": 7.0,
    "extent": 35.0
}
response = requests.post(url, json=payload)
print(response.json())
```

Upon receiving the classification payload, the website automatically:
1. Updates the defect label and category on both User and Staff portals.
2. Recalculates the exact priority score.
3. Dynamically re-ranks the category queue in real time.
