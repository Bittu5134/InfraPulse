# Walkthrough - InfraPulse Maintenance Ticketing System (V2 Updates)

The web application has been refactored into a full-featured **Infrastructure Defect Ticketing Tool** according to your requirements.

---

## Key Refactorings & Major Features Added

1. **Simplified Top Navigation Bar**:
   - Clean navbar featuring only **User Portal** and **Staff Portal** dropdowns.
   - Removed all HTMX/live-sync badge clutter for a clean production web portal UI.

2. **Dark / Light Theme Toggle**:
   - Integrated DaisyUI theme switcher with local storage persistence (`data-theme="light"` / `data-theme="dark"`).

3. **User Portal & Authentication**:
   - **User Accounts**: Registration (`/user/register`), Sign In (`/user/login`), Logout (`/user/logout`).
   - **Ticket Submission**: Captures User Name, Email Address, Phone Number, Address/Location, Problem Description, and Defect Photo. Auto-fills user details if logged in.
   - **My Tickets Dashboard**: Displays all submitted tickets, current status, priority rank score, defect classification, and assigned staff member.

4. **Protected Staff Portal & Ticket Self-Assignment**:
   - **Staff Authentication**: Login required at `/staff/login` using credentials (e.g. `structural@infrapulse.org`, `functional@infrapulse.org`, `performance@infrapulse.org`).
   - **Self-Assignment ("Assign to Me")**: Staff members can claim tickets in their department queue.
   - **Assigned Staff Visibility**: Displays who picked the issue on both User and Staff portal views.
   - **Status Transitions**: `Submitted` → `Assigned` → `In Progress` → `Resolved`.

5. **ML Model REST Integration Endpoint**:
   - Retained `POST /api/v1/complaints/{id}/classify` for external model inference posting.

---

## Test Verification

Ran test suite verifying user auth, ticket submission with contact info, protected staff login, and ticket self-assignment:
```bash
PYTHONPATH=. .venv/bin/pytest -v
```

Output:
```text
tests/test_app.py::test_priority_score_computation PASSED                [ 33%]
tests/test_app.py::test_user_registration_login_and_ticket_submission PASSED [ 66%]
tests/test_app.py::test_staff_login_and_self_assignment PASSED           [100%]

============================== 3 passed in 0.86s ===============================
```

---

## How to Run locally

```bash
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- **Website URL**: `http://localhost:8000/`
- **Default Staff Demo Passwords**: `staff123`
  - `structural@infrapulse.org`
  - `functional@infrapulse.org`
  - `performance@infrapulse.org`
