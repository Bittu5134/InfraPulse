# InfraPulse

**Photo-Based Defect Detection & Priority Maintenance Web System** — Takneek '26 Mid Prep.

A user photographs a building/facility defect; InfraPulse classifies it,
routes it to the right maintenance category's staff queue, ranks it by
priority, and lets users and staff track the live queue.

**Full write-up: [DOCUMENTATION.md](DOCUMENTATION.md)** — approach, detection
logic, priority-ranking method, evaluation results, limitations, and
suggestions for improving accuracy, plus architecture and API reference.
Problem statement: [ReadMe.md](ReadMe.md) / [InfraPulse.pdf](InfraPulse.pdf).

---

## What it does

| Defect (auto-detected from photo) | Category (auto) | Queue priority |
| --- | --- | --- |
| Spalling | Structural | — |
| Stagnant water | Functional | — |
| Cracked tiles | Performance | ranked **above** paint peeling |
| Paint peeling | Performance | |

- **Detection** — MobileNetV2 (ImageNet, frozen) + a small head we trained
  ourselves, running in-process (no external inference API, no
  defect-pretrained checkpoints).
- **Category & routing** — deterministic lookup on the predicted defect, so
  routing can't disagree with the classification.
- **Priority** — fixed defect-type ordering first (cracked tiles > paint
  peeling), then a classical-CV visible-severity score (Canny edge density /
  HSV area) as tie-breaker.
- **Live queues** — one per category, updated in real time over WebSocket;
  status workflow `Submitted → Assigned → In Progress → Resolved`, resolved
  items drop out of the queue automatically.
- **Two portals** — users (submit + track), staff (manage only their
  category's queue).

## Structure

```text
backend/     FastAPI — auth, complaints API, priority queue, WebSocket, in-process ML
frontend/    React (Vite) — separate User and Staff portals
data/        21 held-out real photos for evaluation (never used in training)
notebooks/   earlier Kaggle training notebook, kept for history
```

## Quickstart

### 1-Step Full-Stack Launch (Recommended)

From the project root directory, run:

```bash
npm start
# or: npm run dev
```

This concurrently boots up **both**:
1. **Python FastAPI Backend** on `http://localhost:8005` (API + in-process ML)
2. **React 19 Frontend** on `http://localhost:5173`

---

### Alternative: Running Separately

**Backend:**
```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m app.seed          # optional: create demo accounts (wipes DB)
uvicorn app.main:app --reload --port 8005
```

`head.pt` (the trained classifier) is already committed — no training needed to run. Interactive docs at `http://localhost:8005/docs`.

**Frontend:**
```bash
cd frontend
npm install
npm run dev                 # http://localhost:5173
```

## Demo accounts

`python -m app.seed` creates these — all use password **`demo1234`**:

| Role | Email | Category |
| --- | --- | --- |
| User | `user1@infrapulse.demo` / `user2@…` / `user3@…` | — |
| Staff | `staff.structural@infrapulse.demo` | Structural |
| Staff | `staff.functional@infrapulse.demo` | Functional |
| Staff | `staff.performance1@infrapulse.demo` / `…performance2@…` | Performance |

Users log in at `/`, staff at `/staff/login`.

## Status

End-to-end flow verified locally: auth, photo-upload complaint registration,
in-process classification + auto-routing, classical-CV severity scoring,
per-category live priority queues, full status workflow, WebSocket updates.
All API endpoints and auth/role boundaries pass a manual test sweep (see
[backend/README.md](backend/README.md)).

**Classifier accuracy: 81% (17/21)** on the held-out real-photo set in
`data/`, up from 60% (6/10) for an earlier version trained on large
single-source datasets. The full iteration story — including what *didn't*
work (imbalanced data hurt accuracy; backbone fine-tuning overfit;
search-result titles weren't reliable labels) — is in
[DOCUMENTATION.md §9](DOCUMENTATION.md#9-evaluation-results) and
[backend/app/data/DATASET_NOTES.md](backend/app/data/DATASET_NOTES.md).

**Still to do before submission:** deploy to a public URL (PS requires a
hosted, accessible link); further accuracy work on the `spalling` class if
time allows.
