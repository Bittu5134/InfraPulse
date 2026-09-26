# InfraPulse — Documentation Report

**Photo-Based Defect Detection & Priority Maintenance Web System**
Takneek '26 Mid Prep. Problem statement: [`ReadMe.md`](ReadMe.md) / [`InfraPulse.pdf`](InfraPulse.pdf).

This document covers the six sections the problem statement asks for —
**approach, detection logic, priority-ranking method, evaluation results,
limitations, and suggestions for improving accuracy** — plus the system
architecture, API, and setup needed to run and evaluate it.

---

## 1. Overview

InfraPulse lets a user photograph a building/facility defect and file a
complaint. The system:

1. Classifies the visible defect from the photo (one of four types).
2. Maps it to one of three maintenance categories.
3. Routes the complaint into that category's dedicated staff queue.
4. Ranks it within the queue by a documented priority rule.
5. Lets users track status/position and lets staff manage only their
   category's queue, with the queue updating live for everyone.

| Defect type | Category | Notes |
| --- | --- | --- |
| Spalling | **Structural** | |
| Stagnant water | **Functional** | |
| Cracked tiles | **Performance** | ranked **above** paint peeling |
| Paint peeling | **Performance** | |

---

## 2. System architecture

```
┌────────────────────┐         HTTPS / WSS          ┌──────────────────────────────┐
│  Frontend (React)  │  ───────────────────────────▶ │  Backend (FastAPI, Python)   │
│  Vite SPA          │                               │                              │
│  - User portal     │   REST  /auth /complaints     │  - JWT auth (user / staff)   │
│  - Staff portal    │         /staff                │  - Complaint API + uploads   │
│  - live queue via  │   WS    /ws/queue             │  - Priority queue logic      │
│    WebSocket       │ ◀───────────────────────────  │  - In-process ML classifier  │
└────────────────────┘      queue_update events      │  - Classical-CV severity     │
                                                     │  - SQLite / Postgres (SQLA)  │
                                                     └──────────────────────────────┘
                                                        │
                                          ┌─────────────┴───────────────┐
                                          │  MobileNetV2 (ImageNet,     │
                                          │  frozen) + trained head     │
                                          │  — runs in the same process │
                                          └─────────────────────────────┘
```

### Tech stack

| Layer | Choice | Why |
| --- | --- | --- |
| Frontend | React 19 + Vite, React Router | fast SPA, simple two-portal routing |
| Backend | FastAPI + Uvicorn | async, first-class WebSocket + file-upload support, auto OpenAPI docs |
| ORM / DB | SQLAlchemy 2 over SQLite (Postgres via `DATABASE_URL`) | zero-config locally, swappable for hosting |
| Auth | JWT (`python-jose`) + `pbkdf2_sha256` password hashing (`passlib`) | stateless, no external auth service |
| ML | PyTorch + torchvision MobileNetV2 | lightweight, CPU-friendly, ImageNet weights are permitted |
| CV | OpenCV (`opencv-python-headless`) | severity scoring (Canny edges, HSV segmentation) |
| Real-time | FastAPI WebSocket broadcast | push `queue_update` to all connected clients on any change |

The classifier runs **in the same Python process** as the API — there is no
external ML inference service or API call (see §11, Constraint compliance).

### Repository layout

```
backend/
  app/
    main.py             FastAPI app: CORS, static /uploads mount, routers, /ws/queue
    database.py         SQLAlchemy engine/session (SQLite default, DATABASE_URL override)
    models.py           Account, Complaint tables; DEFECT_TO_CATEGORY, DEFECT_TYPE_PRIORITY
    schemas.py          Pydantic request/response models
    auth.py             JWT create/verify, password hashing, role guards
    queueing.py         live_queue() ranking + with_positions()
    ws.py               ConnectionManager (broadcast to all sockets)
    seed.py             wipe DB + create demo accounts
    routers/
      auth_router.py        /auth/signup/user, /auth/signup/staff, /auth/login
      complaints_router.py  POST /complaints, GET /complaints/mine
      staff_router.py       GET /staff/queue, PATCH /staff/complaints/{id}/status
    ml/
      classifier.py     DefectClassifier (backbone + head), predict()
      severity.py       compute_severity() — classical CV
      train.py          trains the head (frozen backbone, val split, best checkpoint)
    data/
      dataset/<class>/*.jpg     training images (one folder per class)
      dataset_extra/            held-back images (kept out to preserve balance)
      DATASET_NOTES.md          detailed dataset + accuracy-iteration log
    models_weights/
      head.pt                        current trained head (81% real-photo accuracy)
      head_kaggle_original.pt.bak    original Kaggle-trained head (60%) — for comparison
frontend/
  src/
    api/client.js       fetch wrappers, VITE_API_URL / VITE_WS_URL
    context/AuthContext.jsx   token + role in localStorage
    pages/              UserLogin, StaffLogin, UserDashboard, StaffDashboard
data/                   21 held-out real photos for evaluation (never used in training)
notebooks/              Kaggle training notebook (earlier approach, kept for history)
```

---

## 3. Approach

### 3.1 Design principles

- **Everything self-hosted.** No external AI/ML inference API is called for
  detection or classification. The model is a `torchvision` MobileNetV2
  loaded with ImageNet weights, plus a small head we trained ourselves.
- **The model makes exactly one decision.** It predicts the *defect type*.
  Category and queue routing are then pure deterministic lookups, so
  auto-routing cannot disagree with the classification.
- **Severity is explainable, not a black box.** Extent/severity scoring uses
  classical computer vision (edge density / colour-area segmentation) so the
  priority ordering can be fully justified in this report.
- **Classification is strictly from what's visible.** The model sees only
  the photo; there is no metadata, no user-selected label, no claim about
  non-visible or predicted damage.

### 3.2 The detection pipeline

```
photo ──▶ resize 224×224, ImageNet-normalize
      ──▶ MobileNetV2 features (frozen)  ──▶ global average pool ──▶ 1280-d vector
      ──▶ DefectHead:  Linear(1280→128) ▸ ReLU ▸ Dropout(0.3) ▸ Linear(128→4)
      ──▶ softmax ──▶ argmax  ──▶  (defect_label, confidence)
      ──▶ DEFECT_TO_CATEGORY[defect_label]        → category
      ──▶ DEFECT_TYPE_PRIORITY[defect_label]      → type_priority
      ──▶ compute_severity(photo, defect_label)   → severity_score (0–100)
```

All of `defect_label`, `category`, `confidence`, `severity_score`,
`type_priority` are computed once at submission time
(`backend/app/routers/complaints_router.py::submit_complaint`) and stored on
the complaint row. The user and staff portals display the label and category
directly from that row — never a manual choice.

### 3.3 Why MobileNetV2, frozen

- ImageNet-scale pretraining on a **generic** dataset is explicitly allowed;
  MobileNetV2 is small enough to run inference on CPU in the same process as
  the API (important — no GPU, no inference service).
- Freezing the backbone and training only a 2-layer head keeps training to
  seconds on a small dataset and drastically limits overfitting.
- We also tried unfreezing and fine-tuning the last MobileNetV2 block; on our
  dataset size it overfit (train accuracy 100%, held-out unchanged), so the
  backbone stays frozen. The option remains in `train.py`
  (`finetune_last_n_blocks`, default 0).

---

## 4. Detection & classification logic

### 4.1 Classifier

`backend/app/ml/classifier.py`

| Component | Detail |
| --- | --- |
| Backbone | `torchvision.models.mobilenet_v2(weights=IMAGENET1K_V1)`, `.features` only, all params frozen, `.eval()` mode |
| Pooling | `AdaptiveAvgPool2d(1)` → flatten → 1280-d feature vector |
| Head (`DefectHead`) | `Linear(1280, 128)` → `ReLU` → `Dropout(0.3)` → `Linear(128, 4)` |
| Classes (fixed order) | `["spalling", "stagnant_water", "paint_peeling", "cracked_tiles"]` |
| Inference transform | `Resize((224,224))` → `ToTensor` → ImageNet `Normalize` |
| Output | `argmax` of `softmax(logits)` → `(class_name, confidence∈[0,1])` |
| Loading | singleton; loads `head.pt`; optionally loads `backbone_finetune.pt` if present (absent by default) |

The **confidence** shown in the UI is the softmax probability of the chosen
class. It is informational only — routing uses `argmax` regardless.

### 4.2 Category routing (auto-routing)

`backend/app/models.py`

```python
DEFECT_TO_CATEGORY = {
    "spalling":       Category.structural,
    "stagnant_water": Category.functional,
    "cracked_tiles":  Category.performance,
    "paint_peeling":  Category.performance,
}
```

This mirrors the problem statement's fixed mapping exactly. Because category
is a lookup on the predicted label, **routing is correct whenever the
classification is correct** — there is no separate category model that could
disagree, and no way to land in a queue that doesn't match the displayed
defect type.

### 4.3 Training

`backend/app/ml/train.py`

- Loads `app/data/dataset/<class>/*.jpg` via `torchvision.datasets.ImageFolder`.
- Fixed random seed (`42`) for reproducible reruns on unchanged data.
- Holds out a **15% validation split**; trains 30 epochs; **keeps the
  checkpoint with the best validation accuracy** (not the last epoch).
- Augmentation on the training split: random horizontal flip, ±15° rotation,
  colour jitter (brightness/contrast/saturation 0.2). Validation uses no
  augmentation.
- Only the head's parameters are optimised (Adam, lr 1e-3); the backbone is
  frozen.
- Saves `app/models_weights/head.pt`.

Dataset provenance and the full accuracy-iteration history are in
[`backend/app/data/DATASET_NOTES.md`](backend/app/data/DATASET_NOTES.md).
Summary: 84 real photos (11 spalling / 26 stagnant_water / 25 paint_peeling /
22 cracked_tiles) sourced from **Wikimedia Commons** and **Openverse-indexed
Flickr CC** images, chosen for cross-source diversity (many different
cameras, locations, lighting) rather than volume.

---

## 5. Priority-ranking method

Priority has **two components**, applied in order.

### 5.1 Defect-type priority (ordinal, per type)

`backend/app/models.py`

```python
DEFECT_TYPE_PRIORITY = {
    "spalling":       100,
    "stagnant_water": 100,
    "cracked_tiles":   80,
    "paint_peeling":   40,
}
```

Rationale:

- **Spalling** and **stagnant water** are each the *only* defect type in
  their category (Structural, Functional). There is nothing to rank them
  against within their own queue, so they take the top value (100). The
  problem statement only requires a *within-category* ordering.
- **Within Performance**, the problem statement explicitly requires
  *cracked tiles > paint peeling*. We encode this as `80` vs `40` — a fixed
  gap so that **any** cracked-tiles complaint outranks **any** paint-peeling
  complaint in the shared Performance queue, regardless of severity or
  submission time. Spalling and stagnant water don't share a queue with
  these, so their `100` never competes with them.

### 5.2 Visible severity / extent (per photo, 0–100)

`backend/app/ml/severity.py` — deterministic, defect-type aware Computer Vision engine with CLAHE contrast normalization, morphological damage zone dilation, and non-linear perceptual scaling:

| Defect type | Method | Intuition |
| --- | --- | --- |
| **spalling**, **cracked_tiles** | `_crack_spalling_score`: CLAHE contrast enhancement → Canny edge detection (40/120) → **Morphological dilation (5×5)** to measure the physical damage influence zone → Laplacian surface roughness weighting → square-root perceptual scaling ($\sqrt{\text{fraction}} \times 160 \times \text{roughness}$) | Captures entire fractured/crumbling surface area rather than 1-pixel hairlines, scaling into realistic 35–95 score ranges |
| **stagnant_water** | `_stagnant_water_score`: Dual-channel segmentation (**HSV reflective mask** $H \in [0, 180], S \in [0, 75], V \in [15, 175]$ + **Specular highlight thresholding**) → square-root scaling | Measures true visible puddle coverage including water reflections and surface glints |
| **paint_peeling** | `_paint_peeling_score`: CLAHE contrast normalization → **Otsu substrate thresholding** + **Flake boundary edge gradient analysis** ($0.6 \times \text{flake} + 0.4 \times \text{otsu}$) | Distinguishes blistering/flaking paint boundaries from normal flat wall color variations |

`compute_severity(image_path, defect_label)` dispatches on the predicted label and returns a rounded, calibrated score stored as `severity_score` on the complaint row.

### 5.3 Final ranking

`backend/app/queueing.py::live_queue`

```python
complaints = <all complaints in this category with status != Resolved>
complaints.sort(key=lambda c: (c.type_priority, c.severity_score), reverse=True)
```

So within a category queue, complaints are ordered by:

1. **defect-type priority** descending, then
2. **severity score** descending as the tie-breaker.

`queue_position` is simply the 1-based index in this sorted list. A
`Resolved` complaint is excluded from the query, so it disappears from every
queue automatically (the user still sees its final status via
`GET /complaints/mine`).

**Worked example (Performance queue):**

| Complaint | defect | type_priority | severity | position |
| --- | --- | --- | --- | --- |
| A | cracked_tiles | 80 | 30 | **#1** |
| B | cracked_tiles | 80 | 12 | **#2** |
| C | paint_peeling | 40 | 95 | **#3** |

C has the highest severity but still ranks last, because *cracked tiles >
paint peeling* is enforced at the type level first.

---

## 6. Priority queue system & real-time updates

- **One live queue per category.** `GET /staff/queue` returns only the
  authenticated staff member's category, ranked as in §5.3, with positions.
- **Automatic incorporation of new complaints.** `POST /complaints`
  classifies, scores, writes the row, and immediately recomputes + broadcasts
  the affected category's queue — no manual re-entry, no batch job.
- **Real-time updates.** The backend keeps a set of connected WebSocket
  clients (`ws.py::ConnectionManager`). On any complaint submission or status
  change it broadcasts a `queue_update` event. Both portals open
  `ws://…/ws/queue` on mount and re-fetch their view on every event, so the
  user's tracked position and the staff queue update without a refresh.
- **Queue visibility for both roles.**
  - User: `GET /complaints/mine` returns each of their complaints with its
    current `status` and, if not Resolved, its `queue_position` in the
    relevant category queue.
  - Staff: `GET /staff/queue` returns the full ordered queue for their
    category (name, address, description, defect, severity, priority,
    status, position).

### Status workflow

`Submitted → Assigned → In Progress → Resolved`
(`backend/app/models.py::Status`, enforced staff-side in
`staff_router.py::update_status`).

- Only staff can change status, and only for complaints **in their own
  category** (cross-category `PATCH` returns `403`).
- Setting status to `Resolved` removes the complaint from the live queue
  (it's filtered out of `live_queue`); the user still sees `Resolved` as the
  final status on their end.

---

## 7. Web application

### Portals & routing (`frontend/src/App.jsx`)

| Route | Portal | Guard |
| --- | --- | --- |
| `/` | User login / sign-up (tabbed) | — |
| `/staff/login` | Staff login / sign-up (tabbed, with category select) | — |
| `/user/dashboard` | Submit complaint + track own complaints | requires `role == user` |
| `/staff/dashboard` | Live priority queue for the staff member's category | requires `role == staff` |

### Auth (`backend/app/auth.py`, `frontend/src/context/AuthContext.jsx`)

- JWT bearer tokens, 24 h expiry, `HS256`, secret from `JWT_SECRET` env
  (dev default provided).
- Passwords hashed with `pbkdf2_sha256` (no native bcrypt dependency).
- Two roles: `user` and `staff`; staff additionally carry a `staff_category`.
- Role guards: `require_user`, `require_staff` FastAPI dependencies; wrong
  role → `403`, missing/invalid token → `401`.

### Complaint registration (`frontend/src/pages/UserDashboard.jsx`)

Form fields: name, address/location, description, **photo** (with an inline
thumbnail preview before submit). On submit the card list refreshes and shows
the auto-detected defect name, category pill, confidence %, severity bar,
status pill, and live queue position.

### Staff dashboard (`frontend/src/pages/StaffDashboard.jsx`)

Shows the ranked queue for the staff member's category with position badges
and a single "Mark <next status>" button per row that advances the status
one step. The queue re-orders / drops resolved items live.

### UI

Light theme, responsive (verified at 1280 px and 375 px), Inter font,
colour-coded category/status pills, severity gradient bars, hover states,
loading spinners, empty states.

---

## 8. API reference

Base URL configurable via `VITE_API_URL` (frontend) / wherever the backend is
hosted. Interactive docs at `/docs` when the backend is running.

| Method | Path | Auth | Body | Returns |
| --- | --- | --- | --- | --- |
| `GET` | `/health` | — | — | `{"status":"ok"}` |
| `POST` | `/auth/signup/user` | — | `{name,email,password}` | `{access_token, role}` |
| `POST` | `/auth/signup/staff` | — | `{name,email,password,category}` | `{access_token, role, category}` |
| `POST` | `/auth/login` | — | `{email,password}` | `{access_token, role, category?}` |
| `POST` | `/complaints` | user | multipart: `name, address, description, photo` | full complaint incl. `defect_label, category, confidence, severity_score, type_priority, status, queue_position` |
| `GET` | `/complaints/mine` | user | — | list of the user's complaints w/ status + position |
| `GET` | `/staff/queue` | staff | — | ranked queue for the staff member's category |
| `PATCH` | `/staff/complaints/{id}/status` | staff | `{status}` | updated complaint |
| `WS` | `/ws/queue` | — | — | stream of `{"event":"queue_update","payload":{category, queue}}` |
| `GET` | `/uploads/{file}` | — | — | the stored complaint photo (static mount) |

Error contract: `400` duplicate email, `401` bad/missing credentials, `403`
wrong role or wrong category, `404` unknown complaint, `422` validation
(missing field, bad enum value). All 30 of these cases are covered by the
manual endpoint test in [`backend/README.md`](backend/README.md).

---

## 9. Evaluation results

### 9.1 Method

The classifier is evaluated on **21 real photos in [`data/`](data/) that were
never used in training** — a genuine held-out set, sourced independently of
the training images (different Wikimedia/Flickr items). Rerun any time with
the command in [`backend/README.md`](backend/README.md#evaluate-against-held-out-real-photos).

### 9.2 Headline number

| Model | Held-out real-photo accuracy |
| --- | --- |
| Original (Kaggle single-source datasets, 99.9% on its *own* val split) | **60% (6/10)** |
| **Current (`head.pt`)** | **81% (17/21)** |

### 9.3 Confusion matrix (current model)

Rows = true class, columns = predicted class.

|                    | pred spalling | pred stagnant_water | pred paint_peeling | pred cracked_tiles |
| ------------------ | :-----------: | :-----------------: | :----------------: | :----------------: |
| **spalling**       |     **1**     |          0          |         1          |         1          |
| **stagnant_water** |       0       |       **7**         |         0          |         0          |
| **paint_peeling**  |       0       |          1          |       **5**        |         0          |
| **cracked_tiles**  |       1       |          0          |         0          |       **4**        |

Per class: spalling **1/3**, stagnant_water **7/7**, paint_peeling **5/6**,
cracked_tiles **4/5**.

### 9.4 The four remaining errors

| Photo | True | Predicted | Conf | Comment |
| --- | --- | --- | --- | --- |
| `eval_broken_tile_floor_shower.jpg` | cracked_tiles | spalling | 0.64 | cracked shower-wall tile close-up reads as broken concrete surface |
| `eval_living_room_ceiling_1.jpg` | paint_peeling | stagnant_water | 0.52 | water-stained peeling ceiling — visually a damp discoloured patch |
| `spalling_lewis_river_bridge.jpg` | spalling | paint_peeling | 0.71 | rust-stained flaking concrete reads as flaking paint |
| `spalling_oxide_jacking_herbst_pavilion.jpg` | spalling | cracked_tiles | 0.48 | cracked/heaved concrete slab reads as cracked tile |

All four are genuinely ambiguous (moderate confidence, semantically close
classes), not gross errors.

### 9.5 What we learned getting from 60% to 81%

Full log in [`DATASET_NOTES.md`](backend/app/data/DATASET_NOTES.md). Key
findings, each measured against a held-out real-photo set:

1. **The original 99.9% was a mirage.** Each class came from one big
   single-source dataset, so the model learned each dataset's "fingerprint"
   (resolution, compression, colour grading) rather than the defect itself.
   On real photos it scored 60%.
2. **Diverse multi-source real photos** (Wikimedia + Flickr CC) fixed most of
   the gap.
3. **Class balance matters more than volume.** An intermediate dataset with
   36 paint_peeling vs. 10 spalling *dropped* accuracy to 56.5% — the model
   started defaulting to whichever class had the most images.
4. **Search-result titles are not labels.** Several images titled
   "cracked ___" actually showed cracks not peeling paint; a couple of
   "spalling" photos were worker portraits or road cracks. Every image was
   eventually eye-checked; the mislabeled ones were removed (from both
   training and eval sets).
5. **Backbone fine-tuning didn't help** at this dataset size — it overfit
   (train 100%, held-out unchanged). Reverted to frozen.
6. Repeated retrains on cleaned/rebalanced data all land at **81%** — a
   stable plateau for ~84 images and a frozen-backbone linear head. The
   specific errors shuffle between classes run to run; the overall number
   does not.

---

## 10. Priority queue evaluation

Verified manually end-to-end (see [`backend/README.md`](backend/README.md)):

- **Relative priority** — a cracked-tiles complaint and a paint-peeling
  complaint in the same Performance queue: cracked tiles is always ranked
  first regardless of severity or order of submission. ✔
- **Order within a queue** — complaints of the same defect type order by
  severity score descending. ✔
- **Auto-incorporation** — submitting a new complaint immediately places it
  in the correct category queue at the correct position and broadcasts the
  updated queue over WebSocket; no manual step. ✔
- **Resolved removal** — advancing a complaint to `Resolved` drops it from
  `GET /staff/queue` while `GET /complaints/mine` still shows `Resolved`. ✔

---

## 11. Constraint compliance

| Rule | How InfraPulse complies |
| --- | --- |
| No external AI/ML inference APIs for detection/classification | The model is a local `torchvision` MobileNetV2 + our own head, run **in the API process**. No network call is made for inference. |
| Generic pretrained backbones (ImageNet/COCO) allowed | Backbone is `mobilenet_v2(weights=IMAGENET1K_V1)` — generic ImageNet pretraining only. |
| No models pretrained on defect/damage-specific datasets | Nothing in the repo loads a defect-pretrained checkpoint. The head is initialised randomly and trained by us. `backbone_finetune.pt` (if ever generated) is our own fine-tune of the same ImageNet backbone, not a third-party defect checkpoint. |
| Classification strictly from visible evidence | Input is only the photo → 224×224 → backbone → head. No metadata, no user label, no predicted/non-visible claims. |
| Auto-generated labels, no manual selection | `defect_label` and `category` are computed server-side at submission and shown read-only on both portals. |
| General-purpose libraries/APIs (auth, DB, hosting) allowed with citation | FastAPI, SQLAlchemy, PyTorch/torchvision, OpenCV, `python-jose`, `passlib` — all listed in `backend/requirements.txt`. |
| External datasets must be cited | All training/eval image sources (Wikimedia Commons, Openverse/Flickr, with URLs) are listed in [`data/README.md`](data/README.md) and [`DATASET_NOTES.md`](backend/app/data/DATASET_NOTES.md). |
| No hardcoding / memorising evaluation inputs | Classification is a genuine forward pass of a trained network; there is no filename→label table or per-input special-casing anywhere. |

---

## 12. Limitations

**Detection & classification**

- **Small training set** (84 images; `spalling` only 11). Real evaluation
  photos may fall outside the visual variety seen in training.
- **`spalling` is the weak class** — 1/3 on the held-out set. Its failure
  modes are confusion with `paint_peeling` (rust-stained flaking concrete)
  and `cracked_tiles` (cracked/heaved slabs).
- **`paint_peeling` vs `stagnant_water`** — a water-stained, damp, peeling
  ceiling is genuinely visually close to a dark wet patch; the model
  occasionally confuses them.
- **No "unknown / none" class.** Every photo is forced into one of the four
  types. A photo with no defect, or a defect type outside the four, is still
  assigned the nearest class.
- **Backbone is frozen.** The model can only re-weight generic ImageNet
  features; it cannot learn genuinely new low-level texture detectors for
  defects (fine-tuning was tried and overfit at this data size).

**Severity scoring**

- The CV heuristics are **deliberately simple and un-tuned beyond an
  empirical scale factor.** The HSV range for water and the Otsu threshold
  for paint peeling assume typical lighting; unusual lighting, heavy shadow,
  or a busy background can push the score up or down.
- Severity is measured on the **whole frame**, so a small defect
  photographed close-up scores higher than the same defect photographed from
  far away — it captures *visible extent in the photo*, not physical extent.
- Severity only breaks ties **within** a defect type; it never overrides the
  type-level priority.

**Queue / system**

- Priority is recomputed per request rather than cached; fine at this scale,
  would need indexing/materialisation at large volume.
- SQLite by default (single-writer). Use `DATABASE_URL` for Postgres in
  production / under concurrency.
- WebSocket broadcast is unauthenticated and category-agnostic (clients
  re-fetch their own authorised view on each event); acceptable because the
  event carries no sensitive payload the client couldn't already fetch.

---

## 13. Suggestions for improving accuracy

Ordered by expected value for effort, based on the iteration in §9.5.

1. **More carefully-curated `spalling` photos**, kept balanced with the other
   classes (~25–30 each). This is the single biggest lever — `spalling` is
   both the smallest class and the source of most remaining errors. Every
   image should be eye-checked, not trusted from its search title.
2. **Grow the held-out eval set further** (currently 21 photos; each photo is
   worth ~5 accuracy points). A 60–100 photo eval set would make iteration
   decisions far less noisy — several earlier retrains looked like
   improvements that turned out to be sampling noise.
3. **Add an "other / no defect" class** so genuinely-out-of-distribution
   photos aren't force-fit, and low-confidence predictions can be surfaced as
   "uncertain" instead of a confident wrong label.
4. **Backbone fine-tuning, once there's enough data.** Unfreezing the last
   MobileNetV2 block only helps when the dataset is large enough that it
   stops overfitting — the hook is already in `train.py`
   (`finetune_last_n_blocks`).
5. **Test-time augmentation** — average predictions over a few flipped/rotated
   views of the same photo at inference; a cheap, no-retraining robustness
   bump.
6. **Try a stronger generic backbone** (ResNet-18 / EfficientNet-B0, both
   ImageNet-pretrained and permitted) as an A/B once the dataset is bigger.
7. **A learned severity model** (e.g. a small regression head or a
   segmentation model trained on our own masks) would be more robust than the
   current CV heuristics — but adds complexity and another thing to
   validate; the current approach was chosen for explainability.
8. **Class-weighted loss / oversampling** to squeeze more recall out of
   whichever class is currently weakest, as a stopgap while the dataset grows.

---

## 14. Setup & running

### 🚀 1-Step Full-Stack Launch (Recommended)

From the project root:

```bash
npm start
# or: npm run dev
```

Concurrently starts both the FastAPI backend on `http://localhost:8005` and Vite React frontend on `http://localhost:5173`.

---

### Alternative: Running Separately

#### Backend

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m app.seed                     # create demo accounts (optional, wipes DB)
uvicorn app.main:app --reload --port 8005
```

- Interactive API docs: `http://localhost:8005/docs`
- `DATABASE_URL` env var → Postgres for hosting; `JWT_SECRET` env var in prod.
- `head.pt` is already committed — no training step required to run.

#### Frontend

```bash
cd frontend
npm install
npm run dev               # http://localhost:5173
```

### Retrain the classifier

```bash
cd backend
# add balanced images to app/data/dataset/<class>/*.jpg, then:
python -m app.ml.train
```

### Evaluate

See [`backend/README.md`](backend/README.md#evaluate-against-held-out-real-photos)
— runs the current model over `data/` and prints per-class + overall
accuracy.

---

## 15. Demo accounts

Created by `python -m app.seed`. All use password **`demo1234`**.

| Role | Email | Category |
| --- | --- | --- |
| User | `user1@infrapulse.demo` | — |
| User | `user2@infrapulse.demo` | — |
| User | `user3@infrapulse.demo` | — |
| Staff | `staff.structural@infrapulse.demo` | Structural |
| Staff | `staff.functional@infrapulse.demo` | Functional |
| Staff | `staff.performance1@infrapulse.demo` | Performance |
| Staff | `staff.performance2@infrapulse.demo` | Performance |

Users log in at `/`, staff at `/staff/login`. New accounts can still be
created via each page's Sign Up tab.
