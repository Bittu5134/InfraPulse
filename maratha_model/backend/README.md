# InfraPulse backend

FastAPI service: auth, complaints, priority queue, WebSocket live updates,
and the in-process defect classifier (frozen ImageNet MobileNetV2 backbone +
custom-trained head — see `app/ml/`).

Full architecture, detection logic, priority method, evaluation results, and
API reference: [`../DOCUMENTATION.md`](../DOCUMENTATION.md).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train the classifier head

A trained head is already checked in at `app/models_weights/head.pt`,
trained on the real (diverse-source, rebalanced) photos in
`app/data/dataset/` — see `app/data/DATASET_NOTES.md` for provenance and why
class balance matters more than raw volume for this dataset size.

To retrain (e.g. after adding more images):

1. Drop labeled images into `app/data/dataset/<class>/*.jpg` for each of:
   `spalling`, `stagnant_water`, `paint_peeling`, `cracked_tiles`. Keep class
   counts roughly balanced — an imbalanced set measurably hurt real-photo
   accuracy in testing (see `DATASET_NOTES.md`).
2. Run `python -m app.ml.train`.

   Trains the small head on top of the frozen backbone (fixed random seed,
   15% validation split, keeps the best-val checkpoint) and saves weights to
   `app/models_weights/head.pt`.

## Evaluate against held-out real photos

```bash
python -c "
from app.ml.classifier import get_classifier
import glob, os
clf = get_classifier()
correct = total = 0
for path in sorted(glob.glob('../data/*/*.jpg')):
    true_label = os.path.basename(os.path.dirname(path))
    pred, conf = clf.predict(path)
    correct += pred == true_label
    total += 1
    print(pred == true_label, true_label, pred, round(conf, 2), os.path.basename(path))
print(f'{correct}/{total} = {correct/total:.1%}')
"
```

Uses the real, never-trained-on photos in the repo root's `data/` folder
(21 photos across 4 classes) as a held-out sanity check. Current result:
17/21 (81%), up from 6/10 (60%) for the original Kaggle-trained head — see
`app/data/DATASET_NOTES.md` for the full story, including two hard-won
lessons (imbalanced data hurts more than it helps; always eye-check images,
don't trust search-result titles as labels).

## Demo accounts

```bash
python -m app.seed
```

Wipes all accounts/complaints and creates 3 demo users + 4 demo staff (one
per category, two for Performance). See the root [README.md](../README.md#demo-accounts)
for the full credential list — all use password `demo1234`.

## Run the API

```bash
uvicorn app.main:app --reload --port 8000
```

- Docs: http://localhost:8000/docs
- WebSocket for live queue updates: `ws://localhost:8000/ws/queue`
- DB defaults to local SQLite (`infrapulse.db`); set `DATABASE_URL` env var to
  point at Postgres for deployment.
- Set `JWT_SECRET` env var in production.

## Notes on the constraint compliance

- Backbone: `torchvision.models.mobilenet_v2` with `IMAGENET1K_V1` weights —
  generic ImageNet pretraining, explicitly allowed.
- No checkpoint pretrained on defect/damage/crack-specific datasets is used
  anywhere in this repo.
- Classification head (`app/ml/classifier.py::DefectHead`) is trained from
  scratch by the team on a self-collected dataset (`app/ml/train.py`).
- No external AI/ML inference API calls — the model runs in-process.
- Severity/extent scoring (`app/ml/severity.py`) uses classical CV (Canny
  edge density, HSV color-range segmentation), not a learned/black-box model.
