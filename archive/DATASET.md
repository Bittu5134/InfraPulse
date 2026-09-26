# Dataset Restore Guide

The ML image datasets were stripped from this repository to keep it small (they totalled
**~4.2 GB**). Only the trained model weights remain, so the app runs exactly as before.

Nothing in the dataset is required to *run* InfraPulse. The dataset is only needed if you want
to re-run training, re-run benchmarks, or populate the `/test` benchmark page with real images.

---

## What was removed

| Path | Size | Files | What it was | Needed to run app? |
| :--- | ---: | ---: | :--- | :---: |
| `app/model/data/train/` | 112 MB | 560 | Training split, 140/class | No |
| `app/model/data/val/` | 21 MB | 120 | Validation split, 30/class | No |
| `app/model/data/test/` | 24 MB | 120 | Holdout split, 30/class — **served by `/test`** | No (page degrades to "No Images Found") |
| `app/model/data/raw_water_dataset/` | 340 MB | 6030 | "Stagnant Water and Wet Surface" Kaggle set, unfiltered | No |
| `app/model/data/normalized_clean_eval/` | 156 MB | 800 | 200/class phone-photo-filtered eval set | No |
| `app/model/data/external_eval/` | 11 MB | 525 | Ingested external-source eval set | No |
| `app/model/data/external_sources/` | 3.6 GB | 9006 | Raw third-party datasets (see table below) | No |
| **Total** | **~4.2 GB** | **~17,161** | | |

Also removed: `app/model/checkpoints/checkpoints.zip` (80 MB), `app/model/checkpoints/archive/`
(37 MB of superseded `.pt` files), `main_test/` (loose scratch photos), and `maratha_model/`
(a separate, unrelated comparison project).

---

## The `/test` benchmark page after the strip

`app/main.py` mounts the dataset directory at `/test_data` **only if it exists**, so nothing
breaks. The page has two behaviours:

- **With images present** — renders the per-image multi-model comparison grid.
- **With no images present** — renders the "No Images Found" card. The page header, the
  model leaderboard, and the `/test/playground` custom-image upload all still work.

To repopulate it, restore the `test/` split (below). The playground needs no dataset at all.

---

## How to restore

### 1. The `test/` holdout split (populates the `/test` page)

There are two independent routes.

**Route A — from the public BD3 dataset (recommended, fully reproducible).**

`app/model/pull_bd3_dataset.py` downloads and re-splits the BD3 building-defect dataset
from Hugging Face. It is archived at `archive/ml-training/pull_bd3_dataset.py`.

```bash
pip install datasets scikit-learn
# run from the repo root so OUT_ROOT lands on app/model/data (see note below)
python archive/ml-training/pull_bd3_dataset.py
```

Note: the archived script hardcodes `OUT_ROOT = Path("data")` for a different working
directory. To target InfraPulse, edit that one line to:

```python
OUT_ROOT = Path("app/model/data")
```

**Route B — from git history.**

An earlier commit still contains 241 benchmark images under `app/model/data/test/`
(the `bd3_*.jpg` set). Restore just that directory with:

```bash
git checkout 58e93f0 -- app/model/data/test/
```

Verify with `ls app/model/data/test/*/ | wc -l` — you should see images in all four class
folders (`cracked_tiles`, `paint_peeling`, `spalling`, `stagnant_water`).

> **Note on the 120 `test_*.jpg` / `val_*.jpg` / `train_*.jpg` images that were deleted:**
> these were *locally derived* re-splits and were never committed to git, so they are not
> recoverable. They can be regenerated with Route A, or with the archived
> `assemble_max_balanced_dataset.py` script (Route 4) if you have restored the raw sources.

### 2. Training / validation splits

Regenerate with the archived `archive/ml-training/assemble_max_balanced_dataset.py`, which
needs the raw sources in Route 4. It rebuilds all four splits (`train`, `val`, `test`,
`normalized_clean_eval`) at 200 images/class in one pass.

### 3. `raw_water_dataset/` ("New Water Dataset" dropdown on `/test`)

Sourced from the Kaggle dataset **"Stagnant Water and Wet Surface Dataset"**. Manual steps:

1. Download the archive from Kaggle and place the zip at `~/Downloads/Stagnant Water and Wet Surface Dataset 1.zip`.
2. Run `archive/ml-training/normalize_and_clean_dataset.py`, which extracts it to
   `app/model/data/raw_water_dataset/`.

The script also expects `~/Downloads` to be its input directory (`DOWNLOADS_DIR`). Adjust that
constant if you keep downloads elsewhere. The Kaggle slug to search for is
`stagnant water and wet surface dataset`.

### 4. `external_sources/` (3.6 GB of raw third-party data)

These are the upstream downloads that feed the ingestion/assembly scripts. Restore each
source directory from its origin, then re-run the archived pipeline:

| Directory | Size | Origin | Consumed by |
| :--- | ---: | :--- | :--- |
| `external_sources/tile_defect_dataset/` | 109 MB | Kaggle — *Magnetic Tile Defect* | `ingest_external_datasets.py` → `cracked_tiles` |
| `external_sources/spalling_dataset/` | 1.9 MB | Kaggle — *U-spalling* dataset | `ingest_external_datasets.py` → `spalling` |
| `external_sources/building_wall_defects/` | 4.6 MB | Wall/surface defect set | `ingest_external_datasets.py` → `paint_peeling` |
| `external_sources/crack_dataset_hy/` | 3.5 GB | *Concrete Crack (HY)* dataset | `assemble_max_balanced_dataset.py` → `paint_peeling` |
| `external_sources/multiclass_concrete_defect/` | 196 KB | Kaggle — *Multiclass Concrete Defect* | reference only |
| `external_sources/puddle_dataset/`, `external_sources/water_puddle_paper/` | — | Puddle/stagnant-water sets | `ingest_external_datasets.py` → `stagnant_water` (these two were already absent before the strip) |

To rebuild them into the eval set:

```bash
# raw sources -> app/model/data/external_eval/
python archive/ml-training/ingest_external_datasets.py

# raw sources -> balanced train/val/test/normalized_clean_eval (200/class)
python archive/ml-training/assemble_max_balanced_dataset.py
```

Both scripts resolve paths relative to the repo root, so run them from the repo root
(they compute `REPO_ROOT` by walking up four levels from their own file, so **move them back
to `app/model/src/` first**, or fix `REPO_ROOT` accordingly).

### 5. `normalized_clean_eval/`

Rebuilt as a side effect of `normalize_and_clean_dataset.py` (it filters the raw sets down to
real-world smartphone-style photos and caps at 200/class).

---

## Files archived vs. deleted

**Archived (recoverable, in `archive/ml-training/`):**

- `pull_bd3_dataset.py` — HF dataset downloader
- `assemble_max_balanced_dataset.py` — balanced split builder
- `ingest_external_datasets.py` — external-source ingester
- `normalize_and_clean_dataset.py` — clean/filter pipeline
- `dataset.py`, `train.py`, `train_advanced_suite.py`, `evaluate.py`, `inference.py`
- `benchmark_ensemble_combinations.py`, `benchmark_per_category_consensus.py`

**Kept in `app/model/src/`:**

- `model.py` — **the only module imported at runtime** (by `app/model_service.py`)

**Hard-deleted (not recoverable, not worth keeping):**

- `app/model/checkpoints/checkpoints.zip` — a zip archive duplicating checkpoints already in the tree
- `app/model/checkpoints/archive/*.pt` — superseded `old_model_v1.pt` and `multimodal_fusion_infrapulse.pt`; no code path references them
- `main_test/`, `junk/`, `images.jpg` — scratch files from manual testing
- `maratha_model/` — a separate comparison project with no imports from `app/`
- `git-lfs-3.5.1/` + `git-lfs.tar.gz` — the Git LFS *installer*, not a project dependency
