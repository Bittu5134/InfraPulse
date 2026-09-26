# ML Training & Evaluation Pipeline

The scripts that produced the checkpoints in `app/model/checkpoints/`. They are **not**
part of the running application — the app only imports `app/model/src/model.py` (the model
architectures) and loads the `.pt` weights from disk. Nothing in this folder is imported at
runtime, so it has been moved out of the deployed app.

## Scripts

| Script | Role |
| :--- | :--- |
| `pull_bd3_dataset.py` | Downloads the BD3 building-defect dataset from Hugging Face (`chandrabhuma/building_defect_vqa`) and merges it into the `data/{train,val,test}/<class>/` layout. |
| `ingest_external_datasets.py` | Copies/standardises the raw third-party sets in `app/model/data/external_sources/` into `app/model/data/external_eval/`. |
| `assemble_max_balanced_dataset.py` | Walks all raw sources and rebuilds the class-balanced `train`/`val`/`test`/`normalized_clean_eval` splits (200 images per class). |
| `normalize_and_clean_dataset.py` | Extracts and filters the "Stagnant Water and Wet Surface" zip from `~/Downloads` down to real-world phone photos. |
| `dataset.py` | `torchvision` `ImageFolder` + `DataLoader` wiring. |
| `model.py` (archived copy) | Standalone training-time copy of the architectures. **The live copy is `app/model/src/model.py`** — edit that one, not this. |
| `train.py` | Baseline training loop. |
| `train_advanced_suite.py` | Trains and exports the full five-model suite (ConvNeXt-Tiny, Swin-Tiny, MTL dual-branch, INT8 quantized, EfficientNet-B0 baseline). |
| `evaluate.py` | Standalone evaluation entrypoint. |
| `inference.py` | Standalone single-image inference CLI. |
| `benchmark_ensemble_combinations.py` | Grid-searches ensemble weight combinations; produced `ensemble_grid_search_report.json`. |
| `benchmark_per_category_consensus.py` | Optimises the per-category consensus matrix; produced `per_category_consensus_report.json` and the weights now in `app/model/consensus_weights.json`. |
| `priority.py` | `analyze_heatmap` — converts a GradCAM++ map into severity/extent scores. |
| `fallback.py` | `fallback_analysis` — heuristic used only when vision inference fails. |

## Path dependencies (must be fixed if you re-run these)

These scripts were written to live at `app/model/src/`, and several resolve paths by walking
up four levels from their own file:

```python
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
```

From `archive/ml-training/`, that no longer points at the repo root. **Before running any of
them, either move the file back to `app/model/src/` or fix `REPO_ROOT` / `OUT_ROOT` /
`DOWNLOADS_DIR` at the top of the script.** They also rely on the dataset directories that
were stripped — see `../DATASET.md` for how to restore them.

Extra dependencies not in the main `requirements.txt` (see `app/model/requirements.txt` for
the full training set): `scipy`, `tqdm`, `matplotlib`, `grad-cam`, `onnx`, `onnxruntime`,
`datasets`.

## Note on `priority.py` / `fallback.py`

The live app does **not** use these. `app/model_service.py` computes severity and extent with
its own `compute_dynamic_spatial_extent()` and falls back to the rule classifier in
`app/priority_queue.py` (`mock_classify_defect`) when inference fails. These two modules are
only referenced by the archived `inference.py`.
