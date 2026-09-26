# archive/

Everything here is reference material. **None of it is needed to run InfraPulse.**

The app is `app/`, started with:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

This folder collects the reports, comparison scripts, and training pipeline that were
scattered across the repo root, `docs/`, and `app/model/src/` before a cleanup pass, so the
rest of the tree only contains what actually runs.

| Folder / file | Contents |
| :--- | :--- |
| [`DATASET.md`](DATASET.md) | **Start here if you removed the datasets.** Lists every deleted image directory, its size, where it came from, and how to restore it. |
| [`docs/`](docs/) | Design document, technical report (`.tex` + rendered `.pdf`), production architecture report, benchmark/consensus report, code & architecture manual, and the team presentation guide. |
| [`benchmarks/`](benchmarks/) | One-off comparison scripts that benchmarked InfraPulse against two other projects, plus the report generator and their JSON results. See its [README](benchmarks/README.md). |
| [`ml-training/`](ml-training/) | The dataset ingestion, training, evaluation and weight-search pipeline that produced the checkpoints. See its [README](ml-training/README.md). |
| [`scratch-data/`](scratch-data/) | Data rescued during cleanup rather than deleted — a stale demo database and the four complaint photos it referenced. See its [README](scratch-data/README.md). |
| [`SCHEMA_NOTE.md`](SCHEMA_NOTE.md) | Why `schema.sql.stale` must not be run against a real database. |
| `schema.sql.stale` | An out-of-date hand-written DDL file. Superseded by the SQLAlchemy models in `app/models.py`. |

## Why things were moved rather than deleted

Each of these was recoverable and worth keeping as a record of how the final ensemble weights
and architecture choices were reached — which is exactly what the problem statement's
documentation section is scored on. Deleting them would have thrown away that history.

The scripts in `benchmarks/` and `ml-training/` resolve paths by walking up from their own
location, so they need their constants adjusted before they will run again from here. Each
folder's README documents exactly what to change.
