# Benchmarks & Reporting Scripts

Comparison and reporting scripts kept for reference. None of them are part of the running
application — the app itself lives in `app/` and is started with `uvicorn app.main:app`.

| Script | What it did | Needs |
| :--- | :--- | :--- |
| `benchmark_infra_vs_maratha.py` | Ran InfraPulse's ensemble against a separate `maratha_model` project's classifier, image by image. | `maratha_model/` (removed) + `app/model/data/test/` |
| `benchmark_nawabs_vs_shauryas.py` | Three-way comparison against a third-party repo (`shauryas`). | an external repo path + `app/model/data/test/` + `main_test/` |
| `benchmark_submit_ticket_models.py` | Benchmarked the ticket-submission path's models on a custom image set. | `main_test/` |
| `check_maratha_model.py` | Quick sanity check of the maratha classifier's output. | `maratha_model/` (removed) |
| `generate_pdf_report.py` | Rendered the technical report PDF with matplotlib charts. | `reportlab`, `matplotlib`, plus files under an external artifact dir |

`results/` holds the JSON output these scripts produced, kept as a record of the
model comparisons that informed the final ensemble weights.

## Why these no longer run as-is

They are archived, not maintained. The two comparisons against external projects depend on
`maratha_model/` and a third-party checkout, both of which have been removed. The scripts
also referenced a machine-specific scratch path
(`/home/bittu/.gemini/antigravity-cli/brain/...`) that does not exist elsewhere.

`PROJECT_ROOT` in each script was updated to point two levels up (out of `archive/benchmarks/`)
so the paths inside them still resolve to the repo root. To run one, restore its inputs and
adjust the hardcoded constants at the top of the file.

## The in-app equivalent

You don't need any of these to compare models. The app ships a built-in benchmark UI:

- `/test` — side-by-side evaluation of all five checkpoints on the holdout split, with a
  model leaderboard. Needs images in `app/model/data/test/` (see `../DATASET.md`).
- `/test/playground` — upload any photo and get every model's prediction. Needs no dataset.
