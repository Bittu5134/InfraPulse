import sys
import os
import time
import json
import importlib.util
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
SHAURYAS_REPO = PROJECT_ROOT / "maratha_model" / "scratch" / "shauryas_repo"
if not SHAURYAS_REPO.exists():
    SHAURYAS_REPO = Path("/home/bittu/.gemini/antigravity-cli/brain/f1439725-9acf-4d55-abe6-0c8042d0399f/scratch/shauryas_repo")

CLASS_NAMES = ["cracked_tiles", "paint_peeling", "spalling", "stagnant_water"]

LABEL_MAP = {
    "cracked_tiles": "cracked_tiles",
    "paint_peeling": "paint_peeling",
    "spalling": "spalling",
    "stagnant_water": "stagnant_water",
    "Cracked Tiles": "cracked_tiles",
    "Paint Peeling": "paint_peeling",
    "Spalling": "spalling",
    "Stagnant Water": "stagnant_water",
}

print("==========================================================================")
print("     EMPIRICAL BENCHMARK: NAWABS MODEL vs. SHAURYAS MODEL                 ")
print("==========================================================================")

# 1. Load Nawabs Model Service
print("\n[1/3] Initializing Nawabs Model Service (ConvNeXt & Ensemble)...")
sys.path.insert(0, str(PROJECT_ROOT))
try:
    import app.model_service as nawabs_service
    _ = nawabs_service.load_custom_model("convnext_tiny")
    nawabs_loaded = True
    print("[✓] Nawabs Model Ensemble loaded successfully!")
except Exception as e:
    print(f"[!] Error loading Nawabs Model: {e}")
    nawabs_loaded = False

# 2. Load Shauryas Model Analyzer cleanly using importlib
print("\n[2/3] Initializing Shauryas Model Analyzer (ResNet-18)...")
shauryas_loaded = False
try:
    # Prepend shauryas_repo to sys.path and set model env var
    sys.path.insert(0, str(SHAURYAS_REPO))
    os.environ["INFRAPULSE_MODEL"] = str(SHAURYAS_REPO / "model" / "infrapulse_model.pt")

    spec = importlib.util.spec_from_file_location("shauryas_analyzer", SHAURYAS_REPO / "app" / "ml" / "analyzer.py")
    shauryas_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shauryas_mod)

    # Initialize model loading
    shauryas_mod.load_model(str(SHAURYAS_REPO / "model" / "infrapulse_model.pt"))
    shauryas_analyze = shauryas_mod.analyze
    shauryas_loaded = True
    print("[✓] Shauryas Model Analyzer loaded successfully!")
except Exception as e:
    print(f"[!] Error loading Shauryas Model: {e}")

# Restore sys.path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 3. Benchmark on 120-image dataset (app/model/data/test)
TEST_DIR = PROJECT_ROOT / "app" / "model" / "data" / "test"
dataset_files = []
for cat_dir in TEST_DIR.iterdir():
    if cat_dir.is_dir():
        true_label = cat_dir.name
        for ext in ["*.jpg", "*.png", "*.jpeg"]:
            for img_p in cat_dir.glob(ext):
                dataset_files.append((img_p, true_label))

print(f"\n[3/3] Running benchmark on {len(dataset_files)} test dataset images...")

nawabs_dataset_res = {"y_true": [], "y_pred": [], "latencies": [], "severities": [], "extents": []}
shauryas_dataset_res = {"y_true": [], "y_pred": [], "latencies": [], "severities": [], "extents": []}

start_bench_time = time.time()

for idx, (img_path, true_label) in enumerate(dataset_files, 1):
    str_path = str(img_path)
    clean_true = LABEL_MAP.get(true_label, true_label)

    # Nawabs Prediction
    if nawabs_loaded:
        t0 = time.time()
        try:
            r = nawabs_service.predict_single_image(str_path)
            lat = (time.time() - t0) * 1000.0
            pred_d = r.get("defect_name", "").lower().replace(" ", "_")
            pred_d = LABEL_MAP.get(pred_d, pred_d)

            nawabs_dataset_res["y_true"].append(clean_true)
            nawabs_dataset_res["y_pred"].append(pred_d)
            nawabs_dataset_res["latencies"].append(lat)
            nawabs_dataset_res["severities"].append(r.get("severity", 0.0))
            nawabs_dataset_res["extents"].append(r.get("extent", 0.0))
        except Exception as e:
            print(f"[!] Nawabs error: {e}")

    # Shauryas Prediction
    if shauryas_loaded:
        t0 = time.time()
        try:
            r_sh = shauryas_analyze(str_path)
            lat = (time.time() - t0) * 1000.0
            pred_sh = r_sh.get("defect", "").lower().replace(" ", "_")
            pred_sh = LABEL_MAP.get(pred_sh, pred_sh)

            shauryas_dataset_res["y_true"].append(clean_true)
            shauryas_dataset_res["y_pred"].append(pred_sh)
            shauryas_dataset_res["latencies"].append(lat)
            shauryas_dataset_res["severities"].append(r_sh.get("severity", 0.0) * 100.0)
            shauryas_dataset_res["extents"].append(r_sh.get("extent", 0.0) * 100.0)
        except Exception as e:
            print(f"[!] Shauryas error: {e}")

    if idx % 30 == 0 or idx == len(dataset_files):
        print(f"Processed {idx}/{len(dataset_files)} dataset images...")

dataset_bench_duration = time.time() - start_bench_time

# Metric Helper
def compute_metrics(y_true, y_pred, latencies):
    if not y_true or not y_pred:
        return {"accuracy": 0.0, "latency_mean": 0.0, "latency_p95": 0.0, "per_class": {}}
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = float(np.mean(y_true == y_pred)) * 100.0
    lat_mean = float(np.mean(latencies))
    lat_p95 = float(np.percentile(latencies, 95))

    per_class = {}
    for cls in CLASS_NAMES:
        tp = int(np.sum((y_true == cls) & (y_pred == cls)))
        fp = int(np.sum((y_true != cls) & (y_pred == cls)))
        fn = int(np.sum((y_true == cls) & (y_pred != cls)))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[cls] = {
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "f1": round(f1 * 100, 2),
            "support": int(np.sum(y_true == cls))
        }

    return {
        "accuracy": round(acc, 2),
        "latency_mean": round(lat_mean, 2),
        "latency_p95": round(lat_p95, 2),
        "per_class": per_class
    }

m_nawabs = compute_metrics(nawabs_dataset_res["y_true"], nawabs_dataset_res["y_pred"], nawabs_dataset_res["latencies"])
m_shauryas = compute_metrics(shauryas_dataset_res["y_true"], shauryas_dataset_res["y_pred"], shauryas_dataset_res["latencies"])

# 4. Benchmark on main_test images
MAIN_TEST_DIR = PROJECT_ROOT / "main_test"
main_test_files = [p for p in MAIN_TEST_DIR.glob("*") if p.suffix.lower() in [".jpg", ".png", ".jpeg", ".webp"]]
main_test_files.sort()

main_test_comparison = []
for img_p in main_test_files:
    fname = img_p.name
    if "spall" in fname.lower():
        exp = "spalling"
    elif "water" in fname.lower():
        exp = "stagnant_water"
    elif "paint" in fname.lower():
        exp = "paint_peeling"
    elif "crack" in fname.lower():
        exp = "cracked_tiles"
    else:
        exp = "unknown"

    n_d = nawabs_service.predict_single_image(str(img_p)) if nawabs_loaded else {}
    s_d = shauryas_analyze(str(img_p)) if shauryas_loaded else {}

    n_pred = n_d.get("defect_name", "").lower().replace(" ", "_")
    n_pred = LABEL_MAP.get(n_pred, n_pred)

    s_pred = s_d.get("defect", "").lower().replace(" ", "_")
    s_pred = LABEL_MAP.get(s_pred, s_pred)

    main_test_comparison.append({
        "file": fname,
        "expected": exp,
        "nawabs": {
            "defect": n_pred,
            "confidence": n_d.get("confidence", 0.0),
            "severity": n_d.get("severity", 0.0),
            "extent": n_d.get("extent", 0.0),
            "priority_score": n_d.get("priority_score", 0.0)
        },
        "shauryas": {
            "defect": s_pred,
            "confidence": round(s_d.get("confidence", 0.0) * 100.0, 1),
            "severity": round(s_d.get("severity", 0.0) * 100.0, 1),
            "extent": round(s_d.get("extent", 0.0) * 100.0, 1)
        }
    })

# Save JSON Output
benchmark_summary = {
    "dataset_eval": {
        "total_images": len(dataset_files),
        "nawabs": m_nawabs,
        "shauryas": m_shauryas
    },
    "main_test_eval": main_test_comparison
}

with open(PROJECT_ROOT / "nawabs_vs_shauryas_benchmark_results.json", "w") as f:
    json.dump(benchmark_summary, f, indent=2)

print("\n==========================================================================")
print("             FINAL EMPIRICAL SUMMARY: NAWABS vs. SHAURYAS                 ")
print("==========================================================================")

print(f"\n120-Image Dataset Evaluation | Execution Time: {dataset_bench_duration:.2f}s\n")

print(f"{'Model Architecture':<34} | {'Accuracy':<10} | {'Mean Latency':<14} | {'P95 Latency':<12}")
print("-" * 77)
print(f"{'Nawabs Model (ConvNeXt Ensemble)':<34} | {m_nawabs['accuracy']:>8.2f}% | {m_nawabs['latency_mean']:>10.2f} ms | {m_nawabs['latency_p95']:>8.2f} ms")
print(f"{'Shauryas Model (ResNet-18)':<34} | {m_shauryas['accuracy']:>8.2f}% | {m_shauryas['latency_mean']:>10.2f} ms | {m_shauryas['latency_p95']:>8.2f} ms")

print("\n--------------------------------------------------------------------------")
print("PER-CLASS F1-SCORE BREAKDOWN (%)")
print("--------------------------------------------------------------------------")
print(f"{'Defect Category':<20} | {'Nawabs Model':<16} | {'Shauryas Model':<16}")
print("-" * 58)
for cls in CLASS_NAMES:
    f1_naw = m_nawabs["per_class"].get(cls, {}).get("f1", 0.0)
    f1_sha = m_shauryas["per_class"].get(cls, {}).get("f1", 0.0)
    print(f"{cls:<20} | {f1_naw:>14.2f}% | {f1_sha:>14.2f}%")

print("\n==========================================================================")
print("Results saved to nawabs_vs_shauryas_benchmark_results.json")
print("==========================================================================")
