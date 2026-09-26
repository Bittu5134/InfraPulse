import sys
import os
import time
import json
import importlib.util
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
MARATHA_BACKEND = PROJECT_ROOT / "maratha_model" / "backend"

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
print("     EMPIRICAL BENCHMARK: NAWABS MODEL vs. MARATHA MODEL                  ")
print("==========================================================================")

# 1. Load Nawabs Main Model Service
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

# 2. Load Maratha Model Classifier & Severity
print("\n[2/3] Initializing Maratha DefectClassifier & Severity Engine...")
maratha_loaded = False
try:
    cls_spec = importlib.util.spec_from_file_location("maratha_classifier_mod", MARATHA_BACKEND / "app" / "ml" / "classifier.py")
    maratha_cls_mod = importlib.util.module_from_spec(cls_spec)
    cls_spec.loader.exec_module(maratha_cls_mod)

    sev_spec = importlib.util.spec_from_file_location("maratha_severity_mod", MARATHA_BACKEND / "app" / "ml" / "severity.py")
    maratha_sev_mod = importlib.util.module_from_spec(sev_spec)
    sev_spec.loader.exec_module(maratha_sev_mod)

    maratha_classifier = maratha_cls_mod.DefectClassifier()
    compute_maratha_severity = maratha_sev_mod.compute_severity
    maratha_loaded = True
    print("[✓] Maratha DefectClassifier (MobileNetV2 + TTA) loaded successfully!")
except Exception as e:
    print(f"[!] Error loading Maratha Model: {e}")

# 3. Locate Dataset Images
TEST_DIR = PROJECT_ROOT / "app" / "model" / "data" / "test"
dataset_files = []
for cat_dir in TEST_DIR.iterdir():
    if cat_dir.is_dir():
        true_label = cat_dir.name
        for ext in ["*.jpg", "*.png", "*.jpeg"]:
            for img_p in cat_dir.glob(ext):
                dataset_files.append((img_p, true_label))

print(f"\n[3/3] Found {len(dataset_files)} test images across classes: {[d.name for d in TEST_DIR.iterdir() if d.is_dir()]}")

nawabs_results = {"y_true": [], "y_pred": [], "latencies": [], "severities": []}
maratha_tta_results = {"y_true": [], "y_pred": [], "latencies": [], "severities": []}
maratha_single_results = {"y_true": [], "y_pred": [], "latencies": []}

print("\n--------------------------------------------------------------------------")
print(f"Running Inference Benchmark on {len(dataset_files)} Images...")
print("--------------------------------------------------------------------------")

start_bench_time = time.time()

for idx, (img_path, true_label) in enumerate(dataset_files, 1):
    str_path = str(img_path)
    clean_true = LABEL_MAP.get(true_label, true_label)

    # A. Benchmark Nawabs Model
    if nawabs_loaded:
        t0 = time.time()
        try:
            res_nawabs = nawabs_service.predict_single_image(str_path)
            lat_nawabs = (time.time() - t0) * 1000.0
            
            raw_d = res_nawabs.get("defect_name", "").lower().replace(" ", "_")
            pred_nawabs = LABEL_MAP.get(raw_d, raw_d)
            
            nawabs_results["y_true"].append(clean_true)
            nawabs_results["y_pred"].append(pred_nawabs)
            nawabs_results["latencies"].append(lat_nawabs)
            nawabs_results["severities"].append(res_nawabs.get("severity", 0.0))
        except Exception as e:
            print(f"[!] Nawabs error on {img_path.name}: {e}")

    # B. Benchmark Maratha Model (TTA)
    if maratha_loaded:
        t0 = time.time()
        pred_label_tta, conf_tta = maratha_classifier.predict(str_path, use_tta=True)
        lat_tta = (time.time() - t0) * 1000.0
        sev_maratha = compute_maratha_severity(str_path, pred_label_tta)

        maratha_tta_results["y_true"].append(clean_true)
        maratha_tta_results["y_pred"].append(pred_label_tta)
        maratha_tta_results["latencies"].append(lat_tta)
        maratha_tta_results["severities"].append(sev_maratha)

        # C. Benchmark Maratha Model (Single Pass)
        t0 = time.time()
        pred_label_single, conf_single = maratha_classifier.predict(str_path, use_tta=False)
        lat_single = (time.time() - t0) * 1000.0

        maratha_single_results["y_true"].append(clean_true)
        maratha_single_results["y_pred"].append(pred_label_single)
        maratha_single_results["latencies"].append(lat_single)

    if idx % 30 == 0 or idx == len(dataset_files):
        print(f"Processed {idx}/{len(dataset_files)} images...")

total_bench_duration = time.time() - start_bench_time

# Metric Calculation Helper
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

m_nawabs = compute_metrics(nawabs_results["y_true"], nawabs_results["y_pred"], nawabs_results["latencies"])
m_maratha_tta = compute_metrics(maratha_tta_results["y_true"], maratha_tta_results["y_pred"], maratha_tta_results["latencies"])
m_maratha_single = compute_metrics(maratha_single_results["y_true"], maratha_single_results["y_pred"], maratha_single_results["latencies"])

summary = {
    "total_images_evaluated": len(dataset_files),
    "total_duration_sec": round(total_bench_duration, 2),
    "nawabs_model": m_nawabs,
    "maratha_tta_model": m_maratha_tta,
    "maratha_single_model": m_maratha_single
}

with open(PROJECT_ROOT / "benchmark_results.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n==========================================================================")
print("             FINAL BENCHMARK SUMMARY: NAWABS vs. MARATHA                  ")
print("==========================================================================")

print(f"\nEvaluated Dataset: {len(dataset_files)} images | Total Execution Time: {total_bench_duration:.2f}s\n")

print(f"{'Model Architecture':<34} | {'Accuracy':<10} | {'Mean Latency':<14} | {'P95 Latency':<12}")
print("-" * 77)
print(f"{'Nawabs Model (ConvNeXt)':<34} | {m_nawabs['accuracy']:>8.2f}% | {m_nawabs['latency_mean']:>10.2f} ms | {m_nawabs['latency_p95']:>8.2f} ms")
print(f"{'Maratha Model (3-View TTA)':<34} | {m_maratha_tta['accuracy']:>8.2f}% | {m_maratha_tta['latency_mean']:>10.2f} ms | {m_maratha_tta['latency_p95']:>8.2f} ms")
print(f"{'Maratha Model (Single Pass)':<34} | {m_maratha_single['accuracy']:>8.2f}% | {m_maratha_single['latency_mean']:>10.2f} ms | {m_maratha_single['latency_p95']:>8.2f} ms")

print("\n--------------------------------------------------------------------------")
print("PER-CLASS F1-SCORE BREAKDOWN (%)")
print("--------------------------------------------------------------------------")
print(f"{'Defect Category':<20} | {'Nawabs Model':<16} | {'Maratha (TTA)':<16} | {'Maratha (Single)':<16}")
print("-" * 75)
for cls in CLASS_NAMES:
    f1_naw = m_nawabs["per_class"].get(cls, {}).get("f1", 0.0)
    f1_m_tta = m_maratha_tta["per_class"].get(cls, {}).get("f1", 0.0)
    f1_m_s = m_maratha_single["per_class"].get(cls, {}).get("f1", 0.0)
    print(f"{cls:<20} | {f1_naw:>14.2f}% | {f1_m_tta:>14.2f}% | {f1_m_s:>14.2f}%")

print("\n==========================================================================")
print("Benchmark results written to: benchmark_results.json")
print("==========================================================================")
