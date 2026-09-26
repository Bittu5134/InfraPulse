import sys
import os
import time
import json
import importlib.util
from pathlib import Path

# scripts live in archive/benchmarks/, so the repo root is two levels up
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MARATHA_BACKEND = PROJECT_ROOT / "maratha_model" / "backend"
MAIN_TEST_DIR = PROJECT_ROOT / "main_test"

print("==========================================================================")
print("  TICKET SUBMIT BENCHMARK: NAWABS VS MARATHA (EVALUATING MAIN_TEST)       ")
print("==========================================================================")

# 1. Load Nawabs Ticket Submit Model Pipeline
sys.path.insert(0, str(PROJECT_ROOT))
try:
    import app.model_service as nawabs_service
    _ = nawabs_service.load_custom_model("convnext_tiny")
    nawabs_loaded = True
    print("[✓] Nawabs Ticket Submit Model Pipeline (ConvNeXt-Tiny) initialized!")
except Exception as e:
    print(f"[!] Failed to load Nawabs pipeline: {e}")
    nawabs_loaded = False

# 2. Load Maratha Ticket Submit Model Pipeline
maratha_loaded = False
try:
    cls_spec = importlib.util.spec_from_file_location("maratha_cls", MARATHA_BACKEND / "app" / "ml" / "classifier.py")
    maratha_cls_mod = importlib.util.module_from_spec(cls_spec)
    cls_spec.loader.exec_module(maratha_cls_mod)

    sev_spec = importlib.util.spec_from_file_location("maratha_sev", MARATHA_BACKEND / "app" / "ml" / "severity.py")
    maratha_sev_mod = importlib.util.module_from_spec(sev_spec)
    sev_spec.loader.exec_module(maratha_sev_mod)

    maratha_classifier = maratha_cls_mod.DefectClassifier()
    compute_maratha_sev = maratha_sev_mod.compute_severity
    maratha_loaded = True
    print("[✓] Maratha Ticket Submit Model Pipeline (MobileNetV2 + TTA) initialized!")
except Exception as e:
    print(f"[!] Failed to load Maratha pipeline: {e}")

# 3. Process main_test folder images
images = list(MAIN_TEST_DIR.glob("*"))
images = [p for p in images if p.suffix.lower() in [".jpg", ".png", ".jpeg", ".webp"]]
images.sort()

print(f"\n[+] Found {len(images)} images in main_test directory:")
print("--------------------------------------------------------------------------")

results = []

for img_path in images:
    str_p = str(img_path)
    fname = img_path.name
    
    # Expected ground truth inferable from filename
    if "spall" in fname.lower():
        expected = "spalling"
    elif "water" in fname.lower():
        expected = "stagnant_water"
    elif "paint" in fname.lower():
        expected = "paint_peeling"
    elif "crack" in fname.lower():
        expected = "cracked_tiles"
    else:
        expected = "unknown"

    # Run Nawabs Pipeline
    nawabs_res = {}
    if nawabs_loaded:
        t0 = time.time()
        try:
            raw = nawabs_service.predict_single_image(str_p)
            lat = (time.time() - t0) * 1000.0
            nawabs_res = {
                "defect": raw.get("defect_name", "").lower().replace(" ", "_"),
                "category": raw.get("category_str", ""),
                "confidence": raw.get("confidence", 0.0),
                "severity": raw.get("severity", 0.0),
                "extent": raw.get("extent", 0.0),
                "priority_score": raw.get("priority_score", 0.0),
                "latency_ms": round(lat, 1)
            }
        except Exception as e:
            nawabs_res = {"error": str(e)}

    # Run Maratha Pipeline
    maratha_res = {}
    if maratha_loaded:
        t0 = time.time()
        try:
            lbl, conf = maratha_classifier.predict(str_p, use_tta=True)
            sev = compute_maratha_sev(str_p, lbl)
            lat = (time.time() - t0) * 1000.0
            maratha_res = {
                "defect": lbl,
                "confidence": round(conf * 100.0, 1),
                "severity": round(sev, 1),
                "latency_ms": round(lat, 1)
            }
        except Exception as e:
            maratha_res = {"error": str(e)}

    results.append({
        "file": fname,
        "expected": expected,
        "nawabs": nawabs_res,
        "maratha": maratha_res
    })

    print(f"\n📷 File: {fname} (Expected: {expected.upper()})")
    if "defect" in nawabs_res:
        correct_n = "✓" if expected != "unknown" and nawabs_res["defect"] == expected else ("?" if expected == "unknown" else "✗")
        print(f"  └─ NAWABS MODEL : Defect={nawabs_res['defect']} [{correct_n}] (Conf={nawabs_res['confidence']}%, Sev={nawabs_res['severity']}, Priority={nawabs_res['priority_score']}) [{nawabs_res['latency_ms']}ms]")
    else:
        print(f"  └─ NAWABS MODEL : Error: {nawabs_res.get('error')}")

    if "defect" in maratha_res:
        correct_m = "✓" if expected != "unknown" and maratha_res["defect"] == expected else ("?" if expected == "unknown" else "✗")
        print(f"  └─ MARATHA MODEL: Defect={maratha_res['defect']} [{correct_m}] (Conf={maratha_res['confidence']}%, Sev={maratha_res['severity']}) [{maratha_res['latency_ms']}ms]")
    else:
        print(f"  └─ MARATHA MODEL: Error: {maratha_res.get('error')}")

# Save results JSON
with open(PROJECT_ROOT / "main_test_benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n==========================================================================")
print("Benchmarking complete! Summary saved to main_test_benchmark_results.json")
print("==========================================================================")
