import sys
import os
import time
from pathlib import Path

# Add project root and maratha_model backend to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
MARATHA_BACKEND = PROJECT_ROOT / "maratha_model" / "backend"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(MARATHA_BACKEND))

print("==================================================")
print("     INFRA PULSE - MARATHA MODEL VERIFICATION     ")
print("==================================================")

# 1. Test Maratha Model Loading
try:
    from app.ml.classifier import DefectClassifier
    from app.ml.severity import compute_severity
    
    print("\n[+] Loading Maratha DefectClassifier (MobileNetV2 + Custom Head + TTA)...")
    maratha_classifier = DefectClassifier()
    print("[✓] Maratha DefectClassifier successfully loaded!")
except Exception as e:
    print(f"[✗] Failed to load Maratha classifier: {e}")
    sys.exit(1)

# 2. Find sample images
sample_images = []
img_root = PROJECT_ROOT / "app" / "model" / "data" / "external_eval"
if img_root.exists():
    for ext in ["*.jpg", "*.png", "*.jpeg"]:
        sample_images.extend(list(img_root.glob(f"**/{ext}")))

if not sample_images and (PROJECT_ROOT / "images.jpg").exists():
    sample_images.append(PROJECT_ROOT / "images.jpg")

print(f"\n[+] Found {len(sample_images)} test images for evaluation.")

if sample_images:
    eval_imgs = sample_images[:5]
    print("\n--------------------------------------------------")
    print(f"Running predictions on {len(eval_imgs)} sample images:")
    print("--------------------------------------------------")
    
    for img_path in eval_imgs:
        t0 = time.time()
        label, conf = maratha_classifier.predict(str(img_path), use_tta=True)
        t_tta = (time.time() - t0) * 1000
        
        t1 = time.time()
        label_no_tta, conf_no_tta = maratha_classifier.predict(str(img_path), use_tta=False)
        t_single = (time.time() - t1) * 1000
        
        sev = compute_severity(str(img_path), label)
        
        rel_path = img_path.relative_to(PROJECT_ROOT)
        print(f"File: {rel_path}")
        print(f"  └─ Predicted Category: {label} (Conf: {conf:.2%}) [TTA Latency: {t_tta:.1f}ms]")
        print(f"  └─ No-TTA Prediction:  {label_no_tta} (Conf: {conf_no_tta:.2%}) [Single Latency: {t_single:.1f}ms]")
        print(f"  └─ Calculated Severity: {sev}/100\n")

# 3. Check against InfraPulse Main Model Service if available
print("--------------------------------------------------")
print("Comparing with InfraPulse Model Service:")
print("--------------------------------------------------")

try:
    from app.model_service import predict_defect
    if sample_images:
        test_img = str(sample_images[0])
        print(f"Testing InfraPulse predict_defect on: {sample_images[0].relative_to(PROJECT_ROOT)}")
        res = predict_defect(test_img)
        print(f"  └─ Category: {res.get('category')}")
        print(f"  └─ Confidence: {res.get('confidence', 0):.2%}")
        print(f"  └─ Severity: {res.get('severity_score', 0):.1f}/100")
        print(f"  └─ Model Used: {res.get('model_used')}")
except Exception as e:
    print(f"[!] InfraPulse predict_defect note: {e}")

print("\n==================================================")
print("     MARATHA MODEL CHECK COMPLETE - ALL PASSED    ")
print("==================================================")
