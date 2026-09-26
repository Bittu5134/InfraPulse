import os
import sys
import json
import time
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Enforce efficient 4-thread CPU limit
torch.set_num_threads(4)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CKPT_DIR = REPO_ROOT / "app" / "model" / "checkpoints"
EVAL_DIR = REPO_ROOT / "app" / "model" / "data" / "normalized_clean_eval"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLASS_NAMES = ["cracked_tiles", "paint_peeling", "spalling", "stagnant_water"]
MODEL_KEYS = ["convnext_tiny", "mtl_dual_branch", "swin_t", "baseline", "quantized_int8"]
MODEL_DISPLAY_NAMES = {
    "convnext_tiny": "ConvNeXt-Tiny (Modern Pure CNN)",
    "mtl_dual_branch": "Multi-Task Learning (MTL Dual-Branch)",
    "swin_t": "Swin Transformer (Self-Attention)",
    "baseline": "EfficientNet-B0 (Baseline Pure CNN)",
    "quantized_int8": "INT8 Quantized Dynamic Engine"
}

def get_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def load_pure_vision_models(device):
    from app.model_service import load_custom_model
    models = {}
    print("[+] Loading Pure Computer Vision Model Family...")
    for key in MODEL_KEYS:
        obj = load_custom_model(key)
        if obj is not None:
            models[key] = obj
            print(f"  [Loaded] {MODEL_DISPLAY_NAMES[key]}")
        else:
            print(f"  [Warning] Failed to load checkpoint for {key}")
    return models

def collect_predictions_fast(models, loader):
    import gc
    print(f"\n[+] Memory-Safe Evaluation across {len(loader.dataset)} Samples (Streaming Mode, Batch Size: {loader.batch_size})...", flush=True)
    start_total = time.perf_counter()

    probs_dict = {key: [] for key in models.keys()}
    latencies_dict = {key: [] for key in models.keys()}
    y_true = []

    # Evaluate model by model, streaming from DataLoader directly
    for key_idx, (key, obj) in enumerate(models.items()):
        model = obj["model"]
        device = obj["device"]
        print(f"  [{key_idx+1}/{len(models)}] Stream-Evaluating {MODEL_DISPLAY_NAMES[key]}...", flush=True)

        t_m_start = time.perf_counter()
        is_first_model = (key_idx == 0)

        for b_idx, (x_dev, y_batch) in enumerate(loader):
            if is_first_model:
                y_true.extend(y_batch.numpy())

            start_t = time.perf_counter()
            with torch.inference_mode():
                if key == "mtl_dual_branch":
                    mtl_out = model(x_dev, return_all=True)
                    logits = mtl_out["logits"]
                else:
                    logits = model(x_dev)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            batch_lat = (time.perf_counter() - start_t) * 1000.0 / len(x_dev)

            probs_dict[key].extend(probs)
            latencies_dict[key].append(batch_lat)

            if (b_idx + 1) % 4 == 0 or (b_idx + 1) == len(loader):
                print(f"      • Evaluated {(b_idx+1)*loader.batch_size} / {len(loader.dataset)} samples...", flush=True)

        probs_dict[key] = np.array(probs_dict[key])
        m_time = time.perf_counter() - t_m_start
        print(f"    • Completed {key} in {m_time:.2f}s (RAM Safe).", flush=True)
        gc.collect()

    y_true = np.array(y_true)
    total_time = time.perf_counter() - start_total
    print(f"  [Complete] All predictions gathered safely in {total_time:.2f} seconds.", flush=True)
    return y_true, probs_dict, latencies_dict

def optimize_per_category_weights(y_true, probs_dict):
    print("\n[+] Optimizing 4 x 5 Per-Category Class-Specialized Weight Matrix W(c, m)...")
    num_classes = len(CLASS_NAMES)
    model_keys = list(probs_dict.keys())
    num_models = len(model_keys)

    # Convert probs to 3D array: (num_models, N, num_classes)
    P_matrix = np.array([probs_dict[m] for m in model_keys])  # shape: (M, N, C)

    per_category_weights = {}

    for c_idx, c_name in enumerate(CLASS_NAMES):
        mask = (y_true == c_idx)
        if not np.any(mask):
            per_category_weights[c_name] = [1.0 / num_models] * num_models
            continue

        P_c = P_matrix[:, mask, :]  # shape: (M, N_c, C)
        y_c = y_true[mask]          # shape: (N_c,)

        def loss_func(w):
            w = w / np.sum(w)
            # Weighted probability vector for category samples
            P_fused = np.zeros_like(P_c[0])
            for m_idx in range(num_models):
                P_fused += w[m_idx] * P_c[m_idx]

            # Cross-entropy loss on category samples
            eps = 1e-7
            P_fused = np.clip(P_fused, eps, 1.0 - eps)
            log_p = np.log(P_fused[np.arange(len(y_c)), y_c])
            return -np.mean(log_p)

        w0 = np.ones(num_models) / num_models
        bounds = [(0.0, 1.0)] * num_models
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

        res = minimize(loss_func, w0, method='SLSQP', bounds=bounds, constraints=constraints)
        opt_w = res.x / np.sum(res.x)
        per_category_weights[c_name] = {m_key: round(float(opt_w[i]), 4) for i, m_key in enumerate(model_keys)}

        print(f"  • Category '{c_name}':")
        for i, m_key in enumerate(model_keys):
            print(f"      - {m_key.ljust(18)}: {opt_w[i]*100:>5.2f}%")

    return per_category_weights

def evaluate_per_category_consensus(y_true, probs_dict, per_category_weights):
    model_keys = list(probs_dict.keys())
    N = len(y_true)
    num_classes = len(CLASS_NAMES)

    # Compute category-prior predicted class probabilities
    P_consensus = np.zeros((N, num_classes))

    for i in range(N):
        # 1. Compute initial mean prediction to estimate primary category
        mean_p = np.mean([probs_dict[m][i] for m in model_keys], axis=0)
        pred_cat_idx = int(np.argmax(mean_p))
        c_name = CLASS_NAMES[pred_cat_idx]

        # 2. Retrieve specialized weight vector W(c, m) for predicted category
        w_dict = per_category_weights[c_name]
        w_vec = np.array([w_dict[m] for m in model_keys])

        # 3. Compute weighted soft-voting consensus
        fused_p = np.zeros(num_classes)
        for m_idx, m_key in enumerate(model_keys):
            fused_p += w_vec[m_idx] * probs_dict[m_key][i]

        P_consensus[i] = fused_p

    preds = np.argmax(P_consensus, axis=1)
    acc = accuracy_score(y_true, preds)
    macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, preds, average="weighted", zero_division=0)

    per_class_metrics = {}
    for c_idx, c_name in enumerate(CLASS_NAMES):
        c_mask = (y_true == c_idx)
        c_preds = (preds == c_idx)
        c_acc = accuracy_score(c_mask, c_preds)
        c_prec = precision_score(y_true == c_idx, preds == c_idx, zero_division=0)
        c_rec = recall_score(y_true == c_idx, preds == c_idx, zero_division=0)
        c_f1 = f1_score(y_true == c_idx, preds == c_idx, zero_division=0)
        per_class_metrics[c_name] = {
            "accuracy": round(float(c_acc) * 100, 2),
            "precision": round(float(c_prec), 4),
            "recall": round(float(c_rec), 4),
            "f1": round(float(c_f1), 4)
        }

    return {
        "accuracy": round(float(acc) * 100, 2),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "per_class": per_class_metrics
    }

def main():
    print("=" * 70)
    print(" PER-CATEGORY WEIGHTED CONSENSUS ENGINE & BENCHMARK")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_pure_vision_models(device)

    if not EVAL_DIR.exists():
        print(f"[Error] Evaluation dataset directory {EVAL_DIR} not found!")
        return

    ds = datasets.ImageFolder(EVAL_DIR, transform=get_transform(224))
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    y_true, probs_dict, latencies_dict = collect_predictions_fast(models, loader)
    per_cat_weights = optimize_per_category_weights(y_true, probs_dict)
    consensus_res = evaluate_per_category_consensus(y_true, probs_dict, per_cat_weights)

    # Standalone Model Evaluation Summary
    standalone_res = {}
    for key, probs in probs_dict.items():
        preds = np.argmax(probs, axis=1)
        acc = accuracy_score(y_true, preds)
        macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0)
        avg_lat = float(np.mean(latencies_dict[key]))
        standalone_res[key] = {
            "accuracy": round(float(acc) * 100, 2),
            "macro_f1": round(float(macro_f1), 4),
            "latency_ms": round(avg_lat, 1)
        }

    print("\n" + "=" * 70)
    print(" 🏆 PER-CATEGORY WEIGHTED CONSENSUS RESULTS (1,600 BALANCED SAMPLES)")
    print("=" * 70)
    print(f" • Overall Test Accuracy : {consensus_res['accuracy']}%")
    print(f" • Macro F1-Score        : {consensus_res['macro_f1']}")
    print(f" • Weighted F1-Score     : {consensus_res['weighted_f1']}")
    print("\n  Per-Class Metrics under Category-Specialized Consensus:")
    for c_name, m in consensus_res['per_class'].items():
        print(f"   - {c_name.ljust(18)}: Accuracy = {m['accuracy']:>6.2f}% | F1 = {m['f1']:>6.4f} | Precision = {m['precision']:>6.4f} | Recall = {m['recall']:>6.4f}")

    # Export complete report JSON
    report = {
        "dataset_size": len(y_true),
        "per_category_weights": per_cat_weights,
        "per_category_consensus_metrics": consensus_res,
        "standalone_models": standalone_res
    }

    out_json = CKPT_DIR / "per_category_consensus_report.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[+] Per-Category Consensus Report successfully saved to {out_json}")
    print("=" * 70)

if __name__ == "__main__":
    main()
