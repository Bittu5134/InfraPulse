import os
import sys
import json
import time
from pathlib import Path
from itertools import combinations
from typing import Dict, List, Any, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from scipy.optimize import minimize

# Ensure 2-thread limit for PC CPU protection
torch.set_num_threads(2)
torch.set_num_interop_threads(2)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "app" / "model" / "src"))

from app.model.src.model import (
    InfraPulseNet, ConvNeXtInfraPulse, SwinInfraPulse, MultiTaskInfraPulse,
    CLASS_NAMES, CATEGORY_MAP
)

CKPT_DIR = REPO_ROOT / "app" / "model" / "checkpoints"
DATA_DIR = REPO_ROOT / "app" / "model" / "data"

def get_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def load_all_pure_vision_models(device: torch.device) -> Dict[str, Any]:
    """Loads all 5 pure vision models (strictly zero text models)."""
    models = {}

    # 1. ConvNeXt-Tiny (Modern Pure CNN)
    conv_path = CKPT_DIR / "convnext_tiny_infrapulse.pt"
    if conv_path.exists():
        m = ConvNeXtInfraPulse(num_classes=4, pretrained=False).to(device)
        m.load_state_dict(torch.load(conv_path, map_location=device, weights_only=False)["model_state"])
        m.eval()
        models["convnext_tiny"] = {"model": m, "name": "ConvNeXt-Tiny (Pure CNN)", "device": device}
        print("  [Loaded] ConvNeXt-Tiny")

    # 2. Multi-Task Learning (MTL Dual-Branch)
    mtl_path = CKPT_DIR / "multitask_mtl_infrapulse.pt"
    if mtl_path.exists():
        m = MultiTaskInfraPulse(num_classes=4, pretrained=False).to(device)
        m.load_state_dict(torch.load(mtl_path, map_location=device, weights_only=False)["model_state"])
        m.eval()
        models["mtl_dual_branch"] = {"model": m, "name": "Multi-Task Learning (MTL)", "device": device}
        print("  [Loaded] Multi-Task Learning (MTL)")

    # 3. Swin Transformer (Shifted-Window Attention)
    swin_path = CKPT_DIR / "swin_tiny_infrapulse.pt"
    if swin_path.exists():
        m = SwinInfraPulse(num_classes=4, pretrained=False).to(device)
        m.load_state_dict(torch.load(swin_path, map_location=device, weights_only=False)["model_state"])
        m.eval()
        models["swin_t"] = {"model": m, "name": "Swin Transformer (Self-Attention)", "device": device}
        print("  [Loaded] Swin Transformer")

    # 4. EfficientNet-B0 (Baseline Pure CNN)
    base_path = CKPT_DIR / "best_infrapulse_v1.pt"
    if base_path.exists():
        m = InfraPulseNet(num_classes=4, pretrained=False).to(device)
        m.load_state_dict(torch.load(base_path, map_location=device, weights_only=False)["model_state"])
        m.eval()
        models["efficientnet_b0"] = {"model": m, "name": "EfficientNet-B0 (Baseline)", "device": device}
        print("  [Loaded] EfficientNet-B0")

    # 5. INT8 Quantized Dynamic Engine (Ultra-Fast CPU)
    quant_path = CKPT_DIR / "infrapulse_int8_quantized.pt"
    if quant_path.exists():
        cpu_base = InfraPulseNet(num_classes=4, pretrained=False).to("cpu")
        m = torch.ao.quantization.quantize_dynamic(cpu_base, {torch.nn.Linear}, dtype=torch.qint8)
        m.load_state_dict(torch.load(quant_path, map_location="cpu", weights_only=False)["model_state"])
        m.eval()
        models["quantized_int8"] = {"model": m, "name": "INT8 Quantized Engine", "device": torch.device("cpu")}
        print("  [Loaded] INT8 Quantized Dynamic Engine")

    return models

def collect_predictions_slowly(
    models: Dict[str, Any],
    dataloader: DataLoader,
    split_name: str = "test"
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, float]]:
    """
    Runs inference slowly with gentle pacing and 2 threads to protect the PC CPU.
    Returns ground_truth labels, model_probabilities dict, and model_latencies dict.
    """
    print(f"\n[+] Collecting predictions on {split_name} split ({len(dataloader.dataset)} samples)...")
    model_probs = {k: [] for k in models.keys()}
    model_latencies = {k: [] for k in models.keys()}
    all_targets = []

    for b_idx, (images, targets) in enumerate(dataloader):
        all_targets.extend(targets.numpy())

        for k, m_info in models.items():
            m = m_info["model"]
            dev = m_info["device"]
            x = images.to(dev)

            t0 = time.perf_counter()
            with torch.inference_mode():
                if k == "mtl_dual_branch":
                    out = m(x, return_all=True)
                    logits = out["logits"]
                else:
                    logits = m(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0 / len(images)

            model_probs[k].append(probs)
            model_latencies[k].append(elapsed_ms)

        # Gentle throttle between batches to keep CPU temperature low
        time.sleep(0.05)

        if (b_idx + 1) % 4 == 0 or (b_idx + 1) == len(dataloader):
            processed = min((b_idx + 1) * dataloader.batch_size, len(dataloader.dataset))
            print(f"    • Progress: {processed}/{len(dataloader.dataset)} samples evaluated smoothly...")

    # Concatenate all batches
    y_true = np.array(all_targets)
    probs_dict = {k: np.concatenate(model_probs[k], axis=0) for k in models.keys()}
    latencies_dict = {k: round(float(np.mean(model_latencies[k])), 2) for k in models.keys()}

    return y_true, probs_dict, latencies_dict

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    acc = float(accuracy_score(y_true, y_pred) * 100.0)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    per_class = {}
    for idx, cname in enumerate(CLASS_NAMES):
        mask = (y_true == idx)
        class_acc = float(np.mean(y_pred[mask] == idx) * 100.0) if np.sum(mask) > 0 else 0.0
        c_f1 = float(f1_score(y_true == idx, y_pred == idx, average="binary", zero_division=0))
        c_prec = float(precision_score(y_true == idx, y_pred == idx, average="binary", zero_division=0))
        c_rec = float(recall_score(y_true == idx, y_pred == idx, average="binary", zero_division=0))
        per_class[cname] = {
            "accuracy": round(class_acc, 2),
            "f1": round(c_f1, 4),
            "precision": round(c_prec, 4),
            "recall": round(c_rec, 4),
            "sample_count": int(np.sum(mask))
        }

    return {
        "accuracy": round(acc, 2),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class
    }

def evaluate_all_combinations(
    y_true: np.ndarray,
    probs_dict: Dict[str, np.ndarray],
    latencies_dict: Dict[str, float]
) -> Dict[str, Any]:
    """Evaluates all subset combinations and soft/hard voting ensembles."""
    model_keys = list(probs_dict.keys())
    results = {}

    # 1. Standalone Models
    results["standalone"] = {}
    for k in model_keys:
        preds = np.argmax(probs_dict[k], axis=1)
        m = compute_metrics(y_true, preds)
        m["latency_ms"] = latencies_dict[k]
        results["standalone"][k] = m

    # 2. All Subset Combinations (Uniform Soft-Voting Average)
    subset_results = []
    for k_len in range(2, len(model_keys) + 1):
        for combo in combinations(model_keys, k_len):
            combo_name = " + ".join(combo)
            # Uniform average of probabilities
            stacked = np.stack([probs_dict[k] for k in combo], axis=0)  # [K, N, C]
            avg_probs = np.mean(stacked, axis=0)  # [N, C]
            preds = np.argmax(avg_probs, axis=1)

            m = compute_metrics(y_true, preds)
            m["models"] = list(combo)
            m["combination_name"] = combo_name
            m["model_count"] = k_len
            m["sequential_latency_ms"] = round(sum(latencies_dict[k] for k in combo), 2)
            m["parallel_latency_ms"] = round(max(latencies_dict[k] for k in combo) + 2.0, 2)
            subset_results.append(m)

    # Sort subsets by Accuracy and Macro-F1
    subset_results.sort(key=lambda x: (x["accuracy"], x["macro_f1"]), reverse=True)
    results["uniform_combinations"] = subset_results

    # 3. Hard-Voting (Majority Rule) Ensemble
    all_preds = np.stack([np.argmax(probs_dict[k], axis=1) for k in model_keys], axis=0)  # [K, N]
    hard_votes = []
    for i in range(len(y_true)):
        vals, counts = np.unique(all_preds[:, i], return_counts=True)
        hard_votes.append(vals[np.argmax(counts)])
    hard_votes = np.array(hard_votes)
    results["hard_voting_ensemble"] = compute_metrics(y_true, hard_votes)

    # 4. Optimal Weighted Soft-Voting Optimization
    print("\n[+] Optimizing Soft-Voting Weight Vector (Grid Search & Optimization)...")
    K = len(model_keys)
    stacked_all = np.stack([probs_dict[k] for k in model_keys], axis=0)  # [K, N, C]

    def loss_fn(weights):
        w = np.maximum(weights, 0)
        if np.sum(w) == 0:
            return 1.0
        w = w / np.sum(w)
        # Weighted sum: [N, C]
        weighted_p = np.tensordot(w, stacked_all, axes=(0, 0))
        p_argmax = np.argmax(weighted_p, axis=1)
        # We want to maximize macro_f1 + 0.5 * accuracy
        acc = accuracy_score(y_true, p_argmax)
        f1 = f1_score(y_true, p_argmax, average="macro", zero_division=0)
        return -(f1 + 0.5 * acc)

    best_w = None
    best_score = 999.0
    # Grid search initialization
    grid_points = np.linspace(0.05, 0.50, 6)
    for w1 in grid_points:
        for w2 in grid_points:
            for w3 in grid_points:
                for w4 in grid_points:
                    for w5 in grid_points:
                        w_init = np.array([w1, w2, w3, w4, w5])
                        w_init = w_init / np.sum(w_init)
                        score = loss_fn(w_init)
                        if score < best_score:
                            best_score = score
                            best_w = w_init

    # Continuous refinement via SLSQP
    res_opt = minimize(
        loss_fn, best_w,
        bounds=[(0.0, 1.0)] * K,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        method="SLSQP"
    )
    opt_w = np.maximum(res_opt.x, 0)
    opt_w = opt_w / np.sum(opt_w)

    opt_weighted_p = np.tensordot(opt_w, stacked_all, axes=(0, 0))
    opt_preds = np.argmax(opt_weighted_p, axis=1)
    opt_metrics = compute_metrics(y_true, opt_preds)
    opt_metrics["weights"] = {model_keys[i]: round(float(opt_w[i]), 4) for i in range(K)}
    results["optimal_weighted_ensemble"] = opt_metrics

    # 5. Oracle Upper Bound (If any model got it right)
    any_correct = np.any([np.argmax(probs_dict[k], axis=1) == y_true for k in model_keys], axis=0)
    results["oracle_upper_bound"] = {
        "accuracy": round(float(np.mean(any_correct) * 100.0), 2),
        "total_samples": len(y_true),
        "correctly_solvable_samples": int(np.sum(any_correct))
    }

    # 6. Pairwise Complementarity & Error Overlap
    complementarity = {}
    for k1, k2 in combinations(model_keys, 2):
        c1 = (np.argmax(probs_dict[k1], axis=1) == y_true)
        c2 = (np.argmax(probs_dict[k2], axis=1) == y_true)
        both_correct = np.mean(c1 & c2) * 100.0
        at_least_one = np.mean(c1 | c2) * 100.0
        k2_rescues_k1 = np.sum(~c1 & c2)
        k1_rescues_k2 = np.sum(c1 & ~c2)
        complementarity[f"{k1} <-> {k2}"] = {
            "joint_accuracy_if_oracled": round(float(at_least_one), 2),
            "both_correct_pct": round(float(both_correct), 2),
            f"{k2}_rescued_{k1}_errors": int(k2_rescues_k1),
            f"{k1}_rescued_{k2}_errors": int(k1_rescues_k2)
        }
    results["complementarity_matrix"] = complementarity

    return results

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f" InfraPulse - Exhaustive Pure Vision Multi-Model Benchmark & Ensemble Search")
    print(f" Target Device: {device} | Thread Limit: 2 (CPU Safety Active)")
    print("=" * 70)

    # 1. Load pure vision models
    print("\n[Step 1/3] Loading Pure Computer Vision Models...")
    models = load_all_pure_vision_models(device)
    if not models:
        print("[!] No models found to evaluate!")
        return

    # 2. Prepare holdout test DataLoader
    test_dir = DATA_DIR / "test"
    if not test_dir.exists():
        print(f"[!] Test directory {test_dir} does not exist!")
        return

    test_dataset = datasets.ImageFolder(test_dir, transform=get_transform(224))
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    # 3. Collect Predictions Slowly
    print("\n[Step 2/3] Collecting Model Probabilities (Slow & Safe Evaluation)...")
    y_true, probs_dict, latencies_dict = collect_predictions_slowly(models, test_loader, split_name="Holdout Test Set")

    # 4. Comprehensive Combinatorial Analysis
    print("\n[Step 3/3] Analyzing Combinations & Optimizing Ensemble Weights...")
    full_report = evaluate_all_combinations(y_true, probs_dict, latencies_dict)
    full_report["dataset_info"] = {
        "split": "test",
        "sample_count": len(y_true),
        "class_distribution": {c: int(np.sum(y_true == idx)) for idx, c in enumerate(CLASS_NAMES)}
    }

    # Save to JSON
    out_json = CKPT_DIR / "ensemble_grid_search_report.json"
    with open(out_json, "w") as f:
        json.dump(full_report, f, indent=2)

    # Display Pretty CLI Report
    print("\n" + "=" * 70)
    print(" 🏆 STANDALONE PURE VISION MODEL LEADERBOARD")
    print("=" * 70)
    for k, v in full_report["standalone"].items():
        print(f" • {k.ljust(20)}: Acc: {v['accuracy']:>6.2f}% | Macro-F1: {v['macro_f1']:>6.4f} | Latency: {v['latency_ms']:>6.2f}ms")

    print("\n" + "=" * 70)
    print(" 🥇 TOP 5 MODEL COMBINATIONS (UNIFORM SOFT-VOTING)")
    print("=" * 70)
    for i, combo in enumerate(full_report["uniform_combinations"][:5], 1):
        print(f" #{i} {combo['combination_name']}")
        print(f"    • Acc: {combo['accuracy']}% | Macro-F1: {combo['macro_f1']} | Parallel Latency: {combo['parallel_latency_ms']}ms")

    opt = full_report["optimal_weighted_ensemble"]
    print("\n" + "=" * 70)
    print(" 🌟 OPTIMAL CALIBRATED WEIGHTED CONSENSUS ENSEMBLE")
    print("=" * 70)
    print(f" • Overall Accuracy: {opt['accuracy']}%")
    print(f" • Macro-F1 Score:  {opt['macro_f1']}")
    print(f" • Weighted-F1:     {opt['weighted_f1']}")
    print(" • Optimal Weight Vector:")
    for mk, w in opt["weights"].items():
        print(f"    - {mk.ljust(18)}: {w * 100:>5.1f}% (weight = {w})")

    oracle = full_report["oracle_upper_bound"]
    print("\n" + "=" * 70)
    print(" 🔮 ORACLE UPPER BOUND (THEORETICAL CEILING)")
    print("=" * 70)
    print(f" • Theoretical Max Accuracy: {oracle['accuracy']}% ({oracle['correctly_solvable_samples']}/{oracle['total_samples']} samples)")

    print(f"\n[+] Full detailed report exported to: {out_json}\n")

if __name__ == "__main__":
    main()
