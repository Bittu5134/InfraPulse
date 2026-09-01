import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from dataset import make_loaders
from model import InfraPulseNet, CLASS_NAMES

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_class_weights(dataset, device):
    targets = np.array(dataset.targets)
    classes, counts = np.unique(targets, return_counts=True)

    weights = np.ones(len(dataset.classes), dtype=np.float32)
    for c, count in zip(classes, counts):
        weights[c] = len(targets) / (len(classes) * count)

    return torch.tensor(weights, dtype=torch.float32, device=device)

def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    losses = []
    ys_true = []
    ys_pred = []

    pbar = tqdm(loader, leave=False)

    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = criterion(logits, y)

        if is_train:
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        preds = logits.argmax(dim=1)

        ys_true.extend(y.detach().cpu().tolist())
        ys_pred.extend(preds.detach().cpu().tolist())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    acc = accuracy_score(ys_true, ys_pred)
    macro_f1 = f1_score(ys_true, ys_pred, average="macro", zero_division=0)

    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1)
    }

def save_ckpt(path, model, class_to_idx, image_size, score, phase):
    payload = {
        "model_state": model.state_dict(),
        "class_to_idx": class_to_idx,
        "image_size": image_size,
        "score": score,
        "phase": phase,
    }
    torch.save(payload, path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=25,
                    help="Phase-2 fine-tuning epochs")
    ap.add_argument("--phase1-epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--head-lr", type=float, default=1e-4)
    ap.add_argument("--backbone-lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="checkpoints/best_infrapulse_v1.pt")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, class_to_idx = make_loaders(
        args.data,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.workers
    )

    expected = set(CLASS_NAMES)
    actual = set(class_to_idx.keys())
    if expected != actual:
        raise ValueError(
            f"Dataset classes must be exactly {sorted(expected)}, "
            f"but found {sorted(actual)}"
        )

    print("Device:", device)
    print("Classes:", class_to_idx)

    model = InfraPulseNet(num_classes=4, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=compute_class_weights(train_loader.dataset, device),
        label_smoothing=0.05
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------------- PHASE 1 ----------------
    print("\n========== PHASE 1: TRAIN HEAD ==========")
    model.freeze_backbone()

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.head_lr,
        weight_decay=args.weight_decay
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.phase1_epochs)
    )

    best_f1 = -1.0

    for epoch in range(1, args.phase1_epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_metrics = run_epoch(
            model, val_loader, criterion, device
        )

        scheduler.step()

        print(
            f"[Phase 1][{epoch}/{args.phase1_epochs}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_f1={train_metrics['macro_f1']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            save_ckpt(
                out_path, model, class_to_idx,
                args.image_size, best_f1, "phase1"
            )

    # ---------------- PHASE 2 ----------------
    print("\n====== PHASE 2: FINE-TUNE LAST 2 BLOCKS ======")
    model.unfreeze_last_blocks(num_blocks=2)

    head_params = list(model.backbone.classifier.parameters())

    backbone_params = [
        p for p in model.backbone.features.parameters()
        if p.requires_grad
    ]

    optimizer = AdamW(
        [
            {
                "params": backbone_params,
                "lr": args.backbone_lr
            },
            {
                "params": head_params,
                "lr": args.head_lr
            },
        ],
        weight_decay=args.weight_decay
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs)
    )

    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer
        )

        val_metrics = run_epoch(
            model, val_loader, criterion, device
        )

        scheduler.step()

        print(
            f"[Phase 2][{epoch}/{args.epochs}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_f1={train_metrics['macro_f1']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            patience_counter = 0

            save_ckpt(
                out_path, model, class_to_idx,
                args.image_size, best_f1, "phase2"
            )
            print(f"[SAVE] {out_path}  macro-F1={best_f1:.4f}")

        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print("Early stopping.")
            break

    print("\nTraining complete.")
    print("Best validation macro-F1:", round(best_f1, 4))
    print("Checkpoint:", out_path)

if __name__ == "__main__":
    main()
