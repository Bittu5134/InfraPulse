import argparse
import json

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)
from torch import nn
from tqdm import tqdm

from dataset import make_loaders
from model import InfraPulseNet

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    ckpt = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False
    )

    image_size = int(ckpt.get("image_size", 224))

    _, val_loader, test_loader, class_to_idx = make_loaders(
        args.data,
        image_size=image_size,
        batch_size=args.batch_size,
        num_workers=args.workers
    )

    loader = test_loader if test_loader is not None else val_loader

    class_names = [
        name for name, idx in sorted(
            class_to_idx.items(),
            key=lambda kv: kv[1]
        )
    ]

    model = InfraPulseNet(
        num_classes=len(class_names),
        pretrained=False
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    y_true = []
    y_pred = []

    with torch.inference_mode():
        for x, y in tqdm(loader):
            x = x.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1).cpu()

            y_true.extend(y.tolist())
            y_pred.extend(pred.tolist())

    print("\nAccuracy:", accuracy_score(y_true, y_pred))
    print("Macro F1:", f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    ))

    print("\nClassification report:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    ))

    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    main()
