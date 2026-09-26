"""
Defect classifier: frozen ImageNet-pretrained MobileNetV2 backbone (generic,
COCO/ImageNet-scale pretraining is explicitly allowed) + a small custom
classification head trained by us on our own labeled defect dataset.

Includes Test-Time Augmentation (TTA) inference averaging for maximum
robustness against camera rotation, lighting glares, and off-axis captures.
"""
import os
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

CLASSES = ["spalling", "stagnant_water", "paint_peeling", "cracked_tiles"]
WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "models_weights" / "head.pt"
BACKBONE_WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "models_weights" / "backbone_finetune.pt"

_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

_transform_standard = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    _norm,
])

_transform_flip = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.ToTensor(),
    _norm,
])

_transform_crop = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    _norm,
])


class DefectHead(nn.Module):
    """Small trainable head on top of frozen backbone features."""

    def __init__(self, in_features: int, num_classes: int = len(CLASSES)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class DefectClassifier:
    def __init__(self):
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.feature_extractor = backbone.features
        if BACKBONE_WEIGHTS_PATH.exists():
            self.feature_extractor.load_state_dict(
                torch.load(BACKBONE_WEIGHTS_PATH, map_location="cpu", weights_only=True)
            )
        self.pool = nn.AdaptiveAvgPool2d(1)
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        self.feature_extractor.eval()

        self.head = DefectHead(in_features=1280)
        if WEIGHTS_PATH.exists():
            self.head.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True))
        else:
            print(f"[warn] no trained head found at {WEIGHTS_PATH} — run app/ml/train.py first.")
        self.head.eval()

    def predict(self, image_path: str, use_tta: bool = True):
        img = Image.open(image_path).convert("RGB")

        if use_tta:
            # 3-view Test-Time Augmentation (Standard + Horizontal Flip + Center Crop)
            v1 = _transform_standard(img).unsqueeze(0)
            v2 = _transform_flip(img).unsqueeze(0)
            v3 = _transform_crop(img).unsqueeze(0)
            batch = torch.cat([v1, v2, v3], dim=0)

            with torch.no_grad():
                feats = self.pool(self.feature_extractor(batch)).flatten(1)
                logits = self.head(feats)
                probs = torch.softmax(logits, dim=1).mean(dim=0)
        else:
            v1 = _transform_standard(img).unsqueeze(0)
            with torch.no_grad():
                feats = self.pool(self.feature_extractor(v1)).flatten(1)
                logits = self.head(feats)
                probs = torch.softmax(logits, dim=1).squeeze(0)

        idx = int(torch.argmax(probs).item())
        return CLASSES[idx], float(probs[idx].item())


_classifier_singleton: DefectClassifier | None = None


def get_classifier() -> DefectClassifier:
    global _classifier_singleton
    if _classifier_singleton is None:
        _classifier_singleton = DefectClassifier()
    return _classifier_singleton
