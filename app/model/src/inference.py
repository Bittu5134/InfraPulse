import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from model import InfraPulseNet, CATEGORY_MAP
from priority import analyze_heatmap, compute_priority
from fallback import fallback_analysis

try:
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRADCAM_AVAILABLE = True
except Exception:
    GRADCAM_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class InfraPulseInference:
    def __init__(
        self,
        checkpoint,
        device=None,
        allow_fallback=True
    ):
        self.checkpoint = checkpoint
        self.allow_fallback = allow_fallback
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = None
        self.class_names = None
        self.image_size = 224
        self.loaded = False
        self.gradcam = None

        self._load()

    def _load(self):
        ckpt = torch.load(
            self.checkpoint,
            map_location=self.device,
            weights_only=False
        )

        self.image_size = int(ckpt.get("image_size", 224))
        class_to_idx = ckpt["class_to_idx"]

        self.class_names = [
            name for name, idx in sorted(
                class_to_idx.items(),
                key=lambda kv: kv[1]
            )
        ]

        self.model = InfraPulseNet(
            num_classes=len(self.class_names),
            pretrained=False
        ).to(self.device)

        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        if GRADCAM_AVAILABLE:
            self.gradcam = GradCAMPlusPlus(
                model=self.model,
                target_layers=[self.model.get_gradcam_target_layer()]
            )

        self.loaded = True

        logging.info("ML inference pipeline loaded successfully")
        logging.info("MODEL_MODE = ML")
        logging.info("MODEL_LOADED = True")
        logging.info(
            "GRADCAM_AVAILABLE = %s",
            GRADCAM_AVAILABLE
        )

    def _transform(self):
        return transforms.Compose([
            transforms.Resize(int(self.image_size * 1.14)),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def predict(self, image_path, age_hours=0.0):
        try:
            if not self.loaded:
                raise RuntimeError("Model not loaded")

            pil = Image.open(image_path).convert("RGB")
            original_rgb = np.array(pil)

            x = self._transform()(pil).unsqueeze(0).to(self.device)

            with torch.inference_mode():
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1)[0]

            pred_idx = int(torch.argmax(probs).item())
            defect = self.class_names[pred_idx]
            confidence = float(probs[pred_idx].item())

            if not GRADCAM_AVAILABLE:
                raise RuntimeError(
                    "pytorch-grad-cam is unavailable"
                )

            targets = [ClassifierOutputTarget(pred_idx)]

            # GradCAM++ internally requires gradients, so do not wrap this in
            # torch.inference_mode().
            grayscale_cam = self.gradcam(
                input_tensor=x,
                targets=targets
            )[0]

            analysis = analyze_heatmap(
                grayscale_cam,
                original_rgb
            )

            priority = compute_priority(
                defect_class=defect,
                severity=analysis["severity"],
                extent=analysis["extent"],
                age_hours=age_hours
            )

            result = {
                "defect": defect,
                "category": CATEGORY_MAP[defect],
                "confidence": confidence,
                "severity": analysis["severity"],
                "extent": analysis["extent"],
                "priority_score": priority,
                "age_hours": float(age_hours),
                "fallback_used": False,
                "diagnostics": analysis,
            }

            logging.info("FALLBACK_USED = False")
            return result

        except Exception as e:
            logging.exception("Primary ML pipeline failed: %s", e)

            if not self.allow_fallback:
                raise

            pil = Image.open(image_path).convert("RGB")
            rgb = np.array(pil)

            result = fallback_analysis(rgb)
            defect = result["defect"]

            result["category"] = CATEGORY_MAP[defect]
            result["priority_score"] = compute_priority(
                defect,
                result["severity"],
                result["extent"],
                age_hours
            )
            result["age_hours"] = float(age_hours)

            logging.warning("MODEL_MODE = FALLBACK")
            logging.warning("FALLBACK_USED = True")

            return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--age-hours", type=float, default=0.0)
    ap.add_argument(
        "--no-fallback",
        action="store_true",
        help="Crash instead of silently using heuristic fallback"
    )
    args = ap.parse_args()

    engine = InfraPulseInference(
        args.checkpoint,
        allow_fallback=not args.no_fallback
    )

    result = engine.predict(
        args.image,
        age_hours=args.age_hours
    )

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
