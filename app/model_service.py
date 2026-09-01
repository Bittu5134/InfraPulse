import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional

from app.config import BASE_DIR
from app.models import CategoryEnum
from app.priority_queue import compute_priority_score, mock_classify_defect

# Add app/model/src to sys.path so model, priority, fallback imports work
MODEL_SRC_DIR = BASE_DIR / "app" / "model" / "src"
if str(MODEL_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_SRC_DIR))

CHECKPOINT_PATH = BASE_DIR / "app" / "model" / "checkpoints" / "best_infrapulse_v1.pt"

_inference_engine = None
_prediction_cache: Dict[str, Dict[str, Any]] = {}

def get_inference_engine():
    global _inference_engine
    if _inference_engine is None:
        try:
            from inference import InfraPulseInference
            if CHECKPOINT_PATH.exists():
                _inference_engine = InfraPulseInference(
                    checkpoint=str(CHECKPOINT_PATH),
                    allow_fallback=True
                )
        except Exception as e:
            print(f"[ModelService] Failed to load PyTorch inference engine: {e}")
            _inference_engine = None
    return _inference_engine

def predict_single_image(image_path: str, age_hours: float = 0.0) -> Dict[str, Any]:
    """
    Runs ML inference on a given image with in-memory caching to ensure low CPU/memory usage.
    """
    cache_key = f"{image_path}_{age_hours}"
    if cache_key in _prediction_cache:
        return _prediction_cache[cache_key]

    engine = get_inference_engine()
    if engine is not None:
        try:
            result = engine.predict(image_path, age_hours=age_hours)
            
            # Map category string to CategoryEnum
            cat_str = result.get("category", "Performance")
            if cat_str == "Structural":
                cat_enum = CategoryEnum.STRUCTURAL
            elif cat_str == "Functional":
                cat_enum = CategoryEnum.FUNCTIONAL
            else:
                cat_enum = CategoryEnum.PERFORMANCE
                
            formatted = {
                "defect_name": result.get("defect", "").replace("_", " ").title(),
                "category": cat_enum,
                "category_str": cat_str,
                "confidence": round(result.get("confidence", 0.0) * 100, 1),
                "severity": round(result.get("severity", 50.0), 1),
                "extent": round(result.get("extent", 50.0), 1),
                "priority_score": round(result.get("priority_score", 0.0), 2),
                "fallback_used": result.get("fallback_used", False),
                "model_mode": "ML" if not result.get("fallback_used") else "Fallback",
                "diagnostics": result.get("diagnostics", {})
            }
            _prediction_cache[cache_key] = formatted
            return formatted
        except Exception as e:
            print(f"[ModelService] Error during model prediction on {image_path}: {e}")

    # Fallback to rule-based classifier
    fname = Path(image_path).name
    rule_res = mock_classify_defect(fname, fname)
    score = compute_priority_score(
        rule_res["category"],
        rule_res["defect_name"],
        rule_res["severity"],
        rule_res["extent"]
    )
    formatted = {
        "defect_name": rule_res["defect_name"],
        "category": rule_res["category"],
        "category_str": rule_res["category"].value,
        "confidence": 50.0,
        "severity": rule_res["severity"],
        "extent": rule_res["extent"],
        "priority_score": score,
        "fallback_used": True,
        "model_mode": "Rule-Based Mock",
        "diagnostics": {}
    }
    _prediction_cache[cache_key] = formatted
    return formatted

def run_rule_based_classifier(filename: str, description: str = "") -> Dict[str, Any]:
    """
    Runs previous rule-based / keyword matching classifier for direct side-by-side benchmarking.
    """
    res = mock_classify_defect(description or filename, filename)
    score = compute_priority_score(
        res["category"],
        res["defect_name"],
        res["severity"],
        res["extent"]
    )
    return {
        "defect_name": res["defect_name"],
        "category": res["category"],
        "category_str": res["category"].value,
        "severity": res["severity"],
        "extent": res["extent"],
        "priority_score": score
    }
