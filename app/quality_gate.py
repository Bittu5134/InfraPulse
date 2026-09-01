import os
from pathlib import Path
from typing import Dict, Any, Union
import numpy as np
from PIL import Image
import cv2

# Default threshold for Variance of Laplacian (higher = sharper, < 50-60 is typically motion/focus blur)
DEFAULT_BLUR_THRESHOLD = 50.0

def evaluate_sharpness_variance(image_input: Union[str, Path, np.ndarray, Image.Image]) -> float:
    """
    Computes the Variance of Laplacian over a grayscale image.
    Higher values indicate sharp edges; low values indicate blur.
    """
    if isinstance(image_input, (str, Path)):
        if not os.path.exists(image_input):
            return 0.0
        img = cv2.imread(str(image_input))
        if img is None:
            return 0.0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif isinstance(image_input, Image.Image):
        gray = np.array(image_input.convert("L"))
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 3:
            gray = cv2.cvtColor(image_input, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_input
    else:
        return 0.0

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = float(laplacian.var())
    return variance

def check_image_quality(
    image_input: Union[str, Path, np.ndarray, Image.Image],
    blur_threshold: float = DEFAULT_BLUR_THRESHOLD
) -> Dict[str, Any]:
    """
    Phase 1: The Quality Gatekeeper (Pre-Processing)
    Intercepts the uploaded photograph before AI inference.
    Evaluates sharpness using OpenCV Variance of Laplacian.
    """
    sharpness = evaluate_sharpness_variance(image_input)
    is_blurry = sharpness < blur_threshold
    passed = not is_blurry

    if sharpness >= 150.0:
        quality_tier = "Crisp & High Contrast"
        badge_color = "emerald"
    elif sharpness >= blur_threshold:
        quality_tier = "Acceptable Sharpness"
        badge_color = "blue"
    elif sharpness >= 25.0:
        quality_tier = "Slightly Blurry"
        badge_color = "amber"
    else:
        quality_tier = "Severely Blurry"
        badge_color = "rose"

    if passed:
        message = f"Quality gatekeeper passed: Image is sharp (Laplacian variance: {sharpness:.1f} >= {blur_threshold:.1f})."
    else:
        message = f"Quality warning: Photograph appears blurry (Laplacian variance: {sharpness:.1f} < {blur_threshold:.1f}). Please consider capturing a sharper, well-lit photo."

    return {
        "passed": passed,
        "sharpness_score": round(sharpness, 1),
        "blur_threshold": blur_threshold,
        "is_blurry": is_blurry,
        "quality_tier": quality_tier,
        "badge_color": badge_color,
        "message": message
    }
