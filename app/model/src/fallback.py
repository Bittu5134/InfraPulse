import cv2
import numpy as np

CLASS_NAMES = [
    "cracked_tiles",
    "paint_peeling",
    "spalling",
    "stagnant_water",
]

def fallback_analysis(rgb):
    """
    Emergency-only heuristic.
    This is intentionally conservative and should never silently replace ML.
    """

    img = rgb.astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    edges = cv2.Canny(gray, 75, 150)
    edge_density = float((edges > 0).mean())

    saturation = float(hsv[..., 1].mean() / 255.0)
    brightness = float(hsv[..., 2].mean() / 255.0)

    # Simple coarse proxies only.
    # These are NOT intended as production-grade defect classifiers.
    if brightness < 0.35 and edge_density > 0.12:
        defect = "spalling"
    elif edge_density > 0.16:
        defect = "cracked_tiles"
    elif saturation < 0.18 and brightness > 0.50:
        defect = "stagnant_water"
    else:
        defect = "paint_peeling"

    severity = np.clip(
        (0.55 * edge_density + 0.45 * (1.0 - brightness)) * 100.0,
        0,
        100
    )

    extent = np.clip(
        (0.7 * edge_density + 0.3 * saturation) * 100.0,
        0,
        100
    )

    return {
        "defect": defect,
        "confidence": 0.0,
        "severity": float(severity),
        "extent": float(extent),
        "fallback_used": True,
    }
