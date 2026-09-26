"""
Calibrated Computer Vision Severity & Visible Damage Extent Engine (0-100).
Deterministic, defect-type aware, and fully explainable for municipal/facility triage.

Defect-Presence Gating:
- Evaluates true physical defect boundary/edge density first.
- Undamaged surfaces (e.g. clean intact tiles, smooth undamaged walls) naturally score ~0-5.
- Visibly damaged surfaces dynamically scale from 25 to 98 based on dilated damage influence zones.
"""
import cv2
import numpy as np


def _crack_spalling_score(img_bgr: np.ndarray) -> float:
    """Computes damage extent for fracture/spalling networks using dilated damage zones."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = img_bgr.shape[:2]
    total_pixels = float(h * w)

    # 1. CLAHE local adaptive contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # 2. Multi-threshold Canny edge detection
    edges = cv2.Canny(blurred, 50, 140)
    edge_density = np.count_nonzero(edges) / total_pixels

    # Defect-presence gating: if smooth/clean with no fracture edges
    if edge_density < 0.002:
        return round(float(edge_density * 500.0), 2)

    # 3. Morphological dilation to measure physical damage influence zone
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    # 4. Surface roughness variance (Laplacian)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    roughness_factor = float(np.clip(lap_var / 500.0, 0.6, 1.4))

    # 5. Non-linear perceptual scaling
    fraction = np.count_nonzero(dilated) / total_pixels
    score = np.sqrt(fraction) * 160.0 * roughness_factor
    return float(np.clip(score, 0.0, 98.0))


def _stagnant_water_score(img_bgr: np.ndarray) -> float:
    """Computes surface water coverage via reflective HSV mask + specular highlights."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = img_bgr.shape[:2]
    total_pixels = float(h * w)

    # Low saturation reflective dark patches
    lower = np.array([0, 0, 15])
    upper = np.array([180, 75, 175])
    puddle_mask = cv2.inRange(hsv, lower, upper)
    puddle_frac = float(np.count_nonzero(puddle_mask)) / total_pixels

    # Gating: if there are no low-saturation reflective puddle zones, return 0
    if puddle_frac < 0.008:
        return round(float(puddle_frac * 200.0), 2)

    # Specular water surface glints inside/adjacent to puddle regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dilated_puddle = cv2.dilate(puddle_mask, kernel)
    _, specular_mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    water_glints = cv2.bitwise_and(specular_mask, dilated_puddle)

    combined = cv2.bitwise_or(puddle_mask, water_glints)
    area_frac = float(np.count_nonzero(combined)) / total_pixels

    score = np.sqrt(area_frac) * 115.0
    return float(np.clip(score, 0.0, 95.0))


def _paint_peeling_score(img_bgr: np.ndarray) -> float:
    """Computes paint flaking extent combining Otsu contrast and peel boundary gradients."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = img_bgr.shape[:2]
    total_pixels = float(h * w)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    edges = cv2.Canny(enhanced, 60, 160)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated_edges = cv2.dilate(edges, kernel)
    edge_density = np.count_nonzero(edges) / total_pixels

    # Defect-presence gating: smooth walls with no flaking edges score near 0
    if edge_density < 0.003:
        return round(float(edge_density * 800.0), 2)

    # Substrate contrast threshold gated by flake edges
    _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    flake_mask = cv2.bitwise_and(otsu, dilated_edges)
    flake_frac = np.count_nonzero(flake_mask) / total_pixels

    score = np.sqrt(flake_frac) * 180.0
    return float(np.clip(score, 0.0, 98.0))


def compute_severity(image_path: str, defect_label: str) -> float:
    """Entry point: calculates calibrated 0-100 visible damage severity score."""
    img = cv2.imread(image_path)
    if img is None:
        return 0.0

    if defect_label in ("spalling", "cracked_tiles"):
        return round(_crack_spalling_score(img), 2)
    elif defect_label == "stagnant_water":
        return round(_stagnant_water_score(img), 2)
    elif defect_label == "paint_peeling":
        return round(_paint_peeling_score(img), 2)
    return 0.0
