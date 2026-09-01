import cv2
import numpy as np

TYPE_TIER = {
    "cracked_tiles": 1,
    "paint_peeling": 0,
    "spalling": 0,
    "stagnant_water": 0,
}

def _normalize_01(x):
    return float(np.clip(x, 0.0, 1.0))

def analyze_heatmap(heatmap, original_rgb):
    """
    Converts a GradCAM++ map into heuristic visible-severity and extent scores.

    Important:
    This is NOT true segmentation. It is a baseline proxy that measures
    model attention plus local visual structure.
    """

    hmap = heatmap.astype(np.float32)
    if hmap.max() > 1.0:
        hmap /= 255.0

    h, w = original_rgb.shape[:2]

    # Critical fix: GradCAM output must match original image geometry
    if hmap.shape[:2] != (h, w):
        hmap = cv2.resize(
            hmap,
            (w, h),
            interpolation=cv2.INTER_LINEAR
        )

    hmap = np.clip(hmap, 0.0, 1.0)

    # Adaptive focus threshold plus a minimum floor
    threshold = max(0.40, float(np.percentile(hmap, 75)))
    active = (hmap >= threshold).astype(np.uint8)

    coverage = float(active.mean())

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        active, connectivity=8
    )

    component_areas = []
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= max(5, int(0.0005 * h * w)):
            component_areas.append(area)

    component_count = len(component_areas)
    component_score = min(component_count / 6.0, 1.0)

    if active.sum() > 0:
        ys, xs = np.where(active > 0)
        spread_x = (xs.max() - xs.min() + 1) / max(w, 1)
        spread_y = (ys.max() - ys.min() + 1) / max(h, 1)
        spread = float(np.sqrt(spread_x * spread_y))
    else:
        spread = 0.0

    # Extent: how much, how fragmented, how spread out
    extent = (
        0.65 * min(coverage / 0.35, 1.0)
        + 0.15 * component_score
        + 0.20 * spread
    )
    extent = _normalize_01(extent)

    gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(gray, 75, 150)

    if active.sum() > 0:
        active_bool = active.astype(bool)
        mean_activation = float(hmap[active_bool].mean())
        peak_activation = float(hmap[active_bool].max())
        edge_density = float((edges[active_bool] > 0).mean())
    else:
        mean_activation = float(hmap.mean())
        peak_activation = float(hmap.max())
        edge_density = float((edges > 0).mean())

    # Cap expected edge density to a useful range
    edge_score = min(edge_density / 0.25, 1.0)

    severity = (
        0.45 * mean_activation
        + 0.30 * peak_activation
        + 0.25 * edge_score
    )
    severity = _normalize_01(severity)

    return {
        "severity": severity * 100.0,
        "extent": extent * 100.0,
        "coverage_ratio": coverage,
        "component_count": component_count,
        "spread": spread,
        "mean_activation": mean_activation,
        "peak_activation": peak_activation,
        "edge_density": edge_density,
    }

def compute_priority(
    defect_class,
    severity,
    extent,
    age_hours=0.0
):
    """
    PriorityScore =
        TypeTier * 1000
        + Severity * 5
        + Extent * 3
        + TimeBonus

    Severity and Extent expected on [0,100].

    cracked_tiles gets +1000 so it cannot rank below paint_peeling
    because severity+extent maximum contribution is 800.
    """

    tier = TYPE_TIER.get(defect_class, 0)

    severity = float(np.clip(severity, 0.0, 100.0))
    extent = float(np.clip(extent, 0.0, 100.0))

    # Slow starvation-prevention term
    time_bonus = max(0.0, float(age_hours)) * 0.1

    score = (
        tier * 1000.0
        + severity * 5.0
        + extent * 3.0
        + time_bonus
    )

    return float(score)
