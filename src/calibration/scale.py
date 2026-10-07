"""Known-size scale from a standing person or a car."""

from __future__ import annotations

import numpy as np

PERSON_HEIGHT_M = 1.70
CAR_HEIGHT_M = 1.59
PERSON_SIGMA_M = 0.09
CAR_SIGMA_M = 0.21
MIN_SCORE = 0.5


def implied_height_m(box_height_px: float, depth_m: float, focal_y: float) -> float:
    return box_height_px * depth_m / focal_y


def touches_border(box: np.ndarray, width: int, height: int, margin: int = 2) -> bool:
    x1, y1, x2, y2 = box
    return x1 <= margin or y1 <= margin or x2 >= width - margin or y2 >= height - margin


def median_depth(depth: np.ndarray, box: np.ndarray, valid: np.ndarray) -> float | None:
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, depth.shape[1]), min(y2, depth.shape[0])
    if x2 - x1 < 4 or y2 - y1 < 8:
        return None
    region = depth[y1:y2, x1:x2]
    usable = valid[y1:y2, x1:x2] & np.isfinite(region) & (region > 1e-3)
    if np.count_nonzero(usable) < 30:
        return None
    return float(np.median(region[usable]))


def clip_scale(ratios: list[float]) -> float:
    if not ratios:
        return 1.0
    return float(np.median(ratios))
