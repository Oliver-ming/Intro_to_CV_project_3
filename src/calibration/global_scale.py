"""Direction B: one global metric scale fitted on NYU train, frozen for test.

Every estimator reads only (prediction, ground truth) pairs from the training
split. Inputs are flat arrays of valid pixels pooled over training images,
plus per-image arrays for the image-level estimator.
"""

from __future__ import annotations

import numpy as np


def median_of_image_medians(pairs: list[tuple[np.ndarray, np.ndarray]]) -> float:
    """Median over images of median(gt / pred). Each image gets one vote."""
    return float(np.median([np.median(g / p) for p, g in pairs]))


def pooled_log_mean(p: np.ndarray, g: np.ndarray) -> float:
    """exp(mean(log g - log p)): minimizes squared log error."""
    return float(np.exp(np.mean(np.log(g) - np.log(p))))


def pooled_least_squares(p: np.ndarray, g: np.ndarray) -> float:
    """argmin_s sum (s p - g)^2: minimizes squared metric error."""
    return float(np.dot(p, g) / np.dot(p, p))


def pooled_absrel_optimal(p: np.ndarray, g: np.ndarray) -> float:
    """argmin_s sum |s p - g| / g.

    sum |s p - g| / g = sum (p / g) |s - g / p|, so the minimizer is the
    weighted median of g / p with weights p / g.
    """
    ratio = g / p
    weight = p / g
    order = np.argsort(ratio)
    cum = np.cumsum(weight[order])
    return float(ratio[order][np.searchsorted(cum, 0.5 * cum[-1])])


ESTIMATORS = {
    "median_image": None,  # needs per-image pairs
    "log_mean": pooled_log_mean,
    "least_squares": pooled_least_squares,
    "absrel_opt": pooled_absrel_optimal,
}


def fit_all(pairs: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, float]:
    p = np.concatenate([a for a, _ in pairs])
    g = np.concatenate([b for _, b in pairs])
    out = {"median_image": median_of_image_medians(pairs)}
    for name, fn in ESTIMATORS.items():
        if fn is not None:
            out[name] = fn(p, g)
    return out
