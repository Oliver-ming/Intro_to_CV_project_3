"""Scale-invariant boundary F1 from Depth Pro (Bochkovskii et al., ICLR 2025).

Edges are neighboring pixels whose inverse-depth ratio exceeds a threshold.
The score averages F1 over thresholds from 1.05 to 1.25.
"""

from __future__ import annotations

import numpy as np


def _boundary_f1(pred: np.ndarray, gt: np.ndarray, threshold: float) -> float:
    def ratios(depth: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        right = depth[:, 1:] / depth[:, :-1]
        left = depth[:, :-1] / depth[:, 1:]
        bottom = depth[1:, :] / depth[:-1, :]
        top = depth[:-1, :] / depth[1:, :]
        return left > threshold, top > threshold, right > threshold, bottom > threshold

    pred_dirs = ratios(pred)
    gt_dirs = ratios(gt)
    recall_terms = []
    precision_terms = []
    for pred_edge, gt_edge in zip(pred_dirs, gt_dirs):
        hit = np.count_nonzero(pred_edge & gt_edge)
        recall_terms.append(hit / max(np.count_nonzero(gt_edge), 1))
        precision_terms.append(hit / max(np.count_nonzero(pred_edge), 1))
    recall = float(np.mean(recall_terms))
    precision = float(np.mean(precision_terms))
    if recall + precision == 0:
        return 0.0
    return 2 * recall * precision / (recall + precision)


def si_boundary_f1(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> float:
    prediction = pred.astype(np.float64).copy()
    target = gt.astype(np.float64).copy()
    prediction[~valid] = np.nan
    target[~valid] = np.nan
    prediction = 1.0 / np.clip(prediction, 1e-6, None)
    target = 1.0 / np.clip(target, 1e-6, None)
    thresholds = np.linspace(1.05, 1.25, 10)
    weights = thresholds / thresholds.sum()
    scores = np.array([_boundary_f1(prediction, target, float(t)) for t in thresholds])
    return float(np.sum(scores * weights))
