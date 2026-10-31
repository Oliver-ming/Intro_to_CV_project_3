"""Run StereoSGBM on the 15 Middlebury training pairs and write metrics."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.stereo.middlebury import disparity_to_depth, load_pair, training_pairs
from src.stereo.sgbm import left_right_consistent, stereo_sgbm

OUT = ROOT / "outputs" / "stereo"


def bad_pixel_rate(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, thresh: float) -> float:
    valid = mask & np.isfinite(gt) & (gt > 0)
    if not np.any(valid):
        return float("nan")
    err = np.abs(pred[valid] - gt[valid])
    return float(np.mean(err > thresh))


def course_disparity_metrics(pred: np.ndarray, gt: np.ndarray, keep: np.ndarray) -> tuple[float, float]:
    """Course metric on finite, positive GT disparities.

    A prediction is invalid when SGBM returns no match (disparity <= 0) or the
    pixel fails the left-right check. Returns (mean |error| px over valid
    predictions, fraction of GT pixels with a valid prediction).
    """
    gt_ok = np.isfinite(gt) & (gt > 0)
    valid = gt_ok & keep & (pred > 0)
    if not np.any(gt_ok):
        return float("nan"), float("nan")
    epe = float(np.mean(np.abs(pred[valid] - gt[valid]))) if np.any(valid) else float("nan")
    return epe, float(valid.sum() / gt_ok.sum())


def colorize(disparity: np.ndarray) -> np.ndarray:
    finite = disparity[np.isfinite(disparity) & (disparity > 0)]
    vmax = float(np.percentile(finite, 98)) if finite.size else 1.0
    scaled = np.clip(disparity / max(vmax, 1e-3), 0, 1)
    scaled[~np.isfinite(disparity) | (disparity <= 0)] = 0
    return cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for scene in training_pairs():
        left, right, gt, non_occluded, calib = load_pair(scene)
        pred = stereo_sgbm(left, right, calib)
        consistent = left_right_consistent(left, right, pred, calib)
        depth = disparity_to_depth(pred, calib)
        depth[~(consistent & (pred > 0))] = np.nan
        epe, valid_frac = course_disparity_metrics(pred, gt, consistent)
        row = {
            "scene": scene.name,
            "mae_px": epe,
            "valid_frac": valid_frac,
            "bad1_nocc": bad_pixel_rate(pred, gt, non_occluded, 1.0),
            "bad2_nocc": bad_pixel_rate(pred, gt, non_occluded, 2.0),
            "lr_keep": float(np.mean(consistent)),
        }
        rows.append(row)
        cv2.imwrite(str(OUT / f"{scene.name}_disp.png"), colorize(pred))
        cv2.imwrite(str(OUT / f"{scene.name}_invalid.png"), (~consistent).astype(np.uint8) * 255)
        np.save(OUT / f"{scene.name}_depth.npy", depth)
        print(
            f"{scene.name:16s}  mae={row['mae_px']:.2f}px  valid={row['valid_frac']:.3f}  "
            f"bad1={row['bad1_nocc']:.3f}  lr_keep={row['lr_keep']:.3f}"
        )

    fields = ["scene", "mae_px", "valid_frac", "bad1_nocc", "bad2_nocc", "lr_keep"]
    mean_row = {"scene": "MEAN", **{k: float(np.mean([r[k] for r in rows])) for k in fields[1:]}}
    table = OUT / "middlebury_sgbm.csv"
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows + [mean_row])
    print(
        f"mean |disp err| {mean_row['mae_px']:.2f}px  valid {mean_row['valid_frac']:.3f}  "
        f"bad1.0(nocc) {mean_row['bad1_nocc']:.3f}"
    )
    print(f"wrote {table}")


if __name__ == "__main__":
    main()
