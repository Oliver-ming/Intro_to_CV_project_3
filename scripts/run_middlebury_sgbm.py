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
        row = {
            "scene": scene.name,
            "bad1_nocc": bad_pixel_rate(pred, gt, non_occluded, 1.0),
            "bad2_nocc": bad_pixel_rate(pred, gt, non_occluded, 2.0),
            "lr_keep": float(np.mean(consistent)),
        }
        rows.append(row)
        cv2.imwrite(str(OUT / f"{scene.name}_disp.png"), colorize(pred))
        cv2.imwrite(str(OUT / f"{scene.name}_invalid.png"), (~consistent).astype(np.uint8) * 255)
        np.save(OUT / f"{scene.name}_depth.npy", depth)
        print(
            f"{scene.name:16s}  bad1={row['bad1_nocc']:.3f}  "
            f"bad2={row['bad2_nocc']:.3f}  lr_keep={row['lr_keep']:.3f}"
        )

    table = OUT / "middlebury_sgbm.csv"
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scene", "bad1_nocc", "bad2_nocc", "lr_keep"])
        writer.writeheader()
        writer.writerows(rows)
    bad1 = np.mean([r["bad1_nocc"] for r in rows])
    print(f"mean bad1.0 on non-occluded pixels: {bad1:.3f}")
    print(f"wrote {table}")


if __name__ == "__main__":
    main()
