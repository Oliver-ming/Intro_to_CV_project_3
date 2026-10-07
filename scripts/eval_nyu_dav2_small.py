"""NYUv2 test split: Depth Anything V2-Small relative depth with test-time alignment.

Alignment is only for the benchmark. The aligned maps are not metric depth.
The split is the official 654-image test set in splits.mat. The crop and the
10 m cap follow the Eigen protocol used with the Depth Anything V2 metric code.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import scipy.io
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "Depth-Anything-V2"))

from depth_anything_v2.dpt import DepthAnythingV2

MAT = ROOT / "data" / "nyu" / "nyu_depth_v2_labeled.mat"
SPLIT = ROOT / "data" / "nyu" / "splits.mat"
WEIGHT = ROOT / "weights" / "depth_anything_v2_vits.pth"
OUT = ROOT / "outputs" / "nyu_dav2_small"
MIN_DEPTH = 1e-3
MAX_DEPTH = 10.0


def to_hwc(image_chw: np.ndarray) -> np.ndarray:
    # MATLAB v7.3 stores the labeled RGB volume as (3, 640, 480).
    return np.transpose(image_chw, (2, 1, 0))


def to_hw(depth_hw: np.ndarray) -> np.ndarray:
    return np.transpose(depth_hw, (1, 0))


def eigen_mask(gt: np.ndarray) -> np.ndarray:
    mask = np.zeros(gt.shape, dtype=bool)
    mask[45:471, 41:601] = True
    return mask & np.isfinite(gt) & (gt > MIN_DEPTH) & (gt < MAX_DEPTH)


def align_scale_shift(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> np.ndarray:
    prediction = pred[mask].astype(np.float64)
    target = gt[mask].astype(np.float64)
    design = np.stack([prediction, np.ones_like(prediction)], axis=1)
    scale, shift = np.linalg.lstsq(design, target, rcond=None)[0]
    return pred * scale + shift


def align_median(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> np.ndarray:
    scale = np.median(gt[mask]) / np.median(pred[mask])
    return pred * scale


def metrics(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    prediction = pred[mask].astype(np.float64)
    target = gt[mask].astype(np.float64)
    prediction = np.clip(prediction, MIN_DEPTH, None)
    thresh = np.maximum(target / prediction, prediction / target)
    diff = prediction - target
    diff_log = np.log(prediction) - np.log(target)
    return {
        "abs_rel": float(np.mean(np.abs(diff) / target)),
        "sq_rel": float(np.mean(diff**2 / target)),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "rmse_log": float(np.sqrt(np.mean(diff_log**2))),
        "d1": float(np.mean(thresh < 1.25)),
        "d2": float(np.mean(thresh < 1.25**2)),
        "d3": float(np.mean(thresh < 1.25**3)),
    }


def accumulate(total: dict[str, float], one: dict[str, float], n: int) -> None:
    for key, value in one.items():
        total[key] = total.get(key, 0.0) + value / n


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    test_ids = scipy.io.loadmat(SPLIT)["testNdxs"].astype(int).ravel() - 1
    device = "cuda:0"
    model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    model.load_state_dict(torch.load(WEIGHT, map_location="cpu"))
    model = model.to(device).eval()

    scale_shift_sum: dict[str, float] = {}
    median_sum: dict[str, float] = {}
    n = len(test_ids)
    with h5py.File(MAT, "r") as data, torch.inference_mode():
        for step, index in enumerate(test_ids):
            rgb = to_hwc(data["images"][index])
            gt = to_hw(data["depths"][index]).astype(np.float32)
            pred = model.infer_image(rgb[:, :, ::-1].copy(), input_size=518)
            mask = eigen_mask(gt)
            if pred.shape != gt.shape:
                pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
            aligned = align_scale_shift(pred, gt, mask)
            median = align_median(pred, gt, mask)
            accumulate(scale_shift_sum, metrics(aligned, gt, mask), n)
            accumulate(median_sum, metrics(median, gt, mask), n)
            if step == 0:
                vis = rgb[:, :, ::-1]
                cv2.imwrite(str(OUT / "sample_rgb.png"), vis)
                depth_vis = aligned.copy()
                depth_vis[~mask] = 0
                depth_u8 = np.clip(depth_vis / MAX_DEPTH, 0, 1)
                cv2.imwrite(
                    str(OUT / "sample_aligned_depth.png"),
                    cv2.applyColorMap((depth_u8 * 255).astype(np.uint8), cv2.COLORMAP_MAGMA),
                )
            if (step + 1) % 100 == 0 or step + 1 == n:
                print(f"{step + 1}/{n}  abs_rel={scale_shift_sum['abs_rel'] * n / (step + 1):.4f}")

    rows = []
    for name, totals in (("scale_shift", scale_shift_sum), ("median", median_sum)):
        row = {"alignment": name, **totals}
        rows.append(row)
        print(name, {k: round(v, 4) for k, v in totals.items()})
    with (OUT / "nyu_dav2_small.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["alignment", *scale_shift_sum.keys()])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
