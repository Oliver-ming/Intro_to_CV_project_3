"""NYUv2 test split: UniDepthV2-Small metric depth, no test-time alignment.

Intrinsics are the labeled-set Kinect calibration and are passed in at inference.
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
sys.path.insert(0, str(ROOT / "third_party" / "UniDepth"))

from unidepth.models import UniDepthV2
from unidepth.utils.camera import Pinhole

MAT = ROOT / "data" / "nyu" / "nyu_depth_v2_labeled.mat"
SPLIT = ROOT / "data" / "nyu" / "splits.mat"
OUT = ROOT / "outputs" / "nyu_unidepthv2_small"
MIN_DEPTH = 1e-3
MAX_DEPTH = 10.0
# Official NYU Kinect intrinsics for the 640x480 labeled images.
K = torch.tensor(
    [
        [518.8579, 0.0, 325.5824],
        [0.0, 519.4696, 253.7362],
        [0.0, 0.0, 1.0],
    ],
    dtype=torch.float32,
)


def to_hwc(image_chw: np.ndarray) -> np.ndarray:
    return np.transpose(image_chw, (2, 1, 0))


def to_hw(depth_hw: np.ndarray) -> np.ndarray:
    return np.transpose(depth_hw, (1, 0))


def eigen_mask(gt: np.ndarray) -> np.ndarray:
    mask = np.zeros(gt.shape, dtype=bool)
    mask[45:471, 41:601] = True
    return mask & np.isfinite(gt) & (gt > MIN_DEPTH) & (gt < MAX_DEPTH)


def metrics(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    prediction = np.clip(pred[mask].astype(np.float64), MIN_DEPTH, None)
    target = gt[mask].astype(np.float64)
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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    test_ids = scipy.io.loadmat(SPLIT)["testNdxs"].astype(int).ravel() - 1
    model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vits14")
    model.interpolation_mode = "bilinear"
    model = model.to("cuda").eval()
    camera = Pinhole(K=K.unsqueeze(0))
    total: dict[str, float] = {}
    n = len(test_ids)
    with h5py.File(MAT, "r") as data, torch.inference_mode():
        for step, index in enumerate(test_ids):
            rgb = to_hwc(data["images"][index])
            gt = to_hw(data["depths"][index]).astype(np.float32)
            tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)
            pred = model.infer(tensor, camera)["depth"].squeeze().detach().cpu().numpy()
            if pred.shape != gt.shape:
                pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
            one = metrics(pred, gt, eigen_mask(gt))
            for key, value in one.items():
                total[key] = total.get(key, 0.0) + value / n
            if step == 0:
                cv2.imwrite(str(OUT / "sample_rgb.png"), rgb[:, :, ::-1])
                shown = np.clip(pred / MAX_DEPTH, 0, 1)
                cv2.imwrite(
                    str(OUT / "sample_metric_depth.png"),
                    cv2.applyColorMap((shown * 255).astype(np.uint8), cv2.COLORMAP_MAGMA),
                )
            if (step + 1) % 100 == 0 or step + 1 == n:
                print(f"{step + 1}/{n}  abs_rel={total['abs_rel'] * n / (step + 1):.4f}")
    print({k: round(v, 4) for k, v in total.items()})
    with (OUT / "nyu_unidepthv2_small.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["alignment", *total.keys()])
        writer.writeheader()
        writer.writerow({"alignment": "none_metric", **total})


if __name__ == "__main__":
    main()
