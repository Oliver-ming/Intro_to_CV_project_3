"""NYU Depth V2 labeled set under the Spring 2027 course protocol.

Protocol: official splits.mat (trainNdxs 795, testNdxs 654, MATLAB 1-based),
filled depth `depths`, predictions resized to 480x640, Python crop
[45:471, 41:601], GT in (0.001, 10) m, predictions clipped to [0.001, 10],
per-image metrics averaged equally. No scale alignment on test.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import scipy.io

ROOT = Path(__file__).resolve().parents[2]
MAT = ROOT / "data" / "nyu" / "nyu_depth_v2_labeled.mat"
SPLIT = ROOT / "data" / "nyu" / "splits.mat"
MIN_DEPTH = 1e-3
MAX_DEPTH = 10.0
HEIGHT, WIDTH = 480, 640
# Kinect intrinsics for the 640x480 labeled images.
K_NYU = np.array(
    [[518.8579, 0.0, 325.5824], [0.0, 519.4696, 253.7362], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)


def split_ids(name: str) -> np.ndarray:
    """`train` or `test` -> 0-based indices into the labeled .mat."""
    key = {"train": "trainNdxs", "test": "testNdxs"}[name]
    return scipy.io.loadmat(SPLIT)[key].astype(int).ravel() - 1


def open_mat() -> h5py.File:
    return h5py.File(MAT, "r")


def rgb_at(data: h5py.File, index: int) -> np.ndarray:
    return np.ascontiguousarray(data["images"][index].transpose(2, 1, 0))


def depth_at(data: h5py.File, index: int) -> np.ndarray:
    return np.ascontiguousarray(data["depths"][index].T).astype(np.float32)


def scene_type_at(data: h5py.File, index: int) -> str:
    ref = data["sceneTypes"][0][index]
    return "".join(chr(c) for c in data[ref][:].ravel())


def eval_mask(gt: np.ndarray) -> np.ndarray:
    mask = np.zeros(gt.shape, dtype=bool)
    mask[45:471, 41:601] = True
    return mask & np.isfinite(gt) & (gt > MIN_DEPTH) & (gt < MAX_DEPTH)


def metrics(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    p = np.clip(pred[mask].astype(np.float64), MIN_DEPTH, MAX_DEPTH)
    g = gt[mask].astype(np.float64)
    thresh = np.maximum(g / p, p / g)
    diff = p - g
    return {
        "abs_rel": float(np.mean(np.abs(diff) / g)),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "d1": float(np.mean(thresh < 1.25)),
        "sq_rel": float(np.mean(diff**2 / g)),
        "rmse_log": float(np.sqrt(np.mean((np.log(p) - np.log(g)) ** 2))),
        "d2": float(np.mean(thresh < 1.25**2)),
        "d3": float(np.mean(thresh < 1.25**3)),
    }


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
