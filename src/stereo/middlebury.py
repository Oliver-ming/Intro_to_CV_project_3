"""Middlebury Stereo v3 quarter-resolution IO and metric conversion."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "data" / "middlebury" / "MiddEval3" / "trainingQ"


@dataclass(frozen=True)
class Calibration:
    focal_px: float
    baseline_m: float
    doffs: float
    width: int
    height: int
    ndisp: int


def training_pairs() -> list[Path]:
    return sorted(p for p in TRAINING.iterdir() if p.is_dir())


def read_pfm(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        color = handle.readline().decode().strip() == "PF"
        width, height = (int(v) for v in handle.readline().decode().split())
        scale = float(handle.readline().decode().strip())
        endian = "<" if scale < 0 else ">"
        data = np.fromfile(handle, f"{endian}f")
    shape = (height, width, 3) if color else (height, width)
    return np.flipud(data.reshape(shape))


def read_calibration(path: Path) -> Calibration:
    text = path.read_text()

    def grab(name: str) -> str:
        match = re.search(rf"^{name}=(.+)$", text, re.M)
        if match is None:
            raise ValueError(f"{path} has no {name}")
        return match.group(1)

    cam0 = grab("cam0").strip("[]").split(";")[0]
    focal = float(cam0.split()[0])
    return Calibration(
        focal_px=focal,
        baseline_m=float(grab("baseline")) / 1000.0,
        doffs=float(grab("doffs")),
        width=int(grab("width")),
        height=int(grab("height")),
        ndisp=int(grab("ndisp")),
    )


def disparity_to_depth(disparity: np.ndarray, calib: Calibration) -> np.ndarray:
    """Middlebury metric depth: Z = f * B / (d + doffs)."""
    denom = disparity.astype(np.float32) + np.float32(calib.doffs)
    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(denom) & (denom > 1e-3)
    depth[valid] = (calib.focal_px * calib.baseline_m) / denom[valid]
    return depth


def load_pair(scene: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Calibration]:
    left = cv2.imread(str(scene / "im0.png"), cv2.IMREAD_COLOR)
    right = cv2.imread(str(scene / "im1.png"), cv2.IMREAD_COLOR)
    if left is None or right is None:
        raise FileNotFoundError(scene)
    gt = read_pfm(scene / "disp0GT.pfm")
    mask = cv2.imread(str(scene / "mask0nocc.png"), cv2.IMREAD_GRAYSCALE)
    non_occluded = mask > 127
    calib = read_calibration(scene / "calib.txt")
    return left, right, gt, non_occluded, calib
