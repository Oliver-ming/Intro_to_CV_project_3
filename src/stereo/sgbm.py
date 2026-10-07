"""StereoSGBM disparity with a left-right consistency check."""

from __future__ import annotations

import cv2
import numpy as np

from src.stereo.middlebury import Calibration


def _num_disparities(ndisp: int) -> int:
    return max(16, int(np.ceil(ndisp / 16.0) * 16))


def stereo_sgbm(left: np.ndarray, right: np.ndarray, calib: Calibration) -> np.ndarray:
    gray_l = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    channels = 1
    block = 5
    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=_num_disparities(calib.ndisp),
        blockSize=block,
        P1=8 * channels * block * block,
        P2=32 * channels * block * block,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    return matcher.compute(gray_l, gray_r).astype(np.float32) / 16.0


def left_right_consistent(
    left: np.ndarray,
    right: np.ndarray,
    disparity_left: np.ndarray,
    calib: Calibration,
    tolerance: float = 1.0,
) -> np.ndarray:
    """Pixels whose left disparity agrees with the reverse match."""
    disparity_right = np.fliplr(stereo_sgbm(np.fliplr(right), np.fliplr(left), calib))
    height, width = disparity_left.shape
    xs = np.broadcast_to(np.arange(width, dtype=np.float32), (height, width))
    sampled_x = np.rint(xs - disparity_left).astype(np.int32)
    inside = (sampled_x >= 0) & (sampled_x < width) & (disparity_left > 0)
    sampled = np.zeros_like(disparity_left)
    rows = np.arange(height)[:, None]
    safe_x = np.clip(sampled_x, 0, width - 1)
    sampled[inside] = disparity_right[rows, safe_x][inside]
    agree = np.abs(disparity_left - sampled) <= tolerance
    return inside & agree
