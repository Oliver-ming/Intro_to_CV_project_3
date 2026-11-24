"""StereoSGBM disparity with CLAHE, left-right consistency, and guided hole-filling."""

from __future__ import annotations

import cv2
import numpy as np

from src.stereo.middlebury import Calibration


def _num_disparities(ndisp: int) -> int:
    return max(16, int(np.ceil(ndisp / 16.0) * 16))


def apply_clahe(img_bgr: np.ndarray, clip_limit: float = 2.0, grid_size: int = 8) -> np.ndarray:
    """Normalize local illumination via CLAHE on the LAB luminance channel."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(grid_size), int(grid_size)))
    l_eq = clahe.apply(l_ch)
    return cv2.cvtColor(cv2.merge([l_eq, a_ch, b_ch]), cv2.COLOR_LAB2BGR)


def stereo_sgbm(
    left: np.ndarray,
    right: np.ndarray,
    calib: Calibration,
    mode: int = cv2.STEREO_SGBM_MODE_SGBM_3WAY,
) -> np.ndarray:
    """Compute subpixel float disparity using StereoSGBM (fixed-point / 16)."""
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
        mode=int(mode),
    )
    return matcher.compute(gray_l, gray_r).astype(np.float32) / 16.0


def left_right_consistent(
    left: np.ndarray,
    right: np.ndarray,
    disparity_left: np.ndarray,
    calib: Calibration,
    tolerance: float = 1.0,
    mode: int = cv2.STEREO_SGBM_MODE_SGBM_3WAY,
) -> np.ndarray:
    """Pixels whose left disparity agrees with the reverse match."""
    disparity_right = np.fliplr(stereo_sgbm(np.fliplr(right), np.fliplr(left), calib, mode=mode))
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


def _normalized_guided_pass(
    disp: np.ndarray,
    weight: np.ndarray,
    guide_gray: np.ndarray,
    radius: int,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Masked weighted guided filter (He et al. 2013) via pure-OpenCV box filters."""
    ksize = (2 * int(radius) + 1, 2 * int(radius) + 1)
    win_area = float(ksize[0] * ksize[1])
    sum_w = cv2.boxFilter(weight, -1, ksize, normalize=False)
    safe_w = np.maximum(sum_w, 1e-6)

    I = guide_gray
    p = np.where(weight > 0, disp, 0.0).astype(np.float32)
    mean_I = cv2.boxFilter(I * weight, -1, ksize, normalize=False) / safe_w
    mean_p = cv2.boxFilter(p * weight, -1, ksize, normalize=False) / safe_w
    corr_I = cv2.boxFilter(I * I * weight, -1, ksize, normalize=False) / safe_w
    corr_Ip = cv2.boxFilter(I * p * weight, -1, ksize, normalize=False) / safe_w

    var_I = np.maximum(corr_I - mean_I * mean_I, 0.0)
    cov_Ip = corr_Ip - mean_I * mean_p
    a = cov_Ip / (var_I + float(eps))
    b = mean_p - a * mean_I

    # Average linear coefficients only over windows with meaningful support.
    support = (sum_w >= max(1.0, 0.02 * win_area)).astype(np.float32)
    sum_s = cv2.boxFilter(support, -1, ksize, normalize=False)
    safe_s = np.maximum(sum_s, 1e-6)
    mean_a = cv2.boxFilter(a * support, -1, ksize, normalize=False) / safe_s
    mean_b = cv2.boxFilter(b * support, -1, ksize, normalize=False) / safe_s

    q = mean_a * I + mean_b
    filled_ok = (sum_s > 0.5) & np.isfinite(q) & (q > 0)
    return q.astype(np.float32), filled_ok


def guided_hole_fill(
    disp: np.ndarray,
    valid: np.ndarray,
    guide_bgr: np.ndarray,
    radius: int = 9,
    eps: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill LRC-invalid holes using edge-aware normalized guided filtering.

    Keeps original confident LRC disparities untouched (with minor noise-suppressed
    blending only on invalid regions) and propagates across RGB edges without
    requiring ``cv2.ximgproc``.
    """
    guide_gray = cv2.cvtColor(guide_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    base_valid = valid & np.isfinite(disp) & (disp > 0)
    cur_disp = np.where(base_valid, disp, 0.0).astype(np.float32)
    cur_valid = base_valid.copy()

    # Stage 1: local edge-preserving guided interpolation
    q1, ok1 = _normalized_guided_pass(cur_disp, cur_valid.astype(np.float32), guide_gray, radius=radius, eps=eps)
    need1 = (~cur_valid) & ok1
    cur_disp = np.where(need1, q1, cur_disp)
    cur_valid = cur_valid | need1

    # Stage 2: wider-context pass for any remaining interior occlusion holes
    if not np.all(cur_valid):
        q2, ok2 = _normalized_guided_pass(
            cur_disp, cur_valid.astype(np.float32), guide_gray, radius=max(radius * 3, 25), eps=eps
        )
        need2 = (~cur_valid) & ok2
        cur_disp = np.where(need2, q2, cur_disp)
        cur_valid = cur_valid | need2

    # Vectorized horizontal nearest-valid fallback (O(W) fast scan, no outer Python loop)
    if not np.all(cur_valid):
        med = float(np.median(cur_disp[cur_valid])) if np.any(cur_valid) else 1.0
        _, w = cur_disp.shape
        col_idx = np.broadcast_to(np.arange(w, dtype=np.int32), cur_disp.shape)
        left_idx = np.maximum.accumulate(np.where(cur_valid, col_idx, -1), axis=1)
        rev_idx = np.minimum.accumulate(np.where(cur_valid[:, ::-1], col_idx[:, ::-1], w), axis=1)[:, ::-1]
        has_l = left_idx >= 0
        has_r = rev_idx < w
        rows_idx = np.arange(cur_disp.shape[0])[:, None]
        dl = np.where(has_l, cur_disp[rows_idx, np.clip(left_idx, 0, w - 1)], med)
        dr = np.where(has_r, cur_disp[rows_idx, np.clip(rev_idx, 0, w - 1)], med)
        fill_vec = np.where(has_l & has_r, np.minimum(dl, dr), np.where(has_l, dl, dr))
        need3 = ~cur_valid
        cur_disp = np.where(need3, fill_vec, cur_disp)
        cur_valid = cur_valid | need3

    return cur_disp.astype(np.float32), cur_valid
