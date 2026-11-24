"""Physically-based thin-lens depth-of-field with occlusion-aware halo suppression.

Uses the one-sided background thin-lens Circle-of-Confusion law parameterized by
focus distance Z_focus (m) and infinity blur radius R_inf (px), matched to the
baseline 6-level sigma_max=9.0 px pyramid so visual differences come purely from
metric depth physics and occlusion-aware foreground halo suppression.
"""

from __future__ import annotations

import cv2
import numpy as np


def circle_of_confusion_px(
    depth_m: np.ndarray,
    focus_dist_m: float,
    infinity_coc_px: float = 9.0,
    one_sided_bg: bool = True,
) -> np.ndarray:
    """Thin-lens CoC radius.

    If one_sided_bg=True (default for background bokeh), returns
    R_inf * max(0, 1 - Z_focus / Z) clipped to [0, R_inf].
    Otherwise returns symmetric |1 - Z_focus / Z|.
    """
    z = np.clip(depth_m.astype(np.float32), 0.3, 500.0)
    z_f = max(float(focus_dist_m), 0.3)
    rel = 1.0 - z_f / z
    coc = infinity_coc_px * (np.maximum(rel, 0.0) if one_sided_bg else np.abs(rel))
    return np.clip(coc, 0.0, infinity_coc_px).astype(np.float32)


def naive_metric_bokeh(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    focus_dist_m: float,
    levels: int = 6,
    max_sigma: float = 9.0,
) -> np.ndarray:
    """Standard Gaussian pyramid blur without occlusion masking (exhibits edge halos)."""
    coc_bg = circle_of_confusion_px(depth_m, focus_dist_m=focus_dist_m, infinity_coc_px=max_sigma, one_sided_bg=True)
    u = np.clip(coc_bg / max_sigma, 0.0, 1.0) * (levels - 1)
    stack = [rgb.astype(np.float32)] + [
        cv2.GaussianBlur(rgb, (0, 0), max_sigma * k / (levels - 1)).astype(np.float32)
        for k in range(1, levels)
    ]
    stack_arr = np.stack(stack, axis=0)
    low = np.floor(u).astype(np.int32)
    high = np.minimum(low + 1, levels - 1)
    frac = (u - low)[..., None]
    rows, cols = np.indices(low.shape)
    out = (1.0 - frac) * stack_arr[low, rows, cols] + frac * stack_arr[high, rows, cols]
    return np.clip(out, 0, 255).astype(np.uint8)


def occlusion_aware_bokeh(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    focus_dist_m: float,
    levels: int = 6,
    max_sigma: float = 9.0,
) -> np.ndarray:
    """Occlusion-aware thin-lens bokeh suppressing foreground-to-background color bleeding."""
    coc_bg = circle_of_confusion_px(depth_m, focus_dist_m=focus_dist_m, infinity_coc_px=max_sigma, one_sided_bg=True)
    img_f = rgb.astype(np.float32)
    trans_m = max(0.08 * focus_dist_m, 0.25)

    # Background visibility weight: sharp foreground (Z <= Z_focus) gets near-zero weight
    # in the background gather buffer so foreground structures do not bleed into blurred bg!
    bg_vis = np.clip((depth_m - focus_dist_m) / trans_m, 0.0, 1.0).astype(np.float32)
    w_bg = 0.005 + 0.995 * bg_vis
    premult = img_f * w_bg[..., None]

    bg_stack = [img_f]
    for k in range(1, levels):
        sigma = max_sigma * k / (levels - 1)
        num = cv2.GaussianBlur(premult, (0, 0), sigma)
        den = cv2.GaussianBlur(w_bg, (0, 0), sigma)[..., None]
        bg_stack.append(num / np.maximum(den, 1e-4))
    bg_arr = np.stack(bg_stack, axis=0)

    u_bg = np.clip(coc_bg / max_sigma, 0.0, 1.0) * (levels - 1)
    low = np.floor(u_bg).astype(np.int32)
    high = np.minimum(low + 1, levels - 1)
    frac = (u_bg - low)[..., None]
    rows, cols = np.indices(low.shape)
    bg_rendered = (1.0 - frac) * bg_arr[low, rows, cols] + frac * bg_arr[high, rows, cols]

    # Front-to-back anti-aliased alpha compositing of in-focus foreground over clean background
    alpha_fg = np.clip((focus_dist_m + 0.35 * trans_m - depth_m) / (0.5 * trans_m), 0.0, 1.0)[..., None]
    composited = alpha_fg * img_f + (1.0 - alpha_fg) * bg_rendered
    return np.clip(composited, 0, 255).astype(np.uint8)


def compute_halo_bleeding_metrics(
    rgb: np.ndarray,
    naive_blur: np.ndarray,
    occ_blur: np.ndarray,
    depth_m: np.ndarray,
    focus_dist_m: float,
    band_px: int = 9,
) -> dict[str, float]:
    """Quantify foreground-to-background halo bleeding inside a k-px background border ring
    adjacent to in-focus foreground discontinuities (Z <= focus_dist_m).
    """
    fg = (depth_m <= focus_dist_m * 1.02).astype(np.uint8)
    bg = depth_m > focus_dist_m * 1.15
    ksize = 2 * int(band_px) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    fg_dil = cv2.dilate(fg, kernel) > 0
    band = fg_dil & bg
    if int(np.sum(band)) < 50:
        return {
            "band_pixels": int(np.sum(band)),
            "naive_luma_bleed": float("nan"),
            "occ_luma_bleed": float("nan"),
            "luma_bleed_reduction_pct": float("nan"),
            "naive_rgb_bleed": float("nan"),
            "occ_rgb_bleed": float("nan"),
            "rgb_bleed_reduction_pct": float("nan"),
            "fg_sharpness_ratio_naive": float("nan"),
            "fg_sharpness_ratio_occ": float("nan"),
        }

    # Estimate local pure-background reference from uncontaminated interior background pixels
    # when comparing border halo leakage against sharp background rgb[band].
    rgb_f = rgb.astype(np.float32)
    nai_f = naive_blur.astype(np.float32)
    occ_f = occ_blur.astype(np.float32)

    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    y_rgb = rgb_f @ weights
    y_nai = nai_f @ weights
    y_occ = occ_f @ weights

    naive_luma = float(np.mean(np.abs(y_nai[band] - y_rgb[band])))
    occ_luma = float(np.mean(np.abs(y_occ[band] - y_rgb[band])))
    naive_rgb = float(np.mean(np.linalg.norm(nai_f[band] - rgb_f[band], axis=-1)))
    occ_rgb = float(np.mean(np.linalg.norm(occ_f[band] - rgb_f[band], axis=-1)))

    # Foreground edge sharpness preservation relative to input RGB
    fg_mask = fg > 0
    gx = cv2.Sobel(y_rgb, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y_rgb, cv2.CV_32F, 0, 1, ksize=3)
    grad_ref = np.hypot(gx, gy)
    fg_edge = fg_mask & (grad_ref > np.percentile(grad_ref[fg_mask], 70)) if np.any(fg_mask) else fg_mask

    def _mean_grad(y_img: np.ndarray) -> float:
        g1 = cv2.Sobel(y_img, cv2.CV_32F, 1, 0, ksize=3)
        g2 = cv2.Sobel(y_img, cv2.CV_32F, 0, 1, ksize=3)
        return float(np.mean(np.hypot(g1, g2)[fg_edge]))

    ref_g = max(_mean_grad(y_rgb), 1e-6) if np.any(fg_edge) else 1.0
    fg_sharp_naive = _mean_grad(y_nai) / ref_g if np.any(fg_edge) else 1.0
    fg_sharp_occ = _mean_grad(y_occ) / ref_g if np.any(fg_edge) else 1.0

    return {
        "band_pixels": int(np.sum(band)),
        "naive_luma_bleed": naive_luma,
        "occ_luma_bleed": occ_luma,
        "luma_bleed_reduction_pct": float(100.0 * (naive_luma - occ_luma) / max(naive_luma, 1e-6)),
        "naive_rgb_bleed": naive_rgb,
        "occ_rgb_bleed": occ_rgb,
        "rgb_bleed_reduction_pct": float(100.0 * (naive_rgb - occ_rgb) / max(naive_rgb, 1e-6)),
        "fg_sharpness_ratio_naive": float(fg_sharp_naive),
        "fg_sharpness_ratio_occ": float(fg_sharp_occ),
    }
