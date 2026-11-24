"""Run StereoSGBM on the 15 Middlebury trainingQ pairs, evaluate metric depth & O(Z^2) error, and run ablations."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.stereo.middlebury import Calibration, disparity_to_depth, load_pair, training_pairs
from src.stereo.sgbm import apply_clahe, guided_hole_fill, left_right_consistent, stereo_sgbm

OUT = ROOT / "outputs" / "stereo"


def bad_pixel_rate(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, thresh: float) -> float:
    valid = mask & np.isfinite(gt) & (gt > 0)
    if not np.any(valid):
        return float("nan")
    err = np.abs(pred[valid] - gt[valid])
    return float(np.mean(err > thresh))


def mae_on_mask(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    valid = mask & np.isfinite(gt) & (gt > 0)
    if not np.any(valid):
        return float("nan")
    return float(np.mean(np.abs(pred[valid] - gt[valid])))


def valid_disparity_stats(pred: np.ndarray, gt: np.ndarray, keep: np.ndarray) -> dict[str, float]:
    """Metrics on finite, positive GT disparities inside the kept valid mask."""
    gt_ok = np.isfinite(gt) & (gt > 0)
    valid = gt_ok & keep & np.isfinite(pred) & (pred > 0)
    if not np.any(gt_ok) or not np.any(valid):
        return {
            "mae_valid_px": float("nan"),
            "disp_rmse_px": float("nan"),
            "bad1_valid": float("nan"),
            "bad2_valid": float("nan"),
            "valid_frac": float("nan"),
        }
    err = np.abs(pred[valid] - gt[valid])
    return {
        "mae_valid_px": float(np.mean(err)),
        "disp_rmse_px": float(np.sqrt(np.mean(err**2))),
        "bad1_valid": float(np.mean(err > 1.0)),
        "bad2_valid": float(np.mean(err > 2.0)),
        "valid_frac": float(valid.sum() / gt_ok.sum()),
    }


def metric_depth_stats(
    pred_disp: np.ndarray, gt_disp: np.ndarray, keep: np.ndarray, calib: Calibration
) -> dict[str, float]:
    """Evaluate triangulated metric depth Z (m) on valid GT and kept prediction pixels."""
    gt_ok = np.isfinite(gt_disp) & (gt_disp > 0)
    valid = gt_ok & keep & np.isfinite(pred_disp) & (pred_disp > 0)
    if not np.any(valid):
        return {
            "z_med_m": float("nan"),
            "z_mae_m": float("nan"),
            "z_rmse_m": float("nan"),
            "z_absrel": float("nan"),
        }
    z_pred = disparity_to_depth(pred_disp, calib)[valid]
    z_gt = disparity_to_depth(gt_disp, calib)[valid]
    ok = np.isfinite(z_pred) & np.isfinite(z_gt) & (z_gt > 0)
    if not np.any(ok):
        return {
            "z_med_m": float("nan"),
            "z_mae_m": float("nan"),
            "z_rmse_m": float("nan"),
            "z_absrel": float("nan"),
        }
    diff = np.abs(z_pred[ok] - z_gt[ok])
    return {
        "z_med_m": float(np.median(z_gt[ok])),
        "z_mae_m": float(np.mean(diff)),
        "z_rmse_m": float(np.sqrt(np.mean(diff**2))),
        "z_absrel": float(np.mean(diff / z_gt[ok])),
    }


def colorize(disparity: np.ndarray) -> np.ndarray:
    finite = disparity[np.isfinite(disparity) & (disparity > 0)]
    vmax = float(np.percentile(finite, 98)) if finite.size else 1.0
    scaled = np.clip(disparity / max(vmax, 1e-3), 0, 1)
    scaled[~np.isfinite(disparity) | (disparity <= 0)] = 0
    return cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)


def run_ablation_variant(
    pairs_data: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Calibration]],
    use_clahe: bool,
    use_lrc: bool,
    use_fill: bool,
    mode: int,
    label: str,
) -> dict[str, object]:
    # Warm-up 2 pairs
    for _, left, right, _, _, calib in pairs_data[:2]:
        li = apply_clahe(left) if use_clahe else left
        ri = apply_clahe(right) if use_clahe else right
        p = stereo_sgbm(li, ri, calib, mode=mode)
        k = left_right_consistent(li, ri, p, calib, mode=mode) if use_lrc else (p > 0)
        if use_fill:
            _ = guided_hole_fill(p, k & (p > 0), left, radius=9, eps=1e-3)

    rows = []
    times_ms = []
    # 2 repeats for stable CPU latency
    for rep in range(2):
        for name, left, right, gt, non_occluded, calib in pairs_data:
            t0 = time.perf_counter()
            li = apply_clahe(left) if use_clahe else left
            ri = apply_clahe(right) if use_clahe else right
            pred = stereo_sgbm(li, ri, calib, mode=mode)
            keep_lrc = left_right_consistent(li, ri, pred, calib, mode=mode) if use_lrc else (pred > 0)
            keep_lrc = keep_lrc & (pred > 0)
            if use_fill:
                pred_eval, _ = guided_hole_fill(pred, keep_lrc, left, radius=9, eps=1e-3)
            else:
                pred_eval = pred
            times_ms.append((time.perf_counter() - t0) * 1000.0)

            if rep == 0:
                dstats = valid_disparity_stats(pred_eval, gt, keep_lrc)
                zstats_lrc = metric_depth_stats(pred_eval, gt, keep_lrc, calib)
                rows.append(
                    {
                        "mae_valid_px": dstats["mae_valid_px"],
                        "valid_frac": dstats["valid_frac"],
                        "bad1_nocc": bad_pixel_rate(pred_eval, gt, non_occluded, 1.0),
                        "bad2_nocc": bad_pixel_rate(pred_eval, gt, non_occluded, 2.0),
                        "mae_nocc_px": mae_on_mask(pred_eval, gt, non_occluded),
                        "z_mae_m": zstats_lrc["z_mae_m"],
                        "z_absrel": zstats_lrc["z_absrel"],
                    }
                )
    out = {"variant": label}
    for k in rows[0]:
        out[k] = float(np.mean([r[k] for r in rows]))
    out["cpu_ms_per_pair"] = float(np.mean(times_ms))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pairs_data = []
    for scene in training_pairs():
        left, right, gt, non_occluded, calib = load_pair(scene)
        pairs_data.append((scene.name, left, right, gt, non_occluded, calib))

    rows = []
    all_z_gt = []
    all_z_err = []
    all_d_err = []
    adirondack_calib = None

    for name, left, right, gt, non_occluded, calib in pairs_data:
        if name == "Adirondack":
            adirondack_calib = calib
        pred = stereo_sgbm(left, right, calib, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        consistent = left_right_consistent(left, right, pred, calib, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        keep = consistent & (pred > 0)
        depth = disparity_to_depth(pred, calib)
        depth[~keep] = np.nan

        dstats = valid_disparity_stats(pred, gt, keep)
        zstats = metric_depth_stats(pred, gt, keep, calib)
        row = {
            "scene": name,
            "mae_valid_px": dstats["mae_valid_px"],
            "disp_rmse_px": dstats["disp_rmse_px"],
            "bad1_valid": dstats["bad1_valid"],
            "bad2_valid": dstats["bad2_valid"],
            "valid_frac": dstats["valid_frac"],
            "bad1_nocc": bad_pixel_rate(pred, gt, non_occluded, 1.0),
            "bad2_nocc": bad_pixel_rate(pred, gt, non_occluded, 2.0),
            "lr_keep": float(np.mean(consistent)),
            **zstats,
        }
        rows.append(row)

        cv2.imwrite(str(OUT / f"{name}_disp.png"), colorize(pred))
        cv2.imwrite(str(OUT / f"{name}_invalid.png"), (~consistent).astype(np.uint8) * 255)
        np.save(OUT / f"{name}_depth.npy", depth)

        # Accumulate valid pixels for depth error vs GT range curve
        gt_ok = np.isfinite(gt) & (gt > 0) & keep
        if np.any(gt_ok):
            zp = depth[gt_ok]
            zg = disparity_to_depth(gt, calib)[gt_ok]
            ok = np.isfinite(zp) & np.isfinite(zg) & (zg > 0)
            all_z_gt.append(zg[ok])
            all_z_err.append(np.abs(zp[ok] - zg[ok]))
            all_d_err.append(np.abs(pred[gt_ok][ok] - gt[gt_ok][ok]))

        print(
            f"{name:14s} mae={row['mae_valid_px']:.2f}px rmse={row['disp_rmse_px']:.2f}px "
            f"valid={row['valid_frac']:.3f} z_mae={row['z_mae_m']:.3f}m z_abs={100*row['z_absrel']:.2f}%"
        )

    fields = list(rows[0].keys())
    mean_row = {"scene": "MEAN", **{k: float(np.mean([r[k] for r in rows])) for k in fields[1:]}}
    table = OUT / "middlebury_sgbm.csv"
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows + [mean_row])
    print(
        f"MEAN |d err|={mean_row['mae_valid_px']:.3f}px  dRMSE={mean_row['disp_rmse_px']:.3f}px  "
        f"valid={mean_row['valid_frac']:.3f}  Z_MAE={mean_row['z_mae_m']:.4f}m  "
        f"Z_RMSE={mean_row['z_rmse_m']:.4f}m  Z_AbsRel={100*mean_row['z_absrel']:.2f}%"
    )

    # Save empirical O(Z^2) error vs GT range curve
    z_gt_cat = np.concatenate(all_z_gt)
    z_err_cat = np.concatenate(all_z_err)
    d_err_cat = np.concatenate(all_d_err)
    edges = np.arange(1.0, 8.0, 0.5)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (z_gt_cat >= lo) & (z_gt_cat < hi)
        if int(np.sum(in_bin)) < 200:
            continue
        zc = 0.5 * (lo + hi)
        bins.append(
            {
                "z_min_m": float(lo),
                "z_max_m": float(hi),
                "z_mid_m": float(zc),
                "count": int(np.sum(in_bin)),
                "z_mae_m": float(np.mean(z_err_cat[in_bin])),
                "z_median_err_m": float(np.median(z_err_cat[in_bin])),
                "z_rmse_m": float(np.sqrt(np.mean(z_err_cat[in_bin] ** 2))),
                "disp_mae_px": float(np.mean(d_err_cat[in_bin])),
                "theory_adirondack_1px_m": float(
                    (zc**2) / (adirondack_calib.focal_px * adirondack_calib.baseline_m)
                )
                if adirondack_calib is not None
                else float("nan"),
            }
        )
    range_json = {
        "bins": bins,
        "adirondack_f_px": float(adirondack_calib.focal_px) if adirondack_calib else None,
        "adirondack_baseline_m": float(adirondack_calib.baseline_m) if adirondack_calib else None,
    }
    (OUT / "depth_error_vs_range.json").write_text(json.dumps(range_json, indent=2))

    # Run 5-row CPU ablation
    ablation_configs = [
        (False, False, False, cv2.STEREO_SGBM_MODE_SGBM_3WAY, "3WAY (d>0 only, no LRC)"),
        (False, True, False, cv2.STEREO_SGBM_MODE_SGBM_3WAY, "3WAY + 1px LRC (Baseline)"),
        (True, True, False, cv2.STEREO_SGBM_MODE_SGBM_3WAY, "3WAY + CLAHE + 1px LRC"),
        (True, True, True, cv2.STEREO_SGBM_MODE_SGBM_3WAY, "3WAY + CLAHE + Guided Hole-Fill"),
        (True, True, True, cv2.STEREO_SGBM_MODE_HH, "HH (8-path) + CLAHE + Guided Hole-Fill"),
    ]
    abl_rows = [
        run_ablation_variant(pairs_data, u_c, u_l, u_f, m_id, lbl)
        for u_c, u_l, u_f, m_id, lbl in ablation_configs
    ]
    abl_path = OUT / "middlebury_ablation.csv"
    with abl_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(abl_rows[0].keys()))
        writer.writeheader()
        writer.writerows(abl_rows)
    print("\n=== Stereo Ablation Summary ===")
    for r in abl_rows:
        print(
            f"{r['variant']:36s} | mae_val={r['mae_valid_px']:.2f} valid={r['valid_frac']:.3f} "
            f"b1_nocc={r['bad1_nocc']:.3f} b2_nocc={r['bad2_nocc']:.3f} mae_nocc={r['mae_nocc_px']:.2f} "
            f"z_abs={100*r['z_absrel']:.2f}% {r['cpu_ms_per_pair']:.1f}ms"
        )


if __name__ == "__main__":
    main()
