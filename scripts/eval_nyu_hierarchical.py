"""Phase 1 / Step 2 Complete Evaluation: ScaleProbe-2P v2, Feature & Error Attribution, Honest GPU Timing.

Strictly adheres to the course protocol:
- All parameters, scalers, virtual camera kappa, Bayesian sigmas, block weights,
  and Ridge CV regularizations are fitted exclusively on the 795 NYU training images (trainNdxs).
- Zero test-set ground truth or test-batch statistics are used by any non-oracle method.
- Enforces GPU memory cap (0.20) and single-model-at-a-time cleanup.
"""

from __future__ import annotations

import csv
import gc
import json
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import torch
from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights, keypointrcnn_resnet50_fpn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "UniDepth"))

from src.calibration.global_scale import median_of_image_medians
from src.calibration.hierarchical import (
    FoVScaleCalibrator,
    HierarchicalScaleProbe,
    _dual_gram_ridge_fit,
    apply_power_law,
    bayesian_inverse_variance_scale,
    build_cv_pixel_cache,
    extract_geo_uncertainty_features,
    per_image_oracle_params,
    per_image_planar_4p_oracle,
    uncertainty_weighted_image_scale,
)
from src.calibration.scale import PERSON_HEIGHT_M, implied_height_m, median_depth, touches_border
from src.eval.boundary import si_boundary_f1
from src.eval.nyu import (
    HEIGHT,
    K_NYU,
    WIDTH,
    depth_at,
    eval_mask,
    mean_metrics,
    metrics,
    open_mat,
    rgb_at,
    scene_type_at,
    split_ids,
)

CACHE_BASE = ROOT / "outputs" / "nyu_cache"
CACHE_RICH = ROOT / "outputs" / "nyu_cache_rich"
OUT = ROOT / "outputs" / "nyu_hierarchical"
FIGS = ROOT / "paper" / "figs"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_rich_split(data, cam: str, split: str) -> dict[str, object]:
    ids = split_ids(split)
    preds = np.load(CACHE_BASE / f"unidepth_{cam}_{split}.npy").astype(np.float32)
    gts = np.stack([depth_at(data, i) for i in ids])
    masks = np.stack([eval_mask(g) for g in gts])
    scenes = [scene_type_at(data, i) for i in ids]
    prefix = "" if cam == "none" else f"{cam}_"
    intr = np.load(CACHE_RICH / f"intrinsics_{prefix}{split}.npy")
    conf = np.load(CACHE_RICH / f"confidence_{prefix}{split}.npy").astype(np.float32)
    cls_tok = np.load(CACHE_RICH / f"cls_tokens_{prefix}{split}.npy")
    pat_sum = np.load(CACHE_RICH / f"patch_summary_{prefix}{split}.npy")
    dfe = np.load(CACHE_RICH / f"depth_feat_{prefix}{split}.npy")
    dfe_sp = np.load(CACHE_RICH / f"depth_feat_spatial_{prefix}{split}.npy")
    cam_lat = np.load(CACHE_RICH / f"cam_latent_{prefix}{split}.npy")
    geo = extract_geo_uncertainty_features(preds, intr, conf)
    feat_dict = {
        "geo": geo,
        "cls": cls_tok,
        "pat": pat_sum,
        "dfe": dfe,
        "dfe_unc": dfe_sp[:, 6:8, :],
        "dfe_sp": dfe_sp,
        "cam": cam_lat,
    }
    if (CACHE_RICH / f"dav2_cls_{split}.npy").exists():
        feat_dict["dav2_cls"] = np.load(CACHE_RICH / f"dav2_cls_{split}.npy")
        feat_dict["dav2_stats"] = np.load(CACHE_RICH / f"dav2_cross_stats_{split}.npy")
    return {
        "ids": ids,
        "preds": preds,
        "gts": gts,
        "masks": masks,
        "scenes": scenes,
        "intr": intr,
        "conf": conf,
        "feats": feat_dict,
    }


def score_predictions(
    pred_batch: np.ndarray,
    raw_rows: list[dict[str, float]],
    gts: np.ndarray,
    masks: np.ndarray,
    rgb_raw_rows: list[dict[str, float]] | None = None,
    compute_boundary: bool = False,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    rows = [metrics(p, g, m) for p, g, m in zip(pred_batch, gts, masks)]
    agg = mean_metrics(rows)
    raw_ar = np.array([r["abs_rel"] for r in raw_rows])
    cur_ar = np.array([r["abs_rel"] for r in rows])
    agg["improved_frac"] = float(np.mean(cur_ar < raw_ar - 1e-7))
    agg["hurt_frac"] = float(np.mean(cur_ar > raw_ar + 1e-7))
    agg["max_hurt_delta"] = float(np.max(cur_ar - raw_ar))
    if rgb_raw_rows is not None:
        rgb_ar = np.array([r["abs_rel"] for r in rgb_raw_rows])
        agg["hurt_vs_rgb_raw_frac"] = float(np.mean(cur_ar > rgb_ar + 1e-7))
    else:
        agg["hurt_vs_rgb_raw_frac"] = agg["hurt_frac"]
    if compute_boundary:
        agg["si_boundary_f1"] = float(np.mean([si_boundary_f1(p, g, m) for p, g, m in zip(pred_batch, gts, masks)]))
    else:
        agg["si_boundary_f1"] = float("nan")
    return agg, rows


def compute_train_diagnostics(
    preds_tr: np.ndarray,
    gts_tr: np.ndarray,
    masks_tr: np.ndarray,
    intr_tr: np.ndarray,
    conf_tr: np.ndarray,
) -> dict[str, object]:
    """Compute exact train-split physical attribution statistics (no test peeking)."""
    f_tr = np.sqrt(intr_tr[:, 0, 0] * intr_tr[:, 1, 1])
    r_med_tr, r_opt_tr, betas_tr, _ = per_image_oracle_params(preds_tr, gts_tr, masks_tr)

    # Pixel-level correlation between predicted uncertainty u_hat(p) and scale-aligned log error |log(r_i*D) - log G|
    intra_pearson, intra_spearman = [], []
    for i in range(len(preds_tr)):
        m = masks_tr[i]
        p = np.clip(preds_tr[i][m].astype(np.float64), 1e-3, 10.0)
        g = gts_tr[i][m].astype(np.float64)
        u = np.clip(conf_tr[i][m].astype(np.float64), 1e-4, 1e3)
        abs_log_err = np.abs(np.log(r_med_tr[i] * p) - np.log(g))
        # Subsample evenly for fast exact Spearman alongside full Pearson
        c_p = float(np.corrcoef(u, abs_log_err)[0, 1])
        c_s = float(spearmanr(u[::16], abs_log_err[::16]).statistic)
        if np.isfinite(c_p):
            intra_pearson.append(c_p)
        if np.isfinite(c_s):
            intra_spearman.append(c_s)

    return {
        "focal_median_px": float(np.median(f_tr)),
        "focal_q25_px": float(np.percentile(f_tr, 25)),
        "focal_q75_px": float(np.percentile(f_tr, 75)),
        "focal_std_px": float(np.std(f_tr)),
        "true_kinect_f_px": float(np.sqrt(K_NYU[0, 0] * K_NYU[1, 1])),
        "rho_log_f_log_rmed": float(np.corrcoef(np.log(f_tr), np.log(r_med_tr))[0, 1]),
        "rho_log_f_log_ropt": float(np.corrcoef(np.log(f_tr), np.log(r_opt_tr))[0, 1]),
        "unc_mean_intra_pearson": float(np.mean(intra_pearson)),
        "unc_mean_intra_spearman": float(np.mean(intra_spearman)),
        "beta_median": float(np.median(betas_tr)),
        "beta_q25": float(np.percentile(betas_tr, 25)),
        "beta_q75": float(np.percentile(betas_tr, 75)),
    }


def run_feature_and_weight_cv_ablation(
    tr: dict, te: dict, raw_rows_te: list[dict], cv_pixel_cache: dict
) -> tuple[list[dict], dict[str, float]]:
    """Run incremental feature-block ablation strictly selecting weights on Train 5-fold CV."""
    preds_tr, gts_tr, masks_tr, f_tr = tr["preds"], tr["gts"], tr["masks"], tr["feats"]
    preds_te, gts_te, masks_te, f_te = te["preds"], te["gts"], te["masks"], te["feats"]

    candidates = [
        ("geo_only", {"geo": 0.25}),
        ("cls_only", {"cls": 1.0}),
        ("pat_only", {"pat": 1.0}),
        ("dfe_basic_only", {"dfe": 1.0}),
        ("dfe_spatial_only", {"dfe_sp": 1.0}),
        ("cam_latent_only", {"cam": 1.0}),
        ("geo_cls", {"geo": 0.25, "cls": 1.0}),
        ("geo_cls_pat", {"geo": 0.25, "cls": 1.0, "pat": 1.0}),
        ("geo_cls_pat_dfe_basic", {"geo": 0.25, "cls": 1.0, "pat": 1.0, "dfe": 1.0}),
        ("v2_dfe_sp_cam10", {"geo": 0.25, "cls": 1.0, "pat": 1.0, "dfe_sp": 1.0, "cam": 1.0}),
        ("v2_dfe10_unc05_cam15", {"geo": 0.25, "cls": 1.0, "pat": 1.0, "dfe": 1.0, "dfe_unc": 0.5, "cam": 1.5}),
        ("v2_dfe10_unc08_cam20", {"geo": 0.25, "cls": 1.0, "pat": 1.0, "dfe": 1.0, "dfe_unc": 0.8, "cam": 2.0}),
        ("v2_dfe_sp15_cam15", {"geo": 0.25, "cls": 1.0, "pat": 1.0, "dfe_sp": 1.5, "cam": 1.5}),
    ]
    if "dav2_cls" in f_tr:
        candidates.extend([
            ("dav2_only", {"dav2_cls": 1.0, "dav2_stats": 0.25}),
            ("v2_plus_dav2_w05", {"geo": 0.25, "cls": 1.0, "pat": 1.0, "dfe": 1.0, "dfe_unc": 0.5, "cam": 1.5, "dav2_cls": 0.5, "dav2_stats": 0.25}),
            ("v2_plus_dav2_w10", {"geo": 0.25, "cls": 1.0, "pat": 1.0, "dfe": 1.0, "dfe_unc": 0.5, "cam": 1.5, "dav2_cls": 1.0, "dav2_stats": 0.25}),
        ])

    abl_rows = []
    best_v2_cv = float("inf")
    best_v2_weights = {"geo": 0.25, "cls": 1.0, "pat": 1.0, "dfe": 1.0, "dfe_unc": 0.5, "cam": 1.5}
    best_plus_cv = float("inf")
    best_plus_weights = {"geo": 0.25, "cls": 1.0, "pat": 1.0, "dfe": 1.0, "dfe_unc": 0.5, "cam": 1.5, "dav2_cls": 0.5, "dav2_stats": 0.25}

    _, r_opt_tr, betas_tr, _ = per_image_oracle_params(preds_tr, gts_tr, masks_tr)

    # Precompute fast GPU chunks for test scoring in ablation table (zero tuning/selection on test!)
    te_cache = build_cv_pixel_cache(preds_te, gts_te, masks_te, n_chunks=8)
    raw_ar_te = np.array([r["abs_rel"] for r in raw_rows_te])

    def _fast_test_score(s_te: np.ndarray, b_te: np.ndarray) -> dict[str, float]:
        n_te = len(preds_te)
        ar = np.empty(n_te, dtype=np.float64)
        rm = np.empty(n_te, dtype=np.float64)
        d1 = np.empty(n_te, dtype=np.float64)
        dm_arr = te_cache["d_med_full"]
        p_l = te_cache["p_valid_list"]
        g_l = te_cache["g_valid_list"]
        rl_l = te_cache["rel_log_valid_list"]
        s_t = torch.from_numpy(s_te.astype(np.float32)).cuda()
        b_t = torch.from_numpy(b_te.astype(np.float32)).cuda()
        dmed_g = torch.from_numpy(dm_arr.astype(np.float32)).cuda()
        with torch.inference_mode():
            for lo_i, hi_i, p_cpu, g_cpu, rlog_cpu, ids_cpu, inv_cpu in te_cache["chunks_cpu"]:
                p_g = p_cpu.cuda()
                g_g = g_cpu.cuda()
                rlog_g = rlog_cpu.cuda()
                ids_g = ids_cpu.cuda()
                inv_g = inv_cpu.cuda()
                dm_s = (s_t[lo_i:hi_i] * dmed_g[lo_i:hi_i])[ids_g]
                cal = (dm_s * torch.exp(b_t[lo_i:hi_i][ids_g] * rlog_g)).clamp_(1e-3, 10.0)
                diff = (cal - g_g).abs_()
                rel = diff / g_g
                sq = diff * diff
                hit = (torch.maximum(g_g / cal, cal / g_g) < 1.25).float()
                cnt = hi_i - lo_i
                z1 = torch.zeros(cnt, device="cuda")
                z2 = torch.zeros(cnt, device="cuda")
                z3 = torch.zeros(cnt, device="cuda")
                z1.index_add_(0, ids_g, rel)
                z2.index_add_(0, ids_g, sq)
                z3.index_add_(0, ids_g, hit)
                ar[lo_i:hi_i] = (z1 * inv_g).cpu().numpy()
                rm[lo_i:hi_i] = torch.sqrt(z2 * inv_g).cpu().numpy()
                d1[lo_i:hi_i] = (z3 * inv_g).cpu().numpy()
                del p_g, g_g, rlog_g, ids_g, inv_g
        del s_t, b_t, dmed_g
        return {
            "abs_rel": float(np.mean(ar)),
            "rmse": float(np.mean(rm)),
            "d1": float(np.mean(d1)),
            "hurt_frac": float(np.mean(ar > raw_ar_te + 1e-7)),
        }

    for label, bw in candidates:
        probe = HierarchicalScaleProbe(two_param=True, block_weights=bw)
        cv_info = probe.fit_with_cv(
            preds_tr, gts_tr, masks_tr, f_tr, precomputed_targets=(r_opt_tr, betas_tr), cv_pixel_cache=cv_pixel_cache
        )
        s_te, b_te = probe.predict_params(f_te)
        agg_te = _fast_test_score(s_te, b_te)
        abl_rows.append(
            {
                "config": label,
                "feature_dim": cv_info["feature_dim"],
                "train_cv_abs_rel": cv_info["cv_abs_rel"],
                "alpha_scale": cv_info["alpha_scale"],
                "alpha_beta": cv_info["alpha_beta"],
                "test_abs_rel": agg_te["abs_rel"],
                "test_rmse": agg_te["rmse"],
                "test_d1": agg_te["d1"],
                "test_hurt_pct": 100.0 * agg_te["hurt_frac"],
            }
        )
        print(
            f"  [FeatAbl] {label:22s} dim={cv_info['feature_dim']:4d} "
            f"TrainCV={cv_info['cv_abs_rel']:.4f} (a_s={cv_info['alpha_scale']}, a_b={cv_info['alpha_beta']}) "
            f"-> Test AbsRel={agg_te['abs_rel']:.4f}, d1={agg_te['d1']:.4f}",
            flush=True,
        )
        if (label.startswith("v2_") or label == "geo_cls_pat_dfe_basic") and not label.startswith("v2_plus_"):
            if cv_info["cv_abs_rel"] < best_v2_cv:
                best_v2_cv = cv_info["cv_abs_rel"]
                best_v2_weights = dict(bw)
        if label.startswith("v2_plus_"):
            if cv_info["cv_abs_rel"] < best_plus_cv:
                best_plus_cv = cv_info["cv_abs_rel"]
                best_plus_weights = dict(bw)

    del te_cache
    torch.cuda.empty_cache()
    print(f"[CV Winner Single-Backbone v2] weights={best_v2_weights} (TrainCV={best_v2_cv:.4f})")
    if "dav2_cls" in f_tr:
        print(f"[CV Winner Cross-Backbone v2+] weights={best_plus_weights} (TrainCV={best_plus_cv:.4f})")
    return abl_rows, {"v2_weights": best_v2_weights, "plus_weights": best_plus_weights}


def run_error_decomposition(tr: dict, te: dict, best_v2_weights: dict[str, float], cv_pixel_cache: dict) -> list[dict]:
    """Decompose remaining error strictly inside Train 5-fold CV + quantify 4P planar-tilt oracle."""
    preds_tr, gts_tr, masks_tr, f_tr = tr["preds"], tr["gts"], tr["masks"], tr["feats"]
    preds_te, gts_te, masks_te = te["preds"], te["gts"], te["masks"]

    probe = HierarchicalScaleProbe(two_param=True, block_weights=best_v2_weights)
    cv_info = probe.fit_with_cv(preds_tr, gts_tr, masks_tr, f_tr, cv_pixel_cache=cv_pixel_cache)
    a_s, a_b = cv_info["alpha_scale"], cv_info["alpha_beta"]

    _, r_opt_tr, betas_tr, s_2p_tr = per_image_oracle_params(preds_tr, gts_tr, masks_tr)
    thetas_tr, ars_4p_tr = per_image_planar_4p_oracle(preds_tr, gts_tr, masks_tr)
    thetas_te, ars_4p_te = per_image_planar_4p_oracle(preds_te, gts_te, masks_te)

    # Generate clean OOF 5-fold CV predictions on Train for a0, beta, ty, tx
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_s = np.zeros(len(preds_tr), dtype=np.float64)
    oof_b = np.zeros(len(preds_tr), dtype=np.float64)
    oof_theta = np.zeros((len(preds_tr), 4), dtype=np.float64)
    for tr_i, va_i in kf.split(preds_tr):
        pr = HierarchicalScaleProbe(two_param=True, block_weights=best_v2_weights)
        X_t = pr._design_matrix(pr._slice_dict(f_tr, tr_i), fit_scaler=True)
        X_v = pr._design_matrix(pr._slice_dict(f_tr, va_i), fit_scaler=False)
        cs, is_ = _dual_gram_ridge_fit(X_t, np.log(r_opt_tr[tr_i]), a_s)
        cb, ib_ = _dual_gram_ridge_fit(X_t, betas_tr[tr_i], a_b)
        oof_s[va_i] = np.exp(X_v @ cs + is_)
        oof_b[va_i] = np.clip(X_v @ cb + ib_, 0.80, 1.25)
        for col in range(4):
            alpha_c = a_s if col == 0 else a_b
            cc, ic = _dual_gram_ridge_fit(X_t, thetas_tr[tr_i, col], alpha_c)
            oof_theta[va_i, col] = X_v @ cc + ic

    def _eval_tr(s_vec: np.ndarray, b_vec: np.ndarray) -> float:
        errs = []
        for i in range(len(preds_tr)):
            cal = apply_power_law(preds_tr[i], float(s_vec[i]), float(b_vec[i]))
            m = masks_tr[i]
            p_c = np.clip(cal[m], 1e-3, 10.0)
            g_c = gts_tr[i][m]
            errs.append(float(np.mean(np.abs(p_c - g_c) / g_c)))
        return float(np.mean(errs))

    ar_oof = _eval_tr(oof_s, oof_b)
    ar_s_pred_b_ora = _eval_tr(oof_s, betas_tr)
    ar_s_ora_b_pred = _eval_tr(s_2p_tr, oof_b)
    ar_2p_ora_tr = _eval_tr(s_2p_tr, betas_tr)

    r2_names = ["a0_scale", "beta_contrast", "ty_vertical_tilt", "tx_horizontal_tilt"]
    rows = [
        {"diagnostic": "train_5fold_oof_pred_s_pred_beta", "abs_rel": ar_oof, "note": "ScaleProbe-2P v2 Train CV"},
        {"diagnostic": "train_5fold_oof_pred_s_oracle_beta", "abs_rel": ar_s_pred_b_ora, "note": "Fixes contrast exponent only"},
        {"diagnostic": "train_5fold_oracle_s_pred_beta", "abs_rel": ar_s_ora_b_pred, "note": "Fixes global scale only (dominant residual)"},
        {"diagnostic": "train_oracle_2p", "abs_rel": ar_2p_ora_tr, "note": "2P upper bound on train"},
        {"diagnostic": "train_oracle_4p_planar_tilt", "abs_rel": float(np.mean(ars_4p_tr)), "note": "Adds spatial planar tilt (ty, tx)"},
        {"diagnostic": "test_oracle_4p_planar_tilt", "abs_rel": float(np.mean(ars_4p_te)), "note": f"Test #201 bedroom 4P oracle AR = {ars_4p_te[201]:.4f} (tx={thetas_te[201,3]:+.3f})"},
    ]
    for col, nm in enumerate(r2_names):
        y_t = thetas_tr[:, col]
        y_p = oof_theta[:, col]
        ss_res = float(np.sum((y_t - y_p) ** 2))
        ss_tot = float(np.sum((y_t - np.mean(y_t)) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-8)
        rows.append({"diagnostic": f"train_cv_r2_{nm}", "abs_rel": r2, "note": f"Pearson r={np.corrcoef(y_t, y_p)[0,1]:.3f}"})
    return rows


def run_sparsification(preds: np.ndarray, gts: np.ndarray, masks: np.ndarray, conf: np.ndarray) -> tuple[list[dict], float]:
    fractions = np.linspace(0.0, 0.90, 19)
    rng = np.random.default_rng(0)
    rand_curve, mod_curve, ora_curve = [], [], []
    for f in fractions:
        r_vals, m_vals, o_vals = [], [], []
        keep_q = 1.0 - f
        for i in range(len(preds)):
            m = masks[i]
            p = np.clip(preds[i][m], 1e-3, 10.0).astype(np.float64)
            g = gts[i][m].astype(np.float64)
            err = np.abs(p - g) / g
            u = conf[i][m].astype(np.float64)
            k = max(10, int(round(keep_q * len(err))))
            r_vals.append(float(np.mean(err[rng.choice(len(err), size=k, replace=False)])))
            m_vals.append(float(np.mean(err[np.argpartition(u, k - 1)[:k]])))
            o_vals.append(float(np.mean(err[np.argpartition(err, k - 1)[:k]])))
        rand_curve.append(float(np.mean(r_vals)))
        mod_curve.append(float(np.mean(m_vals)))
        ora_curve.append(float(np.mean(o_vals)))

    rows = [
        {"removed_frac": float(f), "random_abs_rel": r, "uncertainty_abs_rel": m, "oracle_abs_rel": o}
        for f, r, m, o in zip(fractions, rand_curve, mod_curve, ora_curve)
    ]
    norm_mod = np.array(mod_curve) / mod_curve[0]
    norm_ora = np.array(ora_curve) / ora_curve[0]
    ause = float(np.trapezoid(norm_mod - norm_ora, fractions))
    return rows, ause


def run_bayesian_height_experiment(
    data,
    tr: dict,
    te: dict,
    s_global: float,
    probe_scales_tr_oof: np.ndarray,
    probe_scales_te: np.ndarray,
) -> tuple[list[dict], dict[str, float]]:
    """Compare hard person-height override vs Bayesian soft-fusion (Zero GT masks & train-estimated sigmas)."""
    eigen_box = np.zeros((HEIGHT, WIDTH), dtype=bool)
    eigen_box[45:471, 41:601] = True

    weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    torch.cuda.reset_peak_memory_stats()
    det = keypointrcnn_resnet50_fpn(weights=weights).to("cuda").eval()

    def _collect_votes(ids: np.ndarray, raw_preds: np.ndarray, intr: np.ndarray) -> list[float | None]:
        obs: list[float | None] = [None] * len(ids)
        with torch.inference_mode():
            for step, idx in enumerate(ids):
                rgb = rgb_at(data, idx)
                ten = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div(255.0).to("cuda")
                out = det([ten])[0]
                boxes = out["boxes"].cpu().numpy()
                scores = out["scores"].cpu().numpy()
                kps = out["keypoints"].cpu().numpy()
                fy = float(intr[step, 1, 1])
                votes = []
                for box, sc, kp in zip(boxes, scores, kps):
                    if sc < 0.75 or touches_border(box, WIDTH, HEIGHT, margin=3):
                        continue
                    head_ok = kp[0, 2] > 0.5 or kp[1, 2] > 0.5 or kp[2, 2] > 0.5
                    ankles_ok = kp[15, 2] > 0.5 and kp[16, 2] > 0.5
                    if not (head_ok and ankles_ok):
                        continue
                    # Use ONLY static Eigen crop box (NEVER GT-derived valid mask!)
                    z_box = median_depth(raw_preds[step], box, eigen_box)
                    if z_box is None:
                        continue
                    h_px = float(max(kp[15, 1], kp[16, 1]) - box[1])
                    if h_px < 40:
                        continue
                    h_imp = implied_height_m(h_px, z_box, fy)
                    if 0.4 < h_imp < 4.5:
                        votes.append(PERSON_HEIGHT_M / h_imp)
                if votes:
                    obs[step] = float(np.median(votes))
        return obs

    obs_tr = _collect_votes(tr["ids"], tr["preds"], tr["intr"])
    obs_te = _collect_votes(te["ids"], te["preds"], te["intr"])
    del det
    gc.collect()
    torch.cuda.empty_cache()

    # Estimate Bayesian sigmas strictly on NYU train split!
    _, r_opt_tr, _, _ = per_image_oracle_params(tr["preds"], tr["gts"], tr["masks"])
    sigma_global_prior = float(np.std(r_opt_tr - s_global))
    sigma_probe_prior = float(np.std(r_opt_tr - probe_scales_tr_oof))
    obs_res_tr = [obs_tr[i] - r_opt_tr[i] for i in range(len(obs_tr)) if obs_tr[i] is not None]
    sigma_obs_train = float(np.sqrt(np.mean(np.square(obs_res_tr)))) if obs_res_tr else 0.35

    n_voted_tr = sum(v is not None for v in obs_tr)
    n_voted_te = sum(v is not None for v in obs_te)
    print(
        f"[Bayesian Train Sigmas] train_voted={n_voted_tr}, test_voted={n_voted_te}, "
        f"sigma_glob={sigma_global_prior:.4f}, sigma_probe={sigma_probe_prior:.4f}, sigma_obs={sigma_obs_train:.4f}"
    )

    configs = [
        ("global_prior_only", [s_global] * len(te["ids"])),
        ("hard_override_person_prior", [v if v is not None else 1.0 for v in obs_te]),
        ("hard_override_with_global_fallback", [v if v is not None else s_global for v in obs_te]),
        (
            "bayesian_soft_fusion_global_prior",
            [bayesian_inverse_variance_scale(s_global, sigma_global_prior, v, sigma_obs_train) for v in obs_te],
        ),
        (
            "bayesian_soft_fusion_probe_prior",
            [
                bayesian_inverse_variance_scale(float(probe_scales_te[i]), sigma_probe_prior, obs_te[i], sigma_obs_train)
                for i in range(len(te["ids"]))
            ],
        ),
    ]
    out_rows = []
    for name, sc_list in configs:
        all_m = [metrics(sc_list[i] * te["preds"][i], te["gts"][i], te["masks"][i]) for i in range(len(te["ids"]))]
        voted_m = [all_m[i] for i in range(len(te["ids"])) if obs_te[i] is not None]
        agg_all = mean_metrics(all_m)
        agg_vot = mean_metrics(voted_m) if voted_m else {"abs_rel": float("nan"), "rmse": float("nan"), "d1": float("nan")}
        out_rows.append(
            {
                "strategy": name,
                "voted_images": n_voted_te,
                "all654_abs_rel": agg_all["abs_rel"],
                "all654_rmse": agg_all["rmse"],
                "all654_d1": agg_all["d1"],
                "voted_subset_abs_rel": agg_vot["abs_rel"],
                "voted_subset_d1": agg_vot["d1"],
            }
        )
    sigma_meta = {
        "train_voted": n_voted_tr,
        "test_voted": n_voted_te,
        "sigma_global_prior": sigma_global_prior,
        "sigma_probe_prior": sigma_probe_prior,
        "sigma_obs": sigma_obs_train,
    }
    return out_rows, sigma_meta


def measure_honest_single_image_gpu_timing(probe_v2: HierarchicalScaleProbe) -> tuple[float, float]:
    """Measure true end-to-end single-image GPU latency (quantile + pooling + linear + power-law) and VRAM."""
    gpu_probe = probe_v2.export_gpu_module("cuda")
    torch.cuda.reset_peak_memory_stats()
    base_mib = torch.cuda.memory_allocated() / 2**20

    depth = torch.rand(HEIGHT, WIDTH, device="cuda") * 6.0 + 0.5
    intr = torch.tensor([[570.0, 0.0, 320.0], [0.0, 570.0, 240.0], [0.0, 0.0, 1.0]], device="cuda")
    conf = torch.rand(HEIGHT, WIDTH, device="cuda") * 0.2 + 0.01
    cls_t = torch.randn(4, 384, device="cuda")
    f_last = torch.randn(35, 46, 384, device="cuda")
    df_map = torch.randn(256, 35, 46, device="cuda")
    cam_t = torch.randn(4, 256, device="cuda")

    for _ in range(25):
        _ = gpu_probe.forward_single(depth, intr, conf, cls_t, f_last, df_map, cam_t)

    torch.cuda.synchronize()
    times = []
    for _ in range(200):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = gpu_probe.forward_single(depth, intr, conf, cls_t, f_last, df_map, cam_t)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    peak_mib = torch.cuda.max_memory_allocated() / 2**20 - base_mib
    return float(np.mean(times)), float(max(peak_mib, 0.0))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.15, device=0)

    ktrain_meta = json.loads((CACHE_RICH / "ktrain_calibration.json").read_text())
    summary_rows = []
    saved: dict[str, object] = {}

    with open_mat() as data:
        # Load RGB-only first so cross-regime hurt_vs_rgb_raw works cleanly
        tr_none = load_rich_split(data, "none", "train")
        te_none = load_rich_split(data, "none", "test")
        rgb_raw_rows_te = [metrics(p, g, m) for p, g, m in zip(te_none["preds"], te_none["gts"], te_none["masks"])]

        train_diag = compute_train_diagnostics(
            tr_none["preds"], tr_none["gts"], tr_none["masks"], tr_none["intr"], tr_none["conf"]
        )
        print("Train diagnostics:", json.dumps(train_diag, indent=2))

        print("\n=== Running Strictly Train-CV Feature & Block-Weight Ablation ===")
        none_tr_cache = build_cv_pixel_cache(tr_none["preds"], tr_none["gts"], tr_none["masks"], n_chunks=10)
        feat_abl_rows, best_w_dict = run_feature_and_weight_cv_ablation(tr_none, te_none, rgb_raw_rows_te, none_tr_cache)
        write_csv(OUT / "feature_ablation.csv", feat_abl_rows)
        best_v2_weights = best_w_dict["v2_weights"]
        best_plus_weights = best_w_dict["plus_weights"]

        print("\n=== Running Train-CV Error Attribution & 4P Planar Oracle ===")
        decomp_rows = run_error_decomposition(tr_none, te_none, best_v2_weights, none_tr_cache)
        write_csv(OUT / "error_decomposition.csv", decomp_rows)

        for cam in ("none", "ktrain", "nyu"):
            tr = tr_none if cam == "none" else load_rich_split(data, cam, "train")
            te = te_none if cam == "none" else load_rich_split(data, cam, "test")
            preds_tr, gts_tr, masks_tr, f_tr = tr["preds"], tr["gts"], tr["masks"], tr["feats"]
            preds_te, gts_te, masks_te, f_te = te["preds"], te["gts"], te["masks"], te["feats"]

            _, r_opt_tr, betas_tr, _ = per_image_oracle_params(preds_tr, gts_tr, masks_tr)
            tr_cache = none_tr_cache if cam == "none" else build_cv_pixel_cache(preds_tr, gts_tr, masks_tr, n_chunks=10)

            # 0. Raw baseline
            raw_rows_te = [metrics(p, g, m) for p, g, m in zip(preds_te, gts_te, masks_te)]
            agg_raw, _ = score_predictions(preds_te, raw_rows_te, gts_te, masks_te, rgb_raw_rows_te, compute_boundary=True)
            summary_rows.append({"camera": cam, "method": "unidepth_raw", "params": 0, **agg_raw})

            # 1A. Frozen median-of-image-medians
            pairs_tr = [
                (np.clip(preds_tr[i][masks_tr[i]], 1e-3, None).astype(np.float64), gts_tr[i][masks_tr[i]].astype(np.float64))
                for i in range(len(preds_tr))
            ]
            s_med = median_of_image_medians(pairs_tr)
            agg_med, glob_rows_te = score_predictions(
                s_med * preds_te, raw_rows_te, gts_te, masks_te, rgb_raw_rows_te, compute_boundary=(cam == "none")
            )
            summary_rows.append({"camera": cam, "method": "global_median_image", "params": 1, **agg_med})

            v1_weights = {"geo": 0.25, "cls": 1.0, "pat": 1.0, "dfe": 1.0}
            if cam == "none":
                s_unc = uncertainty_weighted_image_scale(preds_tr, gts_tr, masks_tr, tr["conf"], gamma=1.0)
                agg_unc, _ = score_predictions(s_unc * preds_te, raw_rows_te, gts_te, masks_te, rgb_raw_rows_te)
                summary_rows.append({"camera": cam, "method": "global_uncertainty_weighted", "params": 1, **agg_unc})

                s0_pow = float(np.median(r_opt_tr))
                b0_pow = float(np.median(betas_tr))
                pow_te = np.stack([apply_power_law(p, s0_pow, b0_pow) for p in preds_te])
                agg_pow, _ = score_predictions(pow_te, raw_rows_te, gts_te, masks_te, rgb_raw_rows_te, compute_boundary=True)
                summary_rows.append({"camera": cam, "method": "global_power_law_2p", "params": 2, **agg_pow})

                fov_cal = FoVScaleCalibrator.fit(tr["intr"], r_opt_tr)
                fov_scales_te = fov_cal.predict_scale(te["intr"])
                agg_fov, _ = score_predictions(fov_scales_te[:, None, None] * preds_te, raw_rows_te, gts_te, masks_te, rgb_raw_rows_te)
                summary_rows.append({"camera": cam, "method": "fov_analytic_2p", "params": 2, **agg_fov})

                # Reproducible v1 reference rows
                p1 = HierarchicalScaleProbe(two_param=False, block_weights=v1_weights)
                p1.fit_with_cv(preds_tr, gts_tr, masks_tr, f_tr, alphas=(0.15, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0), coupled_beta_mult=3.0, precomputed_targets=(r_opt_tr, betas_tr), cv_pixel_cache=tr_cache)
                c1, _, _ = p1.calibrate_batch(preds_te, f_te)
                agg_p1, _ = score_predictions(c1, raw_rows_te, gts_te, masks_te, rgb_raw_rows_te)
                summary_rows.append({"camera": cam, "method": "scale_probe_1p_v1", "params": 1, **agg_p1})

                p2_v1 = HierarchicalScaleProbe(two_param=True, block_weights=v1_weights)
                cv_v1 = p2_v1.fit_with_cv(preds_tr, gts_tr, masks_tr, f_tr, alphas=(0.15, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0), coupled_beta_mult=3.0, precomputed_targets=(r_opt_tr, betas_tr), cv_pixel_cache=tr_cache)
                c2_v1, _, _ = p2_v1.calibrate_batch(preds_te, f_te)
                agg_v1, _ = score_predictions(c2_v1, raw_rows_te, gts_te, masks_te, rgb_raw_rows_te, compute_boundary=True)
                summary_rows.append({"camera": cam, "method": "scale_probe_2p_v1", "params": 2, **agg_v1})
                print(f"[none] ScaleProbe-2P v1 -> AbsRel={agg_v1['abs_rel']:.4f}, d1={agg_v1['d1']:.4f}")

            # ScaleProbe-2P v2 (Single-backbone CV winner!)
            probe_v2 = HierarchicalScaleProbe(two_param=True, block_weights=best_v2_weights)
            cv_v2 = probe_v2.fit_with_cv(preds_tr, gts_tr, masks_tr, f_tr, precomputed_targets=(r_opt_tr, betas_tr), cv_pixel_cache=tr_cache)
            cal2_te, s2_te, _ = probe_v2.calibrate_batch(preds_te, f_te)
            agg_v2, p2_rows_te = score_predictions(cal2_te, raw_rows_te, gts_te, masks_te, rgb_raw_rows_te, compute_boundary=True)
            summary_rows.append({"camera": cam, "method": "scale_probe_2p_v2", "params": 2, **agg_v2})
            print(
                f"[{cam}] ScaleProbe-2P v2 CV={cv_v2['cv_abs_rel']:.4f} -> "
                f"Test AbsRel={agg_v2['abs_rel']:.4f}, RMSE={agg_v2['rmse']:.4f}, d1={agg_v2['d1']:.4f}, "
                f"SI-Bnd={agg_v2['si_boundary_f1']:.4f}, hurt={100*agg_v2['hurt_frac']:.1f}% (vsRGB={100*agg_v2['hurt_vs_rgb_raw_frac']:.1f}%)"
            )

            if cam == "none":
                # Cross-backbone ablation (ScaleProbe-2P+)
                if "dav2_cls" in f_tr:
                    probe_plus = HierarchicalScaleProbe(two_param=True, block_weights=best_plus_weights)
                    cv_plus = probe_plus.fit_with_cv(preds_tr, gts_tr, masks_tr, f_tr, precomputed_targets=(r_opt_tr, betas_tr), cv_pixel_cache=tr_cache)
                    cal_plus_te, _, _ = probe_plus.calibrate_batch(preds_te, f_te)
                    agg_plus, _ = score_predictions(cal_plus_te, raw_rows_te, gts_te, masks_te, rgb_raw_rows_te, compute_boundary=True)
                    summary_rows.append({"camera": cam, "method": "scale_probe_2p_plus_dav2", "params": 2, **agg_plus})
                    print(f"[none] ScaleProbe-2P+ (+DAv2) CV={cv_plus['cv_abs_rel']:.4f} -> Test AbsRel={agg_plus['abs_rel']:.4f}, d1={agg_plus['d1']:.4f}")

                # Leave-One-Scene-Out with full internal CV inside keep_tr (zero leakage)
                l1so_preds = np.empty_like(preds_te)
                tr_scenes_arr = np.array(tr["scenes"])
                for sc_name in sorted(set(te["scenes"])):
                    keep_tr = tr_scenes_arr != sc_name
                    te_idx = [i for i, s in enumerate(te["scenes"]) if s == sc_name]
                    pr_l1 = HierarchicalScaleProbe(two_param=True, block_weights=best_v2_weights)
                    pr_l1.fit_with_cv(
                        preds_tr[keep_tr],
                        gts_tr[keep_tr],
                        masks_tr[keep_tr],
                        pr_l1._slice_dict(f_tr, np.where(keep_tr)[0]),
                        precomputed_targets=(r_opt_tr[keep_tr], betas_tr[keep_tr]),
                    )
                    cal_sub, _, _ = pr_l1.calibrate_batch(preds_te[te_idx], pr_l1._slice_dict(f_te, np.asarray(te_idx)))
                    l1so_preds[te_idx] = cal_sub
                agg_l1so, _ = score_predictions(l1so_preds, raw_rows_te, gts_te, masks_te, rgb_raw_rows_te)
                summary_rows.append({"camera": cam, "method": "scale_probe_2p_v2_unseen_scene", "params": 2, **agg_l1so})
                print(f"[none] LOSO Unseen Scene -> Test AbsRel={agg_l1so['abs_rel']:.4f}, d1={agg_l1so['d1']:.4f}")

                # Test GT Oracles (consistent median anchor -> 0% hurt!)
                r_med_te, r_opt_te, betas_te, s_2p_te = per_image_oracle_params(preds_te, gts_te, masks_te)
                ora1_te = r_opt_te[:, None, None].astype(np.float32) * preds_te
                ora2_te = np.stack([apply_power_law(preds_te[i], float(s_2p_te[i]), float(betas_te[i])) for i in range(len(preds_te))])
                _, ars_4p_te = per_image_planar_4p_oracle(preds_te, gts_te, masks_te)
                agg_o1, _ = score_predictions(ora1_te, raw_rows_te, gts_te, masks_te, rgb_raw_rows_te)
                agg_o2, _ = score_predictions(ora2_te, raw_rows_te, gts_te, masks_te, rgb_raw_rows_te)
                summary_rows.append({"camera": cam, "method": "oracle_1p_test_gt", "params": -1, **agg_o1})
                summary_rows.append({"camera": cam, "method": "oracle_2p_test_gt", "params": -2, **agg_o2})

                spars_rows, ause = run_sparsification(preds_te, gts_te, masks_te, te["conf"])
                write_csv(OUT / "sparsification.csv", spars_rows)

                # Get OOF probe scales on Train for train-fitted Bayesian sigma_prior
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                s_tr_oof = np.zeros(len(preds_tr), dtype=np.float64)
                for tri, vai in kf.split(preds_tr):
                    pr_oof = HierarchicalScaleProbe(two_param=True, block_weights=best_v2_weights)
                    X_t = pr_oof._design_matrix(pr_oof._slice_dict(f_tr, tri), fit_scaler=True)
                    X_v = pr_oof._design_matrix(pr_oof._slice_dict(f_tr, vai), fit_scaler=False)
                    cs, is_ = _dual_gram_ridge_fit(X_t, np.log(r_opt_tr[tri]), cv_v2["alpha_scale"])
                    s_tr_oof[vai] = np.exp(X_v @ cs + is_)

                bayes_rows, sigma_meta = run_bayesian_height_experiment(data, tr, te, s_med, s_tr_oof, s2_te)
                write_csv(OUT / "bayesian_height_prior.csv", bayes_rows)

                gpu_ms, gpu_mib = measure_honest_single_image_gpu_timing(probe_v2)
                print(f"[GPU Single-Image Probe v2] {gpu_ms:.3f} ms/img, {gpu_mib:.2f} MiB delta")

                saved = {
                    "raw_rows_te": raw_rows_te,
                    "glob_rows_te": glob_rows_te,
                    "probe2_rows_te": p2_rows_te,
                    "scenes_te": te["scenes"],
                    "scenes_tr": tr["scenes"],
                    "s_med": s_med,
                    "s_unc": s_unc,
                    "s0_pow": s0_pow,
                    "b0_pow": b0_pow,
                    "fov_b0": fov_cal.b0,
                    "fov_b1": fov_cal.b1,
                    "cv_v1": cv_v1,
                    "cv_v2": cv_v2,
                    "v2_weights": best_v2_weights,
                    "plus_weights": best_plus_weights,
                    "ause": ause,
                    "gpu_ms": gpu_ms,
                    "gpu_mib": gpu_mib,
                    "sigma_meta": sigma_meta,
                    "oracle_4p_test_absrel": float(np.mean(ars_4p_te)),
                    "oracle_4p_bedroom201_absrel": float(ars_4p_te[201]),
                    "raw_bedroom201_absrel": float(raw_rows_te[201]["abs_rel"]),
                }
            elif cam == "ktrain":
                saved["ktrain_p2_rows_te"] = p2_rows_te

        write_csv(OUT / "nyu_hierarchical_summary.csv", summary_rows)

        # Per-scene table
        scenes_te = saved["scenes_te"]
        scenes_tr = saved["scenes_tr"]
        scene_rows = []
        for name in sorted(set(scenes_te)):
            idx = [i for i, s in enumerate(scenes_te) if s == name]
            scene_rows.append(
                {
                    "scene_type": name,
                    "test_images": len(idx),
                    "train_images": sum(s == name for s in scenes_tr),
                    "raw_abs_rel": float(np.mean([saved["raw_rows_te"][i]["abs_rel"] for i in idx])),
                    "global_s_abs_rel": float(np.mean([saved["glob_rows_te"][i]["abs_rel"] for i in idx])),
                    "probe2p_abs_rel": float(np.mean([saved["probe2_rows_te"][i]["abs_rel"] for i in idx])),
                    "ktrain_probe2p_abs_rel": float(np.mean([saved["ktrain_p2_rows_te"][i]["abs_rel"] for i in idx])),
                }
            )
        scene_rows.sort(key=lambda r: -r["test_images"])
        write_csv(OUT / "per_scene_hierarchical.csv", scene_rows)

        meta_out = {
            "ktrain_calibration": ktrain_meta,
            "train_diagnostics": train_diag,
            "s_med_none": saved["s_med"],
            "s_unc_none": saved["s_unc"],
            "global_power_law": {"s0": saved["s0_pow"], "beta0": saved["b0_pow"]},
            "fov_analytic": {"b0": saved["fov_b0"], "b1": saved["fov_b1"]},
            "probe2p_v1_cv": saved["cv_v1"],
            "probe2p_v2_cv": saved["cv_v2"],
            "probe2p_v2_weights": saved["v2_weights"],
            "probe2p_plus_weights": saved["plus_weights"],
            "probe2p_gpu_ms_per_image": saved["gpu_ms"],
            "probe2p_gpu_peak_delta_mib": saved["gpu_mib"],
            "sparsification_ause": saved["ause"],
            "bayesian_sigmas": saved["sigma_meta"],
            "oracle_4p_test_absrel": saved["oracle_4p_test_absrel"],
            "oracle_4p_bedroom201_absrel": saved["oracle_4p_bedroom201_absrel"],
            "raw_bedroom201_absrel": saved["raw_bedroom201_absrel"],
        }
        (OUT / "metadata.json").write_text(json.dumps(meta_out, indent=2))
        print("\nSaved all upgraded NYU hierarchical evaluation results to", OUT)


if __name__ == "__main__":
    main()
