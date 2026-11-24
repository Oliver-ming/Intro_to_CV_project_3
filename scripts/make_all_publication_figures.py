"""Regenerate all publication-ready vector PDF & PNG figures with clean typography."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "Video-Depth-Anything"))

from utils.dc_utils import read_video_frames
from src.application.physical_bokeh import naive_metric_bokeh, occlusion_aware_bokeh
from src.calibration.global_scale import median_of_image_medians
from src.calibration.hierarchical import (
    FoVScaleCalibrator,
    HierarchicalScaleProbe,
    extract_geo_uncertainty_features,
    per_image_oracle_params,
)
from src.eval.nyu import K_NYU, depth_at, eval_mask, metrics, open_mat, rgb_at, scene_type_at, split_ids
from src.stereo.middlebury import TRAINING as MIDD_TRAIN, disparity_to_depth, load_pair
from src.stereo.sgbm import left_right_consistent, stereo_sgbm

FIGS = ROOT / "paper" / "figs"
STEREO_OUT = ROOT / "outputs" / "stereo"
CACHE_BASE = ROOT / "outputs" / "nyu_cache"
CACHE_RICH = ROOT / "outputs" / "nyu_cache_rich"
HIER_OUT = ROOT / "outputs" / "nyu_hierarchical"
VID_BASE = ROOT / "outputs" / "video"
VID_EXT = ROOT / "outputs" / "video_extended"
CLIP = ROOT / "third_party" / "Video-Depth-Anything" / "assets" / "example_videos" / "davis_rollercoaster.mp4"


def read_csv_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def make_pipeline_diagram() -> None:
    stereo_rows = {r["scene"]: r for r in read_csv_rows(STEREO_OUT / "middlebury_sgbm.csv")}
    sgbm_mae = float(stereo_rows["MEAN"]["mae_valid_px"])
    sgbm_val = 100.0 * float(stereo_rows["MEAN"]["valid_frac"])
    sum_rows = {(r["camera"], r["method"]): r for r in read_csv_rows(HIER_OUT / "nyu_hierarchical_summary.csv")}
    prim = sum_rows[("none", "scale_probe_2p_primary")]
    p_ar, p_d1 = float(prim["abs_rel"]), float(prim["d1"])
    abl_map = {r["variant"]: r for r in read_csv_rows(STEREO_OUT / "middlebury_ablation.csv")}
    sgbm_ms = float(abl_map["3WAY + 1px LRC (Baseline)"]["cpu_ms_per_pair"])
    ext_sum = json.loads((VID_EXT / "extended_summary.json").read_text())
    luma_red = ext_sum["coaster_halo_metrics_mean"]["luma_bleed_reduction_pct"]

    fig, ax = plt.subplots(figsize=(7.1, 3.40))
    boxes = [
        # Part 1
        (0.015, 0.72, 0.19, 0.19, "Middlebury trainQ\n15 pairs + calib.txt", "#dbe9f6"),
        (0.255, 0.72, 0.22, 0.19, "3-Way StereoSGBM\n+ 1 px LR Check", "#fde2c8"),
        (0.525, 0.72, 0.46, 0.19, f"Triangulation Z = fB / (d + doffs)\nMAE {sgbm_mae:.2f} px ({sgbm_val:.1f}% valid), {sgbm_ms:.1f} ms CPU", "#e3f1dc"),
        # Part 2 + Part 3 Hierarchical Calibration
        (0.015, 0.41, 0.19, 0.19, "NYU Depth v2 RGB\nTrain 795 / Test 654", "#dbe9f6"),
        (0.255, 0.41, 0.22, 0.19, "Frozen UniDepthV2-S\nDepth + FoV + Uncert + ViT", "#fde2c8"),
        (0.525, 0.41, 0.23, 0.19, "Hierarchical ScaleProbe-2P\n& Self-Calib K_train (Train CV)", "#f6d5e5"),
        (0.795, 0.41, 0.19, 0.19, f"Calibrated Metric Z\nAbsRel {p_ar:.4f}, d1 {p_d1:.3f}", "#e3f1dc"),
        # Video + Bokeh Application (shifted slightly lower so header never collides with vertical arrow)
        (0.015, 0.03, 0.19, 0.19, "Multi-Clip Videos\nCoaster (70f), Still, Moving", "#dbe9f6"),
        (0.255, 0.03, 0.22, 0.19, "DAv2-S vs VDA-S & Metric-VDA\nOffline & KV-Streaming", "#fde2c8"),
        (0.525, 0.03, 0.23, 0.19, "Shared K_clip & Zero-Sky\nSparse Keyframe IRLS", "#f6d5e5"),
        (0.795, 0.03, 0.19, 0.19, f"Occlusion-Aware Bokeh\n-{luma_red:.0f}% Halo Luma Bleed", "#e3f1dc"),
    ]
    arrows = [
        (0.205, 0.815, 0.255), (0.475, 0.815, 0.525),
        (0.205, 0.505, 0.255), (0.475, 0.505, 0.525), (0.755, 0.505, 0.795),
        (0.205, 0.125, 0.255), (0.475, 0.125, 0.525), (0.755, 0.125, 0.795),
    ]
    for x, y, w, h, text, fc in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01", fc=fc, ec="#444", lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.4)
    for x0, y, x1 in arrows:
        ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="->", lw=0.9, color="#333"))

    # Originate vertical metric anchor cleanly from Frozen UniDepthV2-S down to Keyframe IRLS
    # Route along clear open lane right of the 3rd header text (x=0.39 -> y=0.325 -> x=0.69 -> y=0.22)
    ax.plot([0.39, 0.39, 0.69], [0.41, 0.325, 0.325], ls="--", lw=0.95, color="#8e44ad")
    ax.annotate("", xy=(0.69, 0.22), xytext=(0.69, 0.325), arrowprops=dict(arrowstyle="->", lw=0.95, ls="--", color="#8e44ad"))
    ax.text(0.54, 0.340, "every 15f (raw / shared K_clip outdoors)", ha="center", va="bottom", fontsize=5.8, color="#6c3483")

    headers = [
        (0.945, "Part 1: Classical Stereo Geometry Baseline & Illumination/Guided Filtering"),
        (0.635, "Parts 2 & 3: Zero-Shot Metric Depth & Hierarchical Camera-Semantics Calibration"),
        (0.252, "Parts 1-3 & App: Video Consistency & Halo-Free Bokeh"),
    ]
    for y, title in headers:
        ax.text(0.015, y, title, fontsize=7.0, weight="bold", color="#1a1a1a")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.tight_layout(pad=0.1)
    fig.savefig(FIGS / "pipeline.pdf")
    fig.savefig(FIGS / "pipeline.png", dpi=220)
    plt.close(fig)


def make_stereo_panels() -> None:
    # 1. 2x4 Stereo visualization panel with invalid region mask and metric depth (meters)
    fig, axes = plt.subplots(2, 4, figsize=(7.1, 3.15))
    scenes = ["Adirondack", "Jadeplant"]
    for row_i, sname in enumerate(scenes):
        left, right, gt, _, calib = load_pair(MIDD_TRAIN / sname)
        pred = stereo_sgbm(left, right, calib)
        consistent = left_right_consistent(left, right, pred, calib)
        keep = consistent & (pred > 0)
        z_pred = disparity_to_depth(pred, calib)
        z_pred[~keep] = np.nan

        rgb_l = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
        d_show = np.where(keep, pred, np.nan)
        d_vmax = float(np.nanpercentile(pred[pred > 0], 98)) if np.any(pred > 0) else 60.0
        z_vmax = float(np.nanpercentile(z_pred[keep], 98)) if np.any(keep) else 6.0

        inv_vis = np.zeros((*keep.shape, 3), dtype=np.uint8)
        inv_vis[keep] = (238, 244, 248)
        inv_vis[~keep] = (178, 34, 34)  # Deep crimson for invalid/occluded pixels

        cmap_d = matplotlib.colormaps["magma"].copy()
        cmap_d.set_bad(color="black")
        cmap_z = matplotlib.colormaps["viridis_r"].copy()
        cmap_z.set_bad(color="black")

        im_d = axes[row_i, 1].imshow(d_show, cmap=cmap_d, vmin=0, vmax=d_vmax)
        axes[row_i, 0].imshow(rgb_l)
        axes[row_i, 2].imshow(inv_vis)
        im_z = axes[row_i, 3].imshow(z_pred, cmap=cmap_z, vmin=0.5, vmax=z_vmax)

        cbar_d = fig.colorbar(im_d, ax=axes[row_i, 1], fraction=0.044, pad=0.02)
        cbar_d.ax.tick_params(labelsize=5.5)
        cbar_z = fig.colorbar(im_z, ax=axes[row_i, 3], fraction=0.044, pad=0.02)
        cbar_z.ax.tick_params(labelsize=5.5)

        valid_pct = 100.0 * float(np.mean(keep))
        axes[row_i, 0].set_ylabel(sname, fontsize=7.5, weight="bold")
        axes[row_i, 2].set_xlabel(f"Valid: {valid_pct:.1f}%", fontsize=6.2)

    col_titles = ["Left Rectified RGB", "SGBM Disparity (px)", "Invalid Mask (red = rejected)", "Metric Depth Z (m)"]
    for c_i, t in enumerate(col_titles):
        axes[0, c_i].set_title(t, fontsize=7.5, weight="bold")
    for r_i in range(2):
        for c_i in range(4):
            axes[r_i, c_i].set_xticks([])
            axes[r_i, c_i].set_yticks([])

    fig.tight_layout(pad=0.3)
    fig.savefig(FIGS / "stereo_part1_panel.pdf")
    fig.savefig(FIGS / "stereo_part1_panel.png", dpi=220)
    plt.close(fig)

    # 2. Empirical O(Z^2) metric depth error vs GT range plot + Stereo ablation summary bar
    rng_data = json.loads((STEREO_OUT / "depth_error_vs_range.json").read_text())
    bins = rng_data["bins"]
    zm = np.array([b["z_mid_m"] for b in bins])
    zmae = np.array([b["z_mae_m"] for b in bins])
    zmed = np.array([b["z_median_err_m"] for b in bins])
    theory = np.array([b["theory_adirondack_1px_m"] for b in bins])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.35))
    ax1.plot(zm, zmae * 1000.0, "o-", color="#c0392b", lw=1.5, ms=3.5, label="Pooled Mean |Z error| (15 pairs)")
    ax1.plot(zm, zmed * 1000.0, "s--", color="#2980b9", lw=1.3, ms=3.2, label="Pooled Median |Z error|")
    ax1.plot(zm, theory * 1000.0, ":", color="#27ae60", lw=1.5, label=r"Theory $\partial Z/\partial d \cdot 1$px ($fB$ Adirondack)")
    ax1.set_yscale("log")
    ax1.set_xlabel("Ground-Truth Metric Range Z (m)", fontsize=8)
    ax1.set_ylabel("Metric Error (mm, log scale)", fontsize=8)
    ax1.set_title(r"(a) Empirical $O(Z^2)$ Stereo Range Sensitivity", fontsize=8.3, weight="bold")
    ax1.tick_params(labelsize=7)
    ax1.legend(fontsize=6.0, frameon=False, loc="upper left")

    abl_rows = read_csv_rows(STEREO_OUT / "middlebury_ablation.csv")
    labels = ["3WAY\nno LRC", "3WAY\n+LRC", "+CLAHE\n+LRC", "+CLAHE\n+Guided", "HH 8-path\n+Guided"]
    maen = [float(r["mae_nocc_px"]) for r in abl_rows]
    b2n = [100.0 * float(r["bad2_nocc"]) for r in abl_rows]
    x = np.arange(len(labels))
    bars = ax2.bar(x - 0.18, maen, width=0.34, color="#5dade2", edgecolor="#333", lw=0.5, label="nocc MAE (px)")
    ax2.set_ylabel("nocc Disparity MAE (px)", fontsize=7.8)
    ax2.set_ylim(0, 14.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=6.3)
    ax2.tick_params(axis="y", labelsize=7)
    for b, v in zip(bars, maen):
        ax2.text(b.get_x() + b.get_width() / 2 - 0.03, v + 0.28, f"{v:.1f}", ha="center", va="bottom", fontsize=5.4)
    ax2_r = ax2.twinx()
    bars2 = ax2_r.bar(x + 0.18, b2n, width=0.34, color="#e67e22", edgecolor="#333", lw=0.5, label="bad-2.0 (%)")
    ax2_r.set_ylabel("nocc bad-2.0 px (%)", fontsize=7.8)
    ax2_r.set_ylim(0, 50)
    ax2_r.tick_params(axis="y", labelsize=7)
    for b, v in zip(bars2, b2n):
        ax2_r.text(b.get_x() + b.get_width() / 2 + 0.03, v + 0.8, f"{v:.1f}%", ha="center", va="bottom", fontsize=5.3)
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2_r.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=5.8, frameon=False, loc="upper right", ncol=2)
    ax2.set_title("(b) LRC, CLAHE & Guided Hole-Fill Ablation", fontsize=8.3, weight="bold")

    fig.tight_layout(pad=0.45)
    fig.savefig(FIGS / "stereo_error_vs_range.pdf")
    fig.savefig(FIGS / "stereo_error_vs_range.png", dpi=220)
    plt.close(fig)


def make_nyu_figs() -> None:
    meta = json.loads((HIER_OUT / "metadata.json").read_text())
    sum_rows = read_csv_rows(HIER_OUT / "nyu_hierarchical_summary.csv")
    primary_weights = meta.get("primary_weights", {"geo": 0.25, "cls": 1.0, "pat": 1.0, "dfe": 1.0})

    with open_mat() as data:
        tr_ids, te_ids = split_ids("train"), split_ids("test")
        gts_tr = np.stack([depth_at(data, i) for i in tr_ids])
        masks_tr = np.stack([eval_mask(g) for g in gts_tr])
        gts_te = np.stack([depth_at(data, i) for i in te_ids])
        masks_te = np.stack([eval_mask(g) for g in gts_te])

        preds_tr = np.load(CACHE_BASE / "unidepth_none_train.npy").astype(np.float32)
        preds_te = np.load(CACHE_BASE / "unidepth_none_test.npy").astype(np.float32)
        intr_tr = np.load(CACHE_RICH / "intrinsics_train.npy")
        intr_te = np.load(CACHE_RICH / "intrinsics_test.npy")
        conf_tr = np.load(CACHE_RICH / "confidence_train.npy").astype(np.float32)
        conf_te = np.load(CACHE_RICH / "confidence_test.npy").astype(np.float32)

        def _fdict(split: str, preds_arr: np.ndarray, intr_arr: np.ndarray, conf_arr: np.ndarray) -> dict[str, np.ndarray]:
            dfe_sp = np.load(CACHE_RICH / f"depth_feat_spatial_{split}.npy")
            return {
                "geo": extract_geo_uncertainty_features(preds_arr, intr_arr, conf_arr),
                "cls": np.load(CACHE_RICH / f"cls_tokens_{split}.npy"),
                "pat": np.load(CACHE_RICH / f"patch_summary_{split}.npy"),
                "dfe": np.load(CACHE_RICH / f"depth_feat_{split}.npy"),
                "dfe_unc": dfe_sp[:, 6:8, :],
                "dfe_sp": dfe_sp,
                "cam": np.load(CACHE_RICH / f"cam_latent_{split}.npy"),
            }

        f_tr = _fdict("train", preds_tr, intr_tr, conf_tr)
        f_te = _fdict("test", preds_te, intr_te, conf_te)

        r_med_tr, r_opt_tr, betas_tr, _ = per_image_oracle_params(preds_tr, gts_tr, masks_tr)
        fov_cal = FoVScaleCalibrator.fit(intr_tr, r_opt_tr)

        pairs_tr = [(np.clip(preds_tr[i][masks_tr[i]], 1e-3, None).astype(np.float64), gts_tr[i][masks_tr[i]].astype(np.float64)) for i in range(len(preds_tr))]
        s_med = median_of_image_medians(pairs_tr)

        probe_prim = HierarchicalScaleProbe(two_param=True, block_weights=primary_weights)
        probe_prim.fit_with_cv(
            preds_tr,
            gts_tr,
            masks_tr,
            f_tr,
            precomputed_targets=(r_opt_tr, betas_tr),
        )
        cal2_te, _, _ = probe_prim.calibrate_batch(preds_te, f_te)

        raw_ar = np.array([metrics(p, g, m)["abs_rel"] for p, g, m in zip(preds_te, gts_te, masks_te)])
        glo_ar = np.array([metrics(s_med * p, g, m)["abs_rel"] for p, g, m in zip(preds_te, gts_te, masks_te)])
        prb_ar = np.array([metrics(p, g, m)["abs_rel"] for p, g, m in zip(cal2_te, gts_te, masks_te)])

        # Figure: FoV-Scale Coupling + Per-Image Error Change Histogram
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.55))
        f_tr_px = np.sqrt(intr_tr[:, 0, 0] * intr_tr[:, 1, 1])
        corr = float(np.corrcoef(np.log(f_tr_px), np.log(r_med_tr))[0, 1])
        ax1.scatter(f_tr_px, r_med_tr, s=7, alpha=0.35, color="#2b5c8f", edgecolors="none", label=f"NYU train (795), r = {corr:.2f}")
        xs = np.linspace(400, 760, 120)
        ys = np.exp(fov_cal.b0 + fov_cal.b1 * np.log(xs))
        ax1.plot(xs, ys, color="#c0392b", lw=1.6, label=r"Analytic fit $\hat{s}(\hat{f})$")
        ax1.axvline(float(np.sqrt(K_NYU[0, 0] * K_NYU[1, 1])), color="#27ae60", ls="--", lw=1.1, label="True Kinect $f = 519$ px")
        ax1.axhline(1.0, color="#888", ls=":", lw=0.8)
        ax1.set_xlim(400, 770)
        ax1.set_ylim(0.60, 1.36)
        ax1.set_xlabel(r"Predicted focal length $\hat{f}$ (px)", fontsize=8)
        ax1.set_ylabel(r"Optimal scale $r_i = \mathrm{median}(G_i / D_i)$", fontsize=8)
        ax1.set_title("(a) Camera-Depth Bias Coupling", fontsize=8.5, weight="bold")
        ax1.tick_params(labelsize=7)
        ax1.legend(fontsize=6.2, frameon=False, loc="upper right")

        glob_delta = glo_ar - raw_ar
        prb_delta = prb_ar - raw_ar
        bins = np.linspace(-0.16, 0.10, 42)
        ax2.hist(glob_delta, bins=bins, alpha=0.55, color="#e67e22", label=f"Frozen $s_{{\\mathrm{{med}}}}$ (hurts {100*np.mean(glob_delta>0):.1f}%)")
        ax2.hist(prb_delta, bins=bins, alpha=0.60, color="#1f618d", label=f"ScaleProbe-2P (hurts {100*np.mean(prb_delta>0):.1f}%)")
        ax2.axvline(0.0, color="#222", ls="--", lw=1.0)
        ax2.set_xlabel(r"Per-image $\Delta\mathrm{AbsRel}$ (Calibrated $-$ Raw, $<0$ better)", fontsize=8)
        ax2.set_ylabel("Test images (out of 654)", fontsize=8)
        ax2.set_title("(b) Per-Image Error Reduction on NYU Test", fontsize=8.5, weight="bold")
        ax2.tick_params(labelsize=7)
        ax2.legend(fontsize=6.2, frameon=False, loc="upper left")

        fig.tight_layout(pad=0.4)
        fig.savefig(FIGS / "fov_and_gain_hist.pdf")
        fig.savefig(FIGS / "fov_and_gain_hist.png", dpi=220)
        plt.close(fig)

        # Figure: Sparsification + Hierarchical Ladder
        spars_rows = read_csv_rows(HIER_OUT / "sparsification.csv")
        ause = float(meta["sparsification_ause"])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.55))
        rem = [100 * float(r["removed_frac"]) for r in spars_rows]
        ax1.plot(rem, [float(r["random_abs_rel"]) for r in spars_rows], ls="--", color="#7f8c8d", lw=1.2, label="Random removal")
        ax1.plot(rem, [float(r["uncertainty_abs_rel"]) for r in spars_rows], color="#2980b9", lw=1.6, label=f"UniDepthV2 $\\hat{{u}}(p)$ (AUSE={ause:.3f})")
        ax1.plot(rem, [float(r["oracle_abs_rel"]) for r in spars_rows], ls=":", color="#27ae60", lw=1.3, label="True error oracle")
        ax1.set_xlabel("Removed uncertain pixels (%)", fontsize=8)
        ax1.set_ylabel("NYU Test AbsRel", fontsize=8)
        ax1.set_title("(a) Pixel Uncertainty Sparsification", fontsize=8.5, weight="bold")
        ax1.tick_params(labelsize=7)
        ax1.legend(fontsize=6.3, frameon=False, loc="lower left")

        lookup = {(r["camera"], r["method"]): float(r["abs_rel"]) for r in sum_rows}
        ladder_keys = [
            ("none", "unidepth_raw", "Raw\nRGB"),
            ("none", "global_median_image", "Frozen\n$s_{\\mathrm{med}}$"),
            ("none", "global_power_law_2p", "Global\n2P"),
            ("none", "fov_analytic_2p", "FoV\n$\\hat{s}(\\hat{f})$"),
            ("none", "scale_probe_1p_v1", "Probe\n1P"),
            ("none", "scale_probe_2p_primary", "Probe\n2P"),
            ("ktrain", "scale_probe_2p_primary", "$\\hat{K}_{\\mathrm{train}}$\n+2P"),
            ("none", "oracle_1p_test_gt", "1P\nOracle"),
        ]
        labels = [lbl for _, _, lbl in ladder_keys]
        vals = [lookup[(c, m)] for c, m, _ in ladder_keys]
        colors = ["#95a5a6", "#e67e22", "#f39c12", "#5dade2", "#2e86c1", "#1b4f72", "#148f77", "#bdc3c7"]
        bars = ax2.bar(np.arange(len(vals)), vals, color=colors, width=0.65, edgecolor="#333", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.002, f"{v:.4f}", ha="center", va="bottom", fontsize=5.1)
        ax2.set_xticks(np.arange(len(vals)))
        ax2.set_xticklabels(labels, fontsize=6.3)
        ax2.set_ylim(0.0, 0.130)
        ax2.set_ylabel("Test AbsRel (lower is better)", fontsize=8)
        ax2.set_title("(b) Hierarchical Calibration Progression", fontsize=8.5, weight="bold")
        ax2.tick_params(axis="y", labelsize=7)

        fig.tight_layout(pad=0.45)
        fig.savefig(FIGS / "sparsification_and_ladder.pdf")
        fig.savefig(FIGS / "sparsification_and_ladder.png", dpi=220)
        plt.close(fig)

        # Qualitative NYU grid (7 columns INCLUDING Raw UniDepthV2-S depth & error map!)
        gain_order = np.argsort(prb_ar - raw_ar)
        med_gain_idx = int(gain_order[len(gain_order) // 2])
        picks = [med_gain_idx, 184, 42, 201, 550]

        DEPTH_MAP = (np.array(matplotlib.colormaps["magma"].colors) * 255).astype(np.uint8)
        ERR_MAP = (np.array(matplotlib.colormaps["viridis"].colors) * 255).astype(np.uint8)

        def colour(x: np.ndarray, hi: float, cmap: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
            img = cmap[(np.clip(x / hi, 0, 1) * 255).astype(np.uint8)]
            if valid is not None:
                img[~valid] = 0
            return img

        def tile(img: np.ndarray, text: str) -> np.ndarray:
            img = cv2.resize(img, (240, 180), interpolation=cv2.INTER_AREA).copy()
            cv2.rectangle(img, (0, 0), (240, 20), (0, 0, 0), -1)
            cv2.putText(img, text, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (255, 255, 255), 1, cv2.LINE_AA)
            return img

        rows_img = []
        for k in picks:
            idx = te_ids[k]
            rgb, gt = rgb_at(data, idx), depth_at(data, idx)
            mask = eval_mask(gt)
            raw = np.clip(preds_te[k], 1e-3, 10.0)
            glo = np.clip(s_med * raw, 1e-3, 10.0)
            prb = np.clip(cal2_te[k], 1e-3, 10.0)
            hi = float(np.percentile(gt[mask], 98))
            name = scene_type_at(data, idx)
            tiles = [
                tile(rgb, f"#{k} {name}"),
                tile(colour(gt, hi, DEPTH_MAP, mask), f"GT (0-{hi:.1f} m)"),
                tile(colour(raw, hi, DEPTH_MAP), f"Raw UniV2 AR {raw_ar[k]:.3f}"),
                tile(colour(glo, hi, DEPTH_MAP), f"Frozen x{s_med:.3f} AR {glo_ar[k]:.3f}"),
                tile(colour(prb, hi, DEPTH_MAP), f"Probe-2P AR {prb_ar[k]:.3f}"),
                tile(colour(np.abs(raw - gt) / np.maximum(gt, 1e-3), 0.5, ERR_MAP, mask), "|err|/gt Raw (0-0.5)"),
                tile(colour(np.abs(prb - gt) / np.maximum(gt, 1e-3), 0.5, ERR_MAP, mask), "|err|/gt Probe-2P"),
            ]
            rows_img.append(np.concatenate(tiles, axis=1))

        full = np.concatenate(rows_img, axis=0)
        cv2.imwrite(str(FIGS / "nyu_hierarchical_compare.png"), cv2.cvtColor(full, cv2.COLOR_RGB2BGR))


def make_video_and_bokeh_figs() -> None:
    frames, _ = read_video_frames(str(CLIP), 100, -1, 640)
    base_sum = json.loads((VID_BASE / "summary.json").read_text())
    ext_sum = json.loads((VID_EXT / "extended_summary.json").read_text())
    t_csv = np.loadtxt(VID_BASE / "temporal_change.csv", delimiter=",", skiprows=1)
    ext_npz = np.load(VID_EXT / "extended_video_depth.npz")
    z_vid_off = ext_npz["metric_video_offline"].astype(np.float32)

    chg_str_arr = np.array(ext_sum["relative_flow_warped_series"]["vda_streaming"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.5))
    ts = t_csv[:, 0]
    ax1.plot(ts, t_csv[:, 1], lw=1.1, color="#e67e22", label=f"DAv2-S frame-wise ({t_csv[:,1].mean():.4f})")
    ax1.plot(ts, chg_str_arr, lw=1.2, color="#2980b9", label=f"VDA-S streaming ({chg_str_arr.mean():.4f})")
    ax1.plot(ts, t_csv[:, 2], lw=1.2, color="#148f77", label=f"VDA-S offline ({t_csv[:,2].mean():.4f})")
    ax1.set_ylim(0.0105, 0.0320)
    ax1.set_xlabel(r"Frame index $t$", fontsize=8)
    ax1.set_ylabel("Flow-warped |change| (OPW)", fontsize=8)
    ax1.set_title("(a) Temporal Flicker Across 69 Frame Pairs", fontsize=8.4, weight="bold")
    ax1.tick_params(labelsize=7)
    ax1.legend(fontsize=6.1, frameon=False, loc="upper left")

    models_lbl = ["DAv2-S\n(Frame)", "VDA-S\n(Stream)", "M-VDA\n(Stream)", "VDA-S\n(Offline)"]
    mibs = [
        base_sum["efficiency"][0]["peak_gpu_mib"],
        ext_sum["streaming_efficiency"]["peak_gpu_mib"],
        ext_sum["metric_vda_streaming_efficiency"]["peak_gpu_mib"],
        base_sum["efficiency"][1]["peak_gpu_mib"],
    ]
    lats = [
        base_sum["efficiency"][0]["ms_per_frame"],
        ext_sum["streaming_efficiency"]["ms_per_frame"],
        ext_sum["metric_vda_streaming_efficiency"]["ms_per_frame"],
        base_sum["efficiency"][1]["ms_per_frame"],
    ]
    x_pos = np.arange(len(models_lbl))
    bars = ax2.bar(x_pos - 0.17, mibs, width=0.34, color=["#f39c12", "#2980b9", "#8e44ad", "#148f77"], edgecolor="#333", linewidth=0.5)
    ax2.set_yscale("log")
    ax2.set_ylim(100, 24000)
    ax2.set_ylabel("Peak GPU Memory (MiB, log)", fontsize=7.8)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(models_lbl, fontsize=6.5)
    ax2.tick_params(axis="y", labelsize=7)
    for bar, m_val in zip(bars, mibs):
        ax2.text(bar.get_x() + bar.get_width() / 2, m_val * 1.14, f"{m_val:.0f}", ha="center", va="bottom", fontsize=5.8)

    ax2_r = ax2.twinx()
    bars2 = ax2_r.bar(x_pos + 0.17, lats, width=0.34, color="#bdc3c7", edgecolor="#333", linewidth=0.5)
    ax2_r.set_ylim(0, 92)
    ax2_r.set_ylabel("Latency (ms/frame)", fontsize=7.8)
    ax2_r.tick_params(axis="y", labelsize=7)
    for bar, l_val in zip(bars2, lats):
        ax2_r.text(bar.get_x() + bar.get_width() / 2, l_val + 1.8, f"{l_val:.1f}", ha="center", va="bottom", fontsize=5.8)
    ax2.set_title("(b) Memory & Latency Trade-off", fontsize=8.4, weight="bold")

    fig.tight_layout(pad=0.45)
    fig.savefig(FIGS / "video_streaming_tradeoff.pdf")
    fig.savefig(FIGS / "video_streaming_tradeoff.png", dpi=220)
    plt.close(fig)

    # Bokeh Halo Comparison Figure using fixed K_clip + zero-sky metric depth
    focus_m = ext_sum["focus_dist_m"]
    cy, cx, cw, ch = ext_sum["zoom_crop_yxwh"]
    coc_p95 = ext_sum["coc_p95_fixed_px"]
    luma_red = ext_sum["coaster_halo_metrics_mean"]["luma_bleed_reduction_pct"]
    f35 = frames[35]
    b35_naive = naive_metric_bokeh(f35, z_vid_off[35], focus_dist_m=focus_m, levels=6, max_sigma=9.0)
    b35_occ = occlusion_aware_bokeh(f35, z_vid_off[35], focus_dist_m=focus_m, levels=6, max_sigma=9.0)
    z_hi_vis = float(np.percentile(z_vid_off, 95))

    fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.85))
    axes[0, 0].imshow(f35)
    axes[0, 0].add_patch(Rectangle((cx, cy), cw, ch, fill=False, edgecolor="#ff3333", lw=1.6))
    axes[0, 0].set_title("Frame 35 RGB (zoom box)", fontsize=7.5, weight="bold")
    axes[0, 1].imshow(np.clip(z_vid_off[35], 0, z_hi_vis), cmap="magma_r")
    axes[0, 1].set_title(rf"Anchored Metric $Z$ ($Z_f={focus_m:.0f}$ m, $b=0$)", fontsize=7.5, weight="bold")
    axes[0, 2].imshow(b35_occ)
    axes[0, 2].add_patch(Rectangle((cx, cy), cw, ch, fill=False, edgecolor="#2ecc71", lw=1.6))
    axes[0, 2].set_title(rf"Thin-Lens Bokeh ($P_{{95}}\ R_{{\mathrm{{CoC}}}}={coc_p95:.1f}$ px)", fontsize=7.5, weight="bold")

    axes[1, 0].imshow(f35[cy : cy + ch, cx : cx + cw])
    axes[1, 0].set_title("Zoom: Sharp Input Boundary", fontsize=7.4)
    axes[1, 1].imshow(b35_naive[cy : cy + ch, cx : cx + cw])
    axes[1, 1].set_title("Zoom: Naive Blur (dark halo bleed)", fontsize=7.4, color="#c0392b")
    axes[1, 2].imshow(b35_occ[cy : cy + ch, cx : cx + cw])
    axes[1, 2].set_title(f"Zoom: Occlusion-Aware (-{luma_red:.0f}% bleed)", fontsize=7.4, color="#1e8449")

    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout(pad=0.3)
    fig.savefig(FIGS / "bokeh_halo_comparison.pdf")
    fig.savefig(FIGS / "bokeh_halo_comparison.png", dpi=220)
    plt.close(fig)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    make_pipeline_diagram()
    make_stereo_panels()
    make_nyu_figs()
    make_video_and_bokeh_figs()
    print("All publication figures regenerated cleanly!")


if __name__ == "__main__":
    main()
