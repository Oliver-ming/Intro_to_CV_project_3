"""Phase 2 & 3 / Step 3: Streaming KV-cache VDA, Metric-VDA benchmark, shared-camera zero-sky metric anchoring,
causal online streaming anchoring, multi-clip OPW/TAE/drift evaluation, and halo-free thin-lens bokeh.
"""

from __future__ import annotations

import csv
import gc
import json
import sys
import time
from pathlib import Path

import cv2
import imageio
import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
VDA = ROOT / "third_party" / "Video-Depth-Anything"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(VDA))
sys.path.insert(0, str(ROOT / "third_party" / "UniDepth"))

from utils.dc_utils import read_video_frames
from unidepth.models import UniDepthV2
from unidepth.utils.camera import Pinhole

from src.application.physical_bokeh import (
    circle_of_confusion_px,
    compute_halo_bleeding_metrics,
    naive_metric_bokeh,
    occlusion_aware_bokeh,
)
from src.fusion.flow import _sample, fuse_with_flow, load_raft, pairwise_flow
from src.fusion.metric_video import (
    causal_online_metric_anchoring,
    fit_keyframe_metric_alignment,
    run_vda_streaming,
)

CLIP_COASTER = VDA / "assets" / "example_videos" / "davis_rollercoaster.mp4"
CLIP_STILL = ROOT / "data" / "clips" / "still.mp4"
CLIP_MOVING = ROOT / "data" / "clips" / "moving.mp4"
VDA_CKPT = VDA / "checkpoints" / "video_depth_anything_vits.pth"
MVDA_CKPT = VDA / "checkpoints" / "metric_video_depth_anything_vits.pth"
BASE_OUT = ROOT / "outputs" / "video"
OUT = ROOT / "outputs" / "video_extended"
CFG = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}
MAX_LEN, MAX_RES, INPUT_SIZE = 100, 640, 518
MAGMA_R = (np.array(matplotlib.colormaps["magma_r"].colors) * 255).astype(np.uint8)


def clip_range(depths: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(depths, 2)), float(np.percentile(depths, 98))


def normalize(depth: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((depth - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def label(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def compute_flows_cpu(frames: np.ndarray, tolerance: float = 1.0):
    """Compute pairwise RAFT flows and store lightweight tensors on CPU."""
    raft, transforms = load_raft("cuda")
    flows = []
    for t in range(1, len(frames)):
        fw, bw = pairwise_flow(raft, transforms, frames[t - 1], frames[t], "cuda")
        res = torch.linalg.norm(_sample(fw, bw) + bw, dim=1)[0]
        flows.append((fw.cpu(), bw.cpu(), (res <= tolerance).cpu()))
    del raft
    gc.collect()
    torch.cuda.empty_cache()
    return flows


def rel_temporal_change(norm_depths: np.ndarray, flows) -> np.ndarray:
    vals = []
    for t, (_, bw_cpu, mask_cpu) in enumerate(flows, start=1):
        bw = bw_cpu.cuda()
        mask = mask_cpu.cuda()
        prev = torch.from_numpy(norm_depths[t - 1])[None, None].cuda()
        warped = _sample(prev, bw)[0, 0]
        cur = torch.from_numpy(norm_depths[t]).cuda()
        vals.append(float((warped - cur).abs()[mask].mean()))
    return np.array(vals, dtype=np.float32)


def metric_opw_absrel(metric_depths: np.ndarray, flows, min_depth: float = 0.4) -> np.ndarray:
    """Optical-Flow Warping Error (OPW) expressed as flow-warped AbsRel over FB-consistent pixels."""
    vals = []
    for t, (_, bw_cpu, mask_cpu) in enumerate(flows, start=1):
        bw = bw_cpu.cuda()
        mask = mask_cpu.cuda()
        prev = torch.from_numpy(metric_depths[t - 1])[None, None].cuda()
        warped = _sample(prev, bw)[0, 0]
        cur = torch.from_numpy(metric_depths[t]).cuda().clamp(min=min_depth)
        vals.append(float(((warped - cur).abs() / cur)[mask].mean()))
    return np.array(vals, dtype=np.float32)


def identity_pose_tae(metric_depths: np.ndarray, min_depth: float = 0.4) -> float:
    """Exact bidirectional Temporal Alignment Error for stationary camera clips (R=I, T=0)."""
    errs = []
    z = np.clip(metric_depths.astype(np.float64), min_depth, 500.0)
    for t in range(len(z) - 1):
        e1 = np.mean(np.abs(z[t] - z[t + 1]) / z[t + 1])
        e2 = np.mean(np.abs(z[t + 1] - z[t]) / z[t])
        errs.append(0.5 * (e1 + e2))
    return float(np.mean(errs))


def second_order_jitter(metric_depths: np.ndarray, min_depth: float = 0.4) -> float:
    """Mean normalized second-order temporal acceleration |Z_{t+1} - 2Z_t + Z_{t-1}| / Z_t."""
    z = np.clip(metric_depths.astype(np.float64), min_depth, 500.0)
    if len(z) < 3:
        return 0.0
    acc = np.abs(z[2:] - 2.0 * z[1:-1] + z[:-2]) / z[1:-1]
    return float(np.mean(acc))


def median_scale_drift_cv(metric_depths: np.ndarray) -> float:
    """Coefficient of variation (std/mean) of per-frame median depth across the clip."""
    meds = np.median(metric_depths, axis=(1, 2)).astype(np.float64)
    return float(np.std(meds) / max(np.mean(meds), 1e-6))


def run_unidepth_full_and_kclip(frames: np.ndarray, kf_stride: int = 15) -> dict[str, object]:
    """Run UniDepthV2-Small frame-wise (raw RGB) and K_clip-conditioned on keyframes (and full clip for rollercoaster)."""
    h, w = frames.shape[1:3]
    model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vits14")
    model.interpolation_mode = "bilinear"
    model = model.to("cuda").eval()

    with torch.inference_mode():
        for f in frames[:4]:
            _ = model.infer(torch.from_numpy(f).permute(2, 0, 1))
        torch.cuda.reset_peak_memory_stats()
        zs, us, ks, times = [], [], [], []
        for f in frames:
            ten = torch.from_numpy(f).permute(2, 0, 1)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model.infer(ten)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            z = out["depth"].squeeze().float().cpu().numpy()
            u = out["confidence"].squeeze().float().cpu().numpy()
            if z.shape != (h, w):
                z = cv2.resize(z, (w, h), interpolation=cv2.INTER_LINEAR)
                u = cv2.resize(u, (w, h), interpolation=cv2.INTER_LINEAR)
            zs.append(z.astype(np.float32))
            us.append(u.astype(np.float32))
            ks.append(out["intrinsics"][0].float().cpu().numpy())

        peak_mib = torch.cuda.max_memory_allocated() / 2**20
        kf_ids = list(range(0, len(frames), kf_stride))
        k_stack = np.stack(ks, axis=0)
        K_clip = np.median(k_stack[kf_ids], axis=0).astype(np.float32)
        cam_clip = Pinhole(K=torch.from_numpy(K_clip).unsqueeze(0))

        # Re-infer ALL frames conditioned on K_clip cleanly (~1s for 70 frames) so we can compare frame-wise K_clip too
        z_kclip_all, u_kclip_all = [], []
        for f in frames:
            ten = torch.from_numpy(f).permute(2, 0, 1)
            out = model.infer(ten, cam_clip)
            z = out["depth"].squeeze().float().cpu().numpy()
            u = out["confidence"].squeeze().float().cpu().numpy()
            if z.shape != (h, w):
                z = cv2.resize(z, (w, h), interpolation=cv2.INTER_LINEAR)
                u = cv2.resize(u, (w, h), interpolation=cv2.INTER_LINEAR)
            z_kclip_all.append(z.astype(np.float32))
            u_kclip_all.append(u.astype(np.float32))

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "raw_z": np.stack(zs),
        "raw_u": np.stack(us),
        "k_pred": k_stack,
        "K_clip": K_clip,
        "clip_z": np.stack(z_kclip_all),
        "clip_u": np.stack(u_kclip_all),
        "kf_ids": kf_ids,
        "eff": {
            "model": "UniDepthV2-Small",
            "ms_per_frame": 1000.0 * float(np.mean(times[4:])),
            "peak_gpu_mib": float(peak_mib),
        },
    }


def apply_raft_ema_sequence(raw_depths: np.ndarray, flows) -> np.ndarray:
    """Sequential causal RAFT-EMA baseline using src/fusion/flow.py."""
    out = [raw_depths[0].astype(np.float32)]
    for t, (fw_cpu, bw_cpu, _) in enumerate(flows, start=1):
        fused = fuse_with_flow(out[-1], raw_depths[t], fw_cpu.cuda(), bw_cpu.cuda(), tolerance=1.0)
        out.append(fused)
    return np.stack(out, axis=0)


def write_video(path: Path, imgs: list[np.ndarray], fps: float) -> None:
    writer = imageio.get_writer(str(path), fps=fps, macro_block_size=1, codec="libx264", ffmpeg_params=["-crf", "18"])
    for img in imgs:
        writer.append_data(img)
    writer.close()


def find_best_halo_crop(
    naive_img: np.ndarray, occ_img: np.ndarray, depth_m: np.ndarray, focus_m: float, crop_w: int = 150, crop_h: int = 112
) -> tuple[int, int]:
    diff = np.mean(np.abs(occ_img.astype(np.float32) - naive_img.astype(np.float32)), axis=-1)
    is_bg = (depth_m > focus_m * 1.15).astype(np.float32)
    is_fg = (depth_m <= focus_m).astype(np.uint8)
    fg_dil = cv2.dilate(is_fg, np.ones((19, 19), np.uint8)) > 0
    halo_score = diff * is_bg * fg_dil.astype(np.float32)
    h, w = depth_m.shape
    best_val, best_yx = -1.0, (h // 3, w // 3)
    for y in range(16, h - crop_h - 16, 6):
        for x in range(16, w - crop_w - 16, 6):
            s = float(halo_score[y : y + crop_h, x : x + crop_w].mean())
            if s > best_val:
                best_val = s
                best_yx = (y, x)
    return best_yx


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.15, device=0)

    clip_defs = [
        ("davis_rollercoaster", CLIP_COASTER, True),
        ("still", CLIP_STILL, False),
        ("moving", CLIP_MOVING, False),
    ]
    clip_data = {}
    for name, path, has_sky in clip_defs:
        frames, fps = read_video_frames(str(path), MAX_LEN, -1, MAX_RES)
        # Ensure divisible-by-2 H,W for H264 compatibility
        h, w = frames.shape[1:3]
        if h % 2 or w % 2:
            frames = frames[:, : h - (h % 2), : w - (w % 2)]
        clip_data[name] = {"frames": frames, "fps": float(fps), "has_sky": has_sky}
        print(f"Loaded clip {name}: {frames.shape} @ {fps:.1f} fps")

    # Pass 1: RAFT optical flows (one clip at a time, lightweight CPU storage)
    for name in clip_data:
        clip_data[name]["flows"] = compute_flows_cpu(clip_data[name]["frames"], tolerance=1.0)
        print(f"  [RAFT] {name}: {len(clip_data[name]['flows'])} flow pairs")

    # Pass 2: Relative VDA-Small Streaming KV-cache on all 3 clips
    for name in clip_data:
        vda_str, eff_str = run_vda_streaming(clip_data[name]["frames"], VDA_CKPT, CFG, input_size=INPUT_SIZE, fp32=False)
        clip_data[name]["vda_str"] = vda_str
        clip_data[name]["eff_vda_str"] = eff_str
        print(f"  [VDA-S Streaming] {name}: {eff_str['ms_per_frame']:.1f} ms/f, {eff_str['peak_gpu_mib']:.1f} MiB")

    # Pass 3: Official Metric-VDA-Small Streaming KV-cache (CVPR 2025 metric release) on all 3 clips
    for name in clip_data:
        mvda_str, eff_mvda = run_vda_streaming(clip_data[name]["frames"], MVDA_CKPT, CFG, input_size=INPUT_SIZE, fp32=False)
        clip_data[name]["mvda_str"] = mvda_str
        clip_data[name]["eff_mvda_str"] = eff_mvda
        print(
            f"  [Metric-VDA-S Streaming] {name}: {eff_mvda['ms_per_frame']:.1f} ms/f, "
            f"Z p10/p50/p90={np.percentile(mvda_str, 10):.2f}/{np.percentile(mvda_str, 50):.2f}/{np.percentile(mvda_str, 90):.2f} m"
        )

    # Pass 4: UniDepthV2-Small raw + K_clip on all 3 clips
    for name in clip_data:
        clip_data[name]["uni"] = run_unidepth_full_and_kclip(clip_data[name]["frames"], kf_stride=15)
        k_fx = clip_data[name]["uni"]["k_pred"][:, 0, 0]
        print(
            f"  [UniDepthV2] {name}: raw_fx=[{k_fx.min():.1f}..{k_fx.max():.1f}], "
            f"K_clip_fx={clip_data[name]['uni']['K_clip'][0,0]:.1f}"
        )

    # Pass 5: Evaluate anchored & flow-EMA variants cleanly
    bench_rows = []
    # Reuse cached offline VDA-S & DAv2-S on davis_rollercoaster (zero 8GB VDA offline re-run!)
    base_npz = np.load(BASE_OUT / "relative_depth.npz")
    base_sum = json.loads((BASE_OUT / "summary.json").read_text())
    dav2_coaster = base_npz["dav2_small"].astype(np.float32)
    vda_off_coaster = base_npz["vda_small"].astype(np.float32)

    coaster = clip_data["davis_rollercoaster"]
    coaster_flows = coaster["flows"]
    rel_ranges = {
        "dav2_small": clip_range(dav2_coaster),
        "vda_streaming": clip_range(coaster["vda_str"]),
        "vda_offline": clip_range(vda_off_coaster),
    }
    rel_opw = {
        "dav2_small": rel_temporal_change(normalize(dav2_coaster, *rel_ranges["dav2_small"]).astype(np.float32), coaster_flows),
        "vda_streaming": rel_temporal_change(normalize(coaster["vda_str"], *rel_ranges["vda_streaming"]).astype(np.float32), coaster_flows),
        "vda_offline": rel_temporal_change(normalize(vda_off_coaster, *rel_ranges["vda_offline"]).astype(np.float32), coaster_flows),
    }

    # Ablate old uncalibrated keyframe anchoring vs fixed K_clip + zero-sky anchoring on rollercoaster
    kf_ids = coaster["uni"]["kf_ids"]
    z_off_old_raw, align_off_old_raw = fit_keyframe_metric_alignment(
        vda_off_coaster,
        {t: coaster["uni"]["raw_z"][t] for t in kf_ids},
        {t: coaster["uni"]["raw_u"][t] for t in kf_ids},
        zero_sky=False,
    )
    z_off_fixed, align_off_fixed = fit_keyframe_metric_alignment(
        vda_off_coaster,
        {t: coaster["uni"]["clip_z"][t] for t in kf_ids},
        {t: coaster["uni"]["clip_u"][t] for t in kf_ids},
        zero_sky=True,
    )
    z_str_fixed_off, align_str_fixed_off = fit_keyframe_metric_alignment(
        coaster["vda_str"],
        {t: coaster["uni"]["clip_z"][t] for t in kf_ids},
        {t: coaster["uni"]["clip_u"][t] for t in kf_ids},
        zero_sky=True,
    )
    z_str_causal, align_str_causal = causal_online_metric_anchoring(
        coaster["vda_str"],
        kf_ids,
        {t: coaster["uni"]["clip_z"][t] for t in kf_ids},
        {t: coaster["uni"]["clip_u"][t] for t in kf_ids},
        zero_sky=True,
    )

    for cname in ("davis_rollercoaster", "still", "moving"):
        cd = clip_data[cname]
        flows = cd["flows"]
        uni_raw_z = cd["uni"]["raw_z"]
        uni_clip_z = cd["uni"]["clip_z"]
        uni_raft_ema = apply_raft_ema_sequence(uni_raw_z, flows)
        mvda_str = cd["mvda_str"]
        kf = cd["uni"]["kf_ids"]
        has_sky = cd["has_sky"]

        if cname == "davis_rollercoaster":
            z_causal, align_c = z_str_causal, align_str_causal
            z_off_anch = z_off_fixed
        else:
            z_off_anch = None
            z_causal, align_c = causal_online_metric_anchoring(
                cd["vda_str"],
                kf,
                {t: cd["uni"]["clip_z"][t] for t in kf},
                {t: cd["uni"]["clip_u"][t] for t in kf},
                zero_sky=has_sky,
            )

        cd["uni_raft_ema"] = uni_raft_ema
        cd["z_causal"] = z_causal
        cd["align_causal"] = align_c

        models_eval = [
            ("UniDepthV2-S (frame raw)", uni_raw_z),
            ("UniDepthV2-S (shared K_clip)", uni_clip_z),
            ("UniDepthV2-S + RAFT-EMA", uni_raft_ema),
            ("Metric-VDA-S (streaming)", mvda_str),
            ("VDA-S Stream + K_clip Causal", z_causal),
        ]
        if z_off_anch is not None:
            models_eval.append(("VDA-S Offline + K_clip Anchor", z_off_anch))

        for mname, z_seq in models_eval:
            opw_seq = metric_opw_absrel(z_seq, flows)
            row = {
                "clip": cname,
                "method": mname,
                "opw_absrel": float(opw_seq.mean()),
                "tae_identity": identity_pose_tae(z_seq) if cname == "still" else float("nan"),
                "jitter_2nd": second_order_jitter(z_seq),
                "scale_drift_cv": median_scale_drift_cv(z_seq),
                "z_p10_m": float(np.percentile(z_seq, 10)),
                "z_p50_m": float(np.percentile(z_seq, 50)),
                "z_p98_m": float(np.percentile(z_seq, 98)),
            }
            bench_rows.append(row)
            print(
                f"  [{cname:18s}] {mname:30s} OPW={row['opw_absrel']:.4f} "
                f"jitter2={row['jitter_2nd']:.4f} driftCV={row['scale_drift_cv']:.4f} "
                f"Z=[{row['z_p10_m']:.1f}, {row['z_p50_m']:.1f}, {row['z_p98_m']:.1f}]"
            )

    with (OUT / "multi_clip_benchmark.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(bench_rows[0].keys()))
        w.writeheader()
        w.writerows(bench_rows)

    # Compute rollercoaster bokeh, CoC range, & quantitative halo bleeding metrics
    frames_rc = coaster["frames"]
    non_sky_35 = vda_off_coaster[35] > 0.05 * np.percentile(vda_off_coaster, 98)
    focus_rc = float(np.percentile(z_off_fixed[35][non_sky_35], 50))
    coc_sky_old = float(np.percentile(circle_of_confusion_px(z_off_old_raw[35], float(np.percentile(z_off_old_raw[35][non_sky_35], 55)), 9.0, True), 95))
    coc_sky_fixed = float(np.percentile(circle_of_confusion_px(z_off_fixed[35], focus_rc, 9.0, True), 95))
    print(f"\nCoC P95 realization: old(b>0)={coc_sky_old:.2f} px vs fixed(K_clip+zero_sky)={coc_sky_fixed:.2f} px (focus={focus_rc:.1f} m)")

    halo_rows = []
    for t_h in (0, 15, 35, 50, 69):
        b_n = naive_metric_bokeh(frames_rc[t_h], z_off_fixed[t_h], focus_dist_m=focus_rc, levels=6, max_sigma=9.0)
        b_o = occlusion_aware_bokeh(frames_rc[t_h], z_off_fixed[t_h], focus_dist_m=focus_rc, levels=6, max_sigma=9.0)
        hm = compute_halo_bleeding_metrics(frames_rc[t_h], b_n, b_o, z_off_fixed[t_h], focus_dist_m=focus_rc, band_px=9)
        hm["frame"] = t_h
        halo_rows.append(hm)
    coaster_halo_mean = {
        k: float(np.nanmean([r[k] for r in halo_rows]))
        for k in halo_rows[0]
        if k != "frame"
    }
    print("Rollercoaster Bokeh Halo Metrics (mean 5 frames):", json.dumps(coaster_halo_mean, indent=2))

    # Write rollercoaster comparison video with full-CoC fixed depth
    z_hi_vis = float(np.percentile(z_off_fixed, 95))
    bokeh_frames = []
    t_nai, t_occ = [], []
    b35_nai_best, b35_occ_best = None, None
    for t, rgb in enumerate(frames_rc):
        t0 = time.perf_counter()
        bn = naive_metric_bokeh(rgb, z_off_fixed[t], focus_dist_m=focus_rc, levels=6, max_sigma=9.0)
        t1 = time.perf_counter()
        bo = occlusion_aware_bokeh(rgb, z_off_fixed[t], focus_dist_m=focus_rc, levels=6, max_sigma=9.0)
        t2 = time.perf_counter()
        t_nai.append((t1 - t0) * 1000.0)
        t_occ.append((t2 - t1) * 1000.0)
        if t == 35:
            b35_nai_best, b35_occ_best = bn, bo
        z_vis = MAGMA_R[(np.clip(z_off_fixed[t] / max(z_hi_vis, 1.0), 0.0, 1.0) * 255).astype(np.uint8)]
        panel = np.concatenate(
            [
                label(rgb, f"RGB t={t} (Z_focus={focus_rc:.0f}m)"),
                label(z_vis, "Anchored Metric Depth (K_clip, b=0)"),
                label(bn, "Naive Gaussian Bokeh (Halos)"),
                label(bo, "Occlusion-Aware Thin-Lens Bokeh"),
            ],
            axis=1,
        )
        bokeh_frames.append(cv2.resize(panel, (1920, 360), interpolation=cv2.INTER_AREA))
    write_video(OUT / "rollercoaster_physical_bokeh_compare.mp4", bokeh_frames, coaster["fps"])

    cy, cx = find_best_halo_crop(b35_nai_best, b35_occ_best, z_off_fixed[35], focus_rc)

    # Indoor bokeh test + halo metric on moving.mp4 (frame 50)
    mov = clip_data["moving"]
    t_ind = min(50, len(mov["frames"]) - 1)
    z_mov = mov["z_causal"]
    focus_mov = float(np.percentile(z_mov[t_ind], 35))
    bn_mov = naive_metric_bokeh(mov["frames"][t_ind], z_mov[t_ind], focus_dist_m=focus_mov, levels=6, max_sigma=9.0)
    bo_mov = occlusion_aware_bokeh(mov["frames"][t_ind], z_mov[t_ind], focus_dist_m=focus_mov, levels=6, max_sigma=9.0)
    mov_halo = compute_halo_bleeding_metrics(mov["frames"][t_ind], bn_mov, bo_mov, z_mov[t_ind], focus_dist_m=focus_mov, band_px=9)
    z_mov_vis = MAGMA_R[(np.clip(z_mov[t_ind] / max(float(np.percentile(z_mov[t_ind], 98)), 1.0), 0.0, 1.0) * 255).astype(np.uint8)]
    indoor_panel = np.concatenate(
        [
            label(mov["frames"][t_ind], f"Moving Clip t={t_ind} (Z_focus={focus_mov:.2f}m)"),
            label(z_mov_vis, "Causal Stream Anchored Z"),
            label(bn_mov, "Naive Blur (Foreground Bleed)"),
            label(bo_mov, "Occlusion-Aware Thin-Lens"),
        ],
        axis=1,
    )
    cv2.imwrite(str(OUT / "indoor_bokeh_compare.png"), cv2.cvtColor(indoor_panel, cv2.COLOR_RGB2BGR))

    np.savez_compressed(
        OUT / "extended_video_depth.npz",
        vda_streaming=coaster["vda_str"].astype(np.float16),
        mvda_streaming=coaster["mvda_str"].astype(np.float16),
        metric_video_offline=z_off_fixed.astype(np.float16),
        metric_video_streaming=z_str_fixed_off.astype(np.float16),
        metric_video_causal=z_str_causal.astype(np.float16),
    )

    k_fx_rc = coaster["uni"]["k_pred"][coaster["uni"]["kf_ids"], 0, 0]
    ext_summary = {
        "clip": "davis_rollercoaster.mp4",
        "num_frames": int(len(frames_rc)),
        "fps": coaster["fps"],
        "streaming_efficiency": coaster["eff_vda_str"],
        "metric_vda_streaming_efficiency": coaster["eff_mvda_str"],
        "offline_efficiency": base_sum["efficiency"][1],
        "dav2_efficiency_base_fp32": base_sum["efficiency"][0],
        "unidepth_frame_efficiency": coaster["uni"]["eff"],
        "keyframe_amortized_ms_per_frame": float(coaster["uni"]["eff"]["ms_per_frame"] / 15.0),
        "naive_bokeh_ms_per_frame": float(np.mean(t_nai)),
        "occlusion_aware_bokeh_ms_per_frame": float(np.mean(t_occ)),
        "relative_flow_warped_change": {k: float(v.mean()) for k, v in rel_opw.items()},
        "relative_flow_warped_series": {k: [float(x) for x in v] for k, v in rel_opw.items()},
        "keyframe_fx_raw_min": float(k_fx_rc.min()),
        "keyframe_fx_raw_max": float(k_fx_rc.max()),
        "K_clip_fx": float(coaster["uni"]["K_clip"][0, 0]),
        "alignment_old_raw": align_off_old_raw,
        "alignment_offline_fixed": align_off_fixed,
        "alignment_streaming_fixed": align_str_fixed_off,
        "alignment_causal": {k: v for k, v in align_str_causal.items() if k not in ("a_trace", "b_trace")},
        "focus_dist_m": focus_rc,
        "coc_p95_old_raw_px": coc_sky_old,
        "coc_p95_fixed_px": coc_sky_fixed,
        "coaster_halo_metrics_mean": coaster_halo_mean,
        "indoor_moving_halo_metrics": mov_halo,
        "zoom_crop_yxwh": [int(cy), int(cx), 150, 112],
    }
    (OUT / "extended_summary.json").write_text(json.dumps(ext_summary, indent=2))
    print("\nSaved all multi-clip & fixed metric video results to", OUT)


if __name__ == "__main__":
    main()
