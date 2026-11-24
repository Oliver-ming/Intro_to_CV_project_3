"""Streaming KV-cache video depth and sparse-keyframe metric depth anchoring."""

from __future__ import annotations

import gc
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.linear_model import HuberRegressor


def run_vda_streaming(
    frames: np.ndarray,
    vda_ckpt: Path,
    cfg: dict,
    input_size: int = 518,
    fp32: bool = False,
) -> tuple[np.ndarray, dict]:
    """Run Video Depth Anything-Small (or Metric-VDA-Small) in causal KV-cache streaming mode."""
    from video_depth_anything.video_depth_stream import VideoDepthAnything as VDAStream

    warm = VDAStream(**cfg)
    warm.load_state_dict(torch.load(vda_ckpt, map_location="cpu"), strict=True)
    warm = warm.to("cuda").eval()
    with torch.inference_mode():
        for f in frames[:8]:
            _ = warm.infer_video_depth_one(f, input_size=input_size, device="cuda", fp32=fp32)
    del warm
    gc.collect()
    torch.cuda.empty_cache()

    model = VDAStream(**cfg)
    model.load_state_dict(torch.load(vda_ckpt, map_location="cpu"), strict=True)
    model = model.to("cuda").eval()

    torch.cuda.reset_peak_memory_stats()
    depths = []
    times = []
    with torch.inference_mode():
        for rgb in frames:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            d = model.infer_video_depth_one(rgb, input_size=input_size, device="cuda", fp32=fp32)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            depths.append(d.astype(np.float32))

    peak_mib = torch.cuda.max_memory_allocated() / 2**20
    del model
    gc.collect()
    torch.cuda.empty_cache()
    eff = {
        "ckpt": vda_ckpt.name,
        "mode": "streaming_kv_cache",
        "precision": "fp32" if fp32 else "fp16",
        "input_size": int(input_size),
        "ms_per_frame": 1000.0 * float(np.mean(times[5:])),
        "first_frame_ms": 1000.0 * float(times[0]),
        "peak_gpu_mib": float(peak_mib),
    }
    return np.stack(depths, axis=0), eff


def infer_keyframes_shared_camera(
    frames: np.ndarray,
    keyframe_indices: list[int],
) -> dict[str, object]:
    """Run UniDepthV2-Small on sparse keyframes twice:
    1) Raw RGB-only -> reveals per-keyframe focal-length drift and yields K_clip = median(K_pred).
    2) Re-conditioned on single frozen clip camera K_clip -> removes focal drift across keyframes.
    """
    from unidepth.models import UniDepthV2
    from unidepth.utils.camera import Pinhole

    h, w = frames.shape[1:3]
    model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vits14")
    model.interpolation_mode = "bilinear"
    model = model.to("cuda").eval()

    with torch.inference_mode():
        for f in frames[:3]:
            _ = model.infer(torch.from_numpy(f).permute(2, 0, 1))

        torch.cuda.reset_peak_memory_stats()
        raw_z: dict[int, np.ndarray] = {}
        raw_u: dict[int, np.ndarray] = {}
        k_preds = []
        times_kf = []
        for idx in keyframe_indices:
            ten = torch.from_numpy(frames[idx]).permute(2, 0, 1)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model.infer(ten)
            torch.cuda.synchronize()
            times_kf.append(time.perf_counter() - t0)
            z = out["depth"].squeeze().float().cpu().numpy()
            u = out["confidence"].squeeze().float().cpu().numpy()
            if z.shape != (h, w):
                z = cv2.resize(z, (w, h), interpolation=cv2.INTER_LINEAR)
                u = cv2.resize(u, (w, h), interpolation=cv2.INTER_LINEAR)
            raw_z[idx] = z.astype(np.float32)
            raw_u[idx] = u.astype(np.float32)
            k_preds.append(out["intrinsics"][0].float().cpu().numpy())

        K_stack = np.stack(k_preds, axis=0)
        K_clip = np.median(K_stack, axis=0).astype(np.float32)
        cam_clip = Pinhole(K=torch.from_numpy(K_clip).unsqueeze(0))

        clip_z: dict[int, np.ndarray] = {}
        clip_u: dict[int, np.ndarray] = {}
        for idx in keyframe_indices:
            ten = torch.from_numpy(frames[idx]).permute(2, 0, 1)
            out = model.infer(ten, cam_clip)
            z = out["depth"].squeeze().float().cpu().numpy()
            u = out["confidence"].squeeze().float().cpu().numpy()
            if z.shape != (h, w):
                z = cv2.resize(z, (w, h), interpolation=cv2.INTER_LINEAR)
                u = cv2.resize(u, (w, h), interpolation=cv2.INTER_LINEAR)
            clip_z[idx] = z.astype(np.float32)
            clip_u[idx] = u.astype(np.float32)

    peak_mib = torch.cuda.max_memory_allocated() / 2**20
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "K_keyframes": K_stack,
        "K_clip": K_clip,
        "raw_z_kf": raw_z,
        "raw_u_kf": raw_u,
        "clip_z_kf": clip_z,
        "clip_u_kf": clip_u,
        "ms_per_keyframe": 1000.0 * float(np.mean(times_kf)),
        "peak_gpu_mib": float(peak_mib),
    }


def _collect_foreground_samples(
    vda_disp: np.ndarray,
    uni_depths: dict[int, np.ndarray],
    uni_confs: dict[int, np.ndarray],
    disp_p98: float,
    max_samples_per_kf: int = 8000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Extract confident foreground disparity and inverse-depth samples across keyframes."""
    xs, ys, z_list = [], [], []
    sky_fracs = []
    fg_disp_min = max(0.05 * disp_p98, 1e-4)
    for t_idx, z_map in uni_depths.items():
        d_map = vda_disp[t_idx]
        u_map = uni_confs[t_idx]
        sky_fracs.append(float(np.mean(d_map <= 0.02 * max(disp_p98, 1e-4))))
        fg = (d_map > fg_disp_min) & (z_map > 0.25) & (z_map < 350.0) & np.isfinite(u_map)
        if np.sum(fg) < 100:
            fg = (z_map > 0.25) & (z_map < 350.0) & np.isfinite(u_map)
        u_cut = float(np.percentile(u_map[fg], 70))
        valid = fg & (u_map <= u_cut)
        x = d_map[valid]
        y = 1.0 / np.clip(z_map[valid], 0.25, 350.0)
        z_fg = z_map[valid]
        if len(x) > max_samples_per_kf:
            step = max(len(x) // max_samples_per_kf, 1)
            x, y, z_fg = x[::step], y[::step], z_fg[::step]
        xs.append(x)
        ys.append(y)
        z_list.append(z_fg)
    return (
        np.concatenate(xs),
        np.concatenate(ys),
        np.concatenate(z_list),
        float(np.mean(sky_fracs)),
    )


def _fit_huber_zero_intercept(x: np.ndarray, y: np.ndarray, iters: int = 12) -> float:
    """Huber-IRLS proportionality fit y ≈ a * x enforcing physical b = 0 (zero-disp sky -> inf)."""
    pos = (x > 1e-5) & (y > 1e-5)
    xp, yp = x[pos].astype(np.float64), y[pos].astype(np.float64)
    a = float(np.median(yp / xp))
    a = max(a, 1e-6)
    for _ in range(iters):
        res = yp - a * xp
        mad = float(np.median(np.abs(res - np.median(res)))) * 1.4826
        delta = max(1.345 * mad, 1e-6)
        w = np.where(np.abs(res) <= delta, 1.0, delta / np.maximum(np.abs(res), 1e-12))
        denom = float(np.sum(w * xp * xp))
        if denom > 1e-12:
            a = max(float(np.sum(w * xp * yp) / denom), 1e-6)
    return a


def fit_keyframe_metric_alignment(
    vda_disp: np.ndarray,
    uni_depths: dict[int, np.ndarray],
    uni_confs: dict[int, np.ndarray],
    zero_sky: bool | None = None,
    min_z_clip: float = 0.4,
    sky_z_clip: float = 500.0,
    disp_p98: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Fit global inverse-depth alignment 1/Z = a * d_VDA + b on sparse keyframes.

    When zero-disparity sky pixels are detected (zero_sky=True or sky_frac > 5%),
    enforces the physical zero-sky constraint b = 0 so zero disparity maps to infinity
    and thin-lens CoC spans its full physical range [0, R_inf].
    """
    if disp_p98 is None:
        disp_p98 = float(np.percentile(vda_disp, 98))
    x_all, y_all, fg_z_cat, sky_frac = _collect_foreground_samples(
        vda_disp, uni_depths, uni_confs, disp_p98=disp_p98
    )
    use_zero_sky = (sky_frac > 0.05) if zero_sky is None else bool(zero_sky)

    corr_disp_invz = float(np.corrcoef(x_all, y_all)[0, 1]) if len(x_all) > 2 else 0.0
    if use_zero_sky:
        a = _fit_huber_zero_intercept(x_all, y_all)
        b = 0.0
        z_max_clip = sky_z_clip
    else:
        reg = HuberRegressor(max_iter=300, epsilon=1.35, alpha=1e-4)
        reg.fit(x_all.reshape(-1, 1), y_all)
        a = float(reg.coef_[0])
        b = float(reg.intercept_)
        if a <= 1e-6 or b < -0.5 * a * float(np.percentile(x_all, 10)):
            a = _fit_huber_zero_intercept(x_all, y_all)
            b = 0.0
        elif b < 0.0:
            # Clamp non-physical negative intercept cleanly via zero-intercept refit
            a = _fit_huber_zero_intercept(x_all, y_all)
            b = 0.0
        z_max_clip = max(25.0, float(np.percentile(fg_z_cat, 99)) * 2.5)

    inv_z = np.clip(a * vda_disp.astype(np.float32) + b, 1.0 / z_max_clip, 1.0 / min_z_clip)
    z_vid = (1.0 / inv_z).astype(np.float32)
    return z_vid, {
        "scale_a": float(a),
        "shift_b": float(b),
        "zero_sky_enforced": bool(use_zero_sky),
        "sky_frac": float(sky_frac),
        "corr_disp_invz": corr_disp_invz,
        "num_keyframes": len(uni_depths),
        "num_anchor_samples": int(len(y_all)),
        "fg_median_z_m": float(np.median(fg_z_cat)),
        "z_min_clip_m": float(min_z_clip),
        "z_max_clip_m": float(z_max_clip),
    }


def causal_online_metric_anchoring(
    vda_disp: np.ndarray,
    keyframe_ids: list[int],
    uni_depths: dict[int, np.ndarray],
    uni_confs: dict[int, np.ndarray],
    zero_sky: bool | None = None,
    ema_alpha: float = 0.65,
    min_z_clip: float = 0.4,
    sky_z_clip: float = 500.0,
) -> tuple[np.ndarray, dict]:
    """Causal online metric anchoring for KV-cache streaming: updates (a_t, b_t)
    strictly using past/current keyframes t_k <= t with zero future lookahead.
    """
    kf_set = set(keyframe_ids)
    t_len, h, w = vda_disp.shape
    z_vid = np.zeros((t_len, h, w), dtype=np.float32)

    cur_a, cur_b, cur_z_max = 1.0, 0.0, sky_z_clip
    past_kf_z: dict[int, np.ndarray] = {}
    past_kf_u: dict[int, np.ndarray] = {}
    running_p98 = float(np.percentile(vda_disp[0], 98))
    a_trace, b_trace = [], []

    for t in range(t_len):
        p98_t = float(np.percentile(vda_disp[t], 98))
        running_p98 = 0.85 * running_p98 + 0.15 * p98_t
        if t in kf_set:
            past_kf_z[t] = uni_depths[t]
            past_kf_u[t] = uni_confs[t]
            _, fit_info = fit_keyframe_metric_alignment(
                vda_disp[: t + 1],
                past_kf_z,
                past_kf_u,
                zero_sky=zero_sky,
                min_z_clip=min_z_clip,
                sky_z_clip=sky_z_clip,
                disp_p98=running_p98,
            )
            new_a = fit_info["scale_a"]
            new_b = fit_info["shift_b"]
            cur_z_max = fit_info["z_max_clip_m"]
            if len(past_kf_z) == 1:
                cur_a, cur_b = new_a, new_b
            else:
                cur_a = ema_alpha * new_a + (1.0 - ema_alpha) * cur_a
                cur_b = ema_alpha * new_b + (1.0 - ema_alpha) * cur_b

        inv_z = np.clip(cur_a * vda_disp[t].astype(np.float32) + cur_b, 1.0 / cur_z_max, 1.0 / min_z_clip)
        z_vid[t] = (1.0 / inv_z).astype(np.float32)
        a_trace.append(float(cur_a))
        b_trace.append(float(cur_b))

    return z_vid, {
        "final_a": float(cur_a),
        "final_b": float(cur_b),
        "z_max_clip_m": float(cur_z_max),
        "num_keyframes": len(past_kf_z),
        "a_trace": a_trace,
        "b_trace": b_trace,
    }
