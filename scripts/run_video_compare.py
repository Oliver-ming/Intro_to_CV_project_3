"""Part 1/2 video comparison on davis_rollercoaster.mp4 (first 100 frames).

Frame-wise Depth Anything V2-Small vs Video Depth Anything-Small (offline,
FP16), same frames, same palette, one fixed normalization per method over the
whole clip. Both outputs are relative depth (inverse-depth-like disparity);
colors are not a shared metric scale.

The clip has 70 frames, so "first 100 frames" is the whole clip.

Also writes the application output: depth-aware background blur driven by
each method's depth, and an optional flow-warped temporal-change diagnostic.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import imageio
import matplotlib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
VDA = ROOT / "third_party" / "Video-Depth-Anything"
DAV2 = ROOT / "third_party" / "Depth-Anything-V2"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(VDA))
sys.path.insert(0, str(DAV2))

from depth_anything_v2.dpt import DepthAnythingV2
from utils.dc_utils import read_video_frames
from video_depth_anything.video_depth import VideoDepthAnything

from src.fusion.flow import load_raft, pairwise_flow, _sample

CLIP = VDA / "assets" / "example_videos" / "davis_rollercoaster.mp4"
OUT = ROOT / "outputs" / "video"
MAX_LEN, MAX_RES, INPUT_SIZE = 100, 640, 518
CFG = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}


def run_dav2(frames: np.ndarray) -> tuple[np.ndarray, dict]:
    model = DepthAnythingV2(**CFG)
    model.load_state_dict(torch.load(DAV2 / "checkpoints" / "depth_anything_v2_vits.pth", map_location="cpu"))
    model = model.to("cuda").eval()
    torch.cuda.reset_peak_memory_stats()
    out, times = [], []
    with torch.inference_mode():
        for rgb in frames:
            torch.cuda.synchronize(); start = time.perf_counter()
            out.append(model.infer_image(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), INPUT_SIZE))
            torch.cuda.synchronize(); times.append(time.perf_counter() - start)
    eff = {"model": "Depth Anything V2-Small", "precision": "fp32", "input_size": INPUT_SIZE,
           "ms_per_frame": 1000 * float(np.mean(times[5:])), "peak_gpu_mib": torch.cuda.max_memory_allocated() / 2**20}
    del model; torch.cuda.empty_cache()
    return np.stack(out).astype(np.float32), eff


def run_vda(frames: np.ndarray, fps: float) -> tuple[np.ndarray, dict]:
    model = VideoDepthAnything(**CFG, metric=False)
    model.load_state_dict(torch.load(VDA / "checkpoints" / "video_depth_anything_vits.pth", map_location="cpu"), strict=True)
    model = model.to("cuda").eval()
    model.infer_video_depth(frames[:40], fps, input_size=INPUT_SIZE, device="cuda", fp32=False)  # warm-up
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize(); start = time.perf_counter()
    depths, _ = model.infer_video_depth(frames, fps, input_size=INPUT_SIZE, device="cuda", fp32=False)
    torch.cuda.synchronize(); total = time.perf_counter() - start
    eff = {"model": "Video Depth Anything-Small (offline)", "precision": "fp16 (repo default)", "input_size": INPUT_SIZE,
           "ms_per_frame": 1000 * total / len(frames), "peak_gpu_mib": torch.cuda.max_memory_allocated() / 2**20}
    del model; torch.cuda.empty_cache()
    return np.asarray(depths, dtype=np.float32), eff


def clip_range(depths: np.ndarray) -> tuple[float, float]:
    """One normalization for the whole clip (2nd-98th percentile)."""
    return float(np.percentile(depths, 2)), float(np.percentile(depths, 98))


PALETTE = (np.array(matplotlib.colormaps["inferno"].colors) * 255).astype(np.uint8)


def normalize(depth: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)


def colorize(depth: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return PALETTE[(normalize(depth, lo, hi) * 255).astype(np.uint8)]


def label(image: np.ndarray, text: str) -> np.ndarray:
    image = image.copy()
    cv2.rectangle(image, (0, 0), (image.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(image, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def depth_blur(rgb: np.ndarray, disp_norm: np.ndarray, focus: float, levels: int = 6, max_sigma: float = 9.0) -> np.ndarray:
    """Portrait-style blur: pixels farther than the focus plane get blurrier.

    disp_norm is clip-normalized relative disparity (1 = near). Pixels with
    disp_norm >= focus stay sharp; blur grows linearly as disparity drops.
    """
    amount = np.clip((focus - disp_norm) / max(focus, 1e-6), 0, 1) * (levels - 1)
    stack = [rgb.astype(np.float32)] + [cv2.GaussianBlur(rgb, (0, 0), max_sigma * k / (levels - 1)).astype(np.float32) for k in range(1, levels)]
    low = np.floor(amount).astype(int)
    high = np.minimum(low + 1, levels - 1)
    frac = (amount - low)[..., None]
    stack = np.stack(stack)
    rows, cols = np.indices(low.shape)
    out = (1 - frac) * stack[low, rows, cols] + frac * stack[high, rows, cols]
    return out.clip(0, 255).astype(np.uint8)


def flow_masks(frames: np.ndarray, tolerance: float = 1.0) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Backward flow t->t-1 and a forward-backward consistency mask for each t>=1."""
    raft, transforms = load_raft("cuda")
    out = []
    for t in range(1, len(frames)):
        forward, backward = pairwise_flow(raft, transforms, frames[t - 1], frames[t], "cuda")
        residual = torch.linalg.norm(_sample(forward, backward) + backward, dim=1)[0]
        out.append((backward, residual <= tolerance))
    del raft; torch.cuda.empty_cache()
    return out


def temporal_change(norm_depths: np.ndarray, flows) -> np.ndarray:
    """Optional diagnostic: mean |D_t - warp(D_{t-1})| in clip-normalized units, flow-consistent pixels only."""
    values = []
    for t, (backward, mask) in enumerate(flows, start=1):
        prev = torch.from_numpy(norm_depths[t - 1])[None, None].cuda()
        warped = _sample(prev, backward)[0, 0]
        cur = torch.from_numpy(norm_depths[t]).cuda()
        values.append(float((warped - cur).abs()[mask].mean()))
    return np.array(values)


def write_video(path: Path, images: list[np.ndarray], fps: float) -> None:
    writer = imageio.get_writer(str(path), fps=fps, macro_block_size=1, codec="libx264", ffmpeg_params=["-crf", "18"])
    for image in images:
        writer.append_data(image)
    writer.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames, fps = read_video_frames(str(CLIP), MAX_LEN, -1, MAX_RES)
    print(f"{CLIP.name}: {len(frames)} frames at {fps:.1f} fps, {frames.shape[2]}x{frames.shape[1]}")
    dav2, eff_dav2 = run_dav2(frames)
    vda, eff_vda = run_vda(frames, fps)
    np.savez_compressed(OUT / "relative_depth.npz", dav2_small=dav2, vda_small=vda)
    ranges = {"dav2_small": clip_range(dav2), "vda_small": clip_range(vda)}
    norm = {k: normalize(d, *ranges[k]).astype(np.float32) for k, d in (("dav2_small", dav2), ("vda_small", vda))}
    flows = flow_masks(frames)
    change = {k: temporal_change(v, flows) for k, v in norm.items()}
    focus = 0.55  # normalized disparity of the focus plane, same for both methods
    side, blur = [], []
    for t, rgb in enumerate(frames):
        a = label(colorize(dav2[t], *ranges["dav2_small"]), f"DAv2-Small frame-wise (relative)  t={t}")
        b = label(colorize(vda[t], *ranges["vda_small"]), "VDA-Small video (relative)")
        side.append(np.concatenate([label(rgb, "RGB"), a, b], axis=1))
        blur.append(np.concatenate([
            label(depth_blur(rgb, norm["dav2_small"][t], focus), "Blur from DAv2-Small"),
            label(depth_blur(rgb, norm["vda_small"][t], focus), "Blur from VDA-Small")], axis=1))
    write_video(OUT / "rollercoaster_rgb_dav2_vda.mp4", side, fps)
    write_video(OUT / "rollercoaster_depth_blur.mp4", blur, fps)
    for t in (0, len(frames) // 2, len(frames) - 1):
        cv2.imwrite(str(OUT / f"frame_{t:03d}_compare.png"), cv2.cvtColor(side[t], cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(OUT / f"frame_{t:03d}_blur.png"), cv2.cvtColor(blur[t], cv2.COLOR_RGB2BGR))
    np.savetxt(OUT / "temporal_change.csv", np.stack([np.arange(1, len(frames)), change["dav2_small"], change["vda_small"]], 1),
               delimiter=",", header="t,dav2_small,vda_small", comments="", fmt=["%d", "%.5f", "%.5f"])
    summary = {"clip": CLIP.name, "frames": len(frames), "fps": fps, "resolution": f"{frames.shape[2]}x{frames.shape[1]}",
               "max_res": MAX_RES, "ranges": ranges, "gpu": torch.cuda.get_device_name(0),
               "efficiency": [eff_dav2, eff_vda],
               "mean_temporal_change": {k: float(v.mean()) for k, v in change.items()}}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
