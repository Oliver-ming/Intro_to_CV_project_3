"""Run the video pipeline on phone clips in data/clips.

The directory is empty until the two indoor recordings are added. Each clip
should be about 15 seconds. One keeps the camera still. One moves it. A
standing person with head and both ankles visible lets the clip scale vote.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "UniDepth"))

from unidepth.models import UniDepthV2

from src.fusion.flow import flow_temporal_absrel, fuse_with_flow, load_raft, pairwise_flow

CLIPS = ROOT / "data" / "clips"
OUT = ROOT / "outputs" / "clips"


def read_frames(path: Path, limit: int = 450) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) < limit:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    return frames


def run_clip(path: Path, model, raft, transforms, device: str) -> None:
    frames = read_frames(path)
    if len(frames) < 2:
        print(f"skip {path.name}: fewer than 2 frames")
        return
    fused = None
    maps = []
    errors = []
    with torch.inference_mode():
        for frame in frames:
            tensor = torch.from_numpy(frame).permute(2, 0, 1)
            current = model.infer(tensor)["depth"].squeeze().detach().cpu().numpy()
            if fused is None:
                fused = current
            else:
                forward, backward = pairwise_flow(raft, transforms, frames[len(maps) - 1], frame, device)
                errors.append(flow_temporal_absrel(fused, current, backward))
                fused = fuse_with_flow(fused, current, forward, backward)
            maps.append(fused.astype(np.float32))
    scene = OUT / path.stem
    scene.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(scene / "depth.npz", depth=np.stack(maps))
    finite = np.concatenate([item[item > 0] for item in maps])
    vmax = float(np.percentile(finite, 98)) if finite.size else 1.0
    writer = cv2.VideoWriter(
        str(scene / "depth.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        15,
        (maps[0].shape[1], maps[0].shape[0]),
    )
    for depth in maps:
        shown = np.clip(depth / max(vmax, 1e-3), 0, 1)
        writer.write(cv2.applyColorMap((shown * 255).astype(np.uint8), cv2.COLORMAP_MAGMA))
    writer.release()
    rel = float(np.mean(errors)) if errors else float("nan")
    print(f"{path.name}: frames={len(maps)}  flow_temp_absrel={rel:.4f}  range={vmax:.2f}m")


def main() -> None:
    videos = sorted(CLIPS.glob("*.mp4")) + sorted(CLIPS.glob("*.mov"))
    if not videos:
        print(f"No phone clips in {CLIPS}. Add still.mp4 and moving.mp4, then rerun.")
        return
    device = "cuda"
    model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vits14")
    model.interpolation_mode = "bilinear"
    model = model.to(device).eval()
    raft, transforms = load_raft(device)
    OUT.mkdir(parents=True, exist_ok=True)
    for path in videos:
        run_clip(path, model, raft, transforms, device)


if __name__ == "__main__":
    main()
