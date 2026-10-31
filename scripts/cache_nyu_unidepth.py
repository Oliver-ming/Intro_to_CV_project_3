"""Run UniDepthV2-Small on NYU train and test once and cache raw metric depth.

--camera none : RGB-only, model.infer(rgb), as in the course brief (main).
--camera nyu  : Kinect intrinsics passed at inference (ablation).

Writes outputs/nyu_cache/unidepth_<camera>_<split>.npy (float16, N x 480 x 640)
and an efficiency line measured on the test split.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "UniDepth"))

from unidepth.models import UniDepthV2
from unidepth.utils.camera import Pinhole

from src.eval.nyu import HEIGHT, K_NYU, WIDTH, open_mat, rgb_at, split_ids

OUT = ROOT / "outputs" / "nyu_cache"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", choices=["none", "nyu"], default="none")
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vits14")
    model.interpolation_mode = "bilinear"
    model = model.to("cuda").eval()
    camera = Pinhole(K=torch.from_numpy(K_NYU).unsqueeze(0)) if args.camera == "nyu" else None

    with open_mat() as data, torch.inference_mode():
        for split in args.splits:
            ids = split_ids(split)
            preds = np.zeros((len(ids), HEIGHT, WIDTH), dtype=np.float16)
            torch.cuda.reset_peak_memory_stats()
            times = []
            for step, index in enumerate(ids):
                rgb = torch.from_numpy(rgb_at(data, index)).permute(2, 0, 1)
                torch.cuda.synchronize()
                start = time.perf_counter()
                out = model.infer(rgb, camera) if camera is not None else model.infer(rgb)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - start)
                pred = out["depth"].squeeze().float().cpu().numpy()
                if pred.shape != (HEIGHT, WIDTH):
                    pred = cv2.resize(pred, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
                preds[step] = pred
                if (step + 1) % 200 == 0:
                    print(f"{split} {step + 1}/{len(ids)}")
            np.save(OUT / f"unidepth_{args.camera}_{split}.npy", preds)
            eff = {
                "model": "UniDepthV2-Small (vits14)",
                "camera": args.camera,
                "split": split,
                "input": f"{HEIGHT}x{WIDTH} RGB (model resizes internally)",
                "precision": "fp32 weights, model default autocast",
                "ms_per_image": 1000 * float(np.mean(times[10:])),
                "peak_gpu_mib": torch.cuda.max_memory_allocated() / 2**20,
                "gpu": torch.cuda.get_device_name(0),
            }
            (OUT / f"efficiency_unidepth_{args.camera}_{split}.json").write_text(json.dumps(eff, indent=1))
            print(eff)


if __name__ == "__main__":
    main()
