"""NYU qualitative figures for the report.

Rows: RGB | GT | UniDepth raw | + frozen global scale | AbsRel error (raw) | error (scaled).
Picks one typical image and the three test images with the highest raw AbsRel
as difficult cases, plus the image the global scale hurts most. Within a row,
GT and both predictions share one depth range (the GT 98th percentile), and
both error maps share 0-0.5.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.nyu import depth_at, eval_mask, open_mat, rgb_at, split_ids, scene_type_at

CACHE = ROOT / "outputs" / "nyu_cache"
RES = ROOT / "outputs" / "nyu_global_scale"
OUT = ROOT / "outputs" / "figures"
DEPTH_MAP = (np.array(matplotlib.colormaps["magma"].colors) * 255).astype(np.uint8)
ERR_MAP = (np.array(matplotlib.colormaps["viridis"].colors) * 255).astype(np.uint8)


def colour(x: np.ndarray, hi: float, cmap: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    img = cmap[(np.clip(x / hi, 0, 1) * 255).astype(np.uint8)]
    if valid is not None:
        img[~valid] = 0
    return img


def tile(img: np.ndarray, text: str) -> np.ndarray:
    img = cv2.resize(img, (320, 240), interpolation=cv2.INTER_AREA).copy()
    cv2.rectangle(img, (0, 0), (320, 20), (0, 0, 0), -1)
    cv2.putText(img, text, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def row(data, preds, k: int, index: int, scale: float, absrel: np.ndarray) -> np.ndarray:
    rgb, gt = rgb_at(data, index), depth_at(data, index)
    mask = eval_mask(gt)
    raw = np.clip(preds[k].astype(np.float32), 1e-3, 10)
    cal = np.clip(scale * raw, 1e-3, 10)
    hi = float(np.percentile(gt[mask], 98))
    err_raw = np.abs(raw - gt) / np.maximum(gt, 1e-3)
    err_cal = np.abs(cal - gt) / np.maximum(gt, 1e-3)
    name = scene_type_at(data, index)
    tiles = [
        tile(rgb, f"#{k} {name}"),
        tile(colour(gt, hi, DEPTH_MAP, mask), f"GT (0-{hi:.1f} m)"),
        tile(colour(raw, hi, DEPTH_MAP), f"UniDepth raw  AbsRel {absrel[k, 0]:.3f}"),
        tile(colour(cal, hi, DEPTH_MAP), f"x{scale:.3f} frozen  AbsRel {absrel[k, 1]:.3f}"),
        tile(colour(err_raw, 0.5, ERR_MAP, mask), "|err|/gt raw (0-0.5)"),
        tile(colour(err_cal, 0.5, ERR_MAP, mask), "|err|/gt scaled (0-0.5)"),
    ]
    return np.concatenate(tiles, axis=1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    preds = np.load(CACHE / "unidepth_none_test.npy", mmap_mode="r")
    absrel = np.load(RES / "per_image_abs_rel.npy")
    scale = json.loads((RES / "frozen_scale.json").read_text())["scale"]
    ids = split_ids("test")
    order = np.argsort(absrel[:, 0])
    typical = [int(order[len(order) // 2])]
    hard = [int(k) for k in order[::-1][:3]]
    hurt = [int(np.argmax(absrel[:, 1] - absrel[:, 0]))]
    picks = {"typical": typical, "hard": hard, "scale_hurts": hurt}
    with open_mat() as data:
        for name, ks in picks.items():
            figure = np.concatenate([row(data, preds, k, ids[k], scale, absrel) for k in ks], axis=0)
            cv2.imwrite(str(OUT / f"nyu_{name}.png"), cv2.cvtColor(figure, cv2.COLOR_RGB2BGR))
    helped = float(np.mean(absrel[:, 1] < absrel[:, 0]))
    info = {"picks": picks, "fraction_images_improved": helped, "scale": scale}
    (OUT / "nyu_figures.json").write_text(json.dumps(info, indent=1))
    print(info)


if __name__ == "__main__":
    main()
