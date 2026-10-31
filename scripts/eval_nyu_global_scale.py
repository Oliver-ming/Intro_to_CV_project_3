"""Part 2 + Part 3 Direction B on NYU Depth V2.

Baseline: raw UniDepthV2-Small metric depth on the 654 test images.
Improvement: one global scale fitted on the 795 train images only, frozen,
then applied to every test image. Test ground truth is never used for fitting.

Run scripts/cache_nyu_unidepth.py first.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calibration.global_scale import fit_all
from src.eval.nyu import depth_at, eval_mask, mean_metrics, metrics, open_mat, scene_type_at, split_ids

CACHE = ROOT / "outputs" / "nyu_cache"
OUT = ROOT / "outputs" / "nyu_global_scale"
MAIN_ESTIMATOR = "median_image"
SEED = 0


def load_split(data, split: str, camera: str):
    ids = split_ids(split)
    preds = np.load(CACHE / f"unidepth_{camera}_{split}.npy").astype(np.float32)
    gts = np.stack([depth_at(data, i) for i in ids])
    scenes = [scene_type_at(data, i) for i in ids]
    masks = np.stack([eval_mask(g) for g in gts])
    return preds, gts, masks, scenes


def valid_pairs(preds, gts, masks, idx=None):
    idx = range(len(preds)) if idx is None else idx
    return [(np.clip(preds[i][masks[i]], 1e-3, None).astype(np.float64), gts[i][masks[i]].astype(np.float64)) for i in idx]


def evaluate(preds, gts, masks, scale: float = 1.0) -> list[dict[str, float]]:
    return [metrics(scale * p, g, m) for p, g, m in zip(preds, gts, masks)]


def subset_stability(train_pairs, test, sizes=(5, 20, 50, 200, 795), repeats=20):
    """Refit the main estimator on random train subsets; report scale and test AbsRel."""
    rng = np.random.default_rng(SEED)
    rows = []
    for size in sizes:
        scales, absrels = [], []
        for _ in range(1 if size >= len(train_pairs) else repeats):
            pick = rng.choice(len(train_pairs), size=min(size, len(train_pairs)), replace=False)
            s = fit_all([train_pairs[i] for i in pick])[MAIN_ESTIMATOR]
            scales.append(s)
            absrels.append(mean_metrics(evaluate(*test, scale=s))["abs_rel"])
        rows.append({
            "train_images": size, "repeats": len(scales),
            "scale_mean": np.mean(scales), "scale_std": np.std(scales),
            "test_abs_rel_mean": np.mean(absrels), "test_abs_rel_std": np.std(absrels),
        })
    return rows


def by_scene(base_rows, cal_rows, scenes, train_scenes):
    """Test AbsRel per scene type, with how many train images share that type."""
    out = []
    for name in sorted(set(scenes)):
        idx = [i for i, s in enumerate(scenes) if s == name]
        out.append({
            "scene_type": name, "test_images": len(idx),
            "train_images": sum(s == name for s in train_scenes),
            "base_abs_rel": np.mean([base_rows[i]["abs_rel"] for i in idx]),
            "calibrated_abs_rel": np.mean([cal_rows[i]["abs_rel"] for i in idx]),
        })
    return sorted(out, key=lambda r: -r["test_images"])


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_camera(data, camera: str) -> list[dict]:
    train = load_split(data, "train", camera)
    test = load_split(data, "test", camera)
    train_pairs = valid_pairs(*train[:3])
    scales = fit_all(train_pairs)
    per_image_train = np.array([np.median(g / p) for p, g in train_pairs])
    base_rows = evaluate(*test[:3])
    rows = [{"camera": camera, "method": "unidepth_raw", "scale": 1.0, **mean_metrics(base_rows)}]
    for name, s in scales.items():
        rows.append({"camera": camera, "method": f"global_{name}", "scale": s, **mean_metrics(evaluate(*test[:3], scale=s))})
    # Diagnostic upper bound only: per-image median scale from TEST ground truth.
    oracle = [metrics(p * np.median(g[m] / np.clip(p[m], 1e-3, None)), g, m) for p, g, m in zip(*test[:3])]
    rows.append({"camera": camera, "method": "oracle_per_image_test_gt", "scale": float("nan"), **mean_metrics(oracle)})
    print(f"[{camera}] per-image train scale: median {np.median(per_image_train):.3f}  "
          f"IQR {np.percentile(per_image_train, 25):.3f}-{np.percentile(per_image_train, 75):.3f}")
    return rows, train, test, train_pairs, base_rows, scales


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    table = []
    with open_mat() as data:
        for camera in ("none", "nyu"):
            rows, train, test, train_pairs, base_rows, scales = run_camera(data, camera)
            table += rows
            if camera != "none":
                continue
            s = scales[MAIN_ESTIMATOR]
            cal_rows = evaluate(*test[:3], scale=s)
            write_csv(OUT / "per_scene.csv", by_scene(base_rows, cal_rows, test[3], train[3]))
            write_csv(OUT / "subset_stability.csv", subset_stability(train_pairs, test[:3]))
            np.save(OUT / "per_image_abs_rel.npy", np.array([[b["abs_rel"], c["abs_rel"]] for b, c in zip(base_rows, cal_rows)]))
            (OUT / "frozen_scale.json").write_text(json.dumps({"estimator": MAIN_ESTIMATOR, "camera": camera, "scale": s, "all": scales}, indent=1))
    write_csv(OUT / "nyu_global_scale.csv", table)
    for r in table:
        print(f"{r['camera']:4s} {r['method']:26s} s={r['scale']:.4f}  AbsRel={r['abs_rel']:.4f}  RMSE={r['rmse']:.4f}  d1={r['d1']:.4f}")


if __name__ == "__main__":
    main()
