"""NYU test: boundary F1, known-size scale, and the depth application.

UniDepthV2-Small is metric and is not aligned on the test set. A single image
is a one-frame clip, so the clip scale is that frame's median size ratio.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import scipy.io
import torch
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    KeypointRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn_v2,
    keypointrcnn_resnet50_fpn,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "Depth-Anything-V2"))
sys.path.insert(0, str(ROOT / "third_party" / "UniDepth"))

from depth_anything_v2.dpt import DepthAnythingV2
from unidepth.models import UniDepthV2
from unidepth.utils.camera import Pinhole

from src.calibration.scale import (
    CAR_HEIGHT_M,
    CAR_SIGMA_M,
    MIN_SCORE,
    PERSON_HEIGHT_M,
    PERSON_SIGMA_M,
    clip_scale,
    implied_height_m,
    median_depth,
    touches_border,
)
from src.eval.boundary import si_boundary_f1

MAT = ROOT / "data" / "nyu" / "nyu_depth_v2_labeled.mat"
SPLIT = ROOT / "data" / "nyu" / "splits.mat"
DAV2_WEIGHT = ROOT / "weights" / "depth_anything_v2_vits.pth"
OUT = ROOT / "outputs" / "nyu_part3"
MIN_DEPTH = 1e-3
MAX_DEPTH = 10.0
FOCAL_Y = 519.4696
K = torch.tensor(
    [[518.8579, 0.0, 325.5824], [0.0, FOCAL_Y, 253.7362], [0.0, 0.0, 1.0]],
    dtype=torch.float32,
)
# Torchvision COCO category ids for indoor obstacles.
INDOOR = {1, 62, 63, 64, 65, 67, 70, 72, 82}
CAR_IDS = {3}
COLLISION_M = 1.5
NOSE, LEFT_ANKLE, RIGHT_ANKLE = 0, 15, 16


def to_hwc(image_chw: np.ndarray) -> np.ndarray:
    return np.transpose(image_chw, (2, 1, 0))


def to_hw(depth_hw: np.ndarray) -> np.ndarray:
    return np.transpose(depth_hw, (1, 0))


def eigen_mask(gt: np.ndarray) -> np.ndarray:
    mask = np.zeros(gt.shape, dtype=bool)
    mask[45:471, 41:601] = True
    return mask & np.isfinite(gt) & (gt > MIN_DEPTH) & (gt < MAX_DEPTH)


def depth_metrics(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    prediction = np.clip(pred[mask].astype(np.float64), MIN_DEPTH, None)
    target = gt[mask].astype(np.float64)
    thresh = np.maximum(target / prediction, prediction / target)
    diff = prediction - target
    return {
        "abs_rel": float(np.mean(np.abs(diff) / target)),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "d1": float(np.mean(thresh < 1.25)),
    }


def align_scale_shift(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> np.ndarray:
    prediction = pred[mask].astype(np.float64)
    target = gt[mask].astype(np.float64)
    design = np.stack([prediction, np.ones_like(prediction)], axis=1)
    scale, shift = np.linalg.lstsq(design, target, rcond=None)[0]
    return pred * scale + shift


def _boxes(output, height, width):
    boxes = output["boxes"].detach().cpu().numpy()
    scores = output["scores"].detach().cpu().numpy()
    labels = output["labels"].detach().cpu().numpy()
    keep = scores >= MIN_SCORE
    return boxes[keep], scores[keep], labels[keep]


def person_can_vote(box, keypoints, width, height) -> bool:
    if touches_border(box, width, height):
        return False
    nose, left, right = keypoints[NOSE], keypoints[LEFT_ANKLE], keypoints[RIGHT_ANKLE]
    return bool(nose[2] > MIN_SCORE and left[2] > MIN_SCORE and right[2] > MIN_SCORE)


def size_ratios(depth, boxes, labels, keypoints_by_box, valid, prior_shift: float) -> list[float]:
    height, width = depth.shape
    ratios = []
    for box, label in zip(boxes, labels):
        if touches_border(box, width, height):
            continue
        if label == 1:
            key = keypoints_by_box.get(tuple(np.round(box, 1)))
            if key is None or not person_can_vote(box, key, width, height):
                continue
            prior = PERSON_HEIGHT_M + prior_shift * PERSON_SIGMA_M
        elif label in CAR_IDS:
            prior = CAR_HEIGHT_M + prior_shift * CAR_SIGMA_M
        else:
            continue
        depth_m = median_depth(depth, box, valid)
        if depth_m is None:
            continue
        implied = implied_height_m(box[3] - box[1], depth_m, FOCAL_Y)
        if implied > 0.2:
            ratios.append(prior / implied)
    return ratios


def match_keypoints(person_boxes, person_keypoints, det_boxes) -> dict:
    matched = {}
    if len(person_boxes) == 0:
        return matched
    for box in det_boxes:
        center = 0.5 * (box[:2] + box[2:])
        centers = 0.5 * (person_boxes[:, :2] + person_boxes[:, 2:])
        index = int(np.argmin(np.linalg.norm(centers - center, axis=1)))
        if np.linalg.norm(centers[index] - center) < 40:
            matched[tuple(np.round(box, 1))] = person_keypoints[index]
    return matched


class ScoreBag:
    def __init__(self) -> None:
        self.sum = {"abs_rel": 0.0, "rmse": 0.0, "d1": 0.0, "boundary": 0.0}
        self.n = 0
        self.votes = 0
        self.pair_hit = 0
        self.pair_n = 0
        self.obj_abs = 0.0
        self.obj_n = 0
        self.tp = self.fp = self.fn = self.tn = 0

    def add_depth(self, pred, gt, mask) -> None:
        one = depth_metrics(pred, gt, mask)
        one["boundary"] = si_boundary_f1(pred, gt, mask)
        self.n += 1
        for key in self.sum:
            self.sum[key] += one[key]

    def mean(self) -> dict[str, float]:
        return {key: value / max(self.n, 1) for key, value in self.sum.items()}


def obstacle_rows(depth, gt, boxes, labels, valid):
    rows = []
    for box, label in zip(boxes, labels):
        if int(label) not in INDOOR:
            continue
        pred_d = median_depth(depth, box, np.ones(depth.shape, dtype=bool))
        gt_d = median_depth(gt, box, valid)
        if pred_d is None or gt_d is None:
            continue
        rows.append((pred_d, gt_d))
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    test_ids = scipy.io.loadmat(SPLIT)["testNdxs"].astype(int).ravel() - 1
    limit = int(os.environ.get("NYU_LIMIT", "0"))
    if limit:
        test_ids = test_ids[:limit]
    device = "cuda"
    unidepth = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vits14")
    unidepth.interpolation_mode = "bilinear"
    unidepth = unidepth.to(device).eval()
    camera = Pinhole(K=K.unsqueeze(0))
    relative = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    relative.load_state_dict(torch.load(DAV2_WEIGHT, map_location="cpu"))
    relative = relative.to(device).eval()
    detector = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT).to(device).eval()
    poser = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT).to(device).eval()

    bags = {name: ScoreBag() for name in ("unidepth", "calibrated", "plus_sigma", "minus_sigma", "dav2_aligned")}
    torch.cuda.reset_peak_memory_stats()
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    unidepth_ms = 0.0
    timed = 0

    n = len(test_ids)
    with h5py.File(MAT, "r") as data, torch.inference_mode():
        for step, index in enumerate(test_ids):
            rgb = to_hwc(data["images"][index])
            gt = to_hw(data["depths"][index]).astype(np.float32)
            mask = eigen_mask(gt)
            tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)
            if timed < 100:
                starter.record()
            pred = unidepth.infer(tensor, camera)["depth"].squeeze().detach().cpu().numpy()
            if timed < 100:
                ender.record()
                torch.cuda.synchronize()
                unidepth_ms += starter.elapsed_time(ender)
                timed += 1
            if pred.shape != gt.shape:
                pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)

            image = (tensor.float() / 255.0).unsqueeze(0).to(device)
            det = detector(image)[0]
            pose = poser(image)[0]
            boxes, scores, labels = _boxes(det, *gt.shape)
            pose_boxes, _, _ = _boxes(pose, *gt.shape)
            pose_kpts = pose["keypoints"].detach().cpu().numpy()
            pose_keep = pose["scores"].detach().cpu().numpy() >= MIN_SCORE
            keypoints = match_keypoints(pose_boxes, pose_kpts[pose_keep], boxes)

            valid = np.isfinite(gt) & (gt > MIN_DEPTH)
            ratios = size_ratios(pred, boxes, labels, keypoints, valid, 0.0)
            scale = clip_scale(ratios)
            plus = clip_scale(size_ratios(pred, boxes, labels, keypoints, valid, 1.0))
            minus = clip_scale(size_ratios(pred, boxes, labels, keypoints, valid, -1.0))
            bags["unidepth"].add_depth(pred, gt, mask)
            bags["calibrated"].add_depth(pred * scale, gt, mask)
            bags["plus_sigma"].add_depth(pred * plus, gt, mask)
            bags["minus_sigma"].add_depth(pred * minus, gt, mask)
            if ratios:
                bags["calibrated"].votes += 1

            rel = relative.infer_image(rgb[:, :, ::-1].copy(), input_size=518)
            if rel.shape != gt.shape:
                rel = cv2.resize(rel, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
            bags["dav2_aligned"].add_depth(align_scale_shift(rel, gt, mask), gt, mask)

            objects = obstacle_rows(pred * scale, gt, boxes, labels, valid)
            bag = bags["calibrated"]
            if len(objects) >= 2:
                for i in range(len(objects)):
                    for j in range(i + 1, len(objects)):
                        pred_order = objects[i][0] < objects[j][0]
                        gt_order = objects[i][1] < objects[j][1]
                        bag.pair_hit += int(pred_order == gt_order)
                        bag.pair_n += 1
            for pred_d, gt_d in objects:
                bag.obj_abs += abs(pred_d - gt_d) / gt_d
                bag.obj_n += 1
            if objects:
                nearest_pred = min(item[0] for item in objects)
                nearest_gt = min(item[1] for item in objects)
                pred_warn = nearest_pred < COLLISION_M
                gt_warn = nearest_gt < COLLISION_M
                bag.tp += int(pred_warn and gt_warn)
                bag.fp += int(pred_warn and not gt_warn)
                bag.fn += int(not pred_warn and gt_warn)
                bag.tn += int(not pred_warn and not gt_warn)

            if step == 0:
                cv2.imwrite(str(OUT / "sample_rgb.png"), rgb[:, :, ::-1])
            if (step + 1) % 100 == 0 or step + 1 == n:
                print(f"{step + 1}/{n}  votes={bags['calibrated'].votes}")

    peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
    rows = []
    for name, bag in bags.items():
        row = {"method": name, "frames": bag.n, "scale_votes": bag.votes, **bag.mean()}
        if name == "calibrated":
            precision = bag.tp / max(bag.tp + bag.fp, 1)
            recall = bag.tp / max(bag.tp + bag.fn, 1)
            row.update(
                {
                    "pairwise_order": bag.pair_hit / max(bag.pair_n, 1),
                    "obstacle_abs_rel": bag.obj_abs / max(bag.obj_n, 1),
                    "collision_precision": precision,
                    "collision_recall": recall,
                    "collision_f1": 2 * precision * recall / max(precision + recall, 1e-8),
                    "obstacle_count": bag.obj_n,
                    "pair_count": bag.pair_n,
                    "warn_frames": bag.tp + bag.fn,
                }
            )
        rows.append(row)
        print(name, {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
    print(f"unidepth latency {unidepth_ms / max(timed, 1):.1f} ms  peak {peak_mb:.0f} MiB")
    with (OUT / "nyu_part3.csv").open("w", newline="") as handle:
        fields = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "efficiency.txt").write_text(
        f"unidepth_v2_small_ms={unidepth_ms / max(timed, 1):.2f}\n"
        f"timed_frames={timed}\npeak_mib={peak_mb:.1f}\ninput=480x640\n"
    )


if __name__ == "__main__":
    main()
