# Intro_to_CV_project_3

Monocular metric depth and video consistency for AIAA 3201, Spring 2027.

Yuming Zhang (50025933) and Jiayang Liu (50027216).

Given one RGB image or a short monocular video, the system estimates dense depth in meters and keeps it stable over time. Image benchmarks are Middlebury Stereo v3 and the official NYU Depth v2 test split. Video tests are two indoor phone clips.

## Layout

- `src/stereo` runs StereoSGBM and converts disparity with the Middlebury calibration.
- `scripts/eval_nyu_dav2_small.py` scores Depth Anything V2-Small after per-image scale-and-shift alignment. That alignment is not metric depth.
- `scripts/eval_nyu_unidepthv2_small.py` scores UniDepthV2-Small with the NYU Kinect intrinsics and no test-time alignment.
- `scripts/eval_nyu_part3.py` scores boundary F1, known-size scale, and the indoor obstacle application.
- `scripts/run_phone_clips.py` runs UniDepthV2-Small and RAFT fusion on `data/clips`.
- `paper/main.tex` is the CVPR 2026 report draft.

## Environment

Use the course virtualenv, or a Python 3.10 environment with PyTorch, torchvision, OpenCV, h5py, and scipy.

```bash
export PYTHON=/path/to/env_cv/bin/python
```

Upstream code is not stored in this repository. Clone it beside the scripts:

```bash
mkdir -p third_party
git clone --depth 1 https://github.com/DepthAnything/Depth-Anything-V2.git third_party/Depth-Anything-V2
git clone --depth 1 https://github.com/lpiccinelli-eth/UniDepth.git third_party/UniDepth
```

## Data

Benchmarks download from the links in the Spring 2027 brief:

```bash
bash scripts/download_benchmarks.sh
```

That writes `data/middlebury` and `data/nyu/nyu_depth_v2_labeled.mat`. The official NYU split used here is `data/nyu/splits.mat`.

Phone clips go in `data/clips/still.mp4` and `data/clips/moving.mp4`. Weights go in `weights/`. Data, weights, and outputs are not committed.

Depth Anything V2-Small checkpoint:

```bash
curl -L -o weights/depth_anything_v2_vits.pth \
  https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth
```

UniDepthV2-Small is loaded by the scripts from `lpiccinelli/unidepth-v2-vits14`.

## Commands

```bash
python scripts/run_middlebury_sgbm.py
python scripts/eval_nyu_dav2_small.py
python scripts/eval_nyu_unidepthv2_small.py
python scripts/eval_nyu_part3.py
python scripts/run_phone_clips.py
```

Tables are written under `outputs/`.
