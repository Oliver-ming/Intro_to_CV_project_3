# Intro_to_CV_project_3

Monocular metric depth and video consistency for AIAA 3201, Spring 2027.

Yuming Zhang (50025933) and Jiayang Liu (50027216).

The project follows the final course brief (`Project_3__Monocular_Metric_Depth_Estimation_and_Video_Consistency.pdf`):

- **Part 1**: StereoSGBM on the 15 Middlebury trainQ pairs, with disparity converted to meters through `calib.txt`. Depth Anything V2-Small runs frame-wise on `davis_rollercoaster.mp4`.
- **Part 2**: UniDepthV2-Small, RGB-only, scored on the official NYU Depth V2 test split with no alignment. Video Depth Anything-Small runs on the same clip.
- **Part 3 (Direction B)**: one global metric scale, fitted on the 795 NYU training images and frozen for the 654 test images.
- **Application**: a depth-aware background blur driven by each method's depth.

## Results

| NYU test (654), RGB-only | scale | AbsRel | RMSE | δ1 |
|---|---|---|---|---|
| UniDepthV2-Small | 1 | 0.1145 | 0.399 | 0.892 |
| + frozen train scale | 0.956 | **0.1020** | **0.389** | **0.900** |

Middlebury SGBM: mean absolute disparity error 1.42 px over valid predictions, which cover 62.2% of ground-truth pixels. A prediction is invalid when SGBM finds no match or the pixel fails a 1 px left-right check.

## Environment

Python 3.10+, PyTorch with CUDA, torchvision, OpenCV, h5py, scipy, matplotlib, imageio, imageio-ffmpeg, easydict, einops.

```bash
export PYTHON=/path/to/env_cv/bin/python
```

Upstream code is not stored in this repository. Clone it into `third_party/`:

```bash
mkdir -p third_party
git clone --depth 1 https://github.com/DepthAnything/Depth-Anything-V2.git third_party/Depth-Anything-V2
git clone --depth 1 https://github.com/DepthAnything/Video-Depth-Anything.git third_party/Video-Depth-Anything
git clone --depth 1 https://github.com/lpiccinelli-eth/UniDepth.git third_party/UniDepth
```

## Weights

```bash
mkdir -p third_party/Depth-Anything-V2/checkpoints third_party/Video-Depth-Anything/checkpoints
curl -L -o third_party/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth \
  https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth
curl -L -o third_party/Video-Depth-Anything/checkpoints/video_depth_anything_vits.pth \
  https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth
```

The scripts download UniDepthV2-Small from `lpiccinelli/unidepth-v2-vits14` on first use. RAFT-Small comes from torchvision.

## Data

```bash
bash scripts/download_benchmarks.sh   # Middlebury trainQ + GT, NYU labeled .mat
```

This writes `data/middlebury/MiddEval3` and `data/nyu/nyu_depth_v2_labeled.mat`. Place the official `splits.mat` in `data/nyu/`. The video is `third_party/Video-Depth-Anything/assets/example_videos/davis_rollercoaster.mp4`, which comes with the clone. It has 70 frames, so "first 100 frames" means the whole clip.

## Commands

```bash
$PYTHON scripts/run_middlebury_sgbm.py              # Part 1 stereo -> outputs/stereo
$PYTHON scripts/cache_nyu_unidepth.py --camera none # Part 2 metric depth, train+test cache
$PYTHON scripts/cache_nyu_unidepth.py --camera nyu  # ablation: Kinect intrinsics given
$PYTHON scripts/eval_nyu_global_scale.py            # Part 2 + Part 3 tables -> outputs/nyu_global_scale
$PYTHON scripts/make_nyu_figures.py                 # NYU qualitative + failure cases
$PYTHON scripts/run_video_compare.py                # DAv2-S vs VDA-S, blur demo -> outputs/video
$PYTHON scripts/make_pipeline_figure.py             # paper/figs/pipeline.pdf
```

To reproduce the official Video Depth Anything command:

```bash
cd third_party/Video-Depth-Anything
python run.py --input_video assets/example_videos/davis_rollercoaster.mp4 \
  --output_dir ./outputs --encoder vits --max_len 100 --max_res 640 --save_npz
```

Build the report with the CVPR template:

```bash
cd paper && docker run --rm -u $(id -u):$(id -g) -v "$PWD":/work -w /work texlive/texlive latexmk -pdf main.tex
```

## Earlier experiments (appendix only)

These come from the earlier brief. They are kept in the repository but are not part of the main comparison: `eval_nyu_dav2_small.py` (relative depth aligned on test ground truth), `eval_nyu_unidepthv2_small.py` and `eval_nyu_part3.py` (known-height calibration, obstacle ranking, collision warning), and `run_phone_clips.py` (self-recorded clips with RAFT fusion).
