# Workload log

Each entry records one change: what changed, why, and which decision it rests on.

## 2026-10-01 — Glossary for the depth project

- Read the course brief (`Project 3- Monocular Metric Depth Estimation Video Consistency.pdf`).
- Opened `CONTEXT.md` with the terms that are already fixed: metric depth, relative depth, scale alignment, frame-wise depth, video consistency, flicker, scale drift, course sample, additional domain, exploration.
- Scope agreed with the user: high-score tier. The additional domain is one TUM RGB-D sequence. Part 3 is two or three explorations, each with its own ablation. Which explorations are still open.
- Initialized a git repository in `Project_3` and added `.gitignore` for weights, data, and outputs. No commit yet: this machine has no git user name or email, and git config must not be invented.

## 2026-10-01 — Part 3 trio not accepted

- Proposed explorations A (flow-guided fusion), B (metric scale calibration), and C (edge-aware refinement).
- User replied「没有」. The set is not locked. No glossary change.

## 2026-10-01 — Explorations locked to A, B, and F

- User chose flow-guided fusion, metric scale calibration, and a depth-aware application.
- Updated `CONTEXT.md` with those three terms. The concrete application task is still open.
- An ADR is warranted once the application task is chosen: three explorations instead of the brief's one, and the application counting as both an exploration and the demo.

## 2026-10-01 — Explorations revised to A, B, and E

- User said A, B, and E can all be done. Replaced the depth-aware application as an exploration with efficient deployment.
- The application demo stays a required deliverable, and is no longer an exploration. Its concrete form is still open.
- `CONTEXT.md` now defines Efficient Deployment and Application Demo. The depth-aware application entry was removed.

## 2026-10-01 — Sixth direction is the depth application

- User corrected the previous turn: the third exploration is the sixth brief direction, a depth-aware application, not efficient deployment.
- Restored Depth-Aware Application in `CONTEXT.md`. Removed Efficient Deployment and Application Demo.
- The concrete application task is still open.

## 2026-10-01 — Depth application has three outputs

- User asked for all three: distance ranking, collision warning, and a point cloud.
- Updated `CONTEXT.md` with Distance Ranking, Collision Warning, and Point Cloud. The depth-aware application now produces those three.
- What counts as an object for ranking and collision is still open. An ADR for the three explorations and three application outputs is still unwritten.

## 2026-10-01 — Object is a detector box

- User chose detector boxes. An object's distance is the median of the valid metric-depth pixels inside the box.
- Ranking compares that median with the median ground-truth depth. Collision uses the nearest box. The point cloud stays the full depth map.
- Added Object, Valid Depth, and tightened Distance Ranking and Collision Warning in `CONTEXT.md`.
- The collision threshold is still open.

## 2026-10-01 — Collision threshold

- User set the threshold at 1.5 m indoors and 8 m on KITTI driving scenes. A phone video uses the threshold of its scene type.
- A warning is correct when the median ground-truth depth in the same box falls on the same side of that threshold.
- Added Collision Threshold to `CONTEXT.md`. The calibration cue for metric scale is still open.

## 2026-10-01 — Calibration cue is a known object size

- User answered「A」, the recommended cue: a detected person or car with a size prior.
- Metric Scale Calibration in `CONTEXT.md` now uses that cue only. Added Size Prior.
- How those per-object ratios become one depth scale is still open.

## 2026-10-01 — One median scale per frame

- User chose one scale factor per frame: the median of the per-object size ratios. A frame with no person and no car keeps the model's metric depth. Priors are 1.70 m adult height and 1.80 m car width. Ranking, collision, and the point cloud use the scaled depth.
- Added Scale Factor to `CONTEXT.md` and set those numbers on Size Prior.
- Wrote `docs/adr/0001-three-explorations.md`: Part 3 is three explorations, not the brief's one.
- Still open: a person or car entering mid-clip can make that per-frame scale jump.

## 2026-10-01 — One scale for the whole clip

- User chose a clip-level scale: the median of the per-frame medians. A clip with no person and no car keeps the model's metric depth.
- Revised Scale Factor in `CONTEXT.md`. The per-frame multiplier is no longer the definition.
- Wrote `docs/adr/0002-one-scale-per-clip.md` for the seam this avoids, and for the frozen factor when a clip really changes scale.
- Still open: which depth this multiplier is applied to.

## 2026-10-01 — Clip scale multiplies fused UniDepth

- User applies the clip scale to UniDepthV2 metric depth after flow-guided fusion. Size ratios use that depth before scaling. Video Depth Anything stays a temporal comparison and is not metric depth. Ranking, collision, and the point cloud read the scaled result.
- Added System Depth to `CONTEXT.md`.
- Wrote `docs/adr/0003-application-uses-fused-unidepth.md`.
- Still open: the pixel rule inside flow-guided fusion.

## 2026-10-01 — Fusion pixel rule

- User accepted the recommended rule (「直接A」). RAFT warps the previous fused depth onto the current frame. Pixels whose forward and backward flow disagree, or that land outside the frame, keep the current UniDepth prediction. Other pixels mix by flow consistency. The first frame is UniDepth alone.
- Updated Flow-Guided Fusion in `CONTEXT.md`. Added Flow Consistency and Fused Depth. A single image counts as a one-frame clip for the scale factor.
- Still open: which detector boxes count as objects for ranking and collision. Calibration stays limited to people and cars.

## 2026-10-01 — Literature check before the object-set question

- User asked that each recommendation be checked against published standards, using a research subagent first.
- [Depth eval standards](3a7d5c25-fd4a-4259-8d5a-ee59151db573) reviewed KITTI object eval, nuScenes, Zhu & Fang (ICCV 2019), Uhrig et al. (GCPR 2016), and Kirillov et al. (CVPR 2019).
- Median depth inside the 2D box is appropriate here. The draft “every COCO thing” is not: driving benchmarks score a small obstacle subset, and small props are not collision objects.
- 1.5 m / 8 m and the 1.70 m / 1.80 m priors were left as they are. The object-set question is now asked with the revised rule.

## 2026-10-01 — Obstacles depend on the scene

- User accepted scene-conditioned obstacles. Road scenes: person and bicycle, car, motorcycle, bus, truck. Indoor scenes: person and chair, couch, bed, dining table, toilet, tv, refrigerator, potted plant. Ranking and collision share that set. Size calibration stays person and car.
- Added Road Scene, Indoor Scene, and Obstacle to `CONTEXT.md`. Wrote `docs/adr/0004-scene-obstacle-classes.md`.
- Next question, after another literature check: how ranking and collision are scored.

## 2026-10-01 — Scoring check before asking

- Resumed [Depth eval standards](3a7d5c25-fd4a-4259-8d5a-ee59151db573). Pairwise order accuracy matches Chen et al. (NIPS 2016) and later ordinal-depth papers. Object-distance papers (Zhu & Fang, ICCV 2019) lead with AbsRel, not MAE. Frame-level warn/no-warn with precision, recall, and F1 matches a fixed distance threshold when there is no velocity. Spearman is not the primary ordinal metric.
- The scoring question is asked with that revision. Not locked yet.

## 2026-10-01 — Ranking and collision scores

- User accepted the scores. Ranking: pairwise order accuracy on frames with at least two obstacles, plus obstacle AbsRel. Collision: frame-level precision, recall, and F1, plus AbsRel of the nearest obstacle.
- Updated `CONTEXT.md`. Wrote `docs/adr/0005-ranking-and-collision-scores.md`.
- Next question, after a literature check: which people and cars are allowed to vote on the clip scale factor when the box is cut off.

## 2026-10-01 — Retrospective literature audit requested

- User asked for a review of decisions already made, against papers, including ones that were locked before a literature check.
- [Audit locked decisions](b2a3909f-d29d-40bc-8145-1d3853a2cd17) reviewed the choices that were locked before a literature pass.
- Sound: UniDepth as the metric map and Video Depth Anything as a relative temporal comparison; causal RAFT fusion; known-size calibration as a noisy prior; one TUM sequence as the extra domain; an unscored point cloud beside scored ranking and collision.
- Plausible, not a published protocol: one clip-level multiplier with no shift. The 1.5 m / 8 m cuts stay, but 8 m must be described as a proximity threshold, not as a forward-collision-warning standard.
- Reopen: car width 1.80 m. Hoiem, ScaleNet, and FUMET use car height about 1.59 m. That question is now asked. The visibility-of-scale-voters pass is still in flight separately.

## 2026-10-01 — Car prior is height 1.59 m

- User replaced car width 1.80 m with car height 1.59 m, read from the vertical side of the box. Adult height stays 1.70 m. Implied size is pixel extent times depth, divided by focal length.
- Updated Size Prior and added Implied Size in `CONTEXT.md`. Wrote `docs/adr/0006-car-height-prior.md`.

## 2026-10-01 — Who may vote on the scale

- [Depth eval standards](3a7d5c25-fd4a-4259-8d5a-ee59151db573) finished the visibility check. ScaleNet keeps people only when head and both ankles are visible. Border contact is a reasonable truncation proxy, not a numbered standard. Detector score 0.5 and aspect-ratio cuts are sensitivity choices, not published rules. Implied size as pixel extent times depth over focal length matches the pinhole relation used by Casser et al. (AAAI 2019).
- That brief still assumed a car-width prior. The car prior is now height, so a fronto-parallel width gate is not part of the question. The voting question is asked with the height prior.

## 2026-10-01 — Scale votes

- User accepted the voting rule. A box scores at least 0.5 and does not touch the image border. A person also needs the head and both ankles visible. A car votes with its vertical extent. The 0.5 cut is a sensitivity choice. Implied size uses the median valid depth and the vertical focal length.
- Added Scale Vote and tightened Implied Size in `CONTEXT.md`. Wrote `docs/adr/0007-who-may-vote-on-scale.md`.
- Next question, after a literature check: how temporal error and scale drift are scored.

## 2026-10-01 — Temporal-score check before asking

- Resumed [Depth eval standards](3a7d5c25-fd4a-4259-8d5a-ee59151db573). Flow-warped relative error on non-occluded pixels matches Li et al. (ICCVW 2021) and Khan et al. (CVPR 2023, OPW). Video Depth Anything’s TAE needs poses, which this project does not assume. Static and moving clips must be reported separately because flow-warp constancy fails under ego-motion. Median(pred/gt) drift is a GT scale series, not Luo et al.’s 3D-track drift. Phone video has no numeric drift.
- The temporal-score question is asked with that revision. Not locked yet.

## 2026-10-01 — Temporal error and drift score

- User accepted flow-warped relative temporal error, with static and moving clips reported separately. With ground truth, drift is the standard deviation of per-frame median(prediction / ground truth), plus the largest gap from the first frame. Phone video has no numeric drift. A relative baseline gets one video-level scale alignment before drift is read. The optional 1.01-ratio pixel fraction was not adopted.
- Added Temporal Error and Drift Score to `CONTEXT.md`. Wrote `docs/adr/0008-temporal-error-and-drift.md`.
- Next question, after a literature check: how depth-boundary quality is scored. The course requires it, and this project has no edge-refinement exploration.

## 2026-10-01 — Boundary-score check before asking

- Resumed [Depth eval standards](3a7d5c25-fd4a-4259-8d5a-ee59151db573). A 2-pixel hit radius is not the standard. Depth Pro’s scale-invariant boundary F1 compares metric and relative maps without extra edge labels. Hu et al. (WACV 2019) use Sobel precision/recall/F1. iBims DBE needs manual edges and a 10-pixel chamfer. Sparse KITTI LiDAR should not be turned into automatic edges.
- The boundary question is asked with that revision. Not locked yet.

## 2026-10-01 — Boundary score

- User chose Depth Pro's scale-invariant boundary F1 on NYUv2 and TUM. KITTI boundaries stay qualitative. RGB edges are not the reference.
- Added Boundary Score and noted that the point cloud is unscored in `CONTEXT.md`. Wrote `docs/adr/0009-boundary-score.md`.
- Next question, after a literature check: whose focal length is used for implied size and for the point cloud.

## 2026-10-01 — Camera check before asking

- Resumed [Depth eval standards](3a7d5c25-fd4a-4259-8d5a-ee59151db573). UniDepth couples depth and intrinsics: dataset K may be passed in at inference, and the same K must be used for backprojection and for the size equation. Substituting dataset focal length after the fact, onto depth that was predicted with another camera, is not supported. Phone video stays on the predicted camera.
- The camera question is asked with that revision. Not locked yet.

## 2026-10-01 — One camera per clip

- User chose to pass published intrinsics into UniDepth at inference when they exist, and to use that same camera for implied size and the point cloud. Phone video uses the predicted camera. Post-hoc mixing was rejected.
- Added Camera and tightened Implied Size in `CONTEXT.md`. Wrote `docs/adr/0010-one-camera-per-clip.md`.
- Next question, after a literature check: which single TUM RGB-D sequence is the additional domain.

## 2026-10-01 — TUM sequence check before asking

- Resumed [Depth eval standards](3a7d5c25-fd4a-4259-8d5a-ee59151db573). Dynamic SLAM papers use the freiburg3 walking set when the camera moves and people walk. `fr3/walking_xyz` has official synced depth and pose. Sitting and desk sequences are weak for a standing-height prior. It is a SLAM sequence, not a standard video-depth benchmark clip, which is acceptable for one extra domain.
- The sequence question is asked with that result. Not locked yet.

## 2026-10-01 — Additional domain is fr3/walking_xyz

- User chose fr3/walking_xyz. Updated Additional Domain in `CONTEXT.md`. Wrote `docs/adr/0011-tum-walking-xyz.md`.
- Next question, after a literature check: how the noisy size-prior ablation is defined. The course asks for robustness when the calibration cue is noisy.

## 2026-10-01 — Noisy-prior check before asking

- Resumed [Depth eval standards](3a7d5c25-fd4a-4259-8d5a-ee59151db573). The return restated the ScaleNet priors already used here: people 1.70 ± 0.09 m, cars 1.59 ± 0.21 m (Zhu et al., ECCV 2020). It did not add a separate published protocol for a noisy-cue ablation.
- The ablation question therefore uses those published standard deviations as a fixed ±1σ shift, not a random draw. Not locked yet.

## 2026-10-01 — Noisy prior is ±1σ

- User chose a clean prior as the main system, plus fixed +1σ and −1σ shifts (0.09 m adult height, 0.21 m car height). Each shift recomputes the clip scale and is scored with obstacle AbsRel and collision F1.
- Added Noisy Prior to `CONTEXT.md`. Wrote `docs/adr/0012-noisy-prior-is-one-sigma.md`.
- Searched `/home/zhangyuming` to depth 4 for NYUv2, KITTI, TUM, and Depth Anything data, excluding `env_cv`. Nothing matched. The course-sample location is now the open question.

## 2026-10-01 — Spring 2027 brief replaces the image benchmarks

- Read `AIAA3201_Project3_Monocular_Metric_Depth_Video_Consistency_Spring2027.pdf`.
- Passing quantitative images are now Middlebury Stereo v3 (quarter resolution, 15 training pairs) and the NYU Depth V2 labeled set. Video tests are self-recorded clips. The brief no longer requires an extra public dataset for a high score.
- Updated Course Sample in `CONTEXT.md`. Part 3 explorations, scores, and the size prior are unchanged.
- Still open: `fr3/walking_xyz` was locked under the old high-score rule and now conflicts with the new brief.

## 2026-10-01 — Drop TUM; video is self-recorded

- User dropped fr3/walking_xyz. Video tests are one still-camera phone clip and one moving-camera phone clip, with no ground-truth depth.
- Replaced Additional Domain with Self-Recorded Clip. Road and indoor scenes no longer name KITTI or TUM. Boundary score is NYUv2 only. The 8 m cut remains for a self-recorded road scene and is labeled a proximity cut.
- Marked `docs/adr/0011-tum-walking-xyz.md` superseded. Wrote `docs/adr/0013-self-recorded-clips-not-tum.md`.

## 2026-10-01 — Both phone clips are indoor

- User chose two indoor phone clips: one with a still camera, one with a moving camera, each with a standing person. The 1.5 m proximity cut applies. No road clip, so the 8 m cut is unused.
- Updated Self-Recorded Clip in `CONTEXT.md`. Wrote `docs/adr/0014-indoor-phone-clips.md`.

## 2026-10-01 — Phone clips are about 15 seconds

- User set each self-recorded clip at about 15 seconds. The standing person must show head and both ankles so a scale vote is possible.
- Updated Self-Recorded Clip. Added Scored Frame: numeric obstacle, collision, and drift scores run on NYUv2 only. Self-recorded frames carry temporal error and the demo.

## 2026-10-01 — Scaffold and benchmark downloads

- User said to start the layout and the downloads.
- Added `scripts/download_benchmarks.sh` and source folders: `src/stereo`, `src/monocular`, `src/fusion`, `src/calibration`, `src/application`, `src/eval`, plus `configs` and `data/{middlebury,nyu,clips}`.
- Middlebury quarter-resolution input and left-view ground truth come from the official MiddEval3 zips named in the Spring 2027 brief. NYU labeled `.mat` is downloading from the official Silberman URL. Data stays gitignored.
- No git commit: this machine still has no git user name or email.
- NYU labeled download stopped at about 350 MB of 2.8 GB with curl exit 56 (connection reset). A resume from that byte offset is running.
- Resume finished. `data/nyu/nyu_depth_v2_labeled.mat` is 2,972,037,809 bytes and opens as a MATLAB 7.3 file.

## 2026-10-01 — StereoSGBM on Middlebury training pairs

- Stopped waiting on the phone clips and a git identity. Neither blocks the stereo baseline.
- Added `src/stereo` and `scripts/run_middlebury_sgbm.py`. Metric depth uses the Middlebury calibration, Z = f B / (d + doffs), with baseline converted from millimeters.
- Ran StereoSGBM plus a left-right check on all 15 quarter-resolution training pairs. Mean bad-1.0 rate on non-occluded pixels is 0.368. Per-scene numbers are in `outputs/stereo/middlebury_sgbm.csv`. Colored disparity and invalid masks are beside it.

## 2026-10-01 — NYU test, Depth Anything V2-Small

- Official split from `splits.mat`: 654 test images. Eigen crop and a 10 m cap. Metrics match the Depth Anything V2 `eval_depth` definitions.
- Relative V2-Small with per-image scale-and-shift alignment: AbsRel 0.132, RMSE 0.427, δ1 0.845. Median-only scale alignment fails (AbsRel 1.35) because the relative output also has an unknown shift. These aligned numbers are not metric depth.
- Weights: `weights/depth_anything_v2_vits.pth`. Table: `outputs/nyu_dav2_small/nyu_dav2_small.csv`.

## 2026-10-01 — NYU test, UniDepthV2-Small, no alignment

- Metric depth from `lpiccinelli/unidepth-v2-vits14`. The labeled-set Kinect intrinsics are passed in at inference. No scale alignment.
- Same 654-image split, Eigen crop, and 10 m cap as the relative baseline.
- AbsRel 0.093, RMSE 0.380, δ1 0.925. No test-time alignment. Table: `outputs/nyu_unidepthv2_small/nyu_unidepthv2_small.csv`.

## 2026-10-01 — NYU Part 3 and report draft

- Added scale-invariant boundary F1, known-size voting, indoor obstacle scores, and a RAFT fusion path for phone clips (`scripts/eval_nyu_part3.py`, `scripts/run_phone_clips.py`).
- On the 654-image test split, 60 images cast a scale vote. Clean-prior calibration raises AbsRel from 0.093 to 0.273. Boundary F1 stays 0.117 because a global scale does not move inverse-depth ratios. Obstacle order accuracy is 0.932; 1.5 m warning F1 is 0.788. UniDepth latency is 46 ms/frame.
- Wrote the CVPR 2026 draft in `paper/main.tex`. `pdflatex` is not installed, so there is no PDF yet.
- Phone clips are still absent. `scripts/run_phone_clips.py` exits with a message until `data/clips` contains the two videos.

## 2026-10-01 — Capture sheet for the two phone clips

- Wrote `docs/明天拍摄说明.pdf`: two indoor clips of about 15 seconds, `still.mp4` with a fixed camera and `moving.mp4` with a moving camera, each showing a standing person with head and both ankles inside the frame.

## 2026-10-07 — Phone clips from CV资料/Project3

- Still camera: `相机不动/DJI_20261004141828_0025_D.MP4` (19.9 s). Moving camera: `相机移动/DJI_20261004142302_0028_D.MP4` (19.4 s). A 4.8 s moving clip was left out.
- Transcoded both to 1280x720, 15 fps, as `data/clips/still.mp4` and `data/clips/moving.mp4`. Both show a standing person, full body, and a chair indoors.
- Started UniDepthV2-Small plus RAFT fusion on those two clips.
- Finished. Still clip: 298 frames, flow-warped temporal AbsRel 0.021, fixed range 6.03 m. Moving clip: 289 frames, temporal AbsRel 0.041, fixed range 6.79 m. Depth videos are in `outputs/clips/`.

## 2026-10-08 — Repository prepared, push left to the user

- Authors from `演讲者资料.md`: Yuming Zhang (50025933) and Jiayang Liu (50027216). Report author block and README use those names. The public URL in the abstract is `https://github.com/Oliver-ming/Intro_to_CV_project_3`.
- Staged the code, report, and docs. Data, weights, outputs, and `third_party` stay untracked.
- `origin` points at that GitHub URL. The branch is `main`. The push is not done.

## 2026-10-08 — Pushed main with Yuming Zhang's email

- Rewrote the local commits so the author is Yuming Zhang \<1756882063@qq.com\>. Jiayang Liu's author line stays the student id only.
- Pushed `main` to `https://github.com/Oliver-ming/Intro_to_CV_project_3.git`. The token was used once for that push and was not written into the repo or git config.
