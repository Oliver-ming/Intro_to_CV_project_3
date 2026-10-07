# Monocular Metric Depth

Dense depth with a real-world scale, from one RGB image or an ordinary monocular video, kept stable over time.

## Language

**Metric Depth**:
Depth in real-world units. The scale comes from the model or from a calibration cue.
_Avoid_: absolute depth, real depth, aligned depth

**Relative Depth**:
Depth defined only up to an unknown scale and shift. A benchmark fit does not make it metric depth.
_Avoid_: disparity, normalized depth

**Scale Alignment**:
A test-time fit of scale, or of scale and shift, used only to score relative depth on a benchmark.
_Avoid_: calibration, metric recovery

**Frame-wise Depth**:
Depth predicted from a single frame, with no information from any other frame.
_Avoid_: per-frame baseline, image depth

**Video Consistency**:
The same scene keeping a stable depth across frames: little flicker when the camera is still, and little scale drift when the camera moves.
_Avoid_: temporal smoothness, video quality

**Flicker**:
Frame-to-frame depth change on surfaces that are not moving, while the camera is still.
_Avoid_: noise, jitter

**Scale Drift**:
A change in depth scale along a moving-camera clip that the scene itself does not explain.
_Avoid_: scale ambiguity

**Temporal Error**:
The mean relative disagreement of depth between consecutive frames, after one frame is warped onto the other with optical flow. Only pixels with consistent flow and valid depth are counted. Static-camera clips and moving-camera clips are scored apart.
_Avoid_: photometric error, pose reprojection

**Drift Score**:
For a clip with ground-truth depth, the spread of the per-frame median of predicted depth over ground-truth depth, plus the largest gap from the first frame. A clip with no ground truth has no numeric drift score. A relative-depth baseline is given one video-level scale alignment before this score is read.
_Avoid_: flicker, median depth

**Course Sample**:
The Spring 2027 image benchmarks. Stereo uses the Middlebury Stereo v3 quarter-resolution training pairs and their calibration files. Monocular metric depth uses the NYU Depth V2 labeled set on the standard split and the Depth Anything V2 metric-depth protocol.
_Avoid_: KITTI subset, instructor crop, raw NYUv2

**Self-Recorded Clip**:
A phone video recorded for this project. It is indoors, about 15 seconds, and shows a standing person with the head and both ankles visible. One clip keeps the camera still, for flicker. One clip moves the camera, for scale drift. Neither has ground-truth depth.
_Avoid_: TUM sequence, road scene, course video

**Scored Frame**:
A frame with ground-truth depth. Obstacle AbsRel, pairwise order accuracy, collision precision, recall, and F1, and the drift score are computed only on scored frames. NYUv2 frames are scored frames. A self-recorded frame is not.
_Avoid_: phone frame, demo frame

**Exploration**:
A lightweight attack on one measurable limitation of the reproduced system, judged by its own controlled ablation. This project runs three: flow-guided fusion, metric scale calibration, and a depth-aware application.
_Avoid_: improvement, module, feature

**Flow Consistency**:
How well the forward flow and the backward flow agree at a pixel.
_Avoid_: confidence, photometric error

**Flow-Guided Fusion**:
An exploration that warps the previous fused depth onto the current frame and mixes it with the current metric prediction. The mix follows flow consistency. A pixel whose flow disagrees in reverse, or that lands outside the frame, keeps the current prediction. The first frame is the current prediction alone.
_Avoid_: temporal smoothing, video model

**Fused Depth**:
Metric depth after flow-guided fusion and before the clip scale factor.
_Avoid_: system depth, video depth

**System Depth**:
The depth the application uses. It is the flow-guided fusion of UniDepth's metric depth, multiplied by the clip scale factor.
_Avoid_: video depth, relative depth

**Metric Scale Calibration**:
An exploration that sets metric scale from the known size of a detected person or car.
_Avoid_: scale alignment, ground-plane fit, sparse depth

**Size Prior**:
The real-world size assumed for a detected class: 1.70 m for an adult's height, and 1.59 m for a car's height. Both are read from the vertical extent of the box.
_Avoid_: car width, measured size, ground-truth dimension

**Noisy Prior**:
A size prior shifted by one published standard deviation: 0.09 m for adult height, and 0.21 m for car height. The main system uses the unshifted prior. Each shifted prior is its own clip scale, scored with obstacle AbsRel and collision F1.
_Avoid_: random cue, dropout

**Camera**:
The intrinsics used for implied size and for the point cloud. On a benchmark clip they are the published intrinsics, passed into UniDepth at inference. On a phone video they are the intrinsics UniDepth predicts. A depth map is never paired with a different camera after inference.
_Avoid_: focal swap, EXIF

**Implied Size**:
The meters a box would measure if the current metric depth and focal length were right: vertical pixel extent times the median valid metric depth in the box, divided by the vertical focal length from the camera.
_Avoid_: box width, ground-truth size

**Scale Vote**:
A person or car box allowed to contribute a size ratio. It scores at least 0.5, and it does not touch the image border. A person also needs the head and both ankles visible. The ratio is the size prior over the implied size. The score cut is a sensitivity choice, not a metrology standard.
_Avoid_: obstacle, detection

**Scale Factor**:
One multiplier for a whole clip. Each frame first takes the median of size-prior over depth-implied size for its people and cars. The clip multiplier is the median of those frame medians. A clip with neither keeps the model's metric depth. A single image is a clip of one frame.
_Avoid_: per-frame scale, per-object scale, shift

**Depth-Aware Application**:
A downstream use of depth, scored on that use's own output. It produces a distance ranking, a collision warning, and a point cloud, and it is the project's application demo.
_Avoid_: efficient deployment, visualization, qualitative result

**Object**:
A box from a detector. Its distance is the median of the valid metric-depth pixels inside that box.
_Avoid_: instance mask, segment, click

**Road Scene**:
A driving scene in a self-recorded clip.
_Avoid_: KITTI, outdoor, street

**Indoor Scene**:
A scene inside a room, including NYUv2 and an indoor phone video.
_Avoid_: TUM, interior, household

**Obstacle**:
An object that distance ranking and collision warning are allowed to use. In a road scene it is a person or a vehicle: bicycle, car, motorcycle, bus, or truck. In an indoor scene it is a person or a large thing: chair, couch, bed, dining table, toilet, tv, refrigerator, or potted plant.
_Avoid_: COCO thing, stuff, small prop

**Valid Depth**:
A depth pixel the evaluation set marks as usable: inside the valid-depth mask and under the maximum-depth setting.
_Avoid_: inlier, confidence

**Distance Ranking**:
The order of obstacles in one frame by their distance.
_Avoid_: depth sort, click-to-measure

**Pairwise Order Accuracy**:
The fraction of obstacle pairs in a frame whose nearer-or-farther relation matches the ground-truth distances. A frame with fewer than two obstacles is not scored.
_Avoid_: Spearman, Kendall tau

**Obstacle AbsRel**:
The mean of |predicted distance − ground-truth distance| / ground-truth distance over the obstacles in a frame.
_Avoid_: MAE, RMSE

**Collision Warning**:
A frame-level signal that the nearest obstacle's distance is below the collision threshold. It is scored with precision, recall, and F1 against the same decision on ground-truth distances, plus the AbsRel of that nearest obstacle.
_Avoid_: time-to-collision, average precision

**Collision Threshold**:
1.5 m in an indoor scene. 8 m in a self-recorded road scene. It is an application proximity cut, not a forward-collision-warning standard.
_Avoid_: safety distance, FCW standard

**Point Cloud**:
The predicted depth placed in 3D with the camera geometry. It is a view of system depth, not a separate score.
_Avoid_: mesh, reconstruction

**Boundary Score**:
The scale-invariant boundary F1 between edges of the predicted depth and edges of the ground-truth depth. It is computed on NYUv2 only.
_Avoid_: Sobel F1, chamfer, texture agreement, TUM, KITTI
