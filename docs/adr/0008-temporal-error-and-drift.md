# Temporal error is a flow warp; drift needs ground truth

The course asks for a flow-aligned temporal error and a scale drift. The error is the mean relative gap after a RAFT warp, on pixels whose flow agrees in reverse and whose depth is valid. Static-camera and moving-camera clips are separate, because a flow warp assumes the matched depth stays put, which ego-motion breaks. Pose-based reprojection, as in Video Depth Anything, needs camera poses this project does not assume.

Drift is the spread of per-frame median(prediction / ground truth). Phone video has no ground truth, so it gets no numeric drift. A relative baseline is aligned once for the whole video before that series is read, so a per-frame fit cannot hide the drift.
