# Metric calibration uses one scale for the whole clip

Per-frame scales were chosen first, then dropped. A person or car entering mid-clip would change that frame's median and put a seam in the depth. The clip scale is the median of the per-frame medians, and it multiplies every frame. A clip with no person and no car keeps the model's own metric depth.

The cost is that a real scale change inside one clip, such as a move from a room into a street, is frozen at one factor.
