# The application reads fused UniDepth, not the video model

Video Depth Anything is the temporal reference. It stays relative depth, including when a benchmark scale-aligns it. The depth-aware application reads UniDepth's metric depth after flow-guided fusion, then one clip scale factor. Size ratios are computed on that UniDepth depth before the scale is applied.

The video model is the stronger temporal prior, and it was the other candidate. It does not already live in meters, so one multiplier would not make it metric depth.
