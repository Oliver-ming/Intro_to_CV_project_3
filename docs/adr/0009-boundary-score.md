# Boundary quality is a scale-invariant F1 on dense depth only

The course asks for a boundary number, and this project does not add an edge-refinement method. The score is Depth Pro's scale-invariant boundary F1, so a metric map and a relative map can be compared. It is computed on NYUv2 and TUM, where depth is dense. KITTI's sparse LiDAR is shown qualitatively, because automatic edges on sparse returns do not sit on the true discontinuity.

A 2-pixel matching radius and iBims chamfer were set aside. The radius is not a shared standard, and the chamfer wants hand-labeled edges.
