# Implied size and the point cloud share the camera UniDepth was given

UniDepth's metric depth is conditioned on a camera. When NYUv2, KITTI, or TUM publishes intrinsics, that camera is passed in at inference and then used both to turn box height into meters and to lift the point cloud. A phone video has no published camera, so UniDepth's predicted intrinsics are used for both.

Substituting a dataset focal length only in the size equation, after depth was predicted with another camera, was rejected. The depth and the focal length would no longer describe the same projection.
