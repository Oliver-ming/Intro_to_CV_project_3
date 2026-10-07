# The additional domain is fr3/walking_xyz

Status: superseded by ADR 0013. The Spring 2027 brief dropped the extra public dataset.

The extra domain is one TUM RGB-D sequence so metric error and temporal error have ground truth. fr3/walking_xyz has a moving camera and walking people, which is what dynamic SLAM papers use, and the people are standing, so the height prior can vote. Sitting sequences and person-free sequences were rejected for that prior.

It is a SLAM sequence, not a clip from a video-depth benchmark. That is accepted for a single extra domain.
