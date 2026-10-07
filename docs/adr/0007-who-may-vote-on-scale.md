# Only complete people and cars vote on scale

A truncated box shortens the vertical extent and biases the size ratio. People vote only when the head and both ankles are visible, following ScaleNet. Cars vote on height only when the box does not touch the image border. The 0.5 detector score is a sensitivity setting, not a published metrology cut.

Letting every person and car vote would add more frames. It would also treat a cropped torso or a car cut by the frame edge as a full height.
