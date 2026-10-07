# Cars vote with height, not width

The first prior was a car width of 1.80 m. Hoiem, ScaleNet, and FUMET use car height, about 1.59 m, because a 2D box's width mixes length and width as the car yaws. Person height stays 1.70 m. Both classes use vertical box extent, converted as pixel extent times depth over focal length.

Width remains a real-world statistic for new cars. It is the wrong prior for a box that is not known to be fronto-parallel.
