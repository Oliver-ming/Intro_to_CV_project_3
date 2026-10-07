# The noisy size prior is a fixed one-sigma shift

ScaleNet publishes the spread of the priors: 0.09 m for adult height and 0.21 m for car height. The robustness check shifts both priors by that amount, once up and once down, and recomputes the clip scale. Obstacle AbsRel and collision F1 are compared with the unshifted system.

A random draw from those Gaussians was rejected so the table does not depend on a seed.
