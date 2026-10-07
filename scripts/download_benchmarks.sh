#!/usr/bin/env bash
# Official Spring 2027 course downloads. Data stays under data/ and is not committed.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
mb="$root/data/middlebury"
nyu="$root/data/nyu"
mkdir -p "$mb" "$nyu" "$root/data/clips"

curl -L --fail --continue-at - \
  -o "$mb/MiddEval3-data-Q.zip" \
  "https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-data-Q.zip"
curl -L --fail --continue-at - \
  -o "$mb/MiddEval3-GT0-Q.zip" \
  "https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-GT0-Q.zip"
curl -L --fail --continue-at - \
  -o "$nyu/nyu_depth_v2_labeled.mat" \
  "https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
