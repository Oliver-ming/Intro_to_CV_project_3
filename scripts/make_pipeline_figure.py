"""Draw the three-part pipeline diagram into paper/figs/pipeline.pdf."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "paper" / "figs"

BOXES = [
    # x, y, w, h, text, colour
    (0.02, 0.70, 0.20, 0.22, "Middlebury trainQ\nstereo pair + calib.txt", "#dbe9f6"),
    (0.30, 0.70, 0.22, 0.22, "StereoSGBM\n+ left-right check", "#fde2c8"),
    (0.60, 0.70, 0.36, 0.22, "Z = f B / (d + doffs)\nmetric depth, |d error| px", "#e3f1dc"),
    (0.02, 0.38, 0.20, 0.22, "NYU v2 RGB\ntrain 795 / test 654", "#dbe9f6"),
    (0.30, 0.38, 0.22, 0.22, "UniDepthV2-Small\nRGB-only metric", "#fde2c8"),
    (0.60, 0.38, 0.17, 0.22, "fit s on train\n(frozen)", "#f6d5e5"),
    (0.81, 0.38, 0.15, 0.22, "s x depth\nAbsRel/RMSE/d1", "#e3f1dc"),
    (0.02, 0.06, 0.20, 0.22, "davis_rollercoaster\n70 frames", "#dbe9f6"),
    (0.30, 0.06, 0.22, 0.22, "DAv2-Small per frame\nVDA-Small offline", "#fde2c8"),
    (0.60, 0.06, 0.36, 0.22, "fixed clip normalization\nside-by-side + depth blur", "#e3f1dc"),
]
ARROWS = [(0.22, 0.81, 0.30), (0.52, 0.81, 0.60), (0.22, 0.49, 0.30), (0.52, 0.49, 0.60),
          (0.77, 0.49, 0.81), (0.22, 0.17, 0.30), (0.52, 0.17, 0.60)]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    for x, y, w, h, text, colour in BOXES:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01", fc=colour, ec="#444", lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7)
    for x0, y, x1 in ARROWS:
        ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="->", lw=0.9))
    for y, name in ((0.95, "Part 1: stereo geometry"), (0.63, "Part 2 + Part 3 (B): metric depth"), (0.31, "Parts 1-2: video (qualitative) + application")):
        ax.text(0.02, y, name, fontsize=7.5, weight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.tight_layout(pad=0.1)
    fig.savefig(OUT / "pipeline.pdf"); fig.savefig(OUT / "pipeline.png", dpi=200)
    temporal_plot()


def temporal_plot() -> None:
    """Optional diagnostic plot from outputs/video/temporal_change.csv."""
    import numpy as np
    path = OUT.parents[1] / "outputs" / "video" / "temporal_change.csv"
    t, a, b = np.loadtxt(path, delimiter=",", skiprows=1, unpack=True)
    fig, ax = plt.subplots(figsize=(3.3, 1.6))
    ax.plot(t, a, lw=1, label="DAv2-Small (frame-wise)")
    ax.plot(t, b, lw=1, label="VDA-Small (video)")
    ax.set_xlabel("frame t", fontsize=7); ax.set_ylabel("flow-warped |change|", fontsize=7)
    ax.tick_params(labelsize=6); ax.legend(fontsize=6, frameon=False)
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "temporal_change.pdf")


if __name__ == "__main__":
    main()
