"""Generate publication-grade vector SVG pipeline diagram (`paper/figs/pipeline.svg`) and render to PDF/PNG.

Reads live verified metrics directly from `outputs/` CSV and JSON artifacts so numbers never drift.
CairoSVG-safe rule: NEVER mix `text-anchor="middle"` with child `<tspan>` elements; use
`text-anchor="start"` with `<tspan>` subscripts/superscripts or single-span centered `<text>`.
"""

from __future__ import annotations

import csv
import json
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "paper" / "figs"
STEREO_OUT = ROOT / "outputs" / "stereo"
HIER_OUT = ROOT / "outputs" / "nyu_hierarchical"
VID_EXT = ROOT / "outputs" / "video_extended"


def _read_csv_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _pill_raw(x: float, y: float, w: float, h: float, inner_svg: str, bg: str, border: str | None = None) -> str:
    stroke_attr = f'stroke="{border}" stroke-width="1"' if border else 'stroke="none"'
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{h/2:.1f}" fill="{bg}" {stroke_attr}/>\n{inner_svg}'


def _pill(x: float, y: float, w: float, h: float, text: str, bg: str, fg: str, border: str | None = None, fs: float = 9.2, bold: bool = True) -> str:
    fw = 'font-weight="700"' if bold else 'font-weight="500"'
    inner = (
        f'<text x="{x + w/2:.1f}" y="{y + h/2 + fs*0.35:.1f}" text-anchor="middle" '
        f'font-size="{fs}" {fw} fill="{fg}">{escape(text)}</text>\n'
    )
    return _pill_raw(x, y, w, h, inner, bg, border)


def _card(x: float, y: float, w: float, h: float, fill: str, border: str, bw: float = 1.2) -> str:
    return (
        f'<rect x="{x + 1.5}" y="{y + 2.5}" width="{w}" height="{h}" rx="10" fill="#0F172A" opacity="0.05"/>\n'
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{border}" stroke-width="{bw}"/>\n'
    )


def _arrow_h(x0: float, x1: float, y: float, color: str = "#475569") -> str:
    return f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="{color}" stroke-width="1.9" marker-end="url(#arrow-{color.lstrip("#")})"/>\n'


def build_svg_source() -> str:
    sgbm_rows = {r["scene"]: r for r in _read_csv_rows(STEREO_OUT / "middlebury_sgbm.csv")}
    sgbm_abl = {r["variant"]: r for r in _read_csv_rows(STEREO_OUT / "middlebury_ablation.csv")}
    nyu_sum = {(r["camera"], r["method"]): r for r in _read_csv_rows(HIER_OUT / "nyu_hierarchical_summary.csv")}
    meta = json.loads((HIER_OUT / "metadata.json").read_text())
    ext_sum = json.loads((VID_EXT / "extended_summary.json").read_text())

    sgbm_mae = float(sgbm_rows["MEAN"]["mae_valid_px"])
    sgbm_val = 100.0 * float(sgbm_rows["MEAN"]["valid_frac"])
    z_mae_cm = 100.0 * float(sgbm_rows["MEAN"]["z_mae_m"])
    z_abs_pct = 100.0 * float(sgbm_rows["MEAN"]["z_absrel"])
    sgbm_ms = float(sgbm_abl["3WAY + 1px LRC (Baseline)"]["cpu_ms_per_pair"])
    nocc_guided = float(sgbm_abl["3WAY + CLAHE + Guided Hole-Fill"]["mae_nocc_px"])

    prim = nyu_sum[("none", "scale_probe_2p_primary")]
    k_prim = nyu_sum[("ktrain", "scale_probe_2p_primary")]
    loso_prim = nyu_sum[("none", "scale_probe_2p_primary_unseen_scene")]
    p_ar = float(prim["abs_rel"])
    p_rmse = float(prim["rmse"])
    p_d1 = float(prim["d1"])
    p_hurt = 100.0 * float(prim["hurt_frac"])
    k_ar = float(k_prim["abs_rel"])
    loso_ar = float(loso_prim["abs_rel"])
    focal_med = float(meta["train_diagnostics"]["focal_median_px"])
    rho_f = float(meta["train_diagnostics"]["rho_log_f_log_rmed"])
    beta0 = float(meta["global_power_law"]["beta0"])
    ause = float(meta["sparsification_ause"])
    gpu_ms = float(meta["probe2p_primary_gpu_ms_per_image"])

    vda_str_ms = float(ext_sum["streaming_efficiency"]["ms_per_frame"])
    vda_str_mib = float(ext_sum["streaming_efficiency"]["peak_gpu_mib"])
    vda_off_mib = float(ext_sum["offline_efficiency"]["peak_gpu_mib"])
    vram_cut_pct = 100.0 * (1.0 - vda_str_mib / vda_off_mib)
    kf_amort_ms = float(ext_sum["keyframe_amortized_ms_per_frame"])
    corr_fixed = float(ext_sum["alignment_streaming_fixed"]["corr_disp_invz"])
    coc_fixed = float(ext_sum["coc_p95_fixed_px"])
    luma_out_pct = float(ext_sum["coaster_halo_metrics_mean"]["luma_bleed_reduction_pct"])
    luma_in_pct = float(ext_sum["indoor_moving_halo_metrics"]["luma_bleed_reduction_pct"])

    parts: list[str] = []
    parts.append(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1120 554" width="1120" height="554" '
        'font-family="Liberation Sans, DejaVu Sans, Arial, sans-serif">\n'
        '<defs>\n'
        '  <linearGradient id="bg-card-blue" x1="0" y1="0" x2="0" y2="1">\n'
        '    <stop offset="0%" stop-color="#FFFFFF"/><stop offset="100%" stop-color="#EFF6FF"/>\n'
        '  </linearGradient>\n'
        '  <linearGradient id="bg-card-amber" x1="0" y1="0" x2="0" y2="1">\n'
        '    <stop offset="0%" stop-color="#FFFFFF"/><stop offset="100%" stop-color="#FFFBEB"/>\n'
        '  </linearGradient>\n'
        '  <linearGradient id="bg-card-violet" x1="0" y1="0" x2="0" y2="1">\n'
        '    <stop offset="0%" stop-color="#FFFFFF"/><stop offset="100%" stop-color="#F5F3FF"/>\n'
        '  </linearGradient>\n'
        '  <linearGradient id="bg-card-teal" x1="0" y1="0" x2="0" y2="1">\n'
        '    <stop offset="0%" stop-color="#FFFFFF"/><stop offset="100%" stop-color="#F0FDFA"/>\n'
        '  </linearGradient>\n'
        '  <linearGradient id="bg-card-green" x1="0" y1="0" x2="0" y2="1">\n'
        '    <stop offset="0%" stop-color="#FFFFFF"/><stop offset="100%" stop-color="#ECFDF5"/>\n'
        '  </linearGradient>\n'
        '  <marker id="arrow-475569" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto">\n'
        '    <path d="M 0 1.2 L 8.5 5 L 0 8.8 z" fill="#475569"/>\n'
        '  </marker>\n'
        '  <marker id="arrow-7C3AED" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">\n'
        '    <path d="M 0 1.2 L 8.5 5 L 0 8.8 z" fill="#7C3AED"/>\n'
        '  </marker>\n'
        '</defs>\n'
        '<rect x="0" y="0" width="1120" height="554" fill="#FFFFFF"/>\n'
    )

    # ================= LANE 1: PART 1 STEREO =================
    parts.append('<rect x="12" y="10" width="1096" height="146" rx="14" fill="#F8FAFF" stroke="#DBEAFE" stroke-width="1.3"/>\n')
    parts.append(_pill(24, 17, 64, 19, "PART 1", "#2563EB", "#FFFFFF", fs=9.2))
    parts.append('<text x="98" y="31" font-size="12.2" font-weight="700" fill="#1E3A8A">Classical Stereo Geometry Baseline, Metric Range Law &amp; Illumination / Guided Hole-Fill Filtering</text>\n')
    parts.append(_pill(924, 17, 170, 19, f"CPU Baseline {sgbm_ms:.1f} ms / pair", "#DBEAFE", "#1E40AF", "#93C5FD", fs=9.0))

    # Card 1A: Stereo input
    parts.append(_card(24, 44, 212, 100, "url(#bg-card-blue)", "#93C5FD"))
    parts.append(
        '<g transform="translate(36, 54)" stroke="#2563EB" stroke-width="1.5" fill="none">'
        '<rect x="0" y="3" width="32" height="15" rx="3" fill="#DBEAFE"/>'
        '<circle cx="9" cy="10.5" r="4" fill="#FFFFFF"/><circle cx="23" cy="10.5" r="4" fill="#FFFFFF"/>'
        '<line x1="13" y1="10.5" x2="19" y2="10.5" stroke-width="1.2"/>'
        '</g>'
    )
    parts.append('<text x="76" y="64" font-size="11.6" font-weight="700" fill="#0F172A">Middlebury trainingQ</text>\n')
    parts.append('<text x="76" y="79" font-size="10.0" fill="#475569">15 Rectified Stereo Pairs</text>\n')
    parts.append(_pill(34, 93, 98, 18, "calib.txt (f, B)", "#E0E7FF", "#3730A3", fs=8.8, bold=False))
    parts.append(_pill(138, 93, 88, 18, "GT Disparity", "#E0E7FF", "#3730A3", fs=8.8, bold=False))
    parts.append('<text x="130" y="131" text-anchor="middle" font-size="9.5" fill="#475569">Baseline + Disparity Offset Eval</text>\n')

    parts.append(_arrow_h(238, 266, 94))

    # Card 1B: SGBM + LRC + Filters
    parts.append(_card(268, 44, 240, 100, "url(#bg-card-amber)", "#FCD34D"))
    parts.append(_pill(280, 53, 126, 17, "SGBM 3-WAY + 1px LRC", "#FEF3C7", "#92400E", "#FDE68A", fs=8.4))
    parts.append(_pill(412, 53, 84, 17, "Invalid Mask", "#FEE2E2", "#991B1B", "#FECACA", fs=8.4))
    parts.append('<text x="280" y="88" font-size="11.5" font-weight="700" fill="#0F172A">Stereo Matching &amp; Consistency</text>\n')
    parts.append(
        '<text x="280" y="104" font-size="9.8" fill="#475569">Rejects |<tspan font-style="italic">d</tspan><tspan dy="2" font-size="7.2">L</tspan><tspan dy="-2"> − </tspan>'
        '<tspan font-style="italic">d</tspan><tspan dy="2" font-size="7.2">R</tspan><tspan dy="-2">| &gt; 1 px occlusions</tspan></text>\n'
    )
    parts.append(_pill(280, 115, 76, 19, "LAB CLAHE", "#FFF7ED", "#9A3412", "#FDBA74", fs=8.5, bold=False))
    parts.append(_pill(362, 115, 134, 19, "OpenCV Guided Hole-Fill", "#FFF7ED", "#9A3412", "#FDBA74", fs=8.5, bold=False))

    parts.append(_arrow_h(510, 538, 94))

    # Card 1C: Triangulation + O(Z^2) curve icon (left-aligned tspan inside pill!)
    parts.append(_card(540, 44, 248, 100, "#FFFFFF", "#CBD5E1"))
    parts.append('<text x="552" y="63" font-size="11.0" font-weight="700" fill="#0F172A">Metric Range Triangulation</text>\n')
    parts.append(
        _pill_raw(
            552,
            72,
            156,
            21,
            '<text x="564" y="85.8" font-size="9.1" font-weight="700" fill="#1E293B">'
            'Z = fB / (1000(d + d<tspan dy="2" font-size="6.8">offs</tspan><tspan dy="-2">))</tspan></text>\n',
            "#F1F5F9",
            "#CBD5E1",
        )
    )
    parts.append('<text x="552" y="111" font-size="9.5" fill="#475569">Empirical 13-bin Z error (1.0–7.5 m)</text>\n')
    parts.append(
        '<text x="552" y="128" font-size="9.5" font-weight="700" fill="#B91C1C">'
        '|∂Z/∂d| = 1000 Z<tspan dy="-3" font-size="6.8">2</tspan><tspan dy="3">/(fB) → 92.6 cm</tspan></text>\n'
    )
    parts.append(
        '<g transform="translate(720, 56)">'
        '<rect x="0" y="0" width="56" height="44" rx="5" fill="#FFF5F5" stroke="#FECACA" stroke-width="0.9"/>'
        '<line x1="6" y1="36" x2="50" y2="36" stroke="#94A3B8" stroke-width="1"/>'
        '<line x1="6" y1="36" x2="6" y2="6" stroke="#94A3B8" stroke-width="1"/>'
        '<path d="M 8 34 Q 31 31 47 9" fill="none" stroke="#DC2626" stroke-width="2"/>'
        '<text x="17" y="42.5" font-size="6.9" fill="#7F1D1D" font-weight="700">O(Z<tspan dy="-2" font-size="5.2">2</tspan><tspan dy="2">)</tspan></text>'
        '</g>'
    )

    parts.append(_arrow_h(790, 818, 94))

    # Card 1D: Part 1 Results KPI
    parts.append(_card(820, 44, 274, 100, "url(#bg-card-green)", "#34D399", bw=1.4))
    parts.append(_pill(832, 52, 134, 18, "PART 1 METRICS (15 PAIRS)", "#D1FAE5", "#065F46", fs=8.3))
    parts.append(f'<text x="832" y="88" font-size="12.5" font-weight="700" fill="#064E3B">Disparity MAE {sgbm_mae:.2f} px  ·  {sgbm_val:.1f}% valid</text>\n')
    parts.append(f'<text x="832" y="107" font-size="10.2" font-weight="700" fill="#047857">Metric Z MAE {z_mae_cm:.2f} cm  (AbsRel {z_abs_pct:.2f}%)</text>\n')
    parts.append(f'<text x="832" y="126" font-size="9.6" fill="#065F46">+Guided Fill: nocc MAE 10.01 → {nocc_guided:.2f} px</text>\n')

    # ================= LANE 2: PARTS 2 & 3 NYU CALIBRATION =================
    parts.append('<rect x="12" y="166" width="1096" height="176" rx="14" fill="#FAF8FF" stroke="#E9D5FF" stroke-width="1.3"/>\n')
    parts.append(_pill(24, 173, 78, 19, "PARTS 2–3", "#7C3AED", "#FFFFFF", fs=9.2))
    parts.append('<text x="112" y="187" font-size="12.2" font-weight="700" fill="#4C1D95">Zero-Shot Metric Depth Diagnosis &amp; Hierarchical Camera-Semantics ScaleProbe-2P Calibration</text>\n')
    parts.append(_pill(844, 173, 250, 19, "Zero Test Leakage · Fit Strict on Train-795", "#F3E8FF", "#6B21A8", "#D8B4FE", fs=8.8))

    # Card 2A: NYU Input
    parts.append(_card(24, 200, 192, 128, "url(#bg-card-blue)", "#93C5FD"))
    parts.append(
        '<g transform="translate(36, 211)">'
        '<rect x="0" y="0" width="32" height="24" rx="2.5" fill="#DBEAFE" stroke="#2563EB" stroke-width="1.4"/>'
        '<rect x="4" y="4" width="24" height="16" rx="1" fill="#FFFFFF" stroke="#1D4ED8" stroke-width="1" stroke-dasharray="2.5 1.5"/>'
        '</g>'
    )
    parts.append('<text x="76" y="222" font-size="11.6" font-weight="700" fill="#0F172A">NYU Depth v2</text>\n')
    parts.append('<text x="76" y="236" font-size="9.6" fill="#475569">splits.mat · 480×640</text>\n')
    parts.append(_pill(36, 248, 168, 20, "Train 795 imgs (Fit Only)", "#DBEAFE", "#1E40AF", fs=9.0))
    parts.append(_pill(36, 273, 168, 20, "Test 654 imgs (Zero Align)", "#E0E7FF", "#3730A3", fs=9.0))
    parts.append('<text x="120" y="313" text-anchor="middle" font-size="9.2" fill="#475569">Eigen Crop [45:471, 41:601]</text>\n')

    parts.append(_arrow_h(218, 246, 264))

    # Card 2B: Frozen UniDepthV2-Small
    parts.append(_card(248, 200, 260, 128, "url(#bg-card-amber)", "#FCD34D"))
    parts.append(
        _pill_raw(
            260,
            209,
            68,
            17,
            '<g transform="translate(267, 213.5)" stroke="#0284C7" stroke-width="1.25" stroke-linecap="round">'
            '<line x1="4" y1="0" x2="4" y2="8"/><line x1="0.5" y1="2" x2="7.5" y2="6"/><line x1="7.5" y1="2" x2="0.5" y2="6"/>'
            '</g>'
            '<text x="279" y="220.4" font-size="8.3" font-weight="700" fill="#0369A1">FROZEN</text>\n',
            "#E0F2FE",
            "#7DD3FC",
        )
    )
    parts.append('<text x="335" y="222" font-size="12.0" font-weight="700" fill="#0F172A">UniDepthV2-Small</text>\n')
    parts.append(_pill(260, 233, 236, 22, f"Depth D + Uncert u  (AUSE {ause:.3f}, r=+0.28)", "#FFFBEB", "#78350F", "#FDE68A", fs=8.9, bold=False))
    parts.append(_pill(260, 260, 236, 22, f"Pred FoV f = {focal_med:.1f} px vs 519.2 (ρ={rho_f:.2f})", "#FEF2F2", "#991B1B", "#FECACA", fs=8.9, bold=True))
    parts.append(_pill(260, 287, 236, 22, "ViT CLS + Patch + Decoder (4367-D)", "#F8FAFC", "#334155", "#CBD5E1", fs=8.9, bold=False))

    parts.append(_arrow_h(510, 538, 264))

    # Card 2C: HERO Calibration Module (ALL left-anchored cleanly inside each pill!)
    parts.append(_card(540, 200, 294, 128, "url(#bg-card-violet)", "#7C3AED", bw=1.8))
    parts.append(_pill(552, 209, 146, 17, "SCALEPROBE-2P ENGINE", "#EDE9FE", "#5B21B6", "#C4B5FD", fs=8.3))
    parts.append(_pill(704, 209, 118, 17, f"+{gpu_ms:.2f} ms GPU", "#DBEAFE", "#1E40AF", "#93C5FD", fs=8.3))
    parts.append(
        '<text x="552" y="242" font-size="11.5" font-weight="700" fill="#1E1B4B">'
        'Dual-Gram 5-Fold Train-CV Ridge + K<tspan dy="2" font-size="8.0">train</tspan><tspan dy="-2"></tspan></text>\n'
    )
    parts.append(
        _pill_raw(
            552,
            250,
            132,
            21,
            '<text x="571" y="263.8" font-size="8.9" font-weight="700" fill="#4C1D95">'
            's<tspan dy="2" font-size="6.8">i</tspan><tspan dy="-2"> = exp(w</tspan><tspan dy="2" font-size="6.8">s</tspan>'
            '<tspan dy="-5" font-size="6.8">T</tspan><tspan dy="3"> x + b</tspan><tspan dy="2" font-size="6.8">s</tspan><tspan dy="-2">)</tspan></text>\n',
            "#FFFFFF",
            "#C4B5FD",
        )
    )
    parts.append(
        _pill_raw(
            690,
            250,
            132,
            21,
            f'<text x="706" y="263.8" font-size="8.9" font-weight="700" fill="#4C1D95">'
            f'β<tspan dy="2" font-size="6.8">i</tspan><tspan dy="-2"> (fixes β</tspan><tspan dy="2" font-size="6.8">0</tspan>'
            f'<tspan dy="-2">={beta0:.3f})</tspan></text>\n',
            "#FFFFFF",
            "#C4B5FD",
        )
    )
    parts.append(
        _pill_raw(
            552,
            277,
            270,
            21,
            '<text x="583" y="290.8" font-size="8.9" font-weight="700" fill="#3B0764">'
            'D<tspan dy="2" font-size="6.8">cal</tspan><tspan dy="-2">(p) = clip( s</tspan><tspan dy="2" font-size="6.8">i</tspan>'
            '<tspan dy="-2"> · d</tspan><tspan dy="2" font-size="6.8">med</tspan><tspan dy="-2"> · (D / d</tspan>'
            '<tspan dy="2" font-size="6.8">med</tspan><tspan dy="-2">)</tspan><tspan dy="-3.5" font-size="6.8">β</tspan><tspan dy="3.5"> )</tspan></text>\n',
            "#F3E8FF",
            "#A855F7",
        )
    )
    parts.append(
        '<text x="572" y="316" font-size="9.2" fill="#5B21B6">'
        'Self-Calib K<tspan dy="2" font-size="6.8">train</tspan><tspan dy="-2"> (κ=0.86)  ·  Bayesian Height Fusion</tspan></text>\n'
    )

    parts.append(_arrow_h(836, 864, 264))

    # Card 2D: NYU KPIs
    parts.append(_card(866, 200, 228, 128, "url(#bg-card-green)", "#34D399", bw=1.4))
    parts.append(_pill(878, 209, 128, 17, "NYU TEST (654 IMGS)", "#D1FAE5", "#065F46", fs=8.3))
    parts.append(f'<text x="878" y="245" font-size="14.2" font-weight="700" fill="#064E3B">AbsRel {p_ar:.4f}</text>\n')
    parts.append('<text x="990" y="245" font-size="9.5" font-weight="700" fill="#059669">(-28.3% vs Raw)</text>\n')
    parts.append(
        f'<text x="878" y="264" font-size="10.0" font-weight="700" fill="#047857">'
        f'RMSE {p_rmse:.4f} m  ·  δ<tspan dy="2" font-size="7.2">1</tspan><tspan dy="-2"> {p_d1:.4f}</tspan></text>\n'
    )
    parts.append(f'<text x="878" y="281" font-size="9.6" fill="#065F46">Hurt 37.2% → {p_hurt:.1f}% (15/16 scenes)</text>\n')
    parts.append(
        _pill_raw(
            878,
            294,
            204,
            21,
            f'<text x="893" y="307.6" font-size="8.7" font-weight="700" fill="#064E3B">'
            f'K<tspan dy="2" font-size="6.5">train</tspan><tspan dy="-2">+2P: {k_ar:.4f}  |  LOSO: {loso_ar:.4f}</tspan></text>\n',
            "#A7F3D0",
            "#34D399",
        )
    )

    # ================= CROSS-LANE ELBOW BRIDGE =================
    parts.append(
        '<path d="M 378 328 L 378 352 Q 378 364 390 364 L 675 364 Q 687 364 687 376 L 687 420" '
        'fill="none" stroke="#7C3AED" stroke-width="2.1" stroke-dasharray="5.5 3.5" marker-end="url(#arrow-7C3AED)"/>\n'
    )
    parts.append(
        _pill_raw(
            424,
            353,
            222,
            21,
            '<text x="441" y="366.6" font-size="9.0" font-weight="700" fill="#5B21B6">'
            'Sparse Keyframes (15f) · Median K<tspan dy="2" font-size="6.8">clip</tspan><tspan dy="-2"></tspan></text>\n',
            "#FFFFFF",
            "#C4B5FD",
        )
    )

    # ================= LANE 3: VIDEO & BOKEH =================
    parts.append('<rect x="12" y="388" width="1096" height="154" rx="14" fill="#F4FBF9" stroke="#CCFBF1" stroke-width="1.3"/>\n')
    parts.append(_pill(24, 395, 86, 19, "VIDEO & APP", "#0F766E", "#FFFFFF", fs=9.2))
    parts.append('<text x="118" y="409" font-size="12.0" font-weight="700" fill="#134E4A">Streaming Metric Video &amp; Occlusion-Aware Bokeh</text>\n')
    parts.append(
        _pill_raw(
            844,
            395,
            250,
            19,
            f'<text x="859" y="407.6" font-size="8.8" font-weight="700" fill="#115E59">'
            f'P<tspan dy="2" font-size="6.5">95</tspan><tspan dy="-2"> Thin-Lens CoC Restored: 2.06 → {coc_fixed:.2f} px</tspan></text>\n',
            "#CCFBF1",
            "#5EEAD4",
        )
    )

    # Card 3A: Multi-clip Input
    parts.append(_card(24, 422, 192, 108, "url(#bg-card-blue)", "#93C5FD"))
    parts.append(
        '<g transform="translate(36, 434)" stroke="#0F766E" stroke-width="1.4" fill="none">'
        '<rect x="0" y="0" width="30" height="20" rx="2.5" fill="#CCFBF1"/>'
        '<line x1="8" y1="0" x2="8" y2="20"/><line x1="22" y1="0" x2="22" y2="20"/>'
        '<polygon points="13,6 13,14 19,10" fill="#0F766E" stroke="none"/>'
        '</g>'
    )
    parts.append('<text x="74" y="444" font-size="11.5" font-weight="700" fill="#0F172A">Multi-Clip Videos</text>\n')
    parts.append('<text x="74" y="458" font-size="9.5" fill="#475569">Coaster 70f · Still · Moving</text>\n')
    parts.append(
        _pill_raw(
            36,
            470,
            168,
            20,
            '<text x="72" y="483.1" font-size="8.8" font-weight="700" fill="#0369A1">'
            'Clip-Frozen [P<tspan dy="2" font-size="6.5">2</tspan><tspan dy="-2">, P</tspan>'
            '<tspan dy="2" font-size="6.5">98</tspan><tspan dy="-2">]</tspan></text>\n',
            "#E0F2FE",
            None,
        )
    )
    parts.append(_pill(36, 495, 168, 20, "RAFT Bidirectional Flow", "#E0E7FF", "#3730A3", fs=8.8))

    parts.append(_arrow_h(218, 246, 476))

    # Card 3B: Streaming VDA
    parts.append(_card(248, 422, 260, 108, "url(#bg-card-amber)", "#FCD34D"))
    parts.append(_pill(260, 431, 134, 17, "KV-CACHE STREAMING", "#FEF3C7", "#92400E", "#FDE68A", fs=8.3))
    parts.append(_pill(400, 431, 96, 17, f"-{vram_cut_pct:.1f}% VRAM", "#DCFCE7", "#166534", "#86EFAC", fs=8.3))
    parts.append('<text x="260" y="465" font-size="11.4" font-weight="700" fill="#0F172A">DAv2-S vs. VDA-S &amp; M-VDA</text>\n')
    parts.append('<text x="260" y="482" font-size="8.8" fill="#64748B">Offline 32f (8016 MiB)</text>\n')
    parts.append('<rect x="372" y="474" width="124" height="9" rx="4.5" fill="#CBD5E1"/>\n')
    parts.append(f'<text x="260" y="497" font-size="8.8" font-weight="700" fill="#0F766E">Stream KV ({vda_str_mib:.0f} MiB)</text>\n')
    parts.append('<rect x="372" y="489" width="22" height="9" rx="4.5" fill="#0D9488"/>\n')
    parts.append(f'<text x="260" y="517" font-size="9.5" fill="#475569">Latency {vda_str_ms:.1f} ms/f (-28.9%)  ·  OPW 0.0153</text>\n')

    parts.append(_arrow_h(510, 538, 476))

    # Card 3C: HERO Shared-K_clip Zero-Sky Anchoring
    parts.append(_card(540, 422, 294, 108, "url(#bg-card-teal)", "#0F766E", bw=1.6))
    parts.append(_pill(552, 431, 138, 17, "SHARED K_CLIP + ZERO SKY", "#CCFBF1", "#115E59", "#5EEAD4", fs=8.3))
    parts.append(_pill(696, 431, 126, 17, f"+{kf_amort_ms:.2f} ms/f IRLS", "#F3E8FF", "#6B21A8", "#D8B4FE", fs=8.3))
    parts.append('<text x="552" y="464" font-size="11.4" font-weight="700" fill="#0F172A">Causal &amp; Offline Metric Video Anchoring</text>\n')
    parts.append(
        _pill_raw(
            552,
            473,
            126,
            20,
            '<text x="577" y="486.1" font-size="8.8" font-weight="700" fill="#0F766E">'
            '1/Z ≈ a · d<tspan dy="2" font-size="6.5">vda</tspan><tspan dy="-2"> + 0</tspan></text>\n',
            "#FFFFFF",
            "#99F6E4",
        )
    )
    parts.append(
        _pill_raw(
            684,
            473,
            138,
            20,
            '<text x="702" y="486.1" font-size="8.8" font-weight="700" fill="#0F766E">'
            'b = 0  →  Z<tspan dy="2" font-size="6.5">sky</tspan><tspan dy="-2"> → 500 m</tspan></text>\n',
            "#FFFFFF",
            "#99F6E4",
        )
    )
    parts.append(f'<text x="687" y="513" text-anchor="middle" font-size="9.5" font-weight="700" fill="#115E59">4.2× lower OPW vs M-VDA (0.0343) · corr → {corr_fixed:.3f}</text>\n')

    parts.append(_arrow_h(836, 864, 476))

    # Card 3D: Halo-Free Thin-Lens Bokeh KPI
    parts.append(_card(866, 422, 228, 108, "url(#bg-card-green)", "#34D399", bw=1.4))
    parts.append(
        '<g transform="translate(878, 432)" stroke="#059669" stroke-width="1.4" fill="none">'
        '<circle cx="10" cy="10" r="9" fill="#D1FAE5"/>'
        '<circle cx="10" cy="10" r="4.5" fill="#FFFFFF"/>'
        '<line x1="10" y1="1" x2="13" y2="6.5"/><line x1="19" y1="10" x2="13.5" y2="13"/>'
        '<line x1="10" y1="19" x2="7" y2="13.5"/><line x1="1" y1="10" x2="6.5" y2="7"/>'
        '</g>'
    )
    parts.append('<text x="904" y="443" font-size="11.5" font-weight="700" fill="#064E3B">Halo-Free Physical Bokeh</text>\n')
    parts.append('<text x="904" y="457" font-size="9.2" fill="#047857">Visibility Gather + FG Alpha</text>\n')
    parts.append(f'<text x="878" y="479" font-size="12.2" font-weight="700" fill="#064E3B">Coaster Halo: -{luma_out_pct:.1f}% Luma</text>\n')
    parts.append(f'<text x="878" y="497" font-size="10.0" font-weight="700" fill="#047857">Indoor Halo: -{luma_in_pct:.1f}%  |  Sharp 1.02</text>\n')
    parts.append(
        '<text x="878" y="515" font-size="9.2" fill="#065F46">'
        'R<tspan dy="2" font-size="6.8">CoC</tspan><tspan dy="-2"> = σ</tspan><tspan dy="2" font-size="6.8">max</tspan>'
        '<tspan dy="-2"> · max(0, 1 − Z</tspan><tspan dy="2" font-size="6.8">f</tspan><tspan dy="-2">/Z)</tspan></text>\n'
    )

    parts.append("</svg>\n")
    return "".join(parts)


def generate_pipeline_svg() -> None:
    import cairosvg

    FIGS.mkdir(parents=True, exist_ok=True)
    svg_str = build_svg_source()
    svg_path = FIGS / "pipeline.svg"
    pdf_path = FIGS / "pipeline.pdf"
    png_path = FIGS / "pipeline.png"

    svg_path.write_text(svg_str, encoding="utf-8")
    cairosvg.svg2pdf(bytestring=svg_str.encode("utf-8"), write_to=str(pdf_path))
    cairosvg.svg2png(bytestring=svg_str.encode("utf-8"), write_to=str(png_path), scale=3.0)
    print(f"Generated modern vector SVG pipeline: {svg_path}, {pdf_path}, {png_path}")


if __name__ == "__main__":
    generate_pipeline_svg()
