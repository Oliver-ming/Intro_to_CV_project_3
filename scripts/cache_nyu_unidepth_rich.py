"""Phase 0 & Step 2: Cache rich UniDepthV2-Small and DAv2-Small outputs on NYU train & test.

Strictly adheres to VRAM safety (< 1.5 GB peak, memory fraction capped at 0.20,
one model loaded at a time) and zero test-set leakage (RGB-only model features).
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "UniDepth"))
sys.path.insert(0, str(ROOT / "third_party" / "Depth-Anything-V2"))

from depth_anything_v2.dpt import DepthAnythingV2
from unidepth.models import UniDepthV2
from unidepth.utils.camera import Pinhole
from src.eval.nyu import HEIGHT, K_NYU, WIDTH, open_mat, rgb_at, split_ids

CACHE_BASE = ROOT / "outputs" / "nyu_cache"
OUT = ROOT / "outputs" / "nyu_cache_rich"
DAV2_CKPT = ROOT / "weights" / "depth_anything_v2_vits.pth"
VIDEO_CLIP = ROOT / "third_party" / "Video-Depth-Anything" / "assets" / "example_videos" / "davis_rollercoaster.mp4"


def _pool_spatial_depth_features(df: torch.Tensor, conf_map: torch.Tensor) -> torch.Tensor:
    """Pool (256, Hp, Wp) decoder depth_features into (8, 256):
    [global_mean, global_std, TL, TR, BL, BR, inv_unc_weighted_mean, top25_cert_mean].
    """
    _, hp, wp = df.shape
    u_low = F.interpolate(conf_map.view(1, 1, *conf_map.shape), size=(hp, wp), mode="bilinear", align_corners=False)[0, 0]
    u_low = u_low.clamp(min=1e-4, max=1e3)
    inv_w = (1.0 / (u_low + 1e-3))
    inv_w = inv_w / inv_w.sum().clamp(min=1e-6)
    q25 = torch.quantile(u_low.reshape(-1), 0.25)
    top25_mask = (u_low <= q25).float()
    top25_w = top25_mask / top25_mask.sum().clamp(min=1.0)

    df_flat = df.view(df.shape[0], -1)  # (256, Hp*Wp)
    g_mean = df.mean(dim=(1, 2))
    g_std = df.std(dim=(1, 2))
    tl = df[:, : hp // 2, : wp // 2].mean(dim=(1, 2))
    tr = df[:, : hp // 2, wp // 2 :].mean(dim=(1, 2))
    bl = df[:, hp // 2 :, : wp // 2].mean(dim=(1, 2))
    br = df[:, hp // 2 :, wp // 2 :].mean(dim=(1, 2))
    u_wmean = df_flat @ inv_w.reshape(-1)
    u_top25 = df_flat @ top25_w.reshape(-1)
    return torch.stack([g_mean, g_std, tl, tr, bl, br, u_wmean, u_top25], dim=0)


def run_unidepth_cache(data) -> None:
    print("[Cache 1/2] Loading UniDepthV2-Small...")
    model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vits14")
    model.interpolation_mode = "bilinear"
    model = model.to("cuda").eval()

    hook_buf: dict[str, object] = {}

    def encoder_hook(_mod, _inp, out):
        feats, toks = out
        hook_buf["tokens"] = toks
        hook_buf["features"] = feats

    def cta_hook(_mod, _inp, out):
        hook_buf["cta"] = out

    def agg2_hook(_mod, _inp, out):
        hook_buf["agg2"] = out

    h_enc = model.pixel_encoder.register_forward_hook(encoder_hook)
    h_cta = model.pixel_decoder.camera_token_adapter.register_forward_hook(cta_hook)
    h_agg = model.pixel_decoder.camera_layer.aggregate2.register_forward_hook(agg2_hook)

    ktrain_meta = json.loads((OUT / "ktrain_calibration.json").read_text())
    K_train = np.asarray(ktrain_meta["K_train"], dtype=np.float32)

    regimes = [
        ("none", "", None),
        ("ktrain", "ktrain_", K_train),
        ("nyu", "nyu_", K_NYU),
    ]

    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for cam_name, prefix, K_mat in regimes:
            cam_obj = Pinhole(K=torch.from_numpy(K_mat).unsqueeze(0)) if K_mat is not None else None
            for split in ("train", "test"):
                ids = split_ids(split)
                n = len(ids)
                intrinsics = np.zeros((n, 3, 3), dtype=np.float32)
                confidence = np.zeros((n, HEIGHT, WIDTH), dtype=np.float16)
                cls_tokens = np.zeros((n, 4, 384), dtype=np.float32)
                patch_summary = np.zeros((n, 6, 384), dtype=np.float32)
                depth_feat = np.zeros((n, 2, 256), dtype=np.float32)
                depth_feat_spatial = np.zeros((n, 8, 256), dtype=np.float32)
                cam_latent = np.zeros((n, 4, 256), dtype=np.float32)
                cam_adapter = np.zeros((n, 4, 256), dtype=np.float32)

                t0 = time.perf_counter()
                for step, index in enumerate(ids):
                    rgb = torch.from_numpy(rgb_at(data, index)).permute(2, 0, 1)
                    out = model.infer(rgb) if cam_obj is None else model.infer(rgb, cam_obj)

                    intrinsics[step] = out["intrinsics"][0].float().cpu().numpy()
                    conf_t = out["confidence"].squeeze().float()  # on GPU
                    conf_np = conf_t.cpu().numpy()
                    if conf_np.shape != (HEIGHT, WIDTH):
                        conf_np = cv2.resize(conf_np, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
                    confidence[step] = conf_np.astype(np.float16)

                    toks = [hook_buf["tokens"][idx].squeeze().float() for idx in (2, 5, 8, 11)]
                    cls_tokens[step] = torch.stack(toks, dim=0).cpu().numpy()

                    f_last = hook_buf["features"][-1][0].float()
                    hp, wp, _ = f_last.shape
                    patch_summary[step] = torch.stack(
                        [
                            f_last.mean(dim=(0, 1)),
                            f_last.std(dim=(0, 1)),
                            f_last[: hp // 2].mean(dim=(0, 1)),
                            f_last[hp // 2 :].mean(dim=(0, 1)),
                            f_last[:, : wp // 2].mean(dim=(0, 1)),
                            f_last[:, wp // 2 :].mean(dim=(0, 1)),
                        ],
                        dim=0,
                    ).cpu().numpy()

                    df = out["depth_features"][0].float()
                    sp_pool = _pool_spatial_depth_features(df, conf_t)
                    depth_feat_spatial[step] = sp_pool.cpu().numpy()
                    depth_feat[step] = sp_pool[:2].cpu().numpy()

                    cam_latent[step] = hook_buf["agg2"][0].float().cpu().numpy()
                    cam_adapter[step] = torch.stack([x.squeeze().float() for x in hook_buf["cta"]], dim=0).cpu().numpy()

                    if (step + 1) % 400 == 0 or step + 1 == n:
                        print(f"  [UniDepth {cam_name}/{split}] {step + 1}/{n} ({time.perf_counter() - t0:.1f}s)")

                np.save(OUT / f"intrinsics_{prefix}{split}.npy", intrinsics)
                np.save(OUT / f"confidence_{prefix}{split}.npy", confidence)
                np.save(OUT / f"cls_tokens_{prefix}{split}.npy", cls_tokens)
                np.save(OUT / f"patch_summary_{prefix}{split}.npy", patch_summary)
                np.save(OUT / f"depth_feat_{prefix}{split}.npy", depth_feat)
                np.save(OUT / f"depth_feat_spatial_{prefix}{split}.npy", depth_feat_spatial)
                np.save(OUT / f"cam_latent_{prefix}{split}.npy", cam_latent)
                np.save(OUT / f"cam_adapter_{prefix}{split}.npy", cam_adapter)

    print(f"  [UniDepth] peak VRAM: {torch.cuda.max_memory_allocated() / 2**20:.1f} MiB")
    h_enc.remove()
    h_cta.remove()
    h_agg.remove()
    del model
    gc.collect()
    torch.cuda.empty_cache()


def run_dav2_cache_and_benchmark(data) -> None:
    print("[Cache 2/2] Loading DepthAnythingV2-Small...")
    dav2 = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    dav2.load_state_dict(torch.load(DAV2_CKPT, map_location="cpu"))
    dav2 = dav2.to("cuda").eval()

    # 1. Benchmark DAv2-Small FP32 vs FP16 on 640x360 video frames
    cap = cv2.VideoCapture(str(VIDEO_CLIP))
    vid_frames = []
    while len(vid_frames) < 70:
        ok, bgr = cap.read()
        if not ok:
            break
        vid_frames.append(cv2.resize(bgr, (640, 360), interpolation=cv2.INTER_AREA))
    cap.release()

    eff = {"model": "DepthAnythingV2-Small (vits)", "input_size": 518, "frame_res": "640x360"}
    with torch.inference_mode():
        for dtype_name, use_fp16 in (("fp32", False), ("fp16", True)):
            # Warmup
            for f in vid_frames[:10]:
                with torch.amp.autocast("cuda", enabled=use_fp16, dtype=torch.float16):
                    _ = dav2.infer_image(f, input_size=518)
            torch.cuda.reset_peak_memory_stats()
            times = []
            for f in vid_frames:
                img_t, (h, w) = dav2.image2tensor(f, 518)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.amp.autocast("cuda", enabled=use_fp16, dtype=torch.float16):
                    d = dav2(img_t)
                    _ = F.interpolate(d[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]
                torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000.0)
            eff[f"{dtype_name}_ms_per_frame"] = float(np.mean(times))
            eff[f"{dtype_name}_peak_mib"] = float(torch.cuda.max_memory_allocated() / 2**20)
    (OUT / "efficiency_dav2_fp16.json").write_text(json.dumps(eff, indent=2))
    print("  DAv2 video efficiency:", eff)

    # 2. Extract DAv2 ViT-S CLS tokens + RGB-only cross-model agreement stats on NYU train & test
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for split in ("train", "test"):
            ids = split_ids(split)
            n = len(ids)
            uni_preds = np.load(CACHE_BASE / f"unidepth_none_{split}.npy").astype(np.float32)
            dav2_cls = np.zeros((n, 4, 384), dtype=np.float32)
            cross_stats = np.zeros((n, 9), dtype=np.float32)

            t0 = time.perf_counter()
            for step, index in enumerate(ids):
                bgr = cv2.cvtColor(rgb_at(data, index), cv2.COLOR_RGB2BGR)
                img_t, (h, w) = dav2.image2tensor(bgr, 518)
                ph, pw = img_t.shape[-2] // 14, img_t.shape[-1] // 14
                feats = dav2.pretrained.get_intermediate_layers(img_t, [2, 5, 8, 11], return_class_token=True)
                d_out = F.relu(dav2.depth_head(feats, ph, pw).squeeze(1))
                d_resized = F.interpolate(d_out[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]

                dav2_cls[step] = torch.stack([cls_tok.squeeze().float() for _, cls_tok in feats], dim=0).cpu().numpy()

                # Compute RGB-only structural agreement features between DAv2 disp & UniDepth inv-depth
                d_np = np.clip(d_resized.float().cpu().numpy(), 1e-4, 1e3)
                u_z = np.clip(uni_preds[step], 1e-2, 20.0)
                inv_z = 1.0 / u_z

                ld = np.log(d_np)
                lp = np.percentile(ld, [5, 25, 50, 75, 95])

                df_flat = d_np.ravel()
                iz_flat = inv_z.ravel()
                df_c = df_flat - df_flat.mean()
                iz_c = iz_flat - iz_flat.mean()
                pearson_lin = float((df_c @ iz_c) / max(np.linalg.norm(df_c) * np.linalg.norm(iz_c), 1e-8))

                ld_flat = ld.ravel()
                lz_flat = -np.log(u_z).ravel()
                ld_c = ld_flat - ld_flat.mean()
                lz_c = lz_flat - lz_flat.mean()
                pearson_log = float((ld_c @ lz_c) / max(np.linalg.norm(ld_c) * np.linalg.norm(lz_c), 1e-8))

                scale_ratio = np.log(np.clip(d_np * u_z, 1e-4, 1e4))
                sr_med = float(np.median(scale_ratio))
                sr_iqr = float(np.percentile(scale_ratio, 75) - np.percentile(scale_ratio, 25))
                cross_stats[step] = np.array([*lp, pearson_lin, pearson_log, sr_med, sr_iqr], dtype=np.float32)

                if (step + 1) % 400 == 0 or step + 1 == n:
                    print(f"  [DAv2/{split}] {step + 1}/{n} ({time.perf_counter() - t0:.1f}s)")

            np.save(OUT / f"dav2_cls_{split}.npy", dav2_cls)
            np.save(OUT / f"dav2_cross_stats_{split}.npy", cross_stats)

    print(f"  [DAv2] peak VRAM: {torch.cuda.max_memory_allocated() / 2**20:.1f} MiB")
    del dav2
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.20, device=0)
    with open_mat() as data:
        run_unidepth_cache(data)
        run_dav2_cache_and_benchmark(data)
    print("Saved all enriched UniDepthV2 + DAv2 caches to", OUT)


if __name__ == "__main__":
    main()
