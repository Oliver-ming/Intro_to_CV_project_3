"""Hierarchical camera- and semantics-aware metric depth calibration.

Implements progressive post-hoc calibration levels fitted strictly on NYU train:
- Level 1A: Frozen global median-of-image-medians scale (baseline Part 3).
- Level 1B: Inverse-uncertainty weighted global scale & centered 2P power-law.
- Level 2A: FoV-conditioned analytic log-linear scale s(f_hat).
- Level 2B: ScaleProbe-1P / ScaleProbe-2P v1 (FoV/uncertainty + ViT + global depth_features).
- Level 2C: ScaleProbe-2P v2 (+ spatial/uncertainty-pooled depth_features + camera_layer latents
            + decoupled 5-fold CV regularization, strict zero test leakage).
- Level 2D: ScaleProbe-2P+ (cross-backbone ablation adding frozen DAv2-S ViT CLS & agreement stats).
- Level 3 : Self-calibrated virtual camera K_train + residual ScaleProbe-2P v2.
- Appendix: Precision-weighted Bayesian soft-fusion for noisy person-height cues.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, normalize

from src.calibration.global_scale import pooled_absrel_optimal


def per_image_oracle_params(
    preds: np.ndarray,
    gts: np.ndarray,
    masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-image oracle (r_med, r_opt_1p, betas, s_opt_2p) using consistent median anchoring."""
    n = len(preds)
    r_med = np.zeros(n, dtype=np.float64)
    r_opt = np.zeros(n, dtype=np.float64)
    betas = np.zeros(n, dtype=np.float64)
    s_opt_2p = np.zeros(n, dtype=np.float64)
    for i in range(n):
        p_full = np.clip(preds[i].astype(np.float64), 1e-3, 20.0)
        d_med = max(float(np.median(p_full)), 1e-3)
        m = masks[i]
        p = np.clip(p_full[m], 1e-2, 10.0)
        g = gts[i][m].astype(np.float64)
        r_med[i] = np.median(g / p)
        r_opt[i] = pooled_absrel_optimal(p, g)

        lp, lg = np.log(p), np.log(g)
        mp, mg = lp.mean(), lg.mean()
        denom = max(float(np.sum((lp - mp) ** 2)), 1e-6)
        b = float(np.clip(np.sum((lp - mp) * (lg - mg)) / denom, 0.65, 1.35))
        betas[i] = b

        q = d_med * ((p / d_med) ** b)
        s_opt_2p[i] = pooled_absrel_optimal(q, g)
    return r_med, r_opt, betas, s_opt_2p


def per_image_planar_4p_oracle(
    preds: np.ndarray,
    gts: np.ndarray,
    masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Diagnostic-only 4-parameter per-image planar-tilt oracle:
    log(D_hat) = log(d_med) + a0 + beta * (log(D) - log(d_med)) + ty * y_norm + tx * x_norm.
    Returns (thetas (N, 4), calibrated_abs_rel (N,)).
    """
    n, h, w = preds.shape
    yy, xx = np.meshgrid(np.linspace(-1.0, 1.0, h), np.linspace(-1.0, 1.0, w), indexing="ij")
    thetas = np.zeros((n, 4), dtype=np.float64)
    ars = np.zeros(n, dtype=np.float64)
    for i in range(n):
        p_full = np.clip(preds[i].astype(np.float64), 1e-3, 20.0)
        d_med = max(float(np.median(p_full)), 1e-3)
        m = masks[i]
        lp_m = np.log(p_full[m]) - np.log(d_med)
        lg_m = np.log(gts[i][m].astype(np.float64)) - np.log(d_med)
        X = np.column_stack([np.ones(m.sum()), lp_m, yy[m], xx[m]])
        theta, _, _, _ = np.linalg.lstsq(X, lg_m, rcond=None)
        theta[1] = float(np.clip(theta[1], 0.65, 1.35))
        thetas[i] = theta
        cal = np.exp(np.log(d_med) + theta[0] + theta[1] * (np.log(p_full) - np.log(d_med)) + theta[2] * yy + theta[3] * xx)
        p_c = np.clip(cal[m], 1e-3, 10.0)
        g_c = gts[i][m].astype(np.float64)
        ars[i] = float(np.mean(np.abs(p_c - g_c) / g_c))
    return thetas, ars


def uncertainty_weighted_image_scale(
    preds: np.ndarray,
    gts: np.ndarray,
    masks: np.ndarray,
    uncertainties: np.ndarray,
    gamma: float = 1.0,
) -> float:
    """Global median of per-image inverse-uncertainty-weighted optimal scales."""
    per_img = []
    for i in range(len(preds)):
        m = masks[i]
        p = np.clip(preds[i][m], 1e-2, 10.0).astype(np.float64)
        g = gts[i][m].astype(np.float64)
        u = np.clip(uncertainties[i][m].astype(np.float64), 1e-3, 1e3)
        ratio = g / p
        w = (u ** (-gamma)) * (p / g)
        order = np.argsort(ratio)
        cum = np.cumsum(w[order])
        per_img.append(float(ratio[order][np.searchsorted(cum, 0.5 * cum[-1])]))
    return float(np.median(per_img))


def apply_power_law(pred: np.ndarray, scale: float, beta: float) -> np.ndarray:
    """Centered power-law transform preserving median anchor scale * median(pred)."""
    p = np.clip(pred.astype(np.float32), 1e-3, 20.0)
    d_med = float(np.median(p))
    return (scale * d_med * ((p / max(d_med, 1e-3)) ** beta)).astype(np.float32)


def extract_geo_uncertainty_features(
    preds: np.ndarray,
    intrinsics: np.ndarray,
    uncertainties: np.ndarray,
) -> np.ndarray:
    """Test-time self-introspection vector (15-d) from RGB-only model outputs."""
    rows = []
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2] / 640.0
    cy = intrinsics[:, 1, 2] / 480.0
    for i in range(len(preds)):
        p = np.clip(preds[i].astype(np.float32), 1e-2, 10.0)
        u = np.clip(uncertainties[i].astype(np.float32), 1e-4, 1e3)
        lp = np.log(p)
        lu = np.log(u)
        dp = np.percentile(lp, [5, 25, 50, 75, 95])
        up = np.percentile(lu, [10, 25, 50, 75, 90])
        v_tilt = float(np.median(lp[:240, :]) - np.median(lp[240:, :]))
        rows.append([np.log(fx[i]), np.log(fy[i]), cx[i], cy[i], v_tilt, *dp, *up])
    return np.asarray(rows, dtype=np.float32)


@dataclass
class FoVScaleCalibrator:
    """2-parameter focal-length conditioned scale s(f_hat) = exp(b0 + b1 * log f_hat)."""

    b0: float
    b1: float

    @classmethod
    def fit(cls, intrinsics: np.ndarray, target_scales: np.ndarray) -> "FoVScaleCalibrator":
        f_hat = np.sqrt(intrinsics[:, 0, 0] * intrinsics[:, 1, 1])
        x = np.log(f_hat).astype(np.float64)
        y = np.log(target_scales).astype(np.float64)
        b1, b0 = np.polyfit(x, y, 1)
        return cls(b0=float(b0), b1=float(b1))

    def predict_scale(self, intrinsics: np.ndarray) -> np.ndarray:
        f_hat = np.sqrt(intrinsics[:, 0, 0] * intrinsics[:, 1, 1])
        return np.exp(self.b0 + self.b1 * np.log(f_hat)).astype(np.float32)


def build_cv_pixel_cache(
    preds: np.ndarray,
    gts: np.ndarray,
    masks: np.ndarray,
    n_chunks: int = 10,
) -> dict[str, object]:
    """Precompute valid masked depth/GT arrays and GPU chunk tuples once across all CV sweeps."""
    n_img = len(preds)
    p_valid_list = []
    g_valid_list = []
    rel_log_valid_list = []
    d_med_full = np.empty(n_img, dtype=np.float64)
    for i in range(n_img):
        p_full = np.clip(preds[i].astype(np.float64), 1e-3, 20.0)
        dm = max(float(np.median(p_full)), 1e-3)
        d_med_full[i] = dm
        m = masks[i]
        pv = p_full[m]
        gv = gts[i][m].astype(np.float64)
        p_valid_list.append(pv)
        g_valid_list.append(gv)
        rel_log_valid_list.append(np.log(pv / dm))

    use_gpu = torch.cuda.is_available()
    chunks_cpu = None
    if use_gpu:
        step_c = max(1, (n_img + n_chunks - 1) // n_chunks)
        chunks_cpu = []
        for lo_i in range(0, n_img, step_c):
            hi_i = min(lo_i + step_c, n_img)
            idx_rng = range(lo_i, hi_i)
            lens = [len(p_valid_list[i]) for i in idx_rng]
            p_cat = torch.from_numpy(np.concatenate([p_valid_list[i] for i in idx_rng]).astype(np.float32))
            g_cat = torch.from_numpy(np.concatenate([g_valid_list[i] for i in idx_rng]).astype(np.float32))
            rlog_cat = torch.from_numpy(np.concatenate([rel_log_valid_list[i] for i in idx_rng]).astype(np.float32))
            img_ids = torch.repeat_interleave(torch.arange(hi_i - lo_i, dtype=torch.int64), torch.tensor(lens, dtype=torch.int64))
            inv_len = torch.tensor([1.0 / l for l in lens], dtype=torch.float32)
            chunks_cpu.append((lo_i, hi_i, p_cat, g_cat, rlog_cat, img_ids, inv_len))

    return {
        "n_img": n_img,
        "p_valid_list": p_valid_list,
        "g_valid_list": g_valid_list,
        "rel_log_valid_list": rel_log_valid_list,
        "d_med_full": d_med_full,
        "chunks_cpu": chunks_cpu,
    }


def _dual_gram_ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    """Exact Ridge(alpha, fit_intercept=True) closed-form coef & intercept via N x N dual Gram."""
    Xf = X.astype(np.float64)
    yf = y.astype(np.float64)
    mu_x = Xf.mean(axis=0, keepdims=True)
    mu_y = float(yf.mean())
    Xc = Xf - mu_x
    yc = yf - mu_y
    if Xc.shape[1] <= Xc.shape[0]:
        A = Xc.T @ Xc
        A[np.diag_indices_from(A)] += float(alpha)
        coef = np.linalg.solve(A, Xc.T @ yc)
    else:
        G = Xc @ Xc.T
        G[np.diag_indices_from(G)] += float(alpha)
        dual = np.linalg.solve(G, yc)
        coef = Xc.T @ dual
    intercept = mu_y - float(mu_x[0] @ coef)
    return coef.astype(np.float64), intercept


class HierarchicalScaleProbe:
    """Lightweight block-weighted ridge calibrator over FoV, uncertainty, and frozen ViT/decoder tokens.

    Supports arbitrary subsets of standardized low-D statistics (first block if standard_blocks)
    and per-image L2-normalized high-D embedding blocks with Train-CV decoupled regularization.
    """

    def __init__(
        self,
        two_param: bool = True,
        block_weights: dict[str, float] | None = None,
        standard_keys: tuple[str, ...] = ("geo", "dav2_stats"),
    ):
        self.two_param = two_param
        self.block_weights = block_weights or {
            "geo": 0.25,
            "cls": 1.0,
            "pat": 1.0,
            "dfe": 1.0,
        }
        self.standard_keys = set(standard_keys)
        self.scalers: dict[str, StandardScaler] = {}
        self.active_keys: list[str] = []
        self.coef_scale: np.ndarray | None = None
        self.intercept_scale: float = 0.0
        self.coef_beta: np.ndarray | None = None
        self.intercept_beta: float = 1.0
        self.best_alpha_scale: float = 0.15
        self.best_alpha_beta: float = 0.45

    def _design_matrix(self, feat_dict: dict[str, np.ndarray], fit_scaler: bool = False) -> np.ndarray:
        if fit_scaler:
            self.active_keys = [k for k, w in self.block_weights.items() if w > 0 and k in feat_dict]
        parts = []
        for k in self.active_keys:
            w = float(self.block_weights[k])
            arr = feat_dict[k]
            flat = arr.reshape(len(arr), -1).astype(np.float64)
            if k in self.standard_keys:
                if fit_scaler:
                    self.scalers[k] = StandardScaler().fit(flat)
                part = self.scalers[k].transform(flat) * w
            else:
                part = normalize(flat) * w
            parts.append(part.astype(np.float32))
        return np.hstack(parts).astype(np.float32)

    @staticmethod
    def _slice_dict(feat_dict: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, np.ndarray]:
        return {k: v[idx] for k, v in feat_dict.items()}

    def fit_with_cv(
        self,
        preds: np.ndarray,
        gts: np.ndarray,
        masks: np.ndarray,
        feat_dict: dict[str, np.ndarray],
        alphas: tuple[float, ...] = (0.005, 0.01, 0.02, 0.05, 0.08, 0.15, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0),
        coupled_beta_mult: float | None = None,
        seed: int = 42,
        precomputed_targets: tuple[np.ndarray, np.ndarray] | None = None,
        cv_pixel_cache: dict[str, object] | None = None,
    ) -> dict[str, float]:
        """Select regularization strictly inside Train 5-fold CV (zero test leakage).

        Uses exact centered dual-Gram eigendecomposition (bit-identical to Ridge(fit_intercept=True))
        and streamed GPU chunk scoring (<350 MiB peak VRAM) so 12+ alphas run in ~0.2s.
        """
        if precomputed_targets is None:
            _, r_opt, betas, _ = per_image_oracle_params(preds, gts, masks)
        else:
            r_opt, betas = precomputed_targets
        log_r = np.log(r_opt).astype(np.float64)
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        splits = list(kf.split(preds))

        if cv_pixel_cache is None:
            cv_pixel_cache = build_cv_pixel_cache(preds, gts, masks)
        n_img = int(cv_pixel_cache["n_img"])
        p_valid_list = cv_pixel_cache["p_valid_list"]
        g_valid_list = cv_pixel_cache["g_valid_list"]
        rel_log_valid_list = cv_pixel_cache["rel_log_valid_list"]
        d_med_full = cv_pixel_cache["d_med_full"]
        chunks_cpu = cv_pixel_cache["chunks_cpu"]

        # Build fold caches via single dual-Gram eigendecomposition per fold
        fold_cache = []
        for tr, va in splits:
            probe = HierarchicalScaleProbe(
                two_param=self.two_param, block_weights=self.block_weights, standard_keys=tuple(self.standard_keys)
            )
            X_tr = probe._design_matrix(self._slice_dict(feat_dict, tr), fit_scaler=True).astype(np.float64)
            X_va = probe._design_matrix(self._slice_dict(feat_dict, va), fit_scaler=False).astype(np.float64)
            mu_x = X_tr.mean(axis=0, keepdims=True)
            Xc_tr = X_tr - mu_x
            Xc_va = X_va - mu_x
            G_tr = Xc_tr @ Xc_tr.T
            eigvals, Q = np.linalg.eigh(G_tr)
            eigvals = np.maximum(eigvals, 0.0)
            C_va = Xc_va @ Xc_tr.T  # (n_va, n_tr)
            y_s = log_r[tr]
            y_b = betas[tr]
            mu_s, mu_b = float(y_s.mean()), float(y_b.mean())
            q_s = Q.T @ (y_s - mu_s)
            q_b = Q.T @ (y_b - mu_b)
            fold_cache.append((va, C_va @ Q, eigvals, mu_s, q_s, mu_b, q_b))

        def _predict_fold(fc_item, alpha: float, is_beta: bool) -> np.ndarray:
            _, CQ, eigvals, mu_s, q_s, mu_b, q_b = fc_item
            q_coef = (q_b if is_beta else q_s) / (eigvals + float(alpha))
            base = mu_b if is_beta else mu_s
            return base + CQ @ q_coef

        use_gpu_score = chunks_cpu is not None
        if use_gpu_score:
            dmed_gpu = torch.from_numpy(d_med_full.astype(np.float32)).cuda()

            def _eval_batch_absrel(s_mat: np.ndarray, b_mat: np.ndarray | None) -> np.ndarray:
                m_cnt = len(s_mat)
                s_t = torch.from_numpy(s_mat.astype(np.float32)).cuda()
                b_t = None if b_mat is None else torch.from_numpy(b_mat.astype(np.float32)).cuda()
                totals = np.zeros(m_cnt, dtype=np.float64)
                with torch.inference_mode():
                    for lo_i, hi_i, p_cpu, g_cpu, rlog_cpu, ids_cpu, inv_cpu in chunks_cpu:
                        p_g = p_cpu.cuda()
                        g_g = g_cpu.cuda()
                        rlog_g = rlog_cpu.cuda()
                        ids_g = ids_cpu.cuda()
                        inv_g = inv_cpu.cuda()
                        for m_idx in range(m_cnt):
                            if b_t is None:
                                cal = (p_g * s_t[m_idx, lo_i:hi_i][ids_g]).clamp_(1e-3, 10.0)
                            else:
                                dm_s = (s_t[m_idx, lo_i:hi_i] * dmed_gpu[lo_i:hi_i])[ids_g]
                                cal = (dm_s * torch.exp(b_t[m_idx, lo_i:hi_i][ids_g] * rlog_g)).clamp_(1e-3, 10.0)
                            rel = (cal - g_g).abs_() / g_g
                            per_img = torch.zeros(hi_i - lo_i, device="cuda", dtype=torch.float32)
                            per_img.index_add_(0, ids_g, rel)
                            totals[m_idx] += float((per_img * inv_g).sum().item())
                        del p_g, g_g, rlog_g, ids_g, inv_g
                return totals / n_img
        else:
            def _eval_batch_absrel(s_mat: np.ndarray, b_mat: np.ndarray | None) -> np.ndarray:
                out_s = np.zeros(len(s_mat), dtype=np.float64)
                for m_idx in range(len(s_mat)):
                    b_row = None if b_mat is None else b_mat[m_idx]
                    errs = np.empty(n_img, dtype=np.float64)
                    for i in range(n_img):
                        if b_row is None:
                            p_cal = np.clip(s_mat[m_idx, i] * p_valid_list[i], 1e-3, 10.0)
                        else:
                            p_cal = np.clip(s_mat[m_idx, i] * d_med_full[i] * np.exp(b_row[i] * rel_log_valid_list[i]), 1e-3, 10.0)
                        errs[i] = float(np.mean(np.abs(p_cal - g_valid_list[i]) / g_valid_list[i]))
                    out_s[m_idx] = float(np.mean(errs))
                return out_s

        best_s_cv = float("inf")
        best_a_scale = alphas[0]

        if not self.two_param or coupled_beta_mult is None:
            s_cand = np.zeros((len(alphas), n_img), dtype=np.float64)
            for a_i, a in enumerate(alphas):
                for fc in fold_cache:
                    va = fc[0]
                    s_cand[a_i, va] = np.exp(_predict_fold(fc, a, is_beta=False))
            scores_s = _eval_batch_absrel(s_cand, None)
            best_s_idx = int(np.argmin(scores_s))
            best_s_cv = float(scores_s[best_s_idx])
            best_a_scale = alphas[best_s_idx]
            self.best_alpha_scale = best_a_scale

            if not self.two_param:
                self.best_alpha_beta = best_a_scale
                best_cv_absrel = best_s_cv
            else:
                s_best_rep = np.broadcast_to(s_cand[best_s_idx : best_s_idx + 1], (len(alphas), n_img)).copy()
                b_cand = np.zeros((len(alphas), n_img), dtype=np.float64)
                for a_i, ab in enumerate(alphas):
                    for fc in fold_cache:
                        va = fc[0]
                        b_cand[a_i, va] = np.clip(_predict_fold(fc, ab, is_beta=True), 0.80, 1.25)
                scores_b = _eval_batch_absrel(s_best_rep, b_cand)
                best_b_idx = int(np.argmin(scores_b))
                best_cv_absrel = float(scores_b[best_b_idx])
                self.best_alpha_beta = alphas[best_b_idx]
        else:
            s_cand = np.zeros((len(alphas), n_img), dtype=np.float64)
            b_cand = np.zeros((len(alphas), n_img), dtype=np.float64)
            for a_i, a in enumerate(alphas):
                for fc in fold_cache:
                    va = fc[0]
                    s_cand[a_i, va] = np.exp(_predict_fold(fc, a, is_beta=False))
                    b_cand[a_i, va] = np.clip(_predict_fold(fc, a * coupled_beta_mult, is_beta=True), 0.80, 1.25)
            scores_c = _eval_batch_absrel(s_cand, b_cand)
            best_idx = int(np.argmin(scores_c))
            best_cv_absrel = float(scores_c[best_idx])
            self.best_alpha_scale = alphas[best_idx]
            self.best_alpha_beta = alphas[best_idx] * coupled_beta_mult

        if use_gpu_score:
            del dmed_gpu
            torch.cuda.empty_cache()

        X_all = self._design_matrix(feat_dict, fit_scaler=True)
        self.coef_scale, self.intercept_scale = _dual_gram_ridge_fit(X_all, log_r, self.best_alpha_scale)
        if self.two_param:
            self.coef_beta, self.intercept_beta = _dual_gram_ridge_fit(X_all, betas, self.best_alpha_beta)
        return {
            "cv_abs_rel": float(best_cv_absrel),
            "alpha_scale": float(self.best_alpha_scale),
            "alpha_beta": float(self.best_alpha_beta),
            "feature_dim": int(X_all.shape[1]),
        }

    def predict_params(self, feat_dict: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        X = self._design_matrix(feat_dict, fit_scaler=False).astype(np.float64)
        assert self.coef_scale is not None
        scales = np.exp(X @ self.coef_scale + self.intercept_scale).astype(np.float32)
        if self.two_param and self.coef_beta is not None:
            betas = np.clip(X @ self.coef_beta + self.intercept_beta, 0.80, 1.25).astype(np.float32)
        else:
            betas = np.ones_like(scales)
        return scales, betas

    def calibrate_batch(
        self,
        preds: np.ndarray,
        feat_dict: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        scales, betas = self.predict_params(feat_dict)
        out = np.empty_like(preds, dtype=np.float32)
        for i in range(len(preds)):
            if self.two_param:
                out[i] = apply_power_law(preds[i], float(scales[i]), float(betas[i]))
            else:
                out[i] = scales[i] * preds[i]
        return out, scales, betas

    def export_gpu_module(self, device: str = "cuda") -> "GPUScaleProbe":
        """Move fitted Ridge weights & scaler stats onto GPU for honest single-image inference."""
        assert self.coef_scale is not None and "geo" in self.scalers
        sc = self.scalers["geo"]
        w_s = torch.from_numpy(self.coef_scale.astype(np.float32)).to(device)
        b_s = torch.tensor(float(self.intercept_scale), dtype=torch.float32, device=device)
        if self.two_param and self.coef_beta is not None:
            w_b = torch.from_numpy(self.coef_beta.astype(np.float32)).to(device)
            b_b = torch.tensor(float(self.intercept_beta), dtype=torch.float32, device=device)
        else:
            w_b = torch.zeros_like(w_s)
            b_b = torch.tensor(1.0, dtype=torch.float32, device=device)
        geo_mean = torch.from_numpy(sc.mean_.astype(np.float32)).to(device)
        geo_scale = torch.from_numpy(sc.scale_.astype(np.float32)).to(device)
        weights = {k: float(self.block_weights[k]) for k in self.active_keys}
        return GPUScaleProbe(
            w_s=w_s,
            b_s=b_s,
            w_b=w_b,
            b_b=b_b,
            geo_mean=geo_mean,
            geo_scale=geo_scale,
            weights=weights,
            active_keys=list(self.active_keys),
            two_param=self.two_param,
        )


class GPUScaleProbe:
    """Zero-CPU-roundtrip single-image GPU probe performing quantile geo extraction + pooling + power-law."""

    def __init__(
        self,
        w_s: torch.Tensor,
        b_s: torch.Tensor,
        w_b: torch.Tensor,
        b_b: torch.Tensor,
        geo_mean: torch.Tensor,
        geo_scale: torch.Tensor,
        weights: dict[str, float],
        active_keys: list[str],
        two_param: bool = True,
    ):
        self.w_s = w_s
        self.b_s = b_s
        self.w_b = w_b
        self.b_b = b_b
        self.geo_mean = geo_mean
        self.geo_scale = geo_scale
        self.weights = weights
        self.active_keys = active_keys
        self.two_param = two_param
        self.dp_q = torch.tensor([0.05, 0.25, 0.50, 0.75, 0.95], device=w_s.device, dtype=torch.float32)
        self.up_q = torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90], device=w_s.device, dtype=torch.float32)

    @torch.inference_mode()
    def forward_single(
        self,
        depth: torch.Tensor,
        intrinsics: torch.Tensor,
        conf: torch.Tensor,
        cls_tokens: torch.Tensor,
        f_last: torch.Tensor,
        depth_feat_map: torch.Tensor,
        cam_latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """All tensors on GPU directly from single-image UniDepthV2 pass."""
        d_c = depth.clamp(min=1e-2, max=10.0)
        u_c = conf.clamp(min=1e-4, max=1e3)
        lp = torch.log(d_c)
        lu = torch.log(u_c)
        h, w = lp.shape
        v_tilt = lp[: h // 2].median() - lp[h // 2 :].median()
        dp = torch.quantile(lp.reshape(-1), self.dp_q)
        up = torch.quantile(lu.reshape(-1), self.up_q)
        geo = torch.cat(
            [
                torch.stack(
                    [
                        torch.log(intrinsics[0, 0]),
                        torch.log(intrinsics[1, 1]),
                        intrinsics[0, 2] / 640.0,
                        intrinsics[1, 2] / 480.0,
                        v_tilt,
                    ]
                ),
                dp,
                up,
            ]
        )
        g_norm = ((geo - self.geo_mean) / self.geo_scale) * self.weights["geo"]

        hp, wp, _ = f_last.shape
        pat = torch.stack(
            [
                f_last.mean(dim=(0, 1)),
                f_last.std(dim=(0, 1)),
                f_last[: hp // 2].mean(dim=(0, 1)),
                f_last[hp // 2 :].mean(dim=(0, 1)),
                f_last[:, : wp // 2].mean(dim=(0, 1)),
                f_last[:, wp // 2 :].mean(dim=(0, 1)),
            ],
            dim=0,
        )

        # Spatial + uncertainty-weighted pooling of decoder depth_features
        _, hpf, wpf = depth_feat_map.shape
        u_low = F.interpolate(conf.view(1, 1, h, w), size=(hpf, wpf), mode="bilinear", align_corners=False)[0, 0].clamp(min=1e-4, max=1e3)
        inv_w = (1.0 / (u_low + 1e-3))
        inv_w = inv_w / inv_w.sum().clamp(min=1e-6)
        q25 = torch.quantile(u_low.reshape(-1), 0.25)
        top25 = (u_low <= q25).float()
        top25 = top25 / top25.sum().clamp(min=1.0)
        df_flat = depth_feat_map.view(depth_feat_map.shape[0], -1)
        dfe_sp = torch.stack(
            [
                depth_feat_map.mean(dim=(1, 2)),
                depth_feat_map.std(dim=(1, 2)),
                depth_feat_map[:, : hpf // 2, : wpf // 2].mean(dim=(1, 2)),
                depth_feat_map[:, : hpf // 2, wpf // 2 :].mean(dim=(1, 2)),
                depth_feat_map[:, hpf // 2 :, : wpf // 2].mean(dim=(1, 2)),
                depth_feat_map[:, hpf // 2 :, wpf // 2 :].mean(dim=(1, 2)),
                df_flat @ inv_w.reshape(-1),
                df_flat @ top25.reshape(-1),
            ],
            dim=0,
        )

        emb_map = {
            "geo": g_norm,
            "cls": F.normalize(cls_tokens.reshape(-1), dim=0) * self.weights.get("cls", 1.0),
            "pat": F.normalize(pat.reshape(-1), dim=0) * self.weights.get("pat", 1.0),
            "dfe": F.normalize(dfe_sp[:2].reshape(-1), dim=0) * self.weights.get("dfe", 1.0),
            "dfe_unc": F.normalize(dfe_sp[6:8].reshape(-1), dim=0) * self.weights.get("dfe_unc", 1.0),
            "dfe_sp": F.normalize(dfe_sp.reshape(-1), dim=0) * self.weights.get("dfe_sp", 1.0),
        }
        if cam_latent is not None and "cam" in self.weights:
            emb_map["cam"] = F.normalize(cam_latent.reshape(-1), dim=0) * self.weights["cam"]

        x = torch.cat([emb_map[k] for k in self.active_keys], dim=0)
        s = torch.exp(torch.dot(self.w_s, x) + self.b_s)
        if not self.two_param:
            return s * depth
        beta = torch.clamp(torch.dot(self.w_b, x) + self.b_b, 0.80, 1.25)
        p_full = depth.clamp(min=1e-3, max=20.0)
        d_med = p_full.median().clamp(min=1e-3)
        return s * d_med * ((p_full / d_med) ** beta)


def bayesian_inverse_variance_scale(
    s_prior: float,
    sigma_prior: float,
    s_obs: float | None,
    sigma_obs: float,
) -> float:
    """Posterior mean scale combining domain prior N(s_prior, sigma_prior^2) with noisy cue."""
    if s_obs is None or not np.isfinite(s_obs):
        return s_prior
    v0 = sigma_prior**2
    v1 = sigma_obs**2
    return float((v1 * s_prior + v0 * s_obs) / (v0 + v1))
