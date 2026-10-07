"""Causal RAFT fusion of metric depth."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import Raft_Small_Weights, raft_small


def load_raft(device: str = "cuda"):
    weights = Raft_Small_Weights.DEFAULT
    model = raft_small(weights=weights, progress=True).to(device).eval()
    return model, weights.transforms()


def _rgb_tensor(image_rgb: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(image_rgb)).permute(2, 0, 1)


def pairwise_flow(raft, transforms, frame_a: np.ndarray, frame_b: np.ndarray, device: str):
    """Forward flow a->b and backward flow b->a, in pixels."""
    first, second = transforms(_rgb_tensor(frame_a), _rgb_tensor(frame_b))
    second_back, first_back = transforms(_rgb_tensor(frame_b), _rgb_tensor(frame_a))
    first = first.unsqueeze(0).to(device)
    second = second.unsqueeze(0).to(device)
    first_back = first_back.unsqueeze(0).to(device)
    second_back = second_back.unsqueeze(0).to(device)
    with torch.inference_mode():
        forward = raft(first, second)[-1]
        backward = raft(second_back, first_back)[-1]
    return forward, backward


def _sample(field: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Sample field at x + flow. field is NCHW, flow is N2HW."""
    _, _, height, width = field.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=field.device),
        torch.arange(width, device=field.device),
        indexing="ij",
    )
    grid_x = (xx + flow[0, 0]) / max(width - 1, 1) * 2 - 1
    grid_y = (yy + flow[0, 1]) / max(height - 1, 1) * 2 - 1
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
    return F.grid_sample(field, grid, align_corners=True, padding_mode="border")


def fuse_with_flow(
    previous_depth: np.ndarray,
    current_depth: np.ndarray,
    forward_flow: torch.Tensor,
    backward_flow: torch.Tensor,
    tolerance: float = 1.0,
) -> np.ndarray:
    """Warp the previous fused depth onto the current frame and mix it.

    Backward flow takes a current pixel to the previous frame. Pixels whose
    forward and backward flow disagree keep the current prediction.
    """
    device = backward_flow.device
    prev = torch.from_numpy(previous_depth)[None, None].to(device)
    warped = _sample(prev, backward_flow)
    forward_at_source = _sample(forward_flow, backward_flow)
    residual = torch.linalg.norm(forward_at_source + backward_flow, dim=1, keepdim=True)
    inside = (warped > 0) & (backward_flow[:, 0:1].abs() + backward_flow[:, 1:2].abs() > 0)
    consistent = (residual <= tolerance) & (warped > 0)
    weight = torch.exp(-residual).clamp(0, 1)
    current = torch.from_numpy(current_depth)[None, None].to(device)
    mixed = weight * warped + (1.0 - weight) * current
    fused = torch.where(consistent, mixed, current)
    del inside
    return fused.squeeze().detach().cpu().numpy().astype(np.float32)


def flow_temporal_absrel(previous: np.ndarray, current: np.ndarray, backward_flow: torch.Tensor) -> float:
    """Mean relative error after warping the previous depth onto the current frame."""
    prev = torch.from_numpy(previous.astype(np.float32))[None, None].to(backward_flow.device)
    warped = _sample(prev, backward_flow).squeeze().detach().cpu().numpy()
    mask = (warped > 1e-3) & (current > 1e-3)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(warped[mask] - current[mask]) / current[mask]))
