"""
Track reprojection loss for articulation refinement.
L_track = mean_k,t || project(T(q_t) x_k) - track_k,t ||_1
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class TrackReprojectionLoss(nn.Module):
    """
    Measures consistency between predicted articulated 3D motion
    and observed 2D tracks.
    """

    def forward(
        self,
        transformed_points_3d: torch.Tensor,
        gt_tracks_2d: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        camera_extrinsics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            transformed_points_3d: [N, T, 3] articulated 3D points over time
            gt_tracks_2d: [N, T, 2] observed 2D track positions
            camera_intrinsics: [3, 3]
            camera_extrinsics: [T, 4, 4] optional
        """
        N, T, _ = transformed_points_3d.shape

        projected_2d = []
        for t in range(T):
            pts = transformed_points_3d[:, t, :]  # [N, 3]
            if camera_extrinsics is not None:
                R = camera_extrinsics[t, :3, :3]
                tvec = camera_extrinsics[t, :3, 3]
                pts = (R @ pts.T).T + tvec

            z = pts[:, 2:3].clamp(min=1e-6)
            pts_norm = pts / z
            uv = (camera_intrinsics[:2, :3] @ pts_norm.T).T  # [N, 2]
            projected_2d.append(uv)

        projected_2d = torch.stack(projected_2d, dim=1)  # [N, T, 2]
        return F.l1_loss(projected_2d, gt_tracks_2d)


class StateSmoothnessLoss(nn.Module):
    """Encourage smooth joint state over time."""

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [T] or [B, T] joint states over time
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        diff = state[:, 1:] - state[:, :-1]
        return (diff ** 2).mean()


def revolute_transform(points: torch.Tensor, axis: torch.Tensor,
                       pivot: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """
    Apply revolute joint transform.
    x_t = R(u, q)(x - p) + p
    """
    axis = F.normalize(axis, dim=-1)
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)

    centered = points - pivot.unsqueeze(0)

    # Rodrigues' rotation formula
    dot = (centered * axis.unsqueeze(0)).sum(dim=-1, keepdim=True)
    cross = torch.cross(
        axis.unsqueeze(0).expand_as(centered), centered, dim=-1
    )
    rotated = centered * cos_a + cross * sin_a + axis.unsqueeze(0) * dot * (1 - cos_a)
    return rotated + pivot.unsqueeze(0)


def prismatic_transform(points: torch.Tensor, axis: torch.Tensor,
                         displacement: torch.Tensor) -> torch.Tensor:
    """
    Apply prismatic joint transform.
    x_t = x + d * u
    """
    axis = F.normalize(axis, dim=-1)
    return points + displacement * axis.unsqueeze(0)
