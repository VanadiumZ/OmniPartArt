"""
TF2: Continuous kinematic refinement (FreeArt3D-Lite style).

Fix OmniPart geometry, optimize:
  - joint axis
  - joint pivot
  - joint state q_t
  - optional small part SE(3) correction

Using rendered mask / track / collision / smoothness losses.
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RefinementConfig:
    lr: float = 0.01
    num_iterations: int = 200
    optimize_axis: bool = True
    optimize_pivot: bool = True
    optimize_state: bool = True
    optimize_se3: bool = False
    lambda_mask: float = 1.0
    lambda_track: float = 1.0
    lambda_collision: float = 0.3
    lambda_smooth: float = 0.2


class KinematicRefiner:
    """
    Per-instance kinematic optimization of joint parameters.
    """

    def __init__(self, config: Optional[RefinementConfig] = None):
        self.config = config or RefinementConfig()

    def refine(
        self,
        joint_type: str,
        init_axis: np.ndarray,
        init_pivot: np.ndarray,
        init_state: np.ndarray,
        moving_points: np.ndarray,
        static_points: np.ndarray,
        motion_mask: Optional[np.ndarray] = None,
        motion_tracks: Optional[np.ndarray] = None,
        camera_intrinsics: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Optimize joint parameters to minimize reconstruction error.

        Returns:
            dict with refined 'axis', 'pivot', 'state'
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg = self.config

        raw_axis = torch.tensor(init_axis, dtype=torch.float32, device=device, requires_grad=cfg.optimize_axis)
        pivot = torch.tensor(init_pivot, dtype=torch.float32, device=device, requires_grad=cfg.optimize_pivot)
        state = torch.tensor(init_state, dtype=torch.float32, device=device, requires_grad=cfg.optimize_state)

        moving_pts = torch.tensor(moving_points, dtype=torch.float32, device=device)
        static_pts = torch.tensor(static_points, dtype=torch.float32, device=device)

        params = [p for p in [raw_axis, pivot, state] if p.requires_grad]
        if not params:
            return {
                "axis": init_axis,
                "pivot": init_pivot,
                "state": init_state,
            }

        optimizer = torch.optim.Adam(params, lr=cfg.lr)

        if motion_mask is not None:
            mask_tensor = torch.tensor(motion_mask, dtype=torch.float32, device=device)
        if motion_tracks is not None:
            track_tensor = torch.tensor(motion_tracks, dtype=torch.float32, device=device)

        best_loss = float("inf")
        best_params = None

        for it in range(cfg.num_iterations):
            optimizer.zero_grad()

            axis = F.normalize(raw_axis, dim=-1)
            T = state.shape[0]
            loss = torch.tensor(0.0, device=device)

            # Simulate articulated motion
            articulated = []
            for t in range(T):
                if joint_type == "revolute":
                    pts_t = self._revolute_transform(moving_pts, axis, pivot, state[t])
                else:
                    pts_t = self._prismatic_transform(moving_pts, axis, state[t])
                articulated.append(pts_t)

            articulated = torch.stack(articulated)  # [T, N, 3]

            # Collision loss
            if cfg.lambda_collision > 0:
                collision = self._collision_loss(articulated, static_pts)
                loss = loss + cfg.lambda_collision * collision

            # Smoothness loss
            if cfg.lambda_smooth > 0:
                smooth = self._smoothness_loss(state)
                loss = loss + cfg.lambda_smooth * smooth

            # Track loss
            if motion_tracks is not None and cfg.lambda_track > 0:
                track_loss = self._simple_track_loss(articulated, track_tensor)
                loss = loss + cfg.lambda_track * track_loss

            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_params = {
                    "axis": F.normalize(raw_axis, dim=-1).detach().cpu().numpy(),
                    "pivot": pivot.detach().cpu().numpy(),
                    "state": state.detach().cpu().numpy(),
                }

        return best_params or {
            "axis": init_axis,
            "pivot": init_pivot,
            "state": init_state,
        }

    @staticmethod
    def _revolute_transform(
        points: torch.Tensor, axis: torch.Tensor,
        pivot: torch.Tensor, angle: torch.Tensor,
    ) -> torch.Tensor:
        centered = points - pivot.unsqueeze(0)
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        dot = (centered * axis.unsqueeze(0)).sum(dim=-1, keepdim=True)
        cross = torch.cross(
            axis.unsqueeze(0).expand_as(centered), centered, dim=-1
        )
        rotated = centered * cos_a + cross * sin_a + axis.unsqueeze(0) * dot * (1 - cos_a)
        return rotated + pivot.unsqueeze(0)

    @staticmethod
    def _prismatic_transform(
        points: torch.Tensor, axis: torch.Tensor, displacement: torch.Tensor,
    ) -> torch.Tensor:
        return points + displacement * axis.unsqueeze(0)

    @staticmethod
    def _collision_loss(
        articulated: torch.Tensor, static: torch.Tensor,
        threshold: float = 0.01,
    ) -> torch.Tensor:
        T = articulated.shape[0]
        total = torch.tensor(0.0, device=articulated.device)
        for t in range(T):
            dists = torch.cdist(articulated[t], static)
            min_dists = dists.min(dim=-1).values
            pen = torch.relu(threshold - min_dists)
            total = total + pen.mean()
        return total / T

    @staticmethod
    def _smoothness_loss(state: torch.Tensor) -> torch.Tensor:
        diff = state[1:] - state[:-1]
        return (diff ** 2).mean()

    @staticmethod
    def _simple_track_loss(
        articulated: torch.Tensor, tracks: torch.Tensor,
    ) -> torch.Tensor:
        """Simplified 2D track loss using orthographic projection."""
        T = min(articulated.shape[0], tracks.shape[1])
        N = min(articulated.shape[1], tracks.shape[0])
        if N == 0 or T == 0:
            return torch.tensor(0.0, device=articulated.device)

        total = torch.tensor(0.0, device=articulated.device)
        for t in range(T):
            pred_2d = articulated[t, :N, :2]
            gt_2d = tracks[:N, t, :]
            total = total + F.l1_loss(pred_2d, gt_2d)
        return total / T
