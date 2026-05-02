"""
Rendered mask loss for training-free refinement (TF2) and E4.
L_render = 1 - IoU(rendered_moving_mask, gt_or_motion_mask)
"""
import torch
import torch.nn as nn
from typing import Optional


class RenderedMaskLoss(nn.Module):
    """Differentiable mask IoU loss."""

    def forward(
        self,
        pred_mask: torch.Tensor,
        gt_mask: torch.Tensor,
        smooth: float = 1e-6,
    ) -> torch.Tensor:
        """
        Args:
            pred_mask: [B, T, H, W] predicted mask probabilities
            gt_mask: [B, T, H, W] ground truth binary masks
        """
        pred_flat = pred_mask.flatten(start_dim=2)
        gt_flat = gt_mask.flatten(start_dim=2)

        intersection = (pred_flat * gt_flat).sum(dim=-1)
        union = pred_flat.sum(dim=-1) + gt_flat.sum(dim=-1) - intersection
        iou = (intersection + smooth) / (union + smooth)

        return (1.0 - iou).mean()


class SilhouetteLoss(nn.Module):
    """Binary cross-entropy on rendered silhouette vs observed silhouette."""

    def forward(
        self,
        pred_sil: torch.Tensor,
        gt_sil: torch.Tensor,
    ) -> torch.Tensor:
        return torch.nn.functional.binary_cross_entropy(
            pred_sil.clamp(1e-6, 1 - 1e-6), gt_sil, reduction="mean"
        )


class CollisionLoss(nn.Module):
    """
    Penalize interpenetration between moving and static parts.
    Uses signed distance or overlap volume heuristic.
    """

    def forward(
        self,
        moving_points: torch.Tensor,
        static_points: torch.Tensor,
        threshold: float = 0.01,
    ) -> torch.Tensor:
        """
        Simple collision penalty: for each moving point, penalize if
        distance to nearest static point is below threshold.
        """
        if moving_points.shape[0] == 0 or static_points.shape[0] == 0:
            return torch.tensor(0.0, device=moving_points.device)

        dists = torch.cdist(moving_points, static_points)
        min_dists = dists.min(dim=-1).values
        penetration = torch.relu(threshold - min_dists)
        return penetration.mean()
