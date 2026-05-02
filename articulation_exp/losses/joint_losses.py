"""
Joint estimation losses following the experiment plan.

Includes:
  - Moving part classification (CE)
  - Joint type classification (CE)
  - Axis angular loss (direction-agnostic)
  - Pivot L1 loss (bbox-normalized)
  - State loss (sin/cos for revolute, L1 for prismatic)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class AxisAngularLoss(nn.Module):
    """
    Direction-agnostic axis loss:
    L_axis = min(angle(u_pred, u_gt), angle(-u_pred, u_gt))
    """

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        pred = F.normalize(pred, dim=-1)
        gt = F.normalize(gt, dim=-1)
        cos_sim = (pred * gt).sum(dim=-1).clamp(-1 + 1e-7, 1 - 1e-7)
        angle_pos = torch.acos(cos_sim.abs())
        return angle_pos.mean()


class PivotLoss(nn.Module):
    """L1 loss on pivot position, optionally bbox-normalized."""

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        bbox_center: Optional[torch.Tensor] = None,
        bbox_size: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if bbox_center is not None and bbox_size is not None:
            pred_norm = (pred - bbox_center) / (bbox_size / 2 + 1e-6)
            gt_norm = (gt - bbox_center) / (bbox_size / 2 + 1e-6)
            return F.l1_loss(pred_norm, gt_norm)
        return F.l1_loss(pred, gt)


class StateLoss(nn.Module):
    """
    Joint state loss.
    Revolute: L1 on sin/cos pairs.
    Prismatic: L1 on displacement.
    """

    def forward(
        self,
        pred_sincos: torch.Tensor,
        gt_state: torch.Tensor,
        joint_type: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_sincos: [B, T, 2] predicted sin/cos or displacement/0
            gt_state: [B, T] ground truth angle or displacement
            joint_type: [B] 0=fixed, 1=revolute, 2=prismatic
        """
        B, T = gt_state.shape

        gt_sin = torch.sin(gt_state)
        gt_cos = torch.cos(gt_state)

        is_revolute = (joint_type == 1).float().unsqueeze(-1)
        is_prismatic = (joint_type == 2).float().unsqueeze(-1)

        loss_revolute = (
            F.l1_loss(pred_sincos[:, :, 0], gt_sin, reduction="none")
            + F.l1_loss(pred_sincos[:, :, 1], gt_cos, reduction="none")
        )
        loss_prismatic = F.l1_loss(
            pred_sincos[:, :, 0], gt_state, reduction="none"
        )

        loss = (loss_revolute * is_revolute + loss_prismatic * is_prismatic)
        return loss.mean()


class ArticulationLoss(nn.Module):
    """
    Combined articulation estimation loss.

    L = λ_part * L_part + λ_type * L_type + λ_axis * L_axis
      + λ_pivot * L_pivot + λ_state * L_state
    """

    def __init__(
        self,
        lambda_part: float = 1.0,
        lambda_type: float = 1.0,
        lambda_axis: float = 5.0,
        lambda_pivot: float = 2.0,
        lambda_state: float = 1.0,
    ):
        super().__init__()
        self.lambda_part = lambda_part
        self.lambda_type = lambda_type
        self.lambda_axis = lambda_axis
        self.lambda_pivot = lambda_pivot
        self.lambda_state = lambda_state

        self.axis_loss = AxisAngularLoss()
        self.pivot_loss = PivotLoss()
        self.state_loss = StateLoss()

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        losses = {}

        # Moving part classification
        if "moving_logits" in pred and "moving_part_id" in batch:
            moving_target = torch.zeros(
                pred["moving_logits"].shape, device=pred["moving_logits"].device,
            )
            for i, mid in enumerate(batch["moving_part_id"]):
                if isinstance(mid, int) and mid < moving_target.shape[1]:
                    moving_target[i, mid] = 1.0
            losses["part"] = F.binary_cross_entropy_with_logits(
                pred["moving_logits"], moving_target
            )

        # Joint type classification
        if "joint_type_logits" in pred:
            jtype = batch["joint_type"]
            if jtype.dim() == 1:
                B = jtype.shape[0]
                P = pred["joint_type_logits"].shape[1]
                type_target = jtype.unsqueeze(1).expand(B, P)
                losses["type"] = F.cross_entropy(
                    pred["joint_type_logits"].reshape(-1, pred["joint_type_logits"].shape[-1]),
                    type_target.reshape(-1),
                )

        # Axis loss (use moving part prediction)
        if "axis" in pred:
            gt_axis = batch["joint_axis"]
            if gt_axis.dim() == 2 and pred["axis"].dim() == 3:
                pred_axis = pred["axis"][:, 0, :]
            else:
                pred_axis = pred["axis"]
            losses["axis"] = self.axis_loss(pred_axis, gt_axis)

        # Pivot loss
        if "pivot" in pred:
            gt_pivot = batch["joint_pivot"]
            if gt_pivot.dim() == 2 and pred["pivot"].dim() == 3:
                pred_pivot = pred["pivot"][:, 0, :]
            else:
                pred_pivot = pred["pivot"]
            losses["pivot"] = self.pivot_loss(pred_pivot, gt_pivot)

        # State loss
        if "state_sincos" in pred:
            gt_state = batch["joint_state"]
            jtype = batch["joint_type"]
            pred_state = pred["state_sincos"]
            if pred_state.dim() == 4:
                pred_state = pred_state[:, 0, :, :]
            losses["state"] = self.state_loss(pred_state, gt_state, jtype)

        # Weighted total
        total = torch.tensor(0.0, device=list(losses.values())[0].device)
        for key, val in losses.items():
            weight = getattr(self, f"lambda_{key}", 1.0)
            total = total + weight * val

        losses["total"] = total
        return losses
