"""
Joint parameter decoder head.

Takes per-part fused tokens and predicts:
  - moving part logits
  - joint type logits
  - joint axis (normalized)
  - joint pivot
  - joint state sequence (sin/cos for revolute, displacement for prismatic)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class JointDecoder(nn.Module):
    """
    Decode joint parameters from part-level feature tokens.

    Input: part_tokens [B, P, D] where P = number of parts
    Output: dict of predictions
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_parts: int = 10,
        joint_type_classes: int = 3,
        num_frames: int = 8,
    ):
        super().__init__()
        self.num_parts = num_parts
        self.num_frames = num_frames

        self.moving_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.type_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, joint_type_classes),
        )

        self.axis_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

        self.pivot_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

        # Predict sin/cos pairs per frame for revolute, or displacement for prismatic
        self.state_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_frames * 2),
        )

    def forward(self, part_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            part_tokens: [B, P, D] part-level features

        Returns:
            dict with keys:
              moving_logits: [B, P]
              joint_type_logits: [B, P, C]
              axis: [B, P, 3] (normalized)
              pivot: [B, P, 3]
              state_sincos: [B, P, T, 2]
        """
        B, P, D = part_tokens.shape

        moving_logits = self.moving_head(part_tokens).squeeze(-1)  # [B, P]
        type_logits = self.type_head(part_tokens)  # [B, P, C]

        raw_axis = self.axis_head(part_tokens)  # [B, P, 3]
        axis = F.normalize(raw_axis, dim=-1)

        pivot = self.pivot_head(part_tokens)  # [B, P, 3]

        state_raw = self.state_head(part_tokens)  # [B, P, T*2]
        state_sincos = state_raw.view(B, P, self.num_frames, 2)

        return {
            "moving_logits": moving_logits,
            "joint_type_logits": type_logits,
            "axis": axis,
            "pivot": pivot,
            "state_sincos": state_sincos,
        }
