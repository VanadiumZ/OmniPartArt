"""
TF0 Step 1: Determine which part is the moving part.

Methods:
  - IoU between projected part mask and observed motion mask
  - Average track displacement inside each part mask
"""
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple


class MovingPartProposal:
    """Propose the most likely moving part from part meshes and motion cues."""

    def __init__(self, method: str = "mask_iou"):
        self.method = method

    def propose(
        self,
        part_masks_2d: np.ndarray,
        motion_mask: Optional[np.ndarray] = None,
        motion_tracks: Optional[np.ndarray] = None,
    ) -> Tuple[int, np.ndarray]:
        """
        Args:
            part_masks_2d: [P, H, W] binary masks for each part
            motion_mask: [H, W] observed binary motion mask
            motion_tracks: [N, T, 2] 2D tracks

        Returns:
            (moving_part_id, scores_per_part)
        """
        P = part_masks_2d.shape[0]
        scores = np.zeros(P)

        if self.method == "mask_iou" and motion_mask is not None:
            for i in range(P):
                scores[i] = self._compute_iou(part_masks_2d[i], motion_mask)

        elif self.method == "track_displacement" and motion_tracks is not None:
            for i in range(P):
                scores[i] = self._avg_displacement_in_mask(
                    part_masks_2d[i], motion_tracks
                )

        elif motion_mask is not None:
            for i in range(P):
                scores[i] = self._compute_iou(part_masks_2d[i], motion_mask)

        moving_id = int(np.argmax(scores))
        return moving_id, scores

    @staticmethod
    def _compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        intersection = np.logical_and(mask_a > 0, mask_b > 0).sum()
        union = np.logical_or(mask_a > 0, mask_b > 0).sum()
        if union == 0:
            return 0.0
        return float(intersection / union)

    @staticmethod
    def _avg_displacement_in_mask(
        mask: np.ndarray, tracks: np.ndarray
    ) -> float:
        """Average track displacement for points starting inside the mask."""
        H, W = mask.shape
        start_pos = tracks[:, 0, :]  # [N, 2]
        end_pos = tracks[:, -1, :]

        x = np.clip(start_pos[:, 0].astype(int), 0, W - 1)
        y = np.clip(start_pos[:, 1].astype(int), 0, H - 1)
        inside = mask[y, x] > 0

        if inside.sum() == 0:
            return 0.0

        displacements = np.linalg.norm(end_pos[inside] - start_pos[inside], axis=-1)
        return float(displacements.mean())
