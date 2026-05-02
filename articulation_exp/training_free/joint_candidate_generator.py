"""
TF0 Step 2: Generate candidate joints for revolute and prismatic types.

Revolute axis candidates:
  - Principal axes of object/part bboxes
  - PCA axes of moving part points
  - Contact-boundary tangent directions

Revolute pivot candidates:
  - Contact region centroid
  - BBox edge centers
  - Closest static-moving boundary

Prismatic axis candidates:
  - Average track displacement direction
  - PCA direction of motion
  - BBox principal axes
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class JointCandidate:
    joint_type: str  # "revolute" or "prismatic"
    axis: np.ndarray  # [3]
    pivot: np.ndarray  # [3] (relevant for revolute)
    score: float = 0.0
    state_range: Tuple[float, float] = (0.0, 1.5)


class JointCandidateGenerator:
    """Generate a set of candidate joint parameters for scoring."""

    def __init__(
        self,
        num_axis_candidates: int = 20,
        num_pivot_candidates: int = 10,
        joint_types: List[str] = None,
    ):
        self.num_axis = num_axis_candidates
        self.num_pivot = num_pivot_candidates
        self.joint_types = joint_types or ["revolute", "prismatic"]

    def generate(
        self,
        moving_part_points: np.ndarray,
        static_part_points: np.ndarray,
        moving_bbox: np.ndarray,
        object_bbox: np.ndarray,
        motion_tracks: Optional[np.ndarray] = None,
    ) -> List[JointCandidate]:
        candidates = []

        if "revolute" in self.joint_types:
            axes = self._generate_revolute_axes(
                moving_part_points, moving_bbox, object_bbox
            )
            pivots = self._generate_revolute_pivots(
                moving_part_points, static_part_points, moving_bbox
            )
            for ax in axes:
                for pv in pivots:
                    candidates.append(JointCandidate(
                        joint_type="revolute", axis=ax, pivot=pv,
                        state_range=(0.0, np.pi / 2),
                    ))

        if "prismatic" in self.joint_types:
            axes = self._generate_prismatic_axes(
                moving_part_points, moving_bbox, motion_tracks
            )
            pivot = moving_part_points.mean(axis=0) if len(moving_part_points) > 0 else np.zeros(3)
            for ax in axes:
                candidates.append(JointCandidate(
                    joint_type="prismatic", axis=ax, pivot=pivot,
                    state_range=(0.0, 0.3),
                ))

        return candidates

    def _generate_revolute_axes(
        self, points: np.ndarray, part_bbox: np.ndarray, obj_bbox: np.ndarray
    ) -> List[np.ndarray]:
        axes = []
        canonical = [
            np.array([1, 0, 0], dtype=np.float32),
            np.array([0, 1, 0], dtype=np.float32),
            np.array([0, 0, 1], dtype=np.float32),
        ]
        axes.extend(canonical)

        if len(points) >= 3:
            try:
                centered = points - points.mean(axis=0)
                _, _, Vt = np.linalg.svd(centered, full_matrices=False)
                for i in range(min(3, Vt.shape[0])):
                    ax = Vt[i]
                    norm = np.linalg.norm(ax)
                    if norm > 1e-8:
                        axes.append(ax / norm)
            except np.linalg.LinAlgError:
                pass

        # Diagonal and edge directions from bbox
        if part_bbox.shape[0] >= 2:
            bbox_min, bbox_max = part_bbox[0], part_bbox[1] if part_bbox.ndim == 2 else (part_bbox[:3], part_bbox[3:])
            for i in range(3):
                edge = np.zeros(3, dtype=np.float32)
                edge[i] = 1.0
                axes.append(edge)

        return axes[:self.num_axis]

    def _generate_revolute_pivots(
        self, moving_pts: np.ndarray, static_pts: np.ndarray, moving_bbox: np.ndarray
    ) -> List[np.ndarray]:
        pivots = []

        # Contact region centroid
        if len(moving_pts) > 0 and len(static_pts) > 0:
            from scipy.spatial import cKDTree
            try:
                tree = cKDTree(static_pts)
                dists, _ = tree.query(moving_pts, k=1)
                contact_threshold = np.percentile(dists, 10)
                contact_mask = dists < contact_threshold
                if contact_mask.sum() > 0:
                    pivots.append(moving_pts[contact_mask].mean(axis=0))
            except Exception:
                pass

        # BBox edge centers
        if moving_bbox.ndim == 1 and len(moving_bbox) == 6:
            bbox_min = moving_bbox[:3]
            bbox_max = moving_bbox[3:]
            center = (bbox_min + bbox_max) / 2
            for i in range(3):
                for sign in [-1, 1]:
                    pv = center.copy()
                    pv[i] = bbox_min[i] if sign == -1 else bbox_max[i]
                    pivots.append(pv)
        elif moving_bbox.ndim == 2:
            center = moving_bbox.mean(axis=0)
            pivots.append(center)

        # Moving part centroid as fallback
        if len(moving_pts) > 0:
            pivots.append(moving_pts.mean(axis=0))

        return pivots[:self.num_pivot]

    def _generate_prismatic_axes(
        self,
        points: np.ndarray,
        bbox: np.ndarray,
        tracks: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        axes = []

        if tracks is not None and len(tracks) > 0:
            displacement = tracks[:, -1, :] - tracks[:, 0, :]
            mean_disp = displacement.mean(axis=0)
            norm = np.linalg.norm(mean_disp)
            if norm > 1e-8:
                axes.append(np.append(mean_disp / norm, 0.0)[:3])

        canonical = [
            np.array([1, 0, 0], dtype=np.float32),
            np.array([0, 1, 0], dtype=np.float32),
            np.array([0, 0, 1], dtype=np.float32),
        ]
        axes.extend(canonical)

        if len(points) >= 3:
            try:
                centered = points - points.mean(axis=0)
                _, _, Vt = np.linalg.svd(centered, full_matrices=False)
                axes.append(Vt[0] / (np.linalg.norm(Vt[0]) + 1e-8))
            except np.linalg.LinAlgError:
                pass

        return axes[:self.num_axis]
