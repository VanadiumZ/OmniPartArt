"""
TF0/TF1: Score joint candidates using geometry, motion, and video prior.

Scoring components:
  - E_mask: rendered mask IoU with observed motion mask
  - E_track: track reprojection error
  - E_collision: interpenetration penalty
  - E_range: joint range plausibility
  - E_smooth: state smoothness
  - E_wan: (TF1) Wan feature similarity between real and candidate videos
"""
import numpy as np
from typing import Dict, List, Optional
from .joint_candidate_generator import JointCandidate


class CandidateScorer:
    """Score and rank joint candidates."""

    def __init__(
        self,
        lambda_mask: float = 1.0,
        lambda_track: float = 1.0,
        lambda_collision: float = 0.5,
        lambda_range: float = 0.3,
        lambda_smooth: float = 0.2,
        lambda_wan: float = 0.0,
    ):
        self.weights = {
            "mask": lambda_mask,
            "track": lambda_track,
            "collision": lambda_collision,
            "range": lambda_range,
            "smooth": lambda_smooth,
            "wan": lambda_wan,
        }

    def score_candidates(
        self,
        candidates: List[JointCandidate],
        moving_part_points: np.ndarray,
        static_part_points: np.ndarray,
        motion_mask: Optional[np.ndarray] = None,
        motion_tracks: Optional[np.ndarray] = None,
        camera_intrinsics: Optional[np.ndarray] = None,
        num_frames: int = 8,
        wan_features_real: Optional[np.ndarray] = None,
        wan_features_fn=None,
    ) -> List[JointCandidate]:
        """Score each candidate and sort by total score (lower is better)."""

        for cand in candidates:
            scores = {}

            articulated_pts = self._simulate_motion(
                moving_part_points, cand, num_frames
            )

            if motion_mask is not None:
                scores["mask"] = self._mask_score(
                    articulated_pts, motion_mask, camera_intrinsics
                )

            if motion_tracks is not None:
                scores["track"] = self._track_score(
                    articulated_pts, motion_tracks, camera_intrinsics
                )

            scores["collision"] = self._collision_score(
                articulated_pts, static_part_points
            )

            scores["range"] = self._range_score(cand)
            scores["smooth"] = self._smoothness_score(articulated_pts)

            if self.weights["wan"] > 0 and wan_features_real is not None and wan_features_fn:
                scores["wan"] = self._wan_score(
                    articulated_pts, wan_features_real, wan_features_fn
                )

            total = sum(
                self.weights.get(k, 0) * v for k, v in scores.items()
            )
            cand.score = total

        return sorted(candidates, key=lambda c: c.score)

    def _simulate_motion(
        self, points: np.ndarray, cand: JointCandidate, num_frames: int
    ) -> np.ndarray:
        """Simulate articulated motion for the candidate over T frames."""
        T = num_frames
        results = np.zeros((T, len(points), 3))

        q_min, q_max = cand.state_range
        states = np.linspace(q_min, q_max, T)

        axis = cand.axis / (np.linalg.norm(cand.axis) + 1e-8)
        pivot = cand.pivot

        for t, q in enumerate(states):
            if cand.joint_type == "revolute":
                centered = points - pivot
                cos_q = np.cos(q)
                sin_q = np.sin(q)
                dot = (centered * axis).sum(axis=-1, keepdims=True)
                cross = np.cross(axis, centered)
                rotated = centered * cos_q + cross * sin_q + axis * dot * (1 - cos_q)
                results[t] = rotated + pivot
            elif cand.joint_type == "prismatic":
                results[t] = points + q * axis

        return results  # [T, N, 3]

    def _mask_score(
        self, articulated_pts: np.ndarray, motion_mask: np.ndarray,
        camera_K: Optional[np.ndarray] = None,
    ) -> float:
        """Simple projected mask IoU (orthographic approximation)."""
        H, W = motion_mask.shape
        T = articulated_pts.shape[0]

        total_iou = 0.0
        for t in range(T):
            pts = articulated_pts[t]
            x = ((pts[:, 0] + 0.5) * W).astype(int).clip(0, W - 1)
            y = ((pts[:, 1] + 0.5) * H).astype(int).clip(0, H - 1)

            rendered = np.zeros_like(motion_mask)
            rendered[y, x] = 1.0

            intersection = (rendered * motion_mask).sum()
            union = rendered.sum() + motion_mask.sum() - intersection
            if union > 0:
                total_iou += intersection / union

        return 1.0 - total_iou / T

    def _track_score(
        self, articulated_pts: np.ndarray, tracks: np.ndarray,
        camera_K: Optional[np.ndarray] = None,
    ) -> float:
        """Average reprojection error against observed tracks."""
        T = min(articulated_pts.shape[0], tracks.shape[1])
        N = min(articulated_pts.shape[1], tracks.shape[0])

        if N == 0 or T == 0:
            return 1.0

        errors = []
        for t in range(T):
            pts_2d = articulated_pts[t, :N, :2]
            gt_2d = tracks[:N, t, :]
            error = np.linalg.norm(pts_2d - gt_2d, axis=-1).mean()
            errors.append(error)

        return float(np.mean(errors))

    def _collision_score(
        self, articulated_pts: np.ndarray, static_pts: np.ndarray
    ) -> float:
        """Penalize penetration into static parts."""
        if len(static_pts) == 0:
            return 0.0

        T = articulated_pts.shape[0]
        total_pen = 0.0
        threshold = 0.01

        for t in range(T):
            pts = articulated_pts[t]
            if len(pts) == 0:
                continue
            diffs = pts[:, None, :] - static_pts[None, :, :]
            dists = np.linalg.norm(diffs, axis=-1).min(axis=-1)
            penetration = np.maximum(threshold - dists, 0).mean()
            total_pen += penetration

        return total_pen / T

    def _range_score(self, cand: JointCandidate) -> float:
        """Penalize unreasonable joint ranges."""
        q_min, q_max = cand.state_range
        if cand.joint_type == "revolute":
            if q_max - q_min > np.pi:
                return 0.5
            if q_max - q_min < 0.1:
                return 0.3
        return 0.0

    def _smoothness_score(self, articulated_pts: np.ndarray) -> float:
        """Encourage smooth motion trajectory."""
        if articulated_pts.shape[0] < 3:
            return 0.0
        vel = np.diff(articulated_pts, axis=0)
        acc = np.diff(vel, axis=0)
        return float(np.mean(np.linalg.norm(acc, axis=-1)))

    def _wan_score(
        self, articulated_pts: np.ndarray,
        real_features: np.ndarray,
        extract_fn,
    ) -> float:
        """TF1: Compare Wan features of real vs rendered candidate video."""
        try:
            cand_features = extract_fn(articulated_pts)
            dist = np.linalg.norm(real_features - cand_features)
            return float(dist)
        except Exception:
            return 1.0
