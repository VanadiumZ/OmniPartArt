"""
Track filtering and quality assessment for generator-derived tracks.

Filters out low-confidence, stationary, or physically implausible tracks.
"""
import numpy as np
from typing import Tuple, Optional


class TrackFilter:
    """Filter and clean up raw trajectory predictions."""

    def __init__(
        self,
        min_confidence: float = 0.3,
        min_displacement: float = 0.01,
        max_velocity: float = 0.5,
        smoothness_threshold: float = 0.1,
    ):
        self.min_confidence = min_confidence
        self.min_displacement = min_displacement
        self.max_velocity = max_velocity
        self.smoothness_threshold = smoothness_threshold

    def filter(
        self,
        tracks: np.ndarray,
        confidence: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Filter tracks based on quality criteria.

        Args:
            tracks: [N, T, 2] raw tracks
            confidence: [N, T] optional confidence scores

        Returns:
            filtered_tracks: [M, T, 2]
            filtered_confidence: [M, T]
            valid_mask: [N] boolean mask
        """
        N, T, _ = tracks.shape
        valid = np.ones(N, dtype=bool)

        # Confidence filter
        if confidence is not None:
            mean_conf = confidence.mean(axis=1)
            valid &= mean_conf >= self.min_confidence

        # Displacement filter (remove stationary tracks)
        displacement = np.linalg.norm(tracks[:, -1] - tracks[:, 0], axis=-1)
        valid &= displacement >= self.min_displacement

        # Velocity filter (remove physically implausible tracks)
        velocities = np.diff(tracks, axis=1)
        max_vel = np.max(np.linalg.norm(velocities, axis=-1), axis=1)
        valid &= max_vel <= self.max_velocity

        # Smoothness filter
        if T >= 3:
            accelerations = np.diff(velocities, axis=1)
            max_acc = np.max(np.linalg.norm(accelerations, axis=-1), axis=1)
            valid &= max_acc <= self.smoothness_threshold

        filtered_tracks = tracks[valid]
        filtered_conf = confidence[valid] if confidence is not None else np.ones((valid.sum(), T))

        return filtered_tracks, filtered_conf, valid

    @staticmethod
    def separate_moving_static(
        tracks: np.ndarray,
        displacement_threshold: float = 0.05,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Separate tracks into moving and static groups.

        Returns:
            moving_tracks, static_tracks
        """
        displacement = np.linalg.norm(tracks[:, -1] - tracks[:, 0], axis=-1)
        is_moving = displacement >= displacement_threshold
        return tracks[is_moving], tracks[~is_moving]
