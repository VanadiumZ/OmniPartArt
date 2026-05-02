"""
VGT0: DiffTrack-style Video-DiT correspondence extraction.

Extract temporal correspondence from video generator internal features
by computing feature similarity across frames.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple


class DiTCorrespondenceExtractor:
    """
    Extract point trajectories from Video-DiT hidden states
    by computing feature similarity across temporal frames.

    For a query point p_0 in frame 0:
      - Find corresponding video token
      - Compute feature similarity to tokens in later frames
      - Use soft-argmax to get correspondence
    """

    def __init__(
        self,
        temperature: float = 0.1,
        confidence_threshold: float = 0.5,
        topk: int = 5,
    ):
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold
        self.topk = topk

    @torch.no_grad()
    def extract_tracks(
        self,
        dit_features: Dict[str, torch.Tensor],
        query_points: np.ndarray,
        feature_resolution: Tuple[int, int] = (32, 32),
        num_frames: int = 8,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract trajectories from DiT hidden states.

        Args:
            dit_features: dict of layer features, each [1, T*H*W, D]
            query_points: [N, 2] query pixel positions in frame 0
            feature_resolution: spatial resolution of features
            num_frames: number of temporal frames

        Returns:
            tracks: [N, T, 2] predicted trajectories
            confidence: [N, T] confidence scores
        """
        feature_key = self._select_best_feature(dit_features)
        if feature_key is None:
            N = query_points.shape[0]
            return (
                np.tile(query_points[:, None, :], (1, num_frames, 1)),
                np.zeros((N, num_frames)),
            )

        feats = dit_features[feature_key]
        if feats.dim() == 3:
            feats = feats[0]  # Remove batch dim

        H, W = feature_resolution
        T = num_frames
        total_tokens = feats.shape[0]

        if total_tokens != T * H * W:
            T = total_tokens // (H * W)
            if T * H * W != total_tokens:
                N = query_points.shape[0]
                return (
                    np.tile(query_points[:, None, :], (1, T, 1)),
                    np.zeros((N, T)),
                )

        feats = feats.view(T, H, W, -1)
        feats = F.normalize(feats, dim=-1)

        N = query_points.shape[0]
        tracks = np.zeros((N, T, 2))
        confidence = np.zeros((N, T))

        for i in range(N):
            px, py = query_points[i]
            fx = int(np.clip(px * W, 0, W - 1))
            fy = int(np.clip(py * H, 0, H - 1))
            query_feat = feats[0, fy, fx]  # [D]

            for t in range(T):
                frame_feats = feats[t].view(-1, feats.shape[-1])  # [H*W, D]
                sim = torch.matmul(frame_feats, query_feat)  # [H*W]
                sim = sim / self.temperature

                soft_weights = F.softmax(sim, dim=0)  # [H*W]

                grid_y, grid_x = torch.meshgrid(
                    torch.linspace(0, 1, H, device=feats.device),
                    torch.linspace(0, 1, W, device=feats.device),
                    indexing="ij",
                )
                grid_x = grid_x.flatten()
                grid_y = grid_y.flatten()

                track_x = (soft_weights * grid_x).sum().item()
                track_y = (soft_weights * grid_y).sum().item()

                tracks[i, t] = [track_x, track_y]
                confidence[i, t] = sim.max().item()

        # Normalize confidence to [0, 1]
        if confidence.max() > 0:
            confidence = (confidence - confidence.min()) / (confidence.max() - confidence.min() + 1e-8)

        return tracks, confidence

    def _select_best_feature(self, features: Dict[str, torch.Tensor]) -> Optional[str]:
        """Select the best available feature layer."""
        preference = ["dit_l16_mid", "dit_l12", "dit_l16", "dit_l8", "dit_l24"]
        for key in preference:
            for feat_key in features:
                if key in feat_key:
                    return feat_key
        if features:
            return list(features.keys())[0]
        return None

    @staticmethod
    def generate_query_grid(
        resolution: Tuple[int, int] = (256, 256),
        grid_spacing: int = 16,
    ) -> np.ndarray:
        """Generate a regular grid of query points."""
        H, W = resolution
        ys = np.arange(grid_spacing // 2, H, grid_spacing) / H
        xs = np.arange(grid_spacing // 2, W, grid_spacing) / W
        grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
        return grid.astype(np.float32)
