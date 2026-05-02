"""
VGT1: KL-tracing / counterfactual flow extraction.

Perturb a region in the first frame, compare generator outputs
(clean vs perturbed) to find where the perturbation propagates,
yielding correspondence / flow information.
"""
import torch
import numpy as np
from typing import Dict, Optional, Tuple


class KLTracingFlowExtractor:
    """
    Extract flow/correspondence by comparing clean vs perturbed generation.

    Steps:
      1. Select query patches
      2. Add small tracer perturbation to first frame
      3. Run generator twice (clean + perturbed)
      4. Compare outputs to find propagation
      5. Convert to flow/tracks
    """

    def __init__(
        self,
        perturbation_strength: float = 0.1,
        patch_size: int = 8,
    ):
        self.perturbation_strength = perturbation_strength
        self.patch_size = patch_size

    @torch.no_grad()
    def extract_flow(
        self,
        video_latent: torch.Tensor,
        generator_fn=None,
        query_points: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract optical-flow-like signal via perturbation.

        Args:
            video_latent: [1, C, T, H, W] clean video latent
            generator_fn: callable that generates from latent
            query_points: [N, 2] optional query positions

        Returns:
            flow: [T-1, H, W, 2] dense flow
            tracks: [N, T, 2] sparse tracks
        """
        if generator_fn is None:
            C, T, H, W = video_latent.shape[1:]
            return (
                np.zeros((T - 1, H, W, 2), dtype=np.float32),
                np.zeros((0, T, 2), dtype=np.float32),
            )

        clean_output = generator_fn(video_latent)

        C, T, H, W = video_latent.shape[1:]
        flow_all = np.zeros((T - 1, H, W, 2), dtype=np.float32)

        if query_points is None:
            query_points = self._generate_sparse_queries(H, W)

        N = query_points.shape[0]
        tracks = np.tile(query_points[:, None, :], (1, T, 1))

        for qi in range(N):
            px, py = query_points[qi]
            ix = int(np.clip(px * W, 0, W - 1))
            iy = int(np.clip(py * H, 0, H - 1))

            perturbed = video_latent.clone()
            ps = self.patch_size // 2
            y0 = max(0, iy - ps)
            y1 = min(H, iy + ps + 1)
            x0 = max(0, ix - ps)
            x1 = min(W, ix + ps + 1)
            perturbed[:, :, 0, y0:y1, x0:x1] += (
                torch.randn_like(perturbed[:, :, 0, y0:y1, x0:x1])
                * self.perturbation_strength
            )

            perturbed_output = generator_fn(perturbed)

            diff = (clean_output - perturbed_output).abs()
            if diff.dim() == 5:
                diff = diff[0]  # [C, T, H, W]

            for t in range(1, T):
                diff_t = diff[:, t].sum(dim=0)  # [H, W]
                if diff_t.max() > 0:
                    diff_norm = diff_t / diff_t.max()
                    peak_flat = diff_norm.flatten().argmax().item()
                    peak_y = peak_flat // W
                    peak_x = peak_flat % W
                    tracks[qi, t] = [peak_x / W, peak_y / H]

        return flow_all, tracks

    def _generate_sparse_queries(
        self, H: int, W: int, spacing: int = 16
    ) -> np.ndarray:
        ys = np.arange(spacing // 2, H, spacing) / H
        xs = np.arange(spacing // 2, W, spacing) / W
        grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
        return grid.astype(np.float32)
