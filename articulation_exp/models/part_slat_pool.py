"""
Pool SLAT tokens by part ID to get part-level representations.
Supports mean, max, and mean+max pooling strategies.
"""
import torch
import torch.nn as nn
from typing import List, Optional


class PartSLATPool(nn.Module):
    """
    Pool variable-length SLAT tokens per part into fixed-size part tokens.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        max_parts: int = 10,
        pool_method: str = "mean_max",
    ):
        super().__init__()
        self.max_parts = max_parts
        self.pool_method = pool_method

        if pool_method == "mean_max":
            self.proj = nn.Linear(input_dim * 2, output_dim)
        else:
            self.proj = nn.Linear(input_dim, output_dim)

    def forward(
        self,
        slat_feats: torch.Tensor,
        slat_part_ids: torch.Tensor,
        num_parts: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            slat_feats: [N, D] SLAT token features (single sample)
            slat_part_ids: [N] integer part assignment for each token
            num_parts: actual number of parts

        Returns:
            [P, D_out] part-level features, padded to max_parts
        """
        device = slat_feats.device
        D = slat_feats.shape[-1]
        P = num_parts or self.max_parts

        mean_pool = torch.zeros(self.max_parts, D, device=device)
        max_pool = torch.full((self.max_parts, D), -1e9, device=device)
        counts = torch.zeros(self.max_parts, device=device)

        for pid in range(min(P, self.max_parts)):
            mask = slat_part_ids == pid
            if mask.any():
                feats_p = slat_feats[mask]
                mean_pool[pid] = feats_p.mean(dim=0)
                max_pool[pid] = feats_p.max(dim=0).values
                counts[pid] = mask.sum().float()

        max_pool = max_pool.clamp(min=0)

        if self.pool_method == "mean":
            pooled = mean_pool
        elif self.pool_method == "max":
            pooled = max_pool
        else:
            pooled = torch.cat([mean_pool, max_pool], dim=-1)

        return self.proj(pooled)  # [P, D_out]

    def forward_batch(
        self,
        slat_feats_list: List[torch.Tensor],
        slat_part_ids_list: List[torch.Tensor],
        num_parts_list: List[int],
    ) -> torch.Tensor:
        """Batch version: returns [B, P, D_out]."""
        parts = []
        for feats, ids, np_ in zip(slat_feats_list, slat_part_ids_list, num_parts_list):
            parts.append(self.forward(feats, ids, np_))
        return torch.stack(parts)
