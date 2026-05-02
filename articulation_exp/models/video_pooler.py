"""
Temporal-spatial pooling for video features.
Supports mean pooling and Perceiver-style cross-attention pooling.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MeanVideoPooler(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, video_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video_feats: [B, T*H*W, D] or [B, N, D]
        Returns:
            [B, D_out]
        """
        pooled = video_feats.mean(dim=1)
        return self.proj(pooled)


class PerceiverVideoPooler(nn.Module):
    """Perceiver-style cross-attention pooler for video tokens."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_latents: int = 32,
        num_heads: int = 8,
        num_layers: int = 2,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, num_latents, output_dim) * 0.02)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.MultiheadAttention(
                embed_dim=output_dim, num_heads=num_heads, batch_first=True,
            ))
        self.input_proj = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, video_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video_feats: [B, N, D_in]
        Returns:
            [B, L, D_out] where L = num_latents
        """
        B = video_feats.shape[0]
        kv = self.input_proj(video_feats)
        q = self.latents.expand(B, -1, -1)

        for attn in self.layers:
            delta, _ = attn(q, kv, kv)
            q = self.norm(q + delta)

        return q


def build_video_pooler(
    input_dim: int,
    output_dim: int,
    method: str = "mean",
    **kwargs,
) -> nn.Module:
    if method == "mean":
        return MeanVideoPooler(input_dim, output_dim)
    elif method == "perceiver":
        return PerceiverVideoPooler(input_dim, output_dim, **kwargs)
    raise ValueError(f"Unknown pooling method: {method}")
