"""
E2: Video-to-SLAT cross-attention fusion.

Core idea: SLAT tokens as queries, video tokens as keys/values.
Each 3D part token queries video features for motion evidence.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class VideoToSLATCrossAttention(nn.Module):
    """Single cross-attention layer: SLAT queries video tokens."""

    def __init__(self, c_slat: int, c_video: int, d: int = 768, num_heads: int = 8):
        super().__init__()
        self.q_proj = nn.Linear(c_slat, d)
        self.k_proj = nn.Linear(c_video, d)
        self.v_proj = nn.Linear(c_video, d)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d, num_heads=num_heads, batch_first=True,
        )
        self.gate = nn.Sequential(
            nn.Linear(d, d),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Linear(d, c_slat)
        self.norm_q = nn.LayerNorm(d)
        self.norm_out = nn.LayerNorm(c_slat)

    def forward(
        self,
        slat_tokens: torch.Tensor,
        video_tokens: torch.Tensor,
        slat_pos: Optional[torch.Tensor] = None,
        video_pos: Optional[torch.Tensor] = None,
        video_mask: Optional[torch.Tensor] = None,
    ) -> tuple:
        q = self.q_proj(slat_tokens)
        if slat_pos is not None:
            q = q + slat_pos

        k = self.k_proj(video_tokens)
        v = self.v_proj(video_tokens)
        if video_pos is not None:
            k = k + video_pos

        q = self.norm_q(q)
        delta, attn_weights = self.cross_attn(q, k, v, key_padding_mask=video_mask)
        gated = self.gate(delta) * delta
        fused = self.norm_out(slat_tokens + self.out_proj(gated))
        return fused, attn_weights


class CrossAttentionFusionModel(nn.Module):
    """
    Full model: pool SLAT by part -> cross-attend video -> decode joints.
    """

    def __init__(self, config: dict):
        super().__init__()
        cfg = config["model"]
        dec_cfg = config["decoder"]

        slat_dim = cfg.get("slat_dim", 8)
        video_dim = cfg.get("video_dim", 1280)
        fusion_dim = cfg.get("fusion_dim", 768)
        num_heads = cfg.get("num_heads", 8)
        self.max_parts = cfg.get("max_parts", 10)
        num_layers = cfg.get("num_cross_attn_layers", 2)

        from .part_slat_pool import PartSLATPool
        self.slat_pool = PartSLATPool(
            input_dim=slat_dim, output_dim=fusion_dim,
            max_parts=self.max_parts,
        )

        self.video_input_proj = nn.Linear(video_dim, fusion_dim)

        self.cross_attn_layers = nn.ModuleList([
            VideoToSLATCrossAttention(
                c_slat=fusion_dim, c_video=fusion_dim,
                d=fusion_dim, num_heads=num_heads,
            )
            for _ in range(num_layers)
        ])

        self.bbox_proj = nn.Linear(6, fusion_dim)

        from .joint_decoder import JointDecoder
        self.decoder = JointDecoder(
            input_dim=fusion_dim,
            hidden_dim=fusion_dim // 2,
            num_parts=self.max_parts,
            joint_type_classes=dec_cfg.get("joint_type_classes", 3),
        )

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        part_tokens = self.slat_pool.forward_batch(
            batch["slat_feats"], batch["slat_part_ids"], batch["num_parts"],
        )  # [B, P, D]

        part_tokens = part_tokens + self.bbox_proj(batch["part_bboxes"])

        video_feats = batch["video_feats"]
        if video_feats.dim() == 2:
            video_feats = video_feats.unsqueeze(1)
        video_tokens = self.video_input_proj(video_feats)  # [B, N, D]

        attn_weights_all = []
        for layer in self.cross_attn_layers:
            part_tokens, attn_w = layer(part_tokens, video_tokens)
            attn_weights_all.append(attn_w)

        preds = self.decoder(part_tokens)
        preds["attn_weights"] = attn_weights_all
        return preds
