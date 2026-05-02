"""
E1-C: Late fusion baseline.
Concatenates / gates global 3D and video tokens before joint decoding.
"""
import torch
import torch.nn as nn
from typing import Dict
from .part_slat_pool import PartSLATPool
from .video_pooler import build_video_pooler
from .joint_decoder import JointDecoder


class GatedFusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )
        self.proj = nn.Linear(dim * 2, dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x, y], dim=-1)
        g = self.gate(combined)
        return self.proj(combined) * g


class LateFusionModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        cfg = config["model"]
        dec_cfg = config["decoder"]

        self.hidden_dim = cfg.get("hidden_dim", 256)
        self.max_parts = cfg.get("max_parts", 10)

        self.slat_pool = PartSLATPool(
            input_dim=cfg.get("slat_dim", 8),
            output_dim=self.hidden_dim,
            max_parts=self.max_parts,
        )

        self.video_pooler = build_video_pooler(
            input_dim=cfg.get("video_dim", 1280),
            output_dim=self.hidden_dim,
            method="mean",
        )

        fusion_method = cfg.get("fusion_method", "gated")
        if fusion_method == "concat":
            self.fusion = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        elif fusion_method == "add":
            self.fusion = None
        elif fusion_method == "gated":
            self.fusion = GatedFusion(self.hidden_dim)

        self.fusion_method = fusion_method

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim, nhead=cfg.get("num_heads", 8),
            dim_feedforward=self.hidden_dim * 4, batch_first=True,
            dropout=0.1, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.get("num_layers", 4),
        )

        self.decoder = JointDecoder(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            num_parts=self.max_parts,
            joint_type_classes=dec_cfg.get("joint_type_classes", 3),
        )

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        part_tokens = self.slat_pool.forward_batch(
            batch["slat_feats"], batch["slat_part_ids"], batch["num_parts"],
        )  # [B, P, D]

        video_feats = batch["video_feats"]
        if video_feats.dim() == 2:
            video_feats = video_feats.unsqueeze(1)
        video_global = self.video_pooler(video_feats)  # [B, D]
        if video_global.dim() == 2:
            video_global = video_global.unsqueeze(1).expand_as(part_tokens)

        if self.fusion_method == "concat":
            fused = self.fusion(torch.cat([part_tokens, video_global], dim=-1))
        elif self.fusion_method == "add":
            fused = part_tokens + video_global
        elif self.fusion_method == "gated":
            fused = self.fusion(part_tokens, video_global)

        fused = self.transformer(fused)
        return self.decoder(fused)
