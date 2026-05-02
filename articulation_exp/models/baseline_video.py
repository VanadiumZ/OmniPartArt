"""
E1-B: Video-only baseline model.
Uses Wan VAE latent or DiT hidden states to predict joint parameters.
"""
import torch
import torch.nn as nn
from typing import Dict
from .video_pooler import build_video_pooler
from .joint_decoder import JointDecoder


class BaselineVideoModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        cfg = config["model"]
        dec_cfg = config["decoder"]

        self.hidden_dim = cfg.get("hidden_dim", 256)
        video_dim = cfg.get("video_dim", 1280)

        self.video_pooler = build_video_pooler(
            input_dim=video_dim,
            output_dim=self.hidden_dim,
            method=cfg.get("spatial_pool", "mean"),
        )

        self.global_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.decoder = JointDecoder(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            num_parts=1,
            joint_type_classes=dec_cfg.get("joint_type_classes", 3),
        )

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        video_feats = batch["video_feats"]  # [B, N, D]
        if video_feats.dim() == 2:
            video_feats = video_feats.unsqueeze(1)

        if hasattr(self.video_pooler, "forward") and isinstance(
            self.video_pooler, nn.Module
        ):
            pooled = self.video_pooler(video_feats)
        else:
            pooled = video_feats.mean(dim=1)

        if pooled.dim() == 2:
            pooled = pooled.unsqueeze(1)

        global_token = self.global_proj(pooled.mean(dim=1, keepdim=True))
        return self.decoder(global_token)
