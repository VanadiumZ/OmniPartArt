"""
E1-A: 3D-only baseline model.
Uses OmniPart SLAT tokens + part bboxes to predict joint parameters.
"""
import torch
import torch.nn as nn
from typing import Dict, List
from .part_slat_pool import PartSLATPool
from .joint_decoder import JointDecoder


class Baseline3DModel(nn.Module):
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
            pool_method=cfg.get("pool_method", "mean_max"),
        )

        self.bbox_proj = nn.Linear(6, self.hidden_dim)

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
        slat_feats_list = batch["slat_feats"]
        slat_part_ids_list = batch["slat_part_ids"]
        num_parts_list = batch["num_parts"]

        part_tokens = self.slat_pool.forward_batch(
            slat_feats_list, slat_part_ids_list, num_parts_list,
        )  # [B, P, D]

        bbox_feats = self.bbox_proj(batch["part_bboxes"])  # [B, P, D]
        tokens = part_tokens + bbox_feats

        tokens = self.transformer(tokens)  # [B, P, D]
        return self.decoder(tokens)
