"""
E3: Projective video-to-SLAT fusion.

For each 3D SLAT voxel point, project to 2D video frames using
camera pose, sample video features, and fuse with 3D token.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class ProjectiveSampler(nn.Module):
    """Projects 3D points to 2D and samples video features."""

    def __init__(self, video_dim: int, output_dim: int):
        super().__init__()
        self.temporal_pool = nn.Sequential(
            nn.Linear(video_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(
        self,
        points_3d: torch.Tensor,
        video_feature_maps: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        camera_extrinsics: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            points_3d: [B, N, 3] 3D point positions
            video_feature_maps: [B, T, C, H, W] video feature maps
            camera_intrinsics: [B, 3, 3]
            camera_extrinsics: [B, T, 4, 4]

        Returns:
            [B, N, D] projected and aggregated features
        """
        B, N, _ = points_3d.shape
        T = video_feature_maps.shape[1]
        H, W = video_feature_maps.shape[-2:]

        sampled_feats_all = []
        for t in range(T):
            R = camera_extrinsics[:, t, :3, :3]  # [B, 3, 3]
            tvec = camera_extrinsics[:, t, :3, 3:]  # [B, 3, 1]

            pts_cam = torch.bmm(R, points_3d.transpose(1, 2)) + tvec  # [B, 3, N]
            pts_cam = pts_cam.transpose(1, 2)  # [B, N, 3]

            z = pts_cam[:, :, 2:3].clamp(min=1e-6)
            pts_2d_h = pts_cam / z  # [B, N, 3]

            K = camera_intrinsics  # [B, 3, 3]
            uv = torch.bmm(pts_2d_h, K.transpose(1, 2))[:, :, :2]  # [B, N, 2]

            grid_x = 2.0 * uv[:, :, 0] / W - 1.0
            grid_y = 2.0 * uv[:, :, 1] / H - 1.0
            grid = torch.stack([grid_x, grid_y], dim=-1)  # [B, N, 2]
            grid = grid.unsqueeze(1)  # [B, 1, N, 2]

            feat_t = video_feature_maps[:, t]  # [B, C, H, W]
            sampled = F.grid_sample(
                feat_t, grid, mode="bilinear", align_corners=True,
            )  # [B, C, 1, N]
            sampled = sampled.squeeze(2).transpose(1, 2)  # [B, N, C]
            sampled_feats_all.append(sampled)

        stacked = torch.stack(sampled_feats_all, dim=2)  # [B, N, T, C]
        temporal_mean = stacked.mean(dim=2)  # [B, N, C]
        return self.temporal_pool(temporal_mean)


class ProjectiveFusionModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        cfg = config["model"]
        dec_cfg = config["decoder"]

        slat_dim = cfg.get("slat_dim", 8)
        video_dim = cfg.get("video_dim", 1280)
        fusion_dim = cfg.get("fusion_dim", 768)
        self.max_parts = cfg.get("max_parts", 10)

        self.slat_proj = nn.Linear(slat_dim, fusion_dim)
        self.projective_sampler = ProjectiveSampler(video_dim, fusion_dim)
        self.fuse_mlp = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim),
        )

        from .part_slat_pool import PartSLATPool
        self.part_pool = PartSLATPool(
            input_dim=fusion_dim, output_dim=fusion_dim,
            max_parts=self.max_parts,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim, nhead=cfg.get("num_heads", 8),
            dim_feedforward=fusion_dim * 4, batch_first=True,
            dropout=0.1, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=2,
        )

        from .joint_decoder import JointDecoder
        self.decoder = JointDecoder(
            input_dim=fusion_dim,
            hidden_dim=fusion_dim // 2,
            num_parts=self.max_parts,
            joint_type_classes=dec_cfg.get("joint_type_classes", 3),
        )

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Requires batch to contain camera_intrinsics, camera_extrinsics,
        and video_feature_maps in addition to standard fields.
        If camera info missing, falls back to cross-attention style.
        """
        slat_feats_list = batch["slat_feats"]
        slat_xyz_list = batch.get("slat_xyz", None)

        has_camera = "camera_intrinsics" in batch and "video_feature_maps" in batch

        if has_camera and slat_xyz_list is not None:
            B = len(slat_feats_list)
            max_n = max(f.shape[0] for f in slat_feats_list)
            D = slat_feats_list[0].shape[-1]
            device = batch["camera_intrinsics"].device

            padded_feats = torch.zeros(B, max_n, D, device=device)
            padded_xyz = torch.zeros(B, max_n, 3, device=device)
            for i in range(B):
                n = slat_feats_list[i].shape[0]
                padded_feats[i, :n] = slat_feats_list[i].to(device)
                padded_xyz[i, :n] = slat_xyz_list[i].to(device)

            slat_3d = self.slat_proj(padded_feats)
            video_proj = self.projective_sampler(
                padded_xyz, batch["video_feature_maps"],
                batch["camera_intrinsics"], batch["camera_extrinsics"],
            )
            fused = self.fuse_mlp(torch.cat([slat_3d, video_proj], dim=-1))
        else:
            # Fallback: simple projection without camera
            from .part_slat_pool import PartSLATPool
            part_tokens = self.part_pool.forward_batch(
                slat_feats_list, batch["slat_part_ids"], batch["num_parts"],
            )
            fused = self.transformer(part_tokens)
            return self.decoder(fused)

        # Pool fused per-point features to part-level
        part_tokens = self.part_pool.forward_batch(
            [fused[i] for i in range(fused.shape[0])],
            batch["slat_part_ids"],
            batch["num_parts"],
        )
        part_tokens = self.transformer(part_tokens)
        return self.decoder(part_tokens)
