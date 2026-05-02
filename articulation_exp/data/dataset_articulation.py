"""
Articulation dataset for joint estimation experiments.
Loads cached OmniPart SLAT features, Wan video features, and GT joint labels.
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional


class ArticulationDataset(Dataset):
    """
    Dataset that loads pre-cached features for articulation experiments.

    Expected cache layout per sample:
        {root}/{category}/{object_id}/
            gt_joint.json
            omnipart/
                slat_feats.pt          # [N3, C3]
                slat_xyz.pt            # [N3, 3]
                slat_part_ids.pt       # [N3]
                part_bboxes.npy        # [P, 6]
            wan/
                vae_latent.pt
                dit_l16_mid.pt         # optional higher-quality features
            motion/
                seganymotion_masks.npy  # optional
                seganymotion_tracks.npy # optional
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        categories: Optional[List[str]] = None,
        max_parts: int = 10,
        video_feature_key: str = "vae_latent",
        single_moving_part: bool = True,
        num_frames: int = 8,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.max_parts = max_parts
        self.video_feature_key = video_feature_key
        self.single_moving_part = single_moving_part
        self.num_frames = num_frames

        self.samples = self._discover_samples(categories)
        self._apply_split(split)

    def _discover_samples(self, categories: Optional[List[str]]) -> List[Dict]:
        samples = []
        if not os.path.exists(self.root_dir):
            return samples

        for cat in sorted(os.listdir(self.root_dir)):
            cat_dir = os.path.join(self.root_dir, cat)
            if not os.path.isdir(cat_dir):
                continue
            if categories and cat not in categories:
                continue
            for obj_id in sorted(os.listdir(cat_dir)):
                obj_dir = os.path.join(cat_dir, obj_id)
                gt_path = os.path.join(obj_dir, "gt_joint.json")
                if not os.path.isfile(gt_path):
                    continue
                samples.append({
                    "object_id": obj_id,
                    "category": cat,
                    "dir": obj_dir,
                    "gt_path": gt_path,
                })
        return samples

    def _apply_split(self, split: str):
        n = len(self.samples)
        if n == 0:
            return
        np.random.seed(42)
        indices = np.random.permutation(n)
        train_end = int(0.8 * n)
        val_end = int(0.9 * n)

        if split == "train":
            self.samples = [self.samples[i] for i in indices[:train_end]]
        elif split == "val":
            self.samples = [self.samples[i] for i in indices[train_end:val_end]]
        elif split == "test":
            self.samples = [self.samples[i] for i in indices[val_end:]]

    def __len__(self) -> int:
        return len(self.samples)

    def _load_gt(self, gt_path: str) -> Dict:
        with open(gt_path, "r") as f:
            gt = json.load(f)
        return gt

    def _pad_parts(self, tensor: torch.Tensor, target_parts: int, dim: int = 0) -> torch.Tensor:
        """Pad tensor along `dim` to `target_parts`."""
        pad_size = target_parts - tensor.shape[dim]
        if pad_size <= 0:
            return tensor.narrow(dim, 0, target_parts)
        pad_shape = list(tensor.shape)
        pad_shape[dim] = pad_size
        return torch.cat([tensor, torch.zeros(pad_shape, dtype=tensor.dtype)], dim=dim)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        obj_dir = sample["dir"]
        gt = self._load_gt(sample["gt_path"])

        out = {
            "object_id": sample["object_id"],
            "category": sample["category"],
        }

        # --- GT joint labels ---
        out["joint_type"] = self._encode_joint_type(gt.get("joint_type", "fixed"))
        out["joint_axis"] = torch.tensor(gt.get("joint_axis", [0, 0, 1]), dtype=torch.float32)
        out["joint_pivot"] = torch.tensor(gt.get("joint_pivot", [0, 0, 0]), dtype=torch.float32)
        out["moving_part_id"] = gt.get("moving_part_id", 0)
        out["num_parts"] = gt.get("num_parts", 1)

        state = gt.get("joint_state", [0.0] * self.num_frames)
        out["joint_state"] = torch.tensor(state[:self.num_frames], dtype=torch.float32)

        # --- OmniPart features ---
        omni_dir = os.path.join(obj_dir, "omnipart")
        slat_feats_path = os.path.join(omni_dir, "slat_feats.pt")
        if os.path.exists(slat_feats_path):
            out["slat_feats"] = torch.load(slat_feats_path, map_location="cpu")
            out["slat_xyz"] = torch.load(os.path.join(omni_dir, "slat_xyz.pt"), map_location="cpu")
            out["slat_part_ids"] = torch.load(os.path.join(omni_dir, "slat_part_ids.pt"), map_location="cpu")
        else:
            out["slat_feats"] = torch.zeros(64, 8)
            out["slat_xyz"] = torch.zeros(64, 3)
            out["slat_part_ids"] = torch.zeros(64, dtype=torch.long)

        bbox_path = os.path.join(omni_dir, "part_bboxes.npy")
        if os.path.exists(bbox_path):
            bboxes = torch.from_numpy(np.load(bbox_path)).float()
            out["part_bboxes"] = self._pad_parts(bboxes, self.max_parts)
        else:
            out["part_bboxes"] = torch.zeros(self.max_parts, 6)

        # --- Video features ---
        wan_dir = os.path.join(obj_dir, "wan")
        feat_path = os.path.join(wan_dir, f"{self.video_feature_key}.pt")
        if os.path.exists(feat_path):
            out["video_feats"] = torch.load(feat_path, map_location="cpu")
        else:
            out["video_feats"] = torch.zeros(1, 1280)

        # --- Motion cues (optional) ---
        motion_dir = os.path.join(obj_dir, "motion")
        tracks_path = os.path.join(motion_dir, "seganymotion_tracks.npy")
        if os.path.exists(tracks_path):
            out["motion_tracks"] = torch.from_numpy(np.load(tracks_path)).float()
        masks_path = os.path.join(motion_dir, "seganymotion_masks.npy")
        if os.path.exists(masks_path):
            out["motion_masks"] = torch.from_numpy(np.load(masks_path)).float()

        return out

    @staticmethod
    def _encode_joint_type(jtype: str) -> int:
        mapping = {"fixed": 0, "revolute": 1, "prismatic": 2}
        return mapping.get(jtype, 0)

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """Custom collate that handles variable-length SLAT tokens."""
        out = {}
        keys_to_stack = [
            "joint_type", "joint_axis", "joint_pivot", "joint_state",
            "part_bboxes", "video_feats",
        ]
        keys_to_list = [
            "object_id", "category", "slat_feats", "slat_xyz",
            "slat_part_ids", "moving_part_id", "num_parts",
        ]

        for key in keys_to_stack:
            if key in batch[0]:
                vals = [b[key] for b in batch]
                if isinstance(vals[0], torch.Tensor):
                    out[key] = torch.stack(vals)
                else:
                    out[key] = torch.tensor(vals)

        for key in keys_to_list:
            if key in batch[0]:
                out[key] = [b[key] for b in batch]

        # Optional motion cues
        for key in ["motion_tracks", "motion_masks"]:
            if key in batch[0]:
                out[key] = [b[key] for b in batch]

        return out
