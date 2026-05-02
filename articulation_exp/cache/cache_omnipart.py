"""
Hook-based OmniPart SLAT feature extractor and cacher.

Registers forward hooks on OmniPart's pipeline models to intercept
SLAT tokens, sparse coordinates, and part layout information without
modifying the original OmniPart codebase.
"""
import os
import sys
import json
import torch
import numpy as np
from typing import Dict, List, Optional

OMNIPART_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if OMNIPART_ROOT not in sys.path:
    sys.path.insert(0, OMNIPART_ROOT)


class OmniPartHookManager:
    """Manages forward hooks on OmniPart pipeline models."""

    def __init__(self):
        self.cache: Dict[str, object] = {}
        self.handles: List = []

    def _make_hook(self, name: str):
        def hook_fn(module, inp, out):
            if hasattr(out, "feats"):
                self.cache[name] = {
                    "feats": out.feats.detach().cpu(),
                    "coords": out.coords.detach().cpu(),
                }
            elif hasattr(out, "detach"):
                self.cache[name] = out.detach().cpu()
            else:
                self.cache[name] = out
        return hook_fn

    def register(self, module, name: str):
        h = module.register_forward_hook(self._make_hook(name))
        self.handles.append(h)
        return h

    def clear(self):
        self.cache.clear()

    def remove_all(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.cache.clear()


class OmniPartFeatureCache:
    """
    Extracts and caches OmniPart features for articulation experiments.

    Usage:
        cacher = OmniPartFeatureCache(pipeline, save_dir="data_cache/...")
        cacher.process_sample(image, mask, object_id, category, gt_joint)
    """

    def __init__(self, pipeline, save_dir: str):
        self.pipeline = pipeline
        self.save_dir = save_dir
        self.hook_mgr = OmniPartHookManager()

        if hasattr(pipeline, "models"):
            if "slat_flow_model" in pipeline.models:
                self.hook_mgr.register(
                    pipeline.models["slat_flow_model"], "slat_flow_out"
                )
            if "sparse_structure_decoder" in pipeline.models:
                self.hook_mgr.register(
                    pipeline.models["sparse_structure_decoder"], "ss_decoder_out"
                )

    def process_sample(
        self,
        image,
        ordered_mask,
        object_id: str,
        category: str,
        gt_joint: Dict,
        bbox_gen_model=None,
        voxel_coords_path: Optional[str] = None,
        bbox_depth_path: Optional[str] = None,
    ):
        """Run OmniPart inference and cache intermediate features."""
        out_dir = os.path.join(self.save_dir, category, object_id)
        omni_dir = os.path.join(out_dir, "omnipart")
        os.makedirs(omni_dir, exist_ok=True)

        self.hook_mgr.clear()

        try:
            coords = self.pipeline.get_coords(image, save_coords=True)

            if voxel_coords_path and bbox_depth_path and bbox_gen_model:
                from modules.inference_utils import (
                    prepare_bbox_gen_input,
                    prepare_part_synthesis_input,
                )
                bbox_input = prepare_bbox_gen_input(
                    voxel_coords_path, image, ordered_mask
                )
                bboxes = bbox_gen_model.generate(bbox_input)
                np.save(os.path.join(omni_dir, "part_bboxes.npy"), bboxes)

                synth_input = prepare_part_synthesis_input(
                    voxel_coords_path, bbox_depth_path, ordered_mask
                )
                part_layouts = synth_input["part_layouts"]
                synth_coords = synth_input["coords"]
                masks = synth_input["masks"]

                slat_result = self.pipeline.get_slat(
                    image, synth_coords, part_layouts, masks,
                    formats=["mesh"],
                )
        except Exception as e:
            print(f"Error processing {object_id}: {e}")
            return

        if "slat_flow_out" in self.hook_mgr.cache:
            slat_data = self.hook_mgr.cache["slat_flow_out"]
            torch.save(slat_data["feats"], os.path.join(omni_dir, "slat_feats.pt"))
            torch.save(slat_data["coords"], os.path.join(omni_dir, "slat_xyz.pt"))

            if voxel_coords_path and bbox_depth_path:
                part_ids = self._assign_part_ids(slat_data["coords"], part_layouts)
                torch.save(part_ids, os.path.join(omni_dir, "slat_part_ids.pt"))

        with open(os.path.join(out_dir, "gt_joint.json"), "w") as f:
            json.dump(gt_joint, f, indent=2)

    @staticmethod
    def _assign_part_ids(coords: torch.Tensor, part_layouts: List) -> torch.Tensor:
        """Assign part IDs to each SLAT token based on part_layouts slicing."""
        part_ids = torch.zeros(coords.shape[0], dtype=torch.long)
        for pid, layout in enumerate(part_layouts):
            if isinstance(layout, slice):
                part_ids[layout] = pid
        return part_ids

    def cleanup(self):
        self.hook_mgr.remove_all()
