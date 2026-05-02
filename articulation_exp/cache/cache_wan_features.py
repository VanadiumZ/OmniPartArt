"""
Hook-based Wan / Video-DiT feature extractor.

Extracts VAE latents and DiT hidden states from a video diffusion
model (e.g., Wan2.2) by registering forward hooks on selected layers.
"""
import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple


class WanDiTHookManager:
    """
    Manages forward hooks on Wan/Video-DiT transformer layers
    to capture hidden states at specified layers and timesteps.
    """

    def __init__(self):
        self.cache: Dict[str, List[torch.Tensor]] = {}
        self.handles: List = []
        self.active = False
        self.target_timesteps: Optional[List[int]] = None
        self.current_timestep: int = -1

    def _make_layer_hook(self, layer_name: str):
        def hook_fn(module, inp, out):
            if not self.active:
                return
            if self.target_timesteps and self.current_timestep not in self.target_timesteps:
                return
            key = f"{layer_name}_t{self.current_timestep}"
            if hasattr(out, "detach"):
                self.cache[key] = out.detach().cpu()
            elif isinstance(out, tuple) and hasattr(out[0], "detach"):
                self.cache[key] = out[0].detach().cpu()
        return hook_fn

    def register_layer(self, module: nn.Module, layer_name: str):
        h = module.register_forward_hook(self._make_layer_hook(layer_name))
        self.handles.append(h)

    def set_timestep(self, t: int):
        self.current_timestep = t

    def activate(self):
        self.active = True

    def deactivate(self):
        self.active = False

    def clear(self):
        self.cache.clear()

    def remove_all(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.cache.clear()
        self.active = False


class WanFeatureExtractor:
    """
    Extract and cache video features from Wan/Video-DiT.

    Supports:
      - V0: VAE latent only
      - V1-V3: DiT hidden states at shallow/middle/deep layers
      - V4-V6: Different noise timestep features

    Usage:
        extractor = WanFeatureExtractor(wan_model)
        extractor.register_hooks(layers=[8, 16, 24], timesteps=[200, 500, 800])
        features = extractor.extract(video_frames)
        extractor.save_features(features, save_dir)
    """

    LAYER_PRESETS = {
        "shallow": [4, 8],
        "middle": [12, 16],
        "deep": [20, 24],
    }

    TIMESTEP_PRESETS = {
        "high_noise": [800, 900],
        "mid_noise": [400, 500],
        "low_noise": [100, 200],
    }

    def __init__(self, wan_model=None, vae=None, device: str = "cuda"):
        self.wan_model = wan_model
        self.vae = vae
        self.device = device
        self.hook_mgr = WanDiTHookManager()

    def register_hooks(
        self,
        layers: Optional[List[int]] = None,
        timesteps: Optional[List[int]] = None,
        layer_preset: Optional[str] = None,
        timestep_preset: Optional[str] = None,
    ):
        """Register hooks on specified DiT layers."""
        if layer_preset:
            layers = self.LAYER_PRESETS.get(layer_preset, [16])
        if timestep_preset:
            timesteps = self.TIMESTEP_PRESETS.get(timestep_preset, [500])
        if layers is None:
            layers = [16]
        if timesteps is not None:
            self.hook_mgr.target_timesteps = timesteps

        if self.wan_model is not None:
            blocks = self._get_transformer_blocks()
            for layer_idx in layers:
                if layer_idx < len(blocks):
                    self.hook_mgr.register_layer(
                        blocks[layer_idx], f"dit_l{layer_idx}"
                    )

    def _get_transformer_blocks(self) -> nn.ModuleList:
        """Find the transformer block list in the model."""
        if hasattr(self.wan_model, "blocks"):
            return self.wan_model.blocks
        if hasattr(self.wan_model, "transformer"):
            if hasattr(self.wan_model.transformer, "blocks"):
                return self.wan_model.transformer.blocks
        for name, module in self.wan_model.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) > 10:
                return module
        return nn.ModuleList()

    @torch.no_grad()
    def extract_vae_latent(self, video_frames: torch.Tensor) -> torch.Tensor:
        """Encode video frames to VAE latent space."""
        if self.vae is None:
            return torch.zeros(1, 4, video_frames.shape[1], 32, 32)
        video_frames = video_frames.to(self.device)
        latent = self.vae.encode(video_frames).latent_dist.sample()
        return latent.cpu()

    @torch.no_grad()
    def extract_dit_features(
        self,
        video_latent: torch.Tensor,
        timesteps: Optional[List[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Run partial diffusion forward pass to extract DiT hidden states.
        Uses hook-captured intermediate representations.
        """
        if self.wan_model is None:
            return {}

        self.hook_mgr.clear()
        self.hook_mgr.activate()

        if timesteps is None:
            timesteps = self.hook_mgr.target_timesteps or [500]

        video_latent = video_latent.to(self.device)
        features = {}

        for t in timesteps:
            self.hook_mgr.set_timestep(t)
            t_tensor = torch.tensor([t], device=self.device).long()
            noisy = self._add_noise(video_latent, t_tensor)

            try:
                _ = self.wan_model(noisy, t_tensor)
            except Exception:
                pass

        features.update(self.hook_mgr.cache)
        self.hook_mgr.deactivate()
        return features

    @staticmethod
    def _add_noise(latent: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise scaled by timestep (simplified scheduler)."""
        alpha = 1.0 - t.float() / 1000.0
        noise = torch.randn_like(latent)
        return alpha * latent + (1.0 - alpha) * noise

    def save_features(self, features: Dict[str, torch.Tensor], save_dir: str):
        """Save extracted features to disk."""
        wan_dir = os.path.join(save_dir, "wan")
        os.makedirs(wan_dir, exist_ok=True)

        for key, tensor in features.items():
            if isinstance(tensor, torch.Tensor):
                torch.save(tensor, os.path.join(wan_dir, f"{key}.pt"))
            elif isinstance(tensor, np.ndarray):
                np.save(os.path.join(wan_dir, f"{key}.npy"), tensor)

    def cleanup(self):
        self.hook_mgr.remove_all()
