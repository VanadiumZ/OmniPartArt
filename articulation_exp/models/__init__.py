from .joint_decoder import JointDecoder
from .part_slat_pool import PartSLATPool
from .video_pooler import MeanVideoPooler, PerceiverVideoPooler, build_video_pooler
from .fusion_late import LateFusionModel
from .fusion_cross_attn import CrossAttentionFusionModel
from .fusion_projective import ProjectiveFusionModel
from .baseline_3d import Baseline3DModel
from .baseline_video import BaselineVideoModel


def build_model(config: dict):
    """Factory function to build model from config dict."""
    name = config["model"]["name"]
    builders = {
        "baseline_3d": Baseline3DModel,
        "baseline_video": BaselineVideoModel,
        "late_fusion": LateFusionModel,
        "cross_fusion": CrossAttentionFusionModel,
        "projective_fusion": ProjectiveFusionModel,
    }
    if name not in builders:
        raise ValueError(f"Unknown model: {name}. Available: {list(builders.keys())}")
    return builders[name](config)
