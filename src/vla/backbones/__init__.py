"""Vision and language backbone modules.

Provides pretrained encoder wrappers for VLA models:
- Vision backbones: DINOv2, SigLIP, ViT (via timm)
- Language backbones: GPT-2 (from other branches)
"""

from .feature_extractor import DualEncoderVision, MultiScaleFeatureExtractor
from .vision import DINOv2Backbone, SigLIPBackbone, VisionBackbone

__all__ = [
    # Vision
    "VisionBackbone",
    "DINOv2Backbone",
    "SigLIPBackbone",
    "MultiScaleFeatureExtractor",
    "DualEncoderVision",
]
