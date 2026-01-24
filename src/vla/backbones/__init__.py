"""Vision and language backbone modules.

Provides pretrained encoder wrappers for VLA models:
- Vision backbones: DINOv2, SigLIP, ViT (via timm/transformers)
- Language backbones: GPT-2, BERT (via transformers)
"""

from .feature_extractor import DualEncoderVision, MultiScaleFeatureExtractor
from .language import GPT2Backbone, LanguageEncoder
from .vision import DINOv2Backbone, SigLIPBackbone, VisionBackbone

__all__ = [
    # Language
    "GPT2Backbone",
    "LanguageEncoder",
    # Vision
    "VisionBackbone",
    "DINOv2Backbone",
    "SigLIPBackbone",
    "MultiScaleFeatureExtractor",
    "DualEncoderVision",
]
