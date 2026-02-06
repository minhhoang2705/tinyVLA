"""Vision and language backbone modules.

Provides pretrained encoder wrappers for VLA models:
- Vision backbones: DINOv2, SigLIP, ViT (via timm)
- Language backbones: GPT-2, BERT (via transformers)
- Protocol interfaces: Type-safe component contracts
"""

from .feature_extractor import DualEncoderVision, MultiScaleFeatureExtractor
from .language import GPT2Backbone, LanguageEncoder
from .protocols import (
    ActionHeadProtocol,
    FusionModuleProtocol,
    LanguageBackboneProtocol,
    VisionBackboneProtocol,
    validate_action_head,
    validate_fusion_module,
    validate_language_backbone,
    validate_vision_backbone,
)
from .vision import DINOv2Backbone, SigLIPBackbone, VisionBackbone

__all__ = [
    # Vision
    "VisionBackbone",
    "DINOv2Backbone",
    "SigLIPBackbone",
    "MultiScaleFeatureExtractor",
    "DualEncoderVision",
    # Language
    "GPT2Backbone",
    "LanguageEncoder",
    # Protocols
    "VisionBackboneProtocol",
    "LanguageBackboneProtocol",
    "FusionModuleProtocol",
    "ActionHeadProtocol",
    "validate_vision_backbone",
    "validate_language_backbone",
    "validate_fusion_module",
    "validate_action_head",
]
