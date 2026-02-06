"""Registry module for VLA component registration and instantiation.

This module provides a type-safe registry pattern for dynamic component loading,
enabling configuration-driven instantiation of vision, language, fusion, action,
and model components. Integrates seamlessly with Hydra configuration system.

Example:
    >>> from vla.registry import VISION_REGISTRY, build_vision_encoder
    >>> from omegaconf import DictConfig
    >>>
    >>> # Register a custom vision encoder
    >>> @VISION_REGISTRY.register("my_encoder")
    >>> class MyEncoder(nn.Module):
    ...     def __init__(self, dim: int = 512):
    ...         super().__init__()
    ...         self.dim = dim
    >>>
    >>> # Build from config
    >>> cfg = DictConfig({"name": "my_encoder", "dim": 768})
    >>> encoder = build_vision_encoder(cfg)
"""

from .base import (
    ACTION_REGISTRY,
    FUSION_REGISTRY,
    LANGUAGE_REGISTRY,
    MODEL_REGISTRY,
    VISION_REGISTRY,
    Registry,
)
from .factories import (
    build_action_head,
    build_fusion_module,
    build_language_encoder,
    build_model,
    build_vision_encoder,
    build_vla_from_hydra,
)

__all__ = [
    # Registry class and instances
    "Registry",
    "VISION_REGISTRY",
    "LANGUAGE_REGISTRY",
    "FUSION_REGISTRY",
    "ACTION_REGISTRY",
    "MODEL_REGISTRY",
    # Factory functions
    "build_vision_encoder",
    "build_language_encoder",
    "build_fusion_module",
    "build_action_head",
    "build_model",
    "build_vla_from_hydra",
]
