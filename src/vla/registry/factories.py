"""Factory functions for building VLA components from Hydra configs."""

from typing import Any

from hydra.utils import instantiate
from omegaconf import DictConfig

from .base import (
    ACTION_REGISTRY,
    FUSION_REGISTRY,
    LANGUAGE_REGISTRY,
    MODEL_REGISTRY,
    VISION_REGISTRY,
)


def build_vision_encoder(cfg: DictConfig) -> Any:
    """Build vision encoder from Hydra config.

    Supports both registry lookup and direct Hydra instantiation via _target_.

    Args:
        cfg: Vision encoder configuration with either:
            - cfg.name: Registered component name + constructor args
            - cfg._target_: Full class path for Hydra instantiate

    Returns:
        Instantiated vision encoder module

    Example:
        >>> # Registry-based
        >>> cfg = DictConfig({"name": "vit_base", "hidden_dim": 768})
        >>> encoder = build_vision_encoder(cfg)
        >>>
        >>> # Hydra _target_ based
        >>> cfg = DictConfig({"_target_": "timm.create_model", "model_name": "vit_base_patch16_224"})
        >>> encoder = build_vision_encoder(cfg)
    """
    if hasattr(cfg, "_target_"):
        return instantiate(cfg)
    name = cfg.name
    kwargs = {k: v for k, v in cfg.items() if k != "name"}
    return VISION_REGISTRY.get(name, **kwargs)  # type: ignore


def build_language_encoder(cfg: DictConfig) -> Any:
    """Build language encoder from Hydra config.

    Supports both registry lookup and direct Hydra instantiation via _target_.

    Args:
        cfg: Language encoder configuration

    Returns:
        Instantiated language encoder module

    Example:
        >>> cfg = DictConfig({"name": "t5_base", "freeze": True})
        >>> encoder = build_language_encoder(cfg)
    """
    if hasattr(cfg, "_target_"):
        return instantiate(cfg)
    name = cfg.name
    kwargs = {k: v for k, v in cfg.items() if k != "name"}
    return LANGUAGE_REGISTRY.get(name, **kwargs)  # type: ignore


def build_fusion_module(cfg: DictConfig) -> Any:
    """Build fusion module from Hydra config.

    Supports both registry lookup and direct Hydra instantiation via _target_.

    Args:
        cfg: Fusion module configuration

    Returns:
        Instantiated fusion module

    Example:
        >>> cfg = DictConfig({"name": "perceiver", "num_latents": 64})
        >>> fusion = build_fusion_module(cfg)
    """
    if hasattr(cfg, "_target_"):
        return instantiate(cfg)
    name = cfg.name
    kwargs = {k: v for k, v in cfg.items() if k != "name"}
    return FUSION_REGISTRY.get(name, **kwargs)  # type: ignore


def build_action_head(cfg: DictConfig) -> Any:
    """Build action head from Hydra config.

    Supports both registry lookup and direct Hydra instantiation via _target_.

    Args:
        cfg: Action head configuration

    Returns:
        Instantiated action head module

    Example:
        >>> cfg = DictConfig({"name": "discrete", "num_bins": 256, "action_dim": 7})
        >>> head = build_action_head(cfg)
    """
    if hasattr(cfg, "_target_"):
        return instantiate(cfg)
    name = cfg.name
    kwargs = {k: v for k, v in cfg.items() if k != "name"}
    return ACTION_REGISTRY.get(name, **kwargs)  # type: ignore


def build_model(cfg: DictConfig) -> Any:
    """Build complete VLA model from Hydra config.

    Supports both registry lookup and direct Hydra instantiation via _target_.
    This is the main entry point for instantiating full VLA models.

    Args:
        cfg: Model configuration with vision, language, fusion, and action specs

    Returns:
        Instantiated VLA model

    Example:
        >>> cfg = DictConfig({
        ...     "name": "vla_base",
        ...     "vision": {"name": "vit_base"},
        ...     "language": {"name": "t5_base"},
        ...     "fusion": {"name": "perceiver"},
        ...     "action": {"name": "discrete", "action_dim": 7}
        ... })
        >>> model = build_model(cfg)
    """
    if hasattr(cfg, "_target_"):
        return instantiate(cfg)
    name = cfg.name
    kwargs = {k: v for k, v in cfg.items() if k != "name"}
    return MODEL_REGISTRY.get(name, **kwargs)  # type: ignore
