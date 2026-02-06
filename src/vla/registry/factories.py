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


def build_vla_from_hydra(cfg: DictConfig) -> Any:
    """Build complete VLA model from top-level Hydra config.

    Convenience function that bridges Hydra's composed config to VLAModel.
    Handles both base and temporal model variants based on model.name.

    Args:
        cfg: Top-level Hydra DictConfig with keys:
            model, vision, language, fusion, action, train, data, seed, etc.

    Returns:
        Instantiated VLAModel or TemporalVLAModel

    Example:
        >>> from hydra import compose, initialize
        >>> initialize(config_path="../../configs")
        >>> cfg = compose(config_name="config")
        >>> model = build_vla_from_hydra(cfg)
    """
    from vla.models.vla_configs import VLAConfig

    # Build VLAConfig from the composed sub-configs
    config_dict = {
        "vision": {k: v for k, v in cfg.vision.items()},
        "language": {k: v for k, v in cfg.language.items()},
        "fusion": {k: v for k, v in cfg.fusion.items()},
        "action": {k: v for k, v in cfg.action.items()},
        "freeze_vision": cfg.model.get("freeze_vision", True),
        "freeze_language": cfg.model.get("freeze_language", True),
        "action_loss_weight": cfg.model.get("action_loss_weight", 1.0),
        "auxiliary_loss_weight": cfg.model.get("auxiliary_loss_weight", 0.0),
    }

    vla_config = VLAConfig.from_dict(config_dict)

    # Select model class based on model.name
    model_name = cfg.model.get("name", "vla_base")
    if model_name == "vla_temporal":
        num_frames = cfg.model.get("num_frames", 6)
        return MODEL_REGISTRY.get_class("vla_temporal")(vla_config, num_frames=num_frames)
    else:
        return MODEL_REGISTRY.get_class("vla_base")(vla_config)
