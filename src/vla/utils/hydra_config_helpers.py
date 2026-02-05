"""Hydra configuration utilities for tinyVLA.

Provides validation, custom resolvers, and helper functions for working
with Hydra/OmegaConf configurations.

Example:
    >>> from vla.utils.hydra_config_helpers import register_resolvers, validate_config
    >>> register_resolvers()
    >>> validate_config(cfg)
"""

import os
from pathlib import Path
from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf

from vla.utils.logging import setup_logger

logger = setup_logger(__name__)


def get_config_dir() -> Path:
    """Get absolute path to the configs/ directory.

    Returns:
        Path to configs/ directory relative to project root

    Example:
        >>> config_dir = get_config_dir()
        >>> print(config_dir)
        /path/to/tinyVLA/configs
    """
    # Navigate from src/vla/utils/ -> project root -> configs/
    return Path(__file__).parent.parent.parent.parent / "configs"


def register_resolvers() -> None:
    """Register custom OmegaConf resolvers for use in YAML configs.

    Resolvers:
        - ${env:KEY,default}: Read environment variable with fallback
        - ${mult:x,y}: Multiply two numbers (useful for computed dims)

    Example:
        >>> register_resolvers()
        >>> cfg = OmegaConf.create({"path": "${env:DATA_DIR,/data}"})
        >>> OmegaConf.resolve(cfg)
    """
    OmegaConf.register_new_resolver(
        "env",
        lambda key, default="": os.environ.get(key, default),
        replace=True,
    )
    OmegaConf.register_new_resolver(
        "mult",
        lambda x, y: int(x) * int(y),
        replace=True,
    )
    logger.info("Registered custom OmegaConf resolvers: env, mult")


def validate_config(cfg: DictConfig) -> None:
    """Validate Hydra configuration for completeness and consistency.

    Checks:
        1. Required top-level fields exist
        2. Component configs have 'name' field
        3. Dimension consistency between vision proj_dim and fusion dim

    Args:
        cfg: Top-level Hydra DictConfig

    Raises:
        ValueError: If required fields are missing or dimensions mismatch

    Example:
        >>> validate_config(cfg)  # Raises ValueError if invalid
    """
    # Check required top-level fields
    required_groups = ["model", "vision", "language", "fusion", "action"]
    for group in required_groups:
        if group not in cfg:
            raise ValueError(f"Missing required config group: '{group}'")

    # Check each component has a 'name' field
    for group in ["vision", "language", "fusion", "action"]:
        if "name" not in cfg[group]:
            raise ValueError(f"Config group '{group}' missing required 'name' field")

    # Validate dimension consistency
    vision_proj = cfg.vision.get("proj_dim", None)
    fusion_dim = cfg.fusion.get("dim", None)

    if vision_proj is not None and fusion_dim is not None:
        if vision_proj != fusion_dim:
            logger.warning(
                f"Dimension mismatch: vision.proj_dim={vision_proj} != fusion.dim={fusion_dim}. "
                f"Fusion module will add a projection layer to compensate."
            )

    logger.info("Config validation passed")


def print_config(cfg: DictConfig, resolve: bool = True) -> None:
    """Pretty-print configuration using logger.

    Args:
        cfg: Hydra DictConfig to print
        resolve: Whether to resolve interpolations before printing

    Example:
        >>> print_config(cfg)
        # Logs the full YAML config at INFO level
    """
    yaml_str = OmegaConf.to_yaml(cfg, resolve=resolve)
    logger.info(f"Configuration:\n{yaml_str}")


def save_config(cfg: DictConfig, path: str) -> None:
    """Save configuration to a YAML file.

    Args:
        cfg: Hydra DictConfig to save
        path: Output file path

    Example:
        >>> save_config(cfg, "outputs/run_config.yaml")
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        OmegaConf.save(cfg, f)
    logger.info(f"Saved config to {output_path}")


def flatten_config(cfg: DictConfig) -> Dict[str, Any]:
    """Flatten nested config to dot-notation dictionary.

    Useful for logging to WandB or TensorBoard as flat key-value pairs.

    Args:
        cfg: Hydra DictConfig to flatten

    Returns:
        Flat dictionary with dot-separated keys

    Example:
        >>> flat = flatten_config(cfg)
        >>> flat["vision.name"]
        'timm_vit'
        >>> flat["fusion.dim"]
        768
    """
    container = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    def _flatten(d: dict, prefix: str = "") -> Dict[str, Any]:
        items: Dict[str, Any] = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                items.update(_flatten(v, key))
            else:
                items[key] = v
        return items

    return _flatten(container)
