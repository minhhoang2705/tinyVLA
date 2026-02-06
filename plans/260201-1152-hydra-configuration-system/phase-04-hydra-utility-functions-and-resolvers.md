# Phase 04: Hydra Utility Functions and Resolvers

## Context Links
- [Plan Overview](plan.md)
- [Existing utils](../../src/vla/utils/__init__.py)
- [Existing logging util](../../src/vla/utils/logging.py)

## Overview
| Field | Value |
|-------|-------|
| Priority | P2 |
| Status | Complete |
| Effort | 45m |
| Dependencies | Phase 01, Phase 02 |

Implement `src/vla/utils/hydra-config-helpers.py` with config validation, custom OmegaConf resolvers, and print/save helpers. Keep it focused -- no over-engineering.

## Key Insights
- Custom resolvers enable `${env:DATA_DIR,/data}` and `${mult:2,384}` in YAML
- Validation catches dimension mismatches early (before model build fails cryptically)
- `print_config()` is essential for experiment logging/debugging
- Keep everything in one file (~120 LOC) to avoid fragmentation

## Requirements

### Functional
- FR-01: `register_resolvers()` -- env var + math resolvers
- FR-02: `validate_config(cfg)` -- required fields + dimension consistency checks
- FR-03: `print_config(cfg)` -- pretty-print to logger (not print())
- FR-04: `save_config(cfg, path)` -- save resolved config to YAML file
- FR-05: `flatten_config(cfg)` -- flatten nested config to dot-notation dict
- FR-06: `get_config_dir()` -- return Path to configs/ directory

### Non-Functional
- NFR-01: Use `setup_logger(__name__)` for all output (no print())
- NFR-02: All functions have type hints and NumPy-style docstrings
- NFR-03: File stays under 200 LOC

## Architecture

```
src/vla/utils/hydra-config-helpers.py
├── register_resolvers()        # Custom OmegaConf resolvers
├── validate_config(cfg)        # Validate required fields + dimensions
├── print_config(cfg)           # Pretty-print config
├── save_config(cfg, path)      # Save to YAML
├── flatten_config(cfg)         # Flatten to dot-notation
└── get_config_dir()            # Return configs/ path
```

## Related Code Files

### Files to Create
| Path | Purpose | Lines |
|------|---------|-------|
| `src/vla/utils/hydra-config-helpers.py` | Hydra utility functions | ~120 |

### Files to Modify
| Path | Changes |
|------|---------|
| `src/vla/utils/__init__.py` | Export new utility functions |

## Implementation Steps

### Step 1: Create `src/vla/utils/hydra-config-helpers.py` (30 min)

```python
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
```

### Step 2: Update `src/vla/utils/__init__.py` (5 min)

Add imports for new utility functions:

```python
from vla.utils.hydra_config_helpers import (
    flatten_config,
    get_config_dir,
    print_config,
    register_resolvers,
    save_config,
    validate_config,
)
```

And update `__all__` to include the new exports.

**Note on import:** Python converts kebab-case filenames to underscores in import statements. The file is `hydra-config-helpers.py` on disk but imported as `hydra_config_helpers`. However, Python does NOT support hyphens in module names. Therefore, the file MUST be named `hydra_config_helpers.py` (underscores) to be importable.

**CORRECTION:** Since Python module names cannot contain hyphens, name the file `hydra_config_helpers.py` instead. This is a necessary exception to the kebab-case naming convention for Python source files.

### Step 3: Verify imports (5 min)

```bash
python -c "from vla.utils.hydra_config_helpers import register_resolvers, validate_config; print('OK')"
```

## Todo List
- [x] Create `src/vla/utils/hydra_config_helpers.py` with all 6 functions
- [x] Update `src/vla/utils/__init__.py` to export new functions
- [x] Verify imports work

## Success Criteria
1. `register_resolvers()` registers `env` and `mult` resolvers without error
2. `validate_config()` raises `ValueError` for missing fields
3. `validate_config()` warns on dimension mismatch
4. `print_config()` logs config YAML via logger (not print)
5. `save_config()` writes valid YAML to disk
6. `flatten_config()` produces dot-notation dict
7. File is under 200 LOC

## Risk Assessment
| Risk | Impact | Mitigation |
|------|--------|------------|
| Resolver name collision | Low | Use `replace=True` to allow re-registration |
| Path calculation wrong in get_config_dir() | Medium | Test with actual file path |
| Validation too strict | Low | Only validate critical fields; warn on dim mismatch |

## Security Considerations
- `env` resolver reads environment variables -- standard practice, no secrets in YAML files themselves
- `save_config()` writes to user-specified path -- caller responsibility to avoid sensitive locations

## Next Steps
- Phase 05: Test script and pytest tests
