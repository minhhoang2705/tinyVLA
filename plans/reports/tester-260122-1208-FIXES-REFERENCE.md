# Factory Function Fix - Code Reference

## Quick Fix Guide

All fixes follow the same pattern. Replace the problematic `return REGISTRY.get(cfg.name, **cfg)` line with proper kwargs filtering.

---

## File to Modify
**Path:** `/home/minh-ub/projects/tinyVLA/src/vla/registry/factories.py`

---

## Fix #1: build_vision_encoder (Line 41)

**Current Code:**
```python
def build_vision_encoder(cfg: DictConfig) -> Any:
    """Build vision encoder from Hydra config."""
    if hasattr(cfg, "_target_"):
        return instantiate(cfg)
    return VISION_REGISTRY.get(cfg.name, **cfg)  # ← PROBLEMATIC
```

**Fixed Code:**
```python
def build_vision_encoder(cfg: DictConfig) -> Any:
    """Build vision encoder from Hydra config."""
    if hasattr(cfg, "_target_"):
        return instantiate(cfg)
    component_name = cfg.name
    component_kwargs = {k: v for k, v in cfg.items() if k != "name"}
    return VISION_REGISTRY.get(component_name, **component_kwargs)
```

---

## Fix #2: build_language_encoder (Line 61)

**Current Code:**
```python
def build_language_encoder(cfg: DictConfig) -> Any:
    """Build language encoder from Hydra config."""
    if hasattr(cfg, "_target_"):
        return instantiate(cfg)
    return LANGUAGE_REGISTRY.get(cfg.name, **cfg)  # ← PROBLEMATIC
```

**Fixed Code:**
```python
def build_language_encoder(cfg: DictConfig) -> Any:
    """Build language encoder from Hydra config."""
    if hasattr(cfg, "_target_"):
        return instantiate(cfg)
    component_name = cfg.name
    component_kwargs = {k: v for k, v in cfg.items() if k != "name"}
    return LANGUAGE_REGISTRY.get(component_name, **component_kwargs)
```

---

## Fix #3: build_fusion_module (Line 81)

**Current Code:**
```python
def build_fusion_module(cfg: DictConfig) -> Any:
    """Build fusion module from Hydra config."""
    if hasattr(cfg, "_target_"):
        return instantiate(cfg)
    return FUSION_REGISTRY.get(cfg.name, **cfg)  # ← PROBLEMATIC
```

**Fixed Code:**
```python
def build_fusion_module(cfg: DictConfig) -> Any:
    """Build fusion module from Hydra config."""
    if hasattr(cfg, "_target_"):
        return instantiate(cfg)
    component_name = cfg.name
    component_kwargs = {k: v for k, v in cfg.items() if k != "name"}
    return FUSION_REGISTRY.get(component_name, **component_kwargs)
```

---

## Fix #4: build_action_head (Line 101)

**Current Code:**
```python
def build_action_head(cfg: DictConfig) -> Any:
    """Build action head from Hydra config."""
    if hasattr(cfg, "_target_"):
        return instantiate(cfg)
    return ACTION_REGISTRY.get(cfg.name, **cfg)  # ← PROBLEMATIC
```

**Fixed Code:**
```python
def build_action_head(cfg: DictConfig) -> Any:
    """Build action head from Hydra config."""
    if hasattr(cfg, "_target_"):
        return instantiate(cfg)
    component_name = cfg.name
    component_kwargs = {k: v for k, v in cfg.items() if k != "name"}
    return ACTION_REGISTRY.get(component_name, **component_kwargs)
```

---

## Fix #5: build_model (Line 128)

**Current Code:**
```python
def build_model(cfg: DictConfig) -> Any:
    """Build complete VLA model from Hydra config."""
    if hasattr(cfg, "_target_"):
        return instantiate(cfg)
    return MODEL_REGISTRY.get(cfg.name, **cfg)  # ← PROBLEMATIC
```

**Fixed Code:**
```python
def build_model(cfg: DictConfig) -> Any:
    """Build complete VLA model from Hydra config."""
    if hasattr(cfg, "_target_"):
        return instantiate(cfg)
    component_name = cfg.name
    component_kwargs = {k: v for k, v in cfg.items() if k != "name"}
    return MODEL_REGISTRY.get(component_name, **component_kwargs)
```

---

## Complete Fixed File

Here's what the entire fixed file should look like:

```python
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
    component_name = cfg.name
    component_kwargs = {k: v for k, v in cfg.items() if k != "name"}
    return VISION_REGISTRY.get(component_name, **component_kwargs)


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
    component_name = cfg.name
    component_kwargs = {k: v for k, v in cfg.items() if k != "name"}
    return LANGUAGE_REGISTRY.get(component_name, **component_kwargs)


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
    component_name = cfg.name
    component_kwargs = {k: v for k, v in cfg.items() if k != "name"}
    return FUSION_REGISTRY.get(component_name, **component_kwargs)


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
    component_name = cfg.name
    component_kwargs = {k: v for k, v in cfg.items() if k != "name"}
    return ACTION_REGISTRY.get(component_name, **component_kwargs)


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
    component_name = cfg.name
    component_kwargs = {k: v for k, v in cfg.items() if k != "name"}
    return MODEL_REGISTRY.get(component_name, **component_kwargs)
```

---

## Verification Checklist

After applying the fixes:

- [ ] All 5 functions updated with kwargs filtering
- [ ] File has no syntax errors (can be imported)
- [ ] Run tests: `pytest tests/unit/test_registry.py -v`
- [ ] Verify result: **20 passed, 0 failed**
- [ ] Coverage: `pytest tests/unit/test_registry.py --cov=vla.registry --cov-report=term-missing`
- [ ] Verify coverage: **≥92%** for registry module

---

## Testing Commands

**Run Tests:**
```bash
source .venv/bin/activate
cd /home/minh-ub/projects/tinyVLA
pytest tests/unit/test_registry.py -v
```

**Expected Output:**
```
============================= test session starts ==============================
collected 20 items

tests/unit/test_registry.py::TestRegistry::test_register_and_get PASSED  [  5%]
tests/unit/test_registry.py::TestRegistry::test_register_with_default_args PASSED [ 10%]
...
tests/unit/test_registry.py::TestFactoryFunctions::test_build_with_hydra_target PASSED [100%]

============================= 20 passed in X.XXs ==============================
```

**Run with Coverage:**
```bash
pytest tests/unit/test_registry.py -v --cov=vla.registry --cov-report=term-missing
```

---

## Alternative Fix Approaches

If you prefer different implementation styles:

### Approach A: Using OmegaConf to_container
```python
def build_vision_encoder(cfg: DictConfig) -> Any:
    if hasattr(cfg, "_target_"):
        return instantiate(cfg)
    from omegaconf import OmegaConf
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    component_name = cfg_dict.pop("name")
    return VISION_REGISTRY.get(component_name, **cfg_dict)
```

### Approach B: Using dict() constructor
```python
def build_vision_encoder(cfg: DictConfig) -> Any:
    if hasattr(cfg, "_target_"):
        return instantiate(cfg)
    cfg_dict = dict(cfg)
    component_name = cfg_dict.pop("name")
    return VISION_REGISTRY.get(component_name, **cfg_dict)
```

### Approach C: Using OmegaConf.to_object (resolves interpolations)
```python
def build_vision_encoder(cfg: DictConfig) -> Any:
    if hasattr(cfg, "_target_"):
        return instantiate(cfg)
    cfg_dict = OmegaConf.to_object(cfg)
    component_name = cfg_dict.pop("name", cfg.name)
    return VISION_REGISTRY.get(component_name, **cfg_dict)
```

**Recommended:** Approach A (dict comprehension) - simple, clear, no external calls

---

## Commit Message Template

```
fix: filter 'name' parameter in registry factory functions

- Extract 'name' from config before unpacking kwargs
- Prevents duplicate argument error when passing cfg to Registry.get()
- Affects: build_vision_encoder, build_language_encoder,
  build_fusion_module, build_action_head, build_model
- Tests: All 20 tests now pass with proper kwargs handling
```

---

## Review Checklist for Code Reviewer

- [ ] All 5 functions fixed consistently
- [ ] No other code changes introduced
- [ ] Line count reasonable (1-2 added lines per function)
- [ ] Kwargs filtering works for all config structures
- [ ] All 20 tests pass
- [ ] Coverage maintained at ≥92%
- [ ] No new warnings or linting errors
