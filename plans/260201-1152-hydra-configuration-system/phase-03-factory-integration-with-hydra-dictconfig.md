# Phase 03: Factory Integration with Hydra DictConfig

## Context Links
- [Plan Overview](plan.md)
- [Registry Factories](../../src/vla/registry/factories.py) -- current factory code
- [VLA Configs Dataclasses](../../src/vla/models/vla_configs.py) -- VLAConfig.from_dict()
- [VLA Base Model](../../src/vla/models/vla_base.py) -- VLAModel constructor

## Overview
| Field | Value |
|-------|-------|
| Priority | P1 |
| Status | Pending |
| Effort | 45m |
| Dependencies | Phase 01, Phase 02 |

Bridge Hydra's `DictConfig` objects to the existing factory and model construction code. The factories already handle DictConfig via the `name` key pattern. Main work is adding a `VLAConfig.from_hydra()` classmethod and a top-level `build_vla_from_hydra()` convenience function.

## Key Insights

**What already works (no changes needed):**
- `factories.py` all 5 build functions accept `DictConfig` and handle both `_target_` and `name` patterns
- `Registry.get(name, **kwargs)` works with any dict-like kwargs

**What needs bridging:**
- `VLAConfig.from_dict()` expects plain `dict`, but Hydra produces `DictConfig` -- need `OmegaConf.to_container()` conversion
- `VLAModel.__init__` calls `VISION_REGISTRY.get(cfg.name, model_name=..., ...)` directly using config dataclass fields, NOT via factory functions -- so YAML keys must match exactly (verified in Phase 02)
- Need a convenience function to go from full Hydra config -> VLAModel in one call

**Design decision: Keep it simple.**
- Add `VLAConfig.from_hydra(cfg: DictConfig)` that does `OmegaConf.to_container(resolve=True)` then calls `from_dict()`
- Add `build_vla_from_hydra(cfg)` that composes VLAConfig + VLAModel construction
- Do NOT restructure the existing VLAModel.__init__ -- it works fine with the dataclass pattern

## Requirements

### Functional
- FR-01: `VLAConfig.from_hydra(cfg)` creates VLAConfig from Hydra DictConfig
- FR-02: `build_vla_from_hydra(cfg)` builds complete VLAModel from top-level Hydra config
- FR-03: Existing `VLAConfig.from_dict()` and factory functions continue working unchanged

### Non-Functional
- NFR-01: No breaking changes to existing API
- NFR-02: Keep factory module under 200 LOC

## Architecture

```
Hydra config.yaml
       |
       v
  DictConfig (nested)
       |
       v
  build_vla_from_hydra(cfg)
       |
       ├──> OmegaConf.to_container(cfg, resolve=True)
       ├──> Extract vision/language/fusion/action/model sub-dicts
       ├──> VLAConfig.from_dict(resolved_dict)
       └──> VLAModel(config) or TemporalVLAModel(config, num_frames)
```

## Related Code Files

### Files to Modify
| Path | Changes | Current Lines | Est. New Lines |
|------|---------|--------------|----------------|
| `src/vla/models/vla_configs.py` | Add `from_hydra()` classmethod | 187 | +15 |
| `src/vla/registry/factories.py` | Add `build_vla_from_hydra()` function | 139 | +35 |

### Files to Update (exports)
| Path | Changes |
|------|---------|
| `src/vla/registry/__init__.py` | Export `build_vla_from_hydra` |

## Implementation Steps

### Step 1: Add `VLAConfig.from_hydra()` (15 min)

In `src/vla/models/vla_configs.py`, add a classmethod after `from_dict()`:

```python
@classmethod
def from_hydra(cls, cfg: "DictConfig") -> "VLAConfig":
    """Create VLAConfig from Hydra DictConfig.

    Resolves interpolations and converts DictConfig to plain dict,
    then delegates to from_dict().

    Args:
        cfg: Hydra DictConfig with vision, language, fusion, action keys

    Returns:
        VLAConfig instance

    Example:
        >>> from hydra import compose, initialize
        >>> initialize(config_path="../../configs")
        >>> cfg = compose(config_name="config")
        >>> config = VLAConfig.from_hydra(cfg)
    """
    from omegaconf import OmegaConf

    resolved = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    return cls.from_dict(resolved)
```

**Important:** Use string annotation for `DictConfig` type hint to avoid hard import dependency. Add conditional import inside the method body.

### Step 2: Add `build_vla_from_hydra()` (20 min)

In `src/vla/registry/factories.py`, add after `build_model()`:

```python
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
```

**Why not use `build_model()`?** Because `VLAModel.__init__` takes a `VLAConfig` object, not kwargs. The existing `build_model()` passes kwargs to the registry's `get()`, which would pass the whole config as kwargs. We need the explicit VLAConfig construction.

### Step 3: Update exports (5 min)

In `src/vla/registry/__init__.py`, add to imports and `__all__`:

```python
from .factories import (
    build_action_head,
    build_fusion_module,
    build_language_encoder,
    build_model,
    build_vision_encoder,
    build_vla_from_hydra,  # NEW
)

__all__ = [
    # ... existing ...
    "build_vla_from_hydra",
]
```

### Step 4: Verify no breakage (5 min)

Run existing tests to confirm nothing breaks:
```bash
pytest tests/unit/test_vla_model.py -v
```

## Todo List
- [ ] Add `VLAConfig.from_hydra()` classmethod in `vla_configs.py`
- [ ] Add `build_vla_from_hydra()` function in `factories.py`
- [ ] Update `registry/__init__.py` exports
- [ ] Run existing tests to verify no regression

## Success Criteria
1. `VLAConfig.from_hydra(cfg)` correctly creates config from DictConfig
2. `build_vla_from_hydra(cfg)` instantiates VLAModel from full Hydra config
3. Existing tests pass unchanged
4. `factories.py` stays under 200 LOC

## Risk Assessment
| Risk | Impact | Mitigation |
|------|--------|------------|
| OmegaConf resolve fails on unresolved ${} | Medium | Use `throw_on_missing=True` for clear errors |
| Circular import VLAConfig <-> factories | High | Use lazy import inside function body |
| from_dict() rejects extra keys from YAML | Medium | from_dict() uses `.get()` with defaults, safe for extra keys |

## Security Considerations
- No new security surface; this is pure config bridging code

## Next Steps
- Phase 04: Hydra utility functions (validation, resolvers)
