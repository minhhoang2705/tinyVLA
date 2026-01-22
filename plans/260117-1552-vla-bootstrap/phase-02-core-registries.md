# Phase 02: Core Registries and Factories

## Context Links
- [PyTorch VLA Research](../reports/researcher-260118-0228-pytorch-vla.md) - Registry pattern section
- [Hydra Config Research](../reports/researcher-260118-hydra-ml-config.md)

## Overview
| Field | Value |
|-------|-------|
| Priority | P1 - Critical Path |
| Status | ✓ COMPLETE |
| Effort | 2h (actual: ~2.5h) |
| Dependencies | Phase 1 |
| Completion Date | 2026-01-22 |
| Code Review | [code-reviewer-260122-1305](../reports/code-reviewer-260122-1305-phase-02-registry.md) |
| Score | 8.5/10 |

Implement registry pattern for dynamic component loading. Enables config-driven instantiation of all VLA components.

**Review Summary:** Implementation complete with excellent test coverage (92%, 20/20 tests passing). No critical issues. Minor improvements recommended for input validation and logging before production deploy.

## Key Insights
- Registry pattern decouples component definition from instantiation
- Hydra `instantiate()` works seamlessly with registered classes
- Type hints enable IDE autocomplete and static checking
- Factory functions handle complex initialization logic

## Requirements

### Functional
- FR-01: Register components by string name
- FR-02: Retrieve components with kwargs passed to constructor
- FR-03: List all registered components per category
- FR-04: Support Hydra `_target_` instantiation pattern

### Non-Functional
- NFR-01: O(1) lookup time for registered components
- NFR-02: Type-safe registration with generics

## Architecture

```
src/vla/registry/
├── __init__.py          # Public API exports
├── base.py              # Generic Registry class
└── factories.py         # Component factory functions
```

**Registry Pattern Flow:**
```
@Registry.register("vit_base")  →  Registry._registry["vit_base"] = ViTBase
                                          ↓
cfg.model.name = "vit_base"     →  Registry.get("vit_base", **cfg) → ViTBase(**cfg)
```

## Related Code Files

### Files to Create
| Path | Purpose | Lines |
|------|---------|-------|
| `src/vla/registry/__init__.py` | Public exports | ~15 |
| `src/vla/registry/base.py` | Generic registry | ~80 |
| `src/vla/registry/factories.py` | Build functions | ~60 |
| `tests/unit/test_registry.py` | Registry tests | ~80 |

## Implementation Steps

### Step 1: Implement base.py (45 min)
```python
"""Generic registry for component registration and lookup."""
from typing import TypeVar, Generic, Dict, Type, Callable, Any, Optional

T = TypeVar("T")


class Registry(Generic[T]):
    """Type-safe registry for component classes.

    Usage:
        vision_registry = Registry[nn.Module]("vision")

        @vision_registry.register("vit_base")
        class ViTBase(nn.Module):
            ...

        model = vision_registry.get("vit_base", hidden_dim=768)
    """

    def __init__(self, name: str):
        self._name = name
        self._registry: Dict[str, Type[T]] = {}

    def register(self, name: str) -> Callable[[Type[T]], Type[T]]:
        """Decorator to register a class."""
        def wrapper(cls: Type[T]) -> Type[T]:
            if name in self._registry:
                raise ValueError(f"{name} already registered in {self._name}")
            self._registry[name] = cls
            return cls
        return wrapper

    def get(self, name: str, **kwargs: Any) -> T:
        """Instantiate registered class with kwargs."""
        if name not in self._registry:
            available = ", ".join(self._registry.keys())
            raise KeyError(f"{name} not found. Available: {available}")
        return self._registry[name](**kwargs)

    def get_class(self, name: str) -> Type[T]:
        """Get class without instantiating."""
        if name not in self._registry:
            raise KeyError(f"{name} not found in {self._name} registry")
        return self._registry[name]

    def list_available(self) -> list[str]:
        """List all registered component names."""
        return list(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._registry


# Global registries for each component type
VISION_REGISTRY = Registry[Any]("vision")
LANGUAGE_REGISTRY = Registry[Any]("language")
FUSION_REGISTRY = Registry[Any]("fusion")
ACTION_REGISTRY = Registry[Any]("action")
MODEL_REGISTRY = Registry[Any]("model")
```

### Step 2: Implement factories.py (30 min)
```python
"""Factory functions for building VLA components from configs."""
from typing import Any, Dict
from omegaconf import DictConfig
from hydra.utils import instantiate

from .base import (
    VISION_REGISTRY,
    LANGUAGE_REGISTRY,
    FUSION_REGISTRY,
    ACTION_REGISTRY,
    MODEL_REGISTRY,
)


def build_vision_encoder(cfg: DictConfig) -> Any:
    """Build vision encoder from config.

    Supports both registry lookup and Hydra instantiate.
    """
    if hasattr(cfg, "_target_"):
        return instantiate(cfg)
    return VISION_REGISTRY.get(cfg.name, **cfg)


def build_language_encoder(cfg: DictConfig) -> Any:
    """Build language encoder from config."""
    if hasattr(cfg, "_target_"):
        return instantiate(cfg)
    return LANGUAGE_REGISTRY.get(cfg.name, **cfg)


def build_fusion_module(cfg: DictConfig) -> Any:
    """Build fusion module from config."""
    if hasattr(cfg, "_target_"):
        return instantiate(cfg)
    return FUSION_REGISTRY.get(cfg.name, **cfg)


def build_action_head(cfg: DictConfig) -> Any:
    """Build action head from config."""
    if hasattr(cfg, "_target_"):
        return instantiate(cfg)
    return ACTION_REGISTRY.get(cfg.name, **cfg)


def build_model(cfg: DictConfig) -> Any:
    """Build complete VLA model from config."""
    if hasattr(cfg, "_target_"):
        return instantiate(cfg)
    return MODEL_REGISTRY.get(cfg.name, **cfg)
```

### Step 3: Create __init__.py (10 min)
```python
"""Registry module for VLA component registration."""
from .base import (
    Registry,
    VISION_REGISTRY,
    LANGUAGE_REGISTRY,
    FUSION_REGISTRY,
    ACTION_REGISTRY,
    MODEL_REGISTRY,
)
from .factories import (
    build_vision_encoder,
    build_language_encoder,
    build_fusion_module,
    build_action_head,
    build_model,
)

__all__ = [
    "Registry",
    "VISION_REGISTRY",
    "LANGUAGE_REGISTRY",
    "FUSION_REGISTRY",
    "ACTION_REGISTRY",
    "MODEL_REGISTRY",
    "build_vision_encoder",
    "build_language_encoder",
    "build_fusion_module",
    "build_action_head",
    "build_model",
]
```

### Step 4: Write tests (35 min)
```python
"""Tests for registry module."""
import pytest
import torch.nn as nn
from vla.registry import Registry, VISION_REGISTRY


class TestRegistry:
    def test_register_and_get(self):
        registry = Registry[nn.Module]("test")

        @registry.register("dummy")
        class DummyModule(nn.Module):
            def __init__(self, dim: int = 64):
                super().__init__()
                self.dim = dim

        module = registry.get("dummy", dim=128)
        assert module.dim == 128

    def test_duplicate_registration_raises(self):
        registry = Registry[nn.Module]("test")

        @registry.register("same_name")
        class First(nn.Module):
            pass

        with pytest.raises(ValueError, match="already registered"):
            @registry.register("same_name")
            class Second(nn.Module):
                pass

    def test_unknown_component_raises(self):
        registry = Registry[nn.Module]("test")
        with pytest.raises(KeyError, match="not found"):
            registry.get("nonexistent")

    def test_list_available(self):
        registry = Registry[nn.Module]("test")

        @registry.register("a")
        class A(nn.Module):
            pass

        @registry.register("b")
        class B(nn.Module):
            pass

        assert set(registry.list_available()) == {"a", "b"}

    def test_contains(self):
        registry = Registry[nn.Module]("test")

        @registry.register("exists")
        class Exists(nn.Module):
            pass

        assert "exists" in registry
        assert "missing" not in registry
```

## Todo List
- [x] Create registry/__init__.py with exports
- [x] Implement Registry generic class in base.py
- [x] Create global registries (VISION, LANGUAGE, FUSION, ACTION, MODEL)
- [x] Implement factory functions in factories.py
- [x] Write unit tests for registry operations
- [x] Test Hydra instantiate integration
- [x] Document usage patterns

**All tasks completed 2026-01-22**

## Success Criteria
1. Registry register/get cycle works
2. Duplicate registration raises error
3. Unknown component raises descriptive error
4. Factory functions support both registry and Hydra patterns
5. All tests pass

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Circular imports | High | Lazy imports, careful __init__.py |
| Type erasure at runtime | Low | Use TypeVar for static checks only |
| Registry pollution | Medium | Namespace registries by component type |

## Security Considerations
- No dynamic code execution from untrusted sources
- Registry names are static strings, not user input

## Next Steps
- Phase 3: Implement NN primitives that use registry
