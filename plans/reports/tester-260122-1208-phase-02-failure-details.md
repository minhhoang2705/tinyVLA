# Phase 02 Registry Tests - Failure Details & Fix Guide

## Failure Overview
5 out of 20 tests failed, all with identical root cause in factory function argument handling.

**Failure Pattern:** `TypeError: Registry.get() got multiple values for argument 'name'`

---

## Detailed Failure Analysis

### Failure #1: test_build_vision_encoder_from_registry
**File:** `tests/unit/test_registry.py::TestFactoryFunctions::test_build_vision_encoder_from_registry`
**Line:** 219

**Test Code:**
```python
def test_build_vision_encoder_from_registry(self):
    @VISION_REGISTRY.register("test_vision")
    class TestVisionEncoder(nn.Module):
        def __init__(self, dim: int = 512):
            super().__init__()
            self.dim = dim

    cfg = DictConfig({"name": "test_vision", "dim": 768})
    encoder = build_vision_encoder(cfg)  # ← FAILS HERE
    assert isinstance(encoder, TestVisionEncoder)
    assert encoder.dim == 768
```

**Stack Trace:**
```
tests/unit/test_registry.py:219: in test_build_vision_encoder_from_registry
    encoder = build_vision_encoder(cfg)
src/vla/registry/factories.py:41: in build_vision_encoder
    return VISION_REGISTRY.get(cfg.name, **cfg)
E   TypeError: Registry.get() got multiple values for argument 'name'
```

**Root Cause Line:** `/home/minh-ub/projects/tinyVLA/src/vla/registry/factories.py:41`
```python
return VISION_REGISTRY.get(cfg.name, **cfg)
```

**Why it Fails:**
1. `cfg.name` → `"test_vision"` (positional arg to `get()`)
2. `**cfg` → `{"name": "test_vision", "dim": 768}` (keyword args)
3. `get()` signature: `def get(self, name: str, **kwargs: Any)`
4. Result: `name` passed both positionally and as keyword
5. Python raises TypeError for duplicate argument

---

### Failure #2: test_build_language_encoder_from_registry
**File:** `tests/unit/test_registry.py::TestFactoryFunctions::test_build_language_encoder_from_registry`
**Line:** 234

**Issue:** Identical to Failure #1, different component
**Problematic Code:** `/home/minh-ub/projects/tinyVLA/src/vla/registry/factories.py:61`
```python
return LANGUAGE_REGISTRY.get(cfg.name, **cfg)
```

---

### Failure #3: test_build_fusion_module_from_registry
**File:** `tests/unit/test_registry.py::TestFactoryFunctions::test_build_fusion_module_from_registry`
**Line:** 249

**Issue:** Identical to Failure #1, different component
**Problematic Code:** `/home/minh-ub/projects/tinyVLA/src/vla/registry/factories.py:81`
```python
return FUSION_REGISTRY.get(cfg.name, **cfg)
```

---

### Failure #4: test_build_action_head_from_registry
**File:** `tests/unit/test_registry.py::TestFactoryFunctions::test_build_action_head_from_registry`
**Line:** 264

**Issue:** Identical to Failure #1, different component
**Problematic Code:** `/home/minh-ub/projects/tinyVLA/src/vla/registry/factories.py:101`
```python
return ACTION_REGISTRY.get(cfg.name, **cfg)
```

---

### Failure #5: test_build_model_from_registry
**File:** `tests/unit/test_registry.py::TestFactoryFunctions::test_build_model_from_registry`
**Line:** 279

**Issue:** Identical to Failure #1, different component
**Problematic Code:** `/home/minh-ub/projects/tinyVLA/src/vla/registry/factories.py:128`
```python
return MODEL_REGISTRY.get(cfg.name, **cfg)
```

---

## Working Test: test_build_with_hydra_target
**File:** `tests/unit/test_registry.py::TestFactoryFunctions::test_build_with_hydra_target`
**Status:** ✓ PASSED

**Why it Works:**
```python
cfg = DictConfig({"_target_": "tests.unit.test_registry.HydraTestModule", "value": 999})
module = build_vision_encoder(cfg)
```

Uses the `_target_` path in factory function:
```python
if hasattr(cfg, "_target_"):
    return instantiate(cfg)  # ← This branch taken, bypasses Registry.get()
```

This confirms the registry base and Hydra integration are working correctly.

---

## The Fix

### Root Issue in Registry.get()
`Registry.get()` method (line 64 in `base.py`):
```python
def get(self, name: str, **kwargs: Any) -> T:
    """Instantiate a registered class with kwargs.

    Args:
        name: Registered component name
        **kwargs: Arguments passed to component constructor
    ...
    """
    if name not in self._registry:
        ...
    return self._registry[name](**kwargs)  # ← Only uses kwargs, not name
```

The method signature is correct - it takes `name` as positional arg and passes remaining kwargs to constructor. The problem is in how factory functions call it.

### Solution: Filter Out 'name' Before Unpacking

**Fix for all 5 factory functions:**

**Option A (Recommended - Cleaner):**
```python
def build_vision_encoder(cfg: DictConfig) -> Any:
    if hasattr(cfg, "_target_"):
        return instantiate(cfg)

    # Extract name and filter out from kwargs
    component_name = cfg.name
    component_kwargs = {k: v for k, v in cfg.items() if k != "name"}
    return VISION_REGISTRY.get(component_name, **component_kwargs)
```

**Option B (Using OmegaConf API):**
```python
def build_vision_encoder(cfg: DictConfig) -> Any:
    if hasattr(cfg, "_target_"):
        return instantiate(cfg)

    # Use OmegaConf to create filtered config
    from omegaconf import OmegaConf
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    component_name = cfg_dict.pop("name")
    return VISION_REGISTRY.get(component_name, **cfg_dict)
```

**Option C (Most Hydra-native):**
```python
from omegaconf import OmegaConf

def build_vision_encoder(cfg: DictConfig) -> Any:
    if hasattr(cfg, "_target_"):
        return instantiate(cfg)

    cfg_copy = OmegaConf.to_container(cfg, resolve=True)
    name = cfg_copy.pop("name")
    return VISION_REGISTRY.get(name, **cfg_copy)
```

---

## Files to Modify

All changes in: `/home/minh-ub/projects/tinyVLA/src/vla/registry/factories.py`

### Changes Required:

1. **Line 41 (build_vision_encoder)**
   - Remove: `return VISION_REGISTRY.get(cfg.name, **cfg)`
   - Add proper kwargs filtering

2. **Line 61 (build_language_encoder)**
   - Remove: `return LANGUAGE_REGISTRY.get(cfg.name, **cfg)`
   - Add proper kwargs filtering

3. **Line 81 (build_fusion_module)**
   - Remove: `return FUSION_REGISTRY.get(cfg.name, **cfg)`
   - Add proper kwargs filtering

4. **Line 101 (build_action_head)**
   - Remove: `return ACTION_REGISTRY.get(cfg.name, **cfg)`
   - Add proper kwargs filtering

5. **Line 128 (build_model)**
   - Remove: `return MODEL_REGISTRY.get(cfg.name, **cfg)`
   - Add proper kwargs filtering

---

## Verification Steps After Fix

1. Run tests:
```bash
source .venv/bin/activate
pytest tests/unit/test_registry.py -v --cov=vla.registry --cov-report=term-missing
```

2. Expected output:
```
============================= test session starts ==============================
collected 20 items

tests/unit/test_registry.py::TestRegistry::test_register_and_get PASSED  [  5%]
...
tests/unit/test_registry.py::TestFactoryFunctions::test_build_with_hydra_target PASSED [100%]

============================= 20 passed in X.XXs ==============================
Coverage: 92%
```

3. Verify:
   - All 20 tests pass
   - No coverage regression
   - No new warnings or errors

---

## Impact Assessment

**Risk Level:** LOW
- Isolated to factory function implementation
- No changes to Registry class or public API
- Only internal kwargs handling modified
- Tests validate the fix

**Testing Coverage:** HIGH
- 5 dedicated tests for each factory function
- 1 test for Hydra instantiation path (already passing)
- Global registry independence validated

**Backward Compatibility:** YES
- External API unchanged
- Same input/output behavior
- Only internal implementation detail fixed

---

## Time Estimate

- **Fix Implementation:** 10-15 minutes
- **Test Verification:** 2-3 minutes
- **Code Review:** 5 minutes
- **Total:** ~20-25 minutes

---

## References

- **Test File:** `/home/minh-ub/projects/tinyVLA/tests/unit/test_registry.py`
- **Factory Functions:** `/home/minh-ub/projects/tinyVLA/src/vla/registry/factories.py`
- **Registry Base:** `/home/minh-ub/projects/tinyVLA/src/vla/registry/base.py`
- **Coverage Report:** `/home/minh-ub/projects/tinyVLA/htmlcov/index.html`
