# Phase 02: Update VLAConfig

**Status:** Pending
**Depends on:** Phase 01

## Overview

Add `AffordanceConfig` dataclass and wire it into `VLAConfig`. The existing `auxiliary_loss_weight` field already exists — no changes needed there.

## Related Code Files

- **Modify:** `src/vla/models/vla_configs.py`

## Implementation Steps

### 1. Add `AffordanceConfig` dataclass

Insert after `ActionConfig` (~line 121):

```python
@dataclass
class AffordanceConfig:
    """Auxiliary affordance head configuration.

    Args:
        enabled: If True, build AffordanceHead and compute aux loss.
        state_dim: Predicted state dimensionality (3 = block_x, block_y, angle)
        hidden_dim: Hidden layer size in the MLP
    """

    enabled: bool = False
    state_dim: int = 3
    hidden_dim: int = 256
```

### 2. Add `affordance` field to `VLAConfig`

Add after `action` field:

```python
affordance: AffordanceConfig = field(default_factory=AffordanceConfig)
```

### 3. Update `VLAConfig.from_dict()`

Add inside the `return cls(...)` block:

```python
affordance=AffordanceConfig(
    **filter_fields(AffordanceConfig, config_dict.get("affordance", {}))
),
```

### 4. Export `AffordanceConfig`

Ensure `AffordanceConfig` is importable from `vla.models.vla_configs`.

## Key Constraints

- `AffordanceConfig.enabled=False` by default — backward compatible
- `auxiliary_loss_weight=0.0` in `VLAConfig` already exists — start at `0.1` only when enabling

## Success Criteria

- [ ] `VLAConfig()` still instantiates without args (backward compat)
- [ ] `VLAConfig.from_dict({"affordance": {"enabled": True}})` works
- [ ] `AffordanceConfig` importable from `vla.models.vla_configs`
