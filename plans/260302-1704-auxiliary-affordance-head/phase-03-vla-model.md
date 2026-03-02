# Phase 03: Update VLAModel Forward Pass

**Status:** Pending
**Depends on:** Phase 01, 02

## Overview

Wire `AffordanceHead` into `VLAModel.__init__` and `VLAModel.forward()`. The head is built only when `config.affordance.enabled` is True. Forward pass accepts an optional `target_state` for aux loss computation.

## Related Code Files

- **Modify:** `src/vla/models/vla_base.py`

## Implementation Steps

### 1. Add `_build_affordance` helper (after `_build_action`)

```python
def _build_affordance(self, cfg: VLAConfig) -> Optional[nn.Module]:
    """Build affordance head if enabled in config.

    Args:
        cfg: Complete VLA configuration

    Returns:
        AffordanceHead instance, or None if disabled
    """
    if not cfg.affordance.enabled:
        return None
    from vla.policy.affordance_head import AffordanceHead
    logger.info(
        f"Building affordance head: state_dim={cfg.affordance.state_dim}, "
        f"hidden_dim={cfg.affordance.hidden_dim}"
    )
    return AffordanceHead(
        input_dim=cfg.fusion.dim,
        hidden_dim=cfg.affordance.hidden_dim,
        state_dim=cfg.affordance.state_dim,
    )
```

### 2. Call in `__init__` (after `self.action_head = ...`)

```python
self.affordance_head: Optional[nn.Module] = self._build_affordance(config)
```

### 3. Extend `VLAModel.forward()` signature

Add `target_state: Optional[torch.Tensor] = None` parameter.

```python
def forward(
    self,
    images: torch.Tensor,
    texts: Optional[List[str]] = None,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    target_actions: Optional[torch.Tensor] = None,
    target_state: Optional[torch.Tensor] = None,   # ← NEW
) -> Dict[str, torch.Tensor]:
```

### 4. Compute auxiliary loss after `fused_features`

After `fused_features = self.fusion(...)` and before action head:

```python
# Auxiliary affordance prediction (optional)
if self.affordance_head is not None and target_state is not None:
    # Mean-pool handled inside AffordanceHead.forward()
    state_pred = self.affordance_head(fused_features)
    aux_loss = self.affordance_head.compute_loss(state_pred, target_state)
    output["aux_loss"] = aux_loss * self.config.auxiliary_loss_weight
    output["state_pred"] = state_pred
```

### 5. Combine losses in output

The combined loss used for backprop:

```python
# Combine losses
if "loss" in output and "aux_loss" in output:
    output["loss"] = output["loss"] + output["aux_loss"]
```

> **Note:** `output["loss"]` = action_loss × action_weight + aux_loss × aux_weight

### 6. Apply same pattern to `TemporalVLAModel.forward()`

`TemporalVLAModel` inherits `_build_affordance`, so only forward() needs the same `target_state` parameter + aux loss block.

## Key Constraints

- **No torch.compile on affordance_head** — `lightning_module.py` compiles only `action_head`; affordance_head is left uncompiled (see Phase 06)
- **Graceful when `target_state=None`** — skip aux loss silently, no crash
- **Separate loss keys** — keep `action_loss` and `aux_loss` separate in output dict for logging

## Common Pitfalls

- Don't accidentally pass `target_state` through `validate_action_tensor` — it has different shape/range
- Don't modify `_validate_protocols` — affordance head is not a registered protocol component

## Success Criteria

- [ ] `VLAModel(VLAConfig()).forward(imgs, texts=["x"])` works unchanged (no `target_state`)
- [ ] With `affordance.enabled=True`, `output["aux_loss"]` present when `target_state` provided
- [ ] Without `target_state`, `output` has no `aux_loss` key
- [ ] `affordance_head` excluded from `torch.compile` call
