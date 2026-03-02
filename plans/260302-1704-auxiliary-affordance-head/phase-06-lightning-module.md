# Phase 06: Update Lightning Training Module

**Status:** Pending
**Depends on:** Phase 03, 05

## Overview

Update `VLALightningModule._shared_step` to:
1. Extract optional `"states"` from batch and pass as `target_state` to model
2. Log `train/action_loss` and `train/aux_loss` **separately** (not just `train/loss`)
3. Return combined loss for backprop

Also ensure `affordance_head` is **not** compiled in `setup()`.

## Related Code Files

- **Modify:** `src/vla/training/lightning_module.py`

## Implementation Steps

### 1. Update `setup()` — exclude affordance head from compile

Current code compiles only `action_head`. No change needed here since affordance head is a separate attribute. But add a comment to make intent explicit:

```python
def setup(self, stage: str) -> None:
    if stage == "fit":
        # Only compile action_head — affordance_head intentionally excluded:
        # compiling two small heads together degrades performance and
        # mixing compile with gradient checkpointing on the same module
        # causes Inductor graph failures.
        self.model.action_head = torch.compile(
            self.model.action_head,
            backend="inductor",
            mode="reduce-overhead",
        )
        logger.info("torch.compile applied to model.action_head (inductor, reduce-overhead)")
```

### 2. Update `_shared_step()` — separate loss logging

```python
def _shared_step(self, batch: Dict[str, Any], stage: str) -> torch.Tensor:
    images: torch.Tensor = batch["images"]
    target_actions: torch.Tensor = batch["actions"]
    # Optional state supervision (None when dataset has no observation.state)
    target_state: Optional[torch.Tensor] = batch.get("states", None)

    is_train = stage == "train"
    log_kwargs = dict(prog_bar=True, on_step=is_train, on_epoch=True, sync_dist=not is_train)

    # Forward pass
    if "input_ids" in batch:
        output = self.model(
            images,
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            target_actions=target_actions,
            target_state=target_state,
        )
    else:
        output = self.model(
            images,
            texts=batch["texts"],
            target_actions=target_actions,
            target_state=target_state,
        )

    # Log action loss and aux loss separately
    # VLAModel.forward() puts weighted action loss in output["loss"] before combining
    # We re-derive action_loss for separate logging:
    action_loss: torch.Tensor = output["loss"]
    if "aux_loss" in output:
        # output["loss"] = action_loss + aux_loss (already combined by VLAModel)
        # Log them separately for monitoring
        self.log(f"{stage}/action_loss", action_loss - output["aux_loss"], **log_kwargs)
        self.log(f"{stage}/aux_loss", output["aux_loss"], **log_kwargs)
        combined_loss = action_loss  # already combined
    else:
        combined_loss = action_loss

    self.log(f"{stage}/loss", combined_loss, **log_kwargs)
    return combined_loss
```

> **Note on logging:** `VLAModel.forward()` sets `output["loss"] = action_loss + aux_loss` (combined). We back-compute `action_loss_only = combined - aux_loss` for separate logging. This avoids changing VLAModel's output contract.

### Alternative (Cleaner) Approach

Have `VLAModel.forward()` keep `output["action_loss"]` and `output["aux_loss"]` separate, and set `output["loss"] = action_loss + aux_loss`. Then `_shared_step` just reads them directly:

```python
# Cleaner: VLAModel returns separate keys
action_loss = output.get("action_loss", output["loss"])
aux_loss = output.get("aux_loss", None)

if aux_loss is not None:
    self.log(f"{stage}/action_loss", action_loss, **log_kwargs)
    self.log(f"{stage}/aux_loss", aux_loss, **log_kwargs)

self.log(f"{stage}/loss", output["loss"], **log_kwargs)
return output["loss"]
```

**Recommended: use the cleaner approach** — store `output["action_loss"]` separately in VLAModel.forward() (Phase 03 should be updated accordingly).

## Key Constraints

- **`batch.get("states", None)`** — never `batch["states"]`; dataset may not have state
- **Separate metric logging** — `aux_loss` must be logged under its own key to avoid it inflating `train/loss` without visibility
- **`aux_loss` scaling** — already applied inside VLAModel (`* auxiliary_loss_weight`); don't double-scale

## Success Criteria

- [ ] `train/action_loss` and `train/aux_loss` appear as separate metrics in WandB/TensorBoard
- [ ] `train/loss` = combined (action + aux) for optimizer
- [ ] Training step works when batch has no `"states"` key (no crash)
- [ ] affordance_head absent from `torch.compile` call
