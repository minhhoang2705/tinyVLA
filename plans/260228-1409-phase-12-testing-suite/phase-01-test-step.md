# Phase 01: Add test_step to VLALightningModule

**Priority:** HIGH
**Status:** COMPLETE ✓
**File:** `src/vla/training/lightning_module.py`

## Context Links

- Source module: `src/vla/training/lightning_module.py` (200 LOC)
- Existing unit tests: `tests/unit/test_training.py`
- Brainstorm: `plans/reports/brainstorm-phase-12.md`

## Overview

`VLALightningModule` has `training_step` and `validation_step` but no `test_step`.
PyTorch Lightning's `trainer.test()` (used by `eval.py`) silently skips test loops without it.
The test step also adds MSE and MAE metrics — more informative than cross-entropy loss alone for action quality.

## Key Insights

- `_shared_step` handles train/val uniformly; test is slightly different (adds MSE/MAE)
- MSE/MAE are computed on **predicted actions vs target actions** (continuous space after argmax decode)
- The `VLAModel.predict()` method returns de-binned actions in `[-1, 1]` — reuse for metric computation
- All test metrics use `sync_dist=True` (matches val pattern for distributed eval safety)
- `prog_bar=False` for MSE/MAE (keep progress bar clean; only loss in bar)

## Requirements

### Functional
- `test_step(batch, batch_idx)` must call `self.log("test/loss", ...)`
- Must also log `test/mse` and `test/mae` on predicted vs target actions
- `sync_dist=True` for all test metrics (distributed evaluation)
- Method signature mirrors `validation_step`

### Non-functional
- File stays under 200 LOC after changes (~10 lines added)
- No new imports needed (torch already imported)

## Architecture

```
test_step(batch, batch_idx)
  └─ calls _shared_step(batch, "test")       → logs test/loss
  └─ calls model.predict(images, texts)      → predicted_actions [B, 7]
  └─ computes mse = F.mse_loss(pred, target)
  └─ computes mae = F.l1_loss(pred, target)
  └─ logs test/mse, test/mae (sync_dist=True)
  └─ returns loss
```

**Why predict() instead of output["actions"]?**
`_shared_step` calls `model(images, texts, target_actions)` which returns binned logits decoded to actions.
`model.predict()` returns the same actions clamped to `[-1, 1]`. Both give equivalent tensors, but
`predict()` is the public inference API — using it in test_step verifies the full inference path.

**Alternative:** Extend `_shared_step` to accept extra metrics. Rejected — YAGNI; test step is the only
caller needing MSE/MAE. Keep `_shared_step` clean, do extra work inline in `test_step`.

## Related Code Files

**Modify:**
- `src/vla/training/lightning_module.py` — add `test_step` method after `validation_step`

**Read (no changes):**
- `src/vla/models/vla_base.py` — confirm `predict()` signature
- `tests/unit/test_training.py` — ensure existing tests unaffected

## Implementation Steps

1. Open `src/vla/training/lightning_module.py`
2. After `validation_step` (line ~153), add `test_step`:

```python
def test_step(
    self, batch: Dict[str, Any], batch_idx: int
) -> torch.Tensor:
    """Compute loss + action-quality metrics on a test batch.

    In addition to cross-entropy loss (via _shared_step), computes
    MSE and MAE between predicted and target actions to measure
    continuous action quality.

    Args:
        batch: Dict with "images", "texts", "actions"
        batch_idx: Index of current batch (unused, required by PL)

    Returns:
        Scalar test loss
    """
    images: torch.Tensor = batch["images"]
    texts: List[str] = batch["texts"]
    target_actions: torch.Tensor = batch["actions"]

    # Cross-entropy loss (reuses shared logic)
    loss = self._shared_step(batch, "test")

    # Action-quality metrics (MSE / MAE in continuous action space)
    with torch.no_grad():
        predicted_actions = self.model.predict(images, texts)

    mse = torch.nn.functional.mse_loss(predicted_actions, target_actions)
    mae = torch.nn.functional.l1_loss(predicted_actions, target_actions)

    self.log("test/mse", mse, on_epoch=True, sync_dist=True)
    self.log("test/mae", mae, on_epoch=True, sync_dist=True)

    return loss
```

3. No import changes needed (`torch.nn.functional` via `torch` already imported)

## Todo

- [x] Add `test_step` method to `VLALightningModule`
- [x] Run `pytest tests/unit/test_training.py -v` — ensure no regressions
- [x] Run `black src/vla/training/lightning_module.py` + `ruff check`
- [x] Verify file stays under 200 LOC

## Success Criteria

- `test_step` exists and logs `test/loss`, `test/mse`, `test/mae`
- All existing unit tests in `tests/unit/test_training.py` still pass
- File ≤ 200 LOC

## Common Pitfalls

- **Don't call `model(...)` twice** in test_step (once in `_shared_step`, once for metrics). Use `predict()` which is inference-only — no second loss computation.
- **Don't set `on_step=True`** for test metrics — test runs once per epoch; step-level logging is meaningless.
- **Don't forget `torch.no_grad()`** around `predict()` — test step should not accumulate gradients.
