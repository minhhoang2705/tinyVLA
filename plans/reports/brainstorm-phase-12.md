## Brainstorm: Phase 12 Testing Suite

### Current State Assessment
- `tests/unit/test_training.py` — Mocked unit tests for `VLALightningModule`
- `tests/integration/test_vla_pipeline_end_to_end.py` — Raw PyTorch forward/backward

**What's MISSING for Phase 12**:
1. Lightning Trainer smoke test (`pl.Trainer(fast_dev_run=True)`)
2. `scripts/eval.py` evaluation entry point

### Recommended Implementation Plan
1. Add `test_step` to `VLALightningModule` (mirrors validation_step, logs test/loss, test/mse, test/mae)
2. Create `tests/e2e/__init__.py`
3. Create `tests/e2e/test_full_pipeline.py` (ONE test: `pl.Trainer(fast_dev_run=True).fit()`)
4. Create `scripts/eval.py` (`@hydra.main` → load checkpoint → `pl.Trainer.test()`)
