# Plan: Phase 12 Testing Suite

**Created:** 2026-02-28
**Branch:** master
**Source:** plans/reports/brainstorm-phase-12.md

## Overview

Complete Phase 12 (Testing & QA) by adding the two missing pieces:
1. A `test_step` on `VLALightningModule` (with MSE/MAE metrics)
2. An end-to-end Lightning Trainer smoke test (`fast_dev_run=True`)
3. A `scripts/eval.py` Hydra entry point for checkpoint evaluation

## Phases

| Phase | File | Status | Priority |
|-------|------|--------|----------|
| 01 | [phase-01-test-step.md](./phase-01-test-step.md) | COMPLETE | HIGH |
| 02 | [phase-02-e2e-trainer-test.md](./phase-02-e2e-trainer-test.md) | COMPLETE | HIGH |
| 03 | [phase-03-eval-script.md](./phase-03-eval-script.md) | COMPLETE | MEDIUM |

## Key Dependencies

- Phase 01 must complete before Phase 02 (e2e test calls `test_step`)
- Phase 02 must complete before Phase 03 (eval script reuses patterns from smoke test)
- All phases depend on existing `VLALightningModule`, `VLADataModule`, `scripts/train.py`

## Files Touched

```
src/vla/training/lightning_module.py   # Phase 01 — add test_step
tests/e2e/__init__.py                  # Phase 02 — new module
tests/e2e/test_full_pipeline.py        # Phase 02 — new test file
scripts/eval.py                        # Phase 03 — new script
```
