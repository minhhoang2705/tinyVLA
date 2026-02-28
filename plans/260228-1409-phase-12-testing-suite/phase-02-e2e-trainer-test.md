# Phase 02: Lightning Trainer Smoke Test (E2E)

**Priority:** HIGH
**Status:** COMPLETE ✓
**Depends on:** Phase 01 (test_step must exist) — ✓ SATISFIED

## Context Links

- Existing integration tests: `tests/integration/test_vla_pipeline_end_to_end.py`
- Training entry point: `scripts/train.py`
- DataModule: `src/vla/data/` + `VLADataModule`
- Brainstorm: `plans/reports/brainstorm-phase-12.md`

## Overview

The existing integration tests (`TestEndToEndForwardPass`, `TestTrainingLoop`) run raw PyTorch
forward/backward passes. They do NOT exercise the PyTorch Lightning Trainer, which has its own
lifecycle (hooks, callbacks, logging context). `fast_dev_run=True` runs exactly 1 train + 1 val +
1 test batch through the full Trainer loop — the minimal smoke test that catches PL integration bugs.

**One test only.** The purpose is verification, not coverage. Keep it fast (<30s on CPU).

## Key Insights

- `fast_dev_run=True` sets `max_epochs=1`, runs 1 batch each for train/val/test, disables checkpointing
- Need a DataModule with `test_dataloader()` — verify `VLADataModule` supports `setup("test")`
- Use tiny model config (same `_tiny_config()` pattern from `test_training.py`) to keep it fast
- `pl.Trainer(fast_dev_run=True, accelerator="cpu")` — always CPU in tests, no GPU assumption
- `logger=False` in Trainer — suppress CSV/WandB output during tests

## Requirements

### Functional
- ONE test: `test_trainer_fast_dev_run_completes`
- Calls `trainer.fit(module, datamodule=dm)` without raising
- Calls `trainer.test(module, datamodule=dm)` without raising (exercises `test_step`)
- Uses `fast_dev_run=True` to limit to 1 batch per phase

### Non-functional
- File under 80 LOC (single test class, minimal fixtures)
- No mocking — real Trainer, real DataModule, real model
- Must run on CPU in <60 seconds

## Architecture

```
tests/e2e/
├── __init__.py                     # empty, makes e2e a pytest package
└── test_full_pipeline.py
    └── class TestLightningTrainerSmoke
        └── test_trainer_fast_dev_run_completes()
              ├── _tiny_config()         # same minimal config as unit tests
              ├── VLALightningModule(cfg)
              ├── VLADataModule(dataset_type="dummy", batch_size=2, num_workers=0)
              ├── pl.Trainer(fast_dev_run=True, accelerator="cpu", logger=False)
              ├── trainer.fit(module, datamodule=dm)
              └── trainer.test(module, datamodule=dm)
```

**Why separate `tests/e2e/` directory?**
- `tests/unit/` = isolated component tests (mocked dependencies)
- `tests/integration/` = raw PyTorch pipeline (no PL Trainer)
- `tests/e2e/` = full PL Trainer lifecycle (the missing layer)

Keeps test categories distinct and makes CI filtering easy (`pytest tests/e2e/ -v`).

**Why not add to existing integration tests?**
The existing `test_vla_pipeline_end_to_end.py` is 400+ LOC raw PyTorch. Mixing PL Trainer tests in
there would blur the boundary between "raw PyTorch" and "Lightning Trainer" tests.

## Related Code Files

**Create:**
- `tests/e2e/__init__.py` — empty init
- `tests/e2e/test_full_pipeline.py` — smoke test

**Read (no changes):**
- `src/vla/data/__init__.py` — confirm `VLADataModule` export
- `src/vla/training/__init__.py` — confirm `VLALightningModule` export

## Implementation Steps

1. Create `tests/e2e/__init__.py` (empty file)

2. Create `tests/e2e/test_full_pipeline.py`:

```python
"""End-to-end smoke test for PyTorch Lightning Trainer integration.

Exercises the full PL lifecycle (fit + test) with fast_dev_run=True,
which runs exactly 1 train + 1 val + 1 test batch. Catches hook/callback
integration bugs that raw PyTorch tests cannot detect.
"""

import pytorch_lightning as pl
import pytest

from vla.data import VLADataModule
from vla.models.vla_configs import (
    ActionConfig,
    FusionConfig,
    LanguageConfig,
    VisionConfig,
    VLAConfig,
)
from vla.training import VLALightningModule


def _tiny_config() -> VLAConfig:
    """Minimal VLAConfig: no pretrained weights, tiny architecture."""
    return VLAConfig(
        vision=VisionConfig(
            name="timm_vit",
            model_name="vit_tiny_patch16_224",
            pretrained=False,
            frozen=True,
        ),
        language=LanguageConfig(
            name="gpt2",
            model_name="gpt2",
            frozen=True,
        ),
        fusion=FusionConfig(num_latents=8, num_layers=1, num_heads=2),
        action=ActionConfig(action_dim=7, num_bins=16),
        freeze_vision=True,
        freeze_language=True,
    )


class TestLightningTrainerSmoke:
    """Smoke tests for full PyTorch Lightning Trainer lifecycle."""

    def test_trainer_fast_dev_run_completes(self):
        """Trainer.fit() + Trainer.test() must complete without errors.

        fast_dev_run=True limits to 1 batch per phase (train/val/test),
        making this fast enough for CI while still exercising all PL hooks.
        """
        module = VLALightningModule(model_cfg=_tiny_config())
        dm = VLADataModule(
            dataset_type="dummy",
            batch_size=2,
            num_workers=0,  # No multiprocessing in tests
        )

        trainer = pl.Trainer(
            fast_dev_run=True,
            accelerator="cpu",
            logger=False,       # No CSV/WandB output during tests
            enable_checkpointing=False,
        )

        # fit() exercises training_step + validation_step
        trainer.fit(module, datamodule=dm)

        # test() exercises test_step (logs test/loss, test/mse, test/mae)
        trainer.test(module, datamodule=dm)
```

## Todo

- [x] Create `tests/e2e/__init__.py`
- [x] Create `tests/e2e/test_full_pipeline.py`
- [x] Verify `VLADataModule` has a `test_dataloader()` — ✓ confirmed working
- [x] Run `pytest tests/e2e/ -v` — ✓ passing
- [x] Run `pytest tests/ -v` — ✓ all tests passing (15/15)

## Success Criteria

- `test_trainer_fast_dev_run_completes` passes in <60s on CPU
- No PL lifecycle hooks raise exceptions
- `test_step` metrics (`test/loss`, `test/mse`, `test/mae`) logged without error

## Common Pitfalls

- **`num_workers > 0` in tests:** causes multiprocessing fork issues on some platforms. Always `num_workers=0`.
- **Missing `test_dataloader()`:** `VLADataModule` may only define `train_dataloader` + `val_dataloader`. Check `setup("test")` is handled; dummy dataset can reuse val split.
- **`enable_checkpointing=True` (default):** tries to write to disk. Disable in smoke tests to avoid filesystem side effects.
- **GPU assertion:** never assume CUDA in tests. Always `accelerator="cpu"`.
