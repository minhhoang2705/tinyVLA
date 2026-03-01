# Performance Optimization Plan — tinyVLA RTX 4070Ti

**Date:** 2026-03-01
**Ref:** `plans/reports/brainstorm-260301-0925-performance-optimization.md`
**Branch:** master
**Status:** Complete ✅ (implementation done; pytest/ruff pending user shell approval)

## Context

7 performance/quality optimizations identified in `PERFORMANCE_ACTION_PLAN.md`.
Actions 1, 2, 6 are already implemented. This plan covers the remaining work
plus a critical bug fix discovered during codebase analysis.

## Already Implemented (skip)

| Action | Description | File |
|--------|-------------|------|
| Action 1 | ImageNet normalization + antialias | `lerobot_dataset.py:303` |
| Action 2 | Training augmentation + TransformWrapper | `datamodule_lightning.py:162` |
| Action 6 | `torch.no_grad()` on frozen backbones | `vla_base.py:348` |

## Phases

| Phase | Description | Status | Priority |
|-------|-------------|--------|----------|
| [Phase 01](phase-01-fix-dummy-dataset-key-bug.md) | Fix DummyVLADataset key inconsistency | Complete ✅ | Critical |
| [Phase 02](phase-02-cpu-tokenization.md) | Action 3: CPU tokenization in collate_fn | Complete ✅ | High |
| [Phase 03](phase-03-batched-temporal-frames.md) | Action 5: Batched frame forward | Complete ✅ | High |
| [Phase 04](phase-04-compile-and-checkpoint.md) | Actions 4+7: torch.compile + gradient checkpoint | Complete ✅ | Medium |
| [Phase 05](phase-05-testing-validation.md) | Run tests + validate success criteria | Partial ⚠️ | Required |

## Key Architectural Decision (Embedded in Plan)

**Actions 4 and 7 are mutually exclusive on `fusion`.**
For RTX 4070Ti (12GB VRAM): use **gradient checkpointing** on `fusion` (VRAM savings)
+ **torch.compile** on `action_head` only (safe, no conflict).

## Files Modified

| File | Changes |
|------|---------|
| `src/vla/data/dummy_vla_dataset.py` | Phase 01: Singular keys in `__getitem__` |
| `src/vla/data/collate_batch_samples.py` | Phase 02: `make_tokenized_collate_fn` factory |
| `src/vla/data/datamodule_lightning.py` | Phase 02: Tokenizer params + collate_fn wiring |
| `src/vla/training/lightning_module.py` | Phase 02+04: `input_ids` routing + `setup()` compile |
| `src/vla/models/vla_configs.py` | Phase 04: `use_gradient_checkpointing` in `FusionConfig` |
| `src/vla/models/vla_base.py` | Phase 03+04: `batch_temporal` + conditional checkpoint kwarg |
| `src/vla/fusion/perceiver.py` | Phase 04: Gradient checkpoint loop in `PerceiverResampler` |

## Success Criteria

- [ ] All existing tests pass after each phase
- [ ] `ruff check` + `mypy` pass
- [ ] Training throughput ≥ 20% improvement (steps/sec)
- [ ] VRAM ≤ 10GB for default config (B=32, 4-layer fusion)
- [ ] No regressions in `test_step` MSE/MAE

## Pending Actions

Run these to complete Phase 05:
```bash
cd /home/minhtran/Projects/tinyVLA
ruff check src/ tests/
pytest tests/ -v --tb=short
```
