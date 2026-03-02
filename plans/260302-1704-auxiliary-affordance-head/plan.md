# Plan: Auxiliary Affordance Head

**Status:** Complete
**Priority:** Medium
**Date:** 2026-03-02
**Branch:** master

## Context

- Brainstorm: [`plans/reports/brainstorm-auxiliary-head.md`](../reports/brainstorm-auxiliary-head.md)
- `observation.state` in LeRobot PushT contains `[block_x, block_y, block_angle]` — currently discarded
- Goal: Add a small MLP affordance head that predicts block state as an auxiliary supervision signal

## Phases

| Phase | Description | Status | File |
|-------|-------------|--------|------|
| 01 | Create AffordanceHead module | Complete | [phase-01](./phase-01-affordance-head-module.md) |
| 02 | Update VLAConfig | Complete | [phase-02](./phase-02-vla-config.md) |
| 03 | Update VLAModel forward pass | Complete | [phase-03](./phase-03-vla-model.md) |
| 04 | Update LeRobot dataset adapter | Complete | [phase-04](./phase-04-lerobot-dataset.md) |
| 05 | Update collate functions | Complete | [phase-05](./phase-05-collate.md) |
| 06 | Update Lightning training module | Complete | [phase-06](./phase-06-lightning-module.md) |
| 07 | Write tests | Complete | [phase-07](./phase-07-tests.md) |

## Key Decisions

- AffordanceHead lives in `src/vla/policy/affordance-head.py` (not action registry)
- `"state"` key is **Optional** throughout — no crash if absent
- Normalize state via dataset stats (mean/std from `meta.stats`) with fallback to fixed scale
- `auxiliary_loss_weight` default stays `0.0`; set to `0.1` to activate
- Do NOT compile affordance head in lightning setup step

## Files Modified

- `src/vla/policy/affordance-head.py` ← **new**
- `src/vla/models/vla_configs.py`
- `src/vla/models/vla_base.py`
- `src/vla/data/lerobot_dataset.py`
- `src/vla/data/collate_batch_samples.py`
- `src/vla/training/lightning_module.py`
- `tests/unit/test_policy.py`
- `tests/unit/test_vla_model.py`
- `tests/unit/test_data_pipeline.py`
