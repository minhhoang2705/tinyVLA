# tinyVLA Project Memory

## Active Plan
- **Plan:** `plans/260301-1045-performance-optimization/`
- **Goal:** 7 performance optimizations from `PERFORMANCE_ACTION_PLAN.md`

## Implementation Status (as of 2026-03-01)

### Already Done ✅
- **Action 1:** ImageNet normalization → `lerobot_dataset.py:303`
- **Action 2:** Training augmentation + `TransformWrapper` → `datamodule_lightning.py:162`
- **Action 6:** `torch.no_grad()` on frozen backbones → `vla_base.py:348`

### Remaining ❌
- **Bug (Phase 01):** `DummyVLADataset.__getitem__` returns plural keys (`images/texts/actions`) but `vla_collate_fn` expects singular (`image/text/action`) → runtime KeyError
- **Action 3 (Phase 02):** Move tokenization to `collate_fn` workers (15–30% throughput)
- **Action 5 (Phase 03):** Batched temporal frame forward in `TemporalVLAModel`
- **Actions 4+7 (Phase 04):** gradient checkpoint on `fusion`, torch.compile on `action_head` only

## Key Architecture Facts
- 5 registries: VISION, LANGUAGE, FUSION, ACTION, MODEL
- Backbones (vision/language) are ALWAYS frozen during training
- `VLAModel.forward()` supports both `texts=` and `input_ids=` paths ✅

## Critical Design Decision
**torch.compile vs gradient checkpoint on fusion are MUTUALLY EXCLUSIVE.**
Resolution: checkpoint `model.fusion` + compile `model.action_head` only.
Reason: RTX 4070Ti is VRAM-bound → VRAM savings > compute speedup.

## Key File Locations
- Data: `src/vla/data/` — lerobot_dataset, dummy_vla_dataset, datamodule_lightning, collate_batch_samples, transform_wrapper
- Models: `src/vla/models/vla_base.py` — VLAModel + TemporalVLAModel
- Training: `src/vla/training/lightning_module.py`
- Fusion: `src/vla/fusion/perceiver.py` — PerceiverResampler + PerceiverBlock
- Configs: `src/vla/models/vla_configs.py`

## Naming Conventions
- Sample keys (singular): `image`, `text`, `action`
- Batch keys (plural): `images`, `texts`/`input_ids`/`attention_mask`, `actions`

## torch.compile Placement
- MUST be in `setup(stage="fit")` hook, NOT `__init__`
- Reason: PL moves model to device between `__init__` and `fit`

## Gradient Checkpointing
- Use `use_reentrant=False` — required for PyTorch >= 2.0 + PL + AMP
- Guard with `if self.training:` — inference doesn't need backward
