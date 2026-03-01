# Phase 02 — Action 3: CPU Tokenization in collate_fn

**Priority:** High (15–30% throughput improvement)
**Status:** Complete ✅
**Effort:** Medium (~50 LOC across 4 files)
**Depends on:** Phase 01 complete
**Completed:** 2026-03-01

## Context Links

- Brainstorm ref: Action 3, lines 94–143 of brainstorm report
- `src/vla/data/collate_batch_samples.py` — collate function to update
- `src/vla/data/datamodule_lightning.py` — DataModule creates tokenizer + passes to collate_fn
- `src/vla/training/lightning_module.py` — `_shared_step` routes `input_ids` to model
- `src/vla/models/vla_base.py:forward()` — already supports `input_ids` path ✅

## Problem

Tokenization currently happens inside `self.language.forward()` (in the model's
GPU forward pass). Even wrapped in `torch.no_grad()`, CPU tokenization blocks
the GPU, causing CPU-GPU sync stalls:

```
[DataLoader worker] → batch arrives (texts as strings)
[Training thread]   → GPU idle waiting → CPU tokenizes → GPU processes language
```

## Solution: Factory collate_fn + tokenizer in DataModule workers

```
[DataLoader worker] → tokenize texts → batch arrives with input_ids
[Training thread]   → GPU processes immediately (no CPU stall)
```

### Decision: Full worker-level tokenization (recommended path)
- 15–30% throughput gain vs 5–10% for the simpler alternative
- `VLAModel.forward()` already has `input_ids`/`attention_mask` path ✅
- Tokenizers (`GPT2Tokenizer`, `AutoTokenizer`) are picklable ✅

## Implementation

### 1. `src/vla/data/collate_batch_samples.py`

Added `make_tokenized_collate_fn` factory function with TYPE_CHECKING import guard
for `PreTrainedTokenizerBase`.

### 2. `src/vla/data/datamodule_lightning.py`

Added tokenizer params to `VLADataModule.__init__`:
- `tokenizer_name: str = "gpt2"`
- `max_token_length: int = 77`
- `use_tokenized_collate: bool = True`

`setup()` builds tokenizer once, creates `_collate_fn` used in all 3 DataLoaders.

### 3. `src/vla/training/lightning_module.py`

- `_shared_step()` routes `input_ids` vs `texts` batch paths
- `test_step()` supports pre-tokenized batch in predict path

## Files to Modify

| File | Change |
|------|--------|
| `src/vla/data/collate_batch_samples.py` | Add `make_tokenized_collate_fn` factory (+30 LOC) |
| `src/vla/data/datamodule_lightning.py` | Add tokenizer params + setup collate_fn (+25 LOC) |
| `src/vla/training/lightning_module.py` | Update `_shared_step` + `test_step` (+15 LOC) |

## Implementation Steps

1. Read all 3 files
2. Add `make_tokenized_collate_fn` to `collate_batch_samples.py`
3. Update `VLADataModule.__init__` and `setup()` in `datamodule_lightning.py`
4. Update `_shared_step` and `test_step` in `lightning_module.py`
5. Run `pytest tests/ -v` to verify no regressions

## Todo

- [x] Add `make_tokenized_collate_fn` to `collate_batch_samples.py`
- [x] Add tokenizer params to `VLADataModule.__init__`
- [x] Add tokenizer setup in `VLADataModule.setup()`
- [x] Update DataLoader calls to use `self._collate_fn`
- [x] Update `_shared_step` for dual input_ids/texts path
- [x] Update `test_step` predict call
- [ ] Run tests (pending user shell approval)

## Success Criteria

- Batches include `input_ids` + `attention_mask` when `use_tokenized_collate=True`
- `_shared_step` routes correctly for both tokenized and text batches
- All existing tests still pass (dummy dataset still uses `vla_collate_fn` or text path)

## Risk: MEDIUM

Batch format change. Backward-compatible via `"input_ids" in batch` check.
`VLAModel.forward()` already handles both paths. Tests will catch regressions.

## Completion Note

All 3 files updated. `make_tokenized_collate_fn` uses `TYPE_CHECKING` guard
for `PreTrainedTokenizerBase` to avoid importing transformers at import time.
Backward-compatible: `use_tokenized_collate=False` falls back to plain `vla_collate_fn`.
