# Phase 01 — Fix DummyVLADataset Key Inconsistency

**Priority:** Critical (runtime KeyError)
**Status:** Complete ✅
**Effort:** Small (~15 LOC)
**Completed:** 2026-03-01

## Context Links

- Bug location: `src/vla/data/dummy_vla_dataset.py:133-136`
- Collate function: `src/vla/data/collate_batch_samples.py:48-54`
- Reference: `lerobot_dataset.py:273` uses singular keys (correct)

## Problem

`DummyVLADataset.__getitem__` returns plural keys:
```python
return {"images": image, "texts": text, "actions": action}
```

But `vla_collate_fn` expects singular keys:
```python
images = torch.stack([sample["image"] for sample in batch])  # KeyError!
texts = [sample["text"] for sample in batch]                 # KeyError!
actions = torch.stack([sample["action"] for sample in batch])  # KeyError!
```

`DummyTemporalVLADataset.__getitem__` has the same issue:
```python
return {"image_sequence": ..., "texts": text, "actions": action}
# temporal_vla_collate_fn expects sample["text"] and sample["action"] (singular)
```

`DummyVLADataset.get_batch()` also uses singular keys to read from `__getitem__`
result — broken at line 158-160.

## Fix

### File: `src/vla/data/dummy_vla_dataset.py`

**`DummyVLADataset.__getitem__` (line 133):** Change plural → singular keys:
```python
# Before
return {"images": image, "texts": text, "actions": action}

# After
return {"image": image, "text": text, "action": action}
```

**`DummyVLADataset.get_batch()` (line 158):** Fix to read from the correct keys
and return the batch format matching collate_fn output:
```python
# Before (broken - reads "image"/"text"/"action" from a result that uses plurals)
"images": torch.stack([s["image"] for s in samples]),
"texts": [s["text"] for s in samples],
"actions": torch.stack([s["action"] for s in samples]),

# After (reads correct singular keys from fixed __getitem__)
"images": torch.stack([s["image"] for s in samples]),
"texts": [s["text"] for s in samples],
"actions": torch.stack([s["action"] for s in samples]),
```
> get_batch() return keys stay plural (batch-level: "images", "texts", "actions") —
> this is correct since it mimics what collate_fn returns.

**`DummyTemporalVLADataset.__getitem__` (line 238):** Change plural → singular:
```python
# Before
return {"image_sequence": image_sequence, "texts": text, "actions": action}

# After
return {"image_sequence": image_sequence, "text": text, "action": action}
```

> **Note:** "image_sequence" stays (it's a special key for temporal data, not
> singular/plural confusion).

## Implementation Steps

1. Read `dummy_vla_dataset.py`
2. Fix `DummyVLADataset.__getitem__` return dict keys (plural → singular)
3. Verify `get_batch()` reads correct keys after fix (should be fine)
4. Fix `DummyTemporalVLADataset.__getitem__` return dict keys
5. Run `pytest tests/unit/ -v` to catch any test failures

## Todo

- [x] Fix `DummyVLADataset.__getitem__` return keys
- [x] Fix `DummyTemporalVLADataset.__getitem__` return keys
- [x] Verify `get_batch()` consistency
- [ ] Run unit tests (pending user shell approval)

## Success Criteria

- `vla_collate_fn` can process dummy dataset batches without `KeyError`
- All existing unit tests pass
- `DummyVLADataset[0]` returns `{"image": ..., "text": ..., "action": ...}`

## Risk: LOW

Pure renaming fix. No logic changes. Tests will catch regressions immediately.

## Completion Note

Fixed working-tree regression: `__getitem__` had reverted to plural keys.
Restored correct singular keys for both `DummyVLADataset` and `DummyTemporalVLADataset`.
