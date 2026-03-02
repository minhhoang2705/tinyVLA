# Phase 05: Update Collate Functions

**Status:** Pending
**Depends on:** Phase 04

## Overview

Update `vla_collate_fn` and `make_tokenized_collate_fn` to handle the optional `"state"` key. Only stack state tensors if **all** samples in the batch contain the key — otherwise omit it silently.

## Related Code Files

- **Modify:** `src/vla/data/collate_batch_samples.py`

## Implementation Steps

### 1. Update `vla_collate_fn`

Add state stacking after actions:

```python
def vla_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor | List[str]]:
    images = torch.stack([sample["image"] for sample in batch])
    texts = [sample["text"] for sample in batch]
    actions = torch.stack([sample["action"] for sample in batch])

    result: Dict[str, Any] = {
        "images": images,
        "texts": texts,
        "actions": actions,
    }

    # Stack state only if every sample has it (dataset may not provide state)
    if all("state" in sample for sample in batch):
        result["states"] = torch.stack([sample["state"] for sample in batch])

    return result
```

> **Key:** output key is `"states"` (plural) to match batch tensor convention. `lightning_module.py` reads `batch.get("states")`.

### 2. Update `make_tokenized_collate_fn` inner function

Same pattern inside the returned `collate_fn`:

```python
def collate_fn(batch):
    images = torch.stack([sample["image"] for sample in batch])
    texts = [sample["text"] for sample in batch]
    actions = torch.stack([sample["action"] for sample in batch])

    encoded = tokenizer(texts, padding="max_length", truncation=True,
                        max_length=max_length, return_tensors="pt")

    result = {
        "images": images,
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "actions": actions,
        "texts": texts,
    }

    if all("state" in sample for sample in batch):
        result["states"] = torch.stack([sample["state"] for sample in batch])

    return result
```

### 3. Leave `temporal_vla_collate_fn` unchanged

Temporal datasets don't use LeRobot PushT state; skip to keep changes minimal.

## Key Constraints

- **All-or-nothing stacking** — partial batches (some samples have state, some don't) are treated as no-state to avoid shape errors
- **Key name `"states"` (plural)** — consistent with `"images"`, `"actions"` naming convention in existing collate output

## Success Criteria

- [ ] Batch from PushT dataset contains `"states"` key with shape `[B, 3]`
- [ ] Batch from dataset without state has no `"states"` key (no KeyError)
- [ ] Mixed batch (some w/ state, some w/o) drops state silently
