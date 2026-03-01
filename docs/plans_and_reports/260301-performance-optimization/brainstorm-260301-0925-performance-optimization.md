---
title: Performance Optimization Brainstorm
date: 2026-03-01
type: brainstorm
status: completed
plan_ref: PERFORMANCE_ACTION_PLAN.md
---

# Performance Optimization Brainstorm — tinyVLA RTX 4070Ti

## Problem Statement

Implement 7 performance/quality optimizations in `PERFORMANCE_ACTION_PLAN.md`.
Key concern: `torch.compile` and DataLoader changes may introduce
PyTorch Lightning (PL) compatibility bugs, especially with AMP, FSDP, and
gradient flow.

---

## Action Items — Risk & Trade-off Analysis

### 🔴 Action 1: ImageNet Normalization (`lerobot_dataset.py`)

**Current state:**
- `_process_image()` returns `img[:3].float()` — range `[0,1]`, no normalization
- `DummyVLADataset.__getitem__()` also returns raw `[0,1]` images
- `TF.resize()` missing `antialias=True`

**Risk: LOW**
- Isolated change; no downstream API changes
- Tests use `DummyVLADataset` — dummy images don't need normalization for shape tests to pass
- BUT: existing trained checkpoints will be **incompatible** (shifted input distribution)

**Trade-offs:**
| Approach | Pros | Cons |
|---|---|---|
| Normalize in `_process_image()` | Close to data source, clear | `DummyVLADataset` also needs it for consistency |
| Normalize in `collate_fn` | Single place, covers all datasets | Mixes preprocessing concern with batching |
| Normalize in model `forward()` | One canonical location | Overhead inside autocast region; fights `torch.compile` fusion |

**Recommended:** Apply in `_process_image()` (LeRobot) and `__getitem__()` (Dummy).
Also add `antialias=True` to `TF.resize()` call.

**Dependency:** Independent; implement first.

---

### 🔴 Action 2: Training Augmentation (`datamodule_lightning.py`)

**Current state:**
- `VLADataModule.setup()` calls `random_split(full_dataset, ...)` on a single `LeRobotVLADataset`
- Neither `LeRobotVLADataset` nor `DummyVLADataset` accept a `transform` parameter
- Applying augmentation post-split is currently impossible without a wrapper

**Risk: MEDIUM — Architectural friction**

The core problem: `random_split()` returns views of the **same dataset object**.
If augmentation is added to the dataset class, it applies identically to train AND val.

**Options:**

Option A — `TransformWrapper` dataset (Recommended)
```python
class TransformWrapper(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset; self.transform = transform
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        sample["image"] = self.transform(sample["image"])
        return sample
```
After `random_split`, wrap only `train_dataset` with augmentation transforms.
- Pros: Clean, no changes to dataset classes
- Cons: Extra class, but minimal (< 15 LOC)

Option B — Pass `is_train` flag to dataset
- Pros: All in one class
- Cons: Hack; the same flag can't work after `random_split`

**Augmentation pipeline for train only:**
```python
from torchvision import transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
])
```

**Dependency:** Needs `TransformWrapper` helper (new file `data/transform_wrapper.py`).
Should be implemented after Action 1.

---

### 🟡 Action 3: CPU Tokenization Bottleneck (`language.py`)

**Current state:**
- `GPT2Backbone.forward()` calls `self.tokenize(texts, device)` inside the GPU forward pass
- Tokenizer sits inside language backbone — not accessible to DataLoader workers
- CPU tokenization happens in the main process, blocking GPU-CPU data transfer

**Risk: HIGH — Most impactful change; cascades to many files**

Correct fix requires these cascading changes:
1. `collate_batch_samples.py` — `vla_collate_fn` must tokenize texts and include `input_ids`, `attention_mask` in batch
2. `lightning_module.py` — `_shared_step()` and `test_step()` must pass `input_ids` instead of `texts`
3. Tokenizer must be decoupled from language backbone (or passed separately)

**The decoupling problem:**
The `GPT2Backbone` and `LanguageEncoder` both store the tokenizer internally.
To tokenize in `collate_fn` (which runs in DataLoader workers), the tokenizer
must be **serializable** (picklable) and **accessible outside the model**.

Both `GPT2Tokenizer` and `AutoTokenizer` are picklable — this is safe.

**Recommended approach:**
1. Create a `collate_fn` factory that closes over a tokenizer instance:
```python
def make_tokenized_collate_fn(tokenizer, max_length=77):
    def collate_fn(batch):
        tokens = tokenizer([s["text"] for s in batch], ...)
        return {"images": ..., "input_ids": tokens.input_ids, ...}
    return collate_fn
```
2. `VLADataModule` creates the tokenizer (same model name as language backbone)
3. Pass tokenized `input_ids`/`attention_mask` through the batch
4. `lightning_module._shared_step()` routes to `model(images, input_ids=..., attention_mask=...)`

Note: `VLAModel.forward()` already supports `input_ids` + `attention_mask` input path.

**Alternative (simpler but less efficient):**
Keep texts in batch but tokenize in `_shared_step()` before model call.
This moves tokenization out of the GPU forward pass but still runs in main process.
No DataLoader worker benefit, but removes CPU-GPU sync stall.

**Trade-off summary:**
| Approach | Speedup | Complexity | Risk |
|---|---|---|---|
| Tokenize in DataLoader workers (recommended) | 15-30% throughput | High | Medium |
| Tokenize in `_shared_step()` before model | 5-10% | Low | Low |
| Status quo (in model `forward()`) | 0% | None | None |

**Dependency:** Should be implemented before Actions 4/7 (those depend on stable forward() contract).

---

### 🟡 Action 4: `torch.compile` (`lightning_module.py`)

**Current state:**
- No compilation applied anywhere
- Model components use standard PyTorch eager mode

**Risk: MEDIUM — PyTorch Lightning + compile gotchas**

**Key PL Compatibility Issues:**

**Issue 1 — Where to apply compile:**
```python
# ❌ BAD: in __init__ (before Trainer.fit, before device move)
self.model.fusion = torch.compile(self.model.fusion, backend="inductor")

# ✅ GOOD: in setup() (called by Trainer after device placement)
def setup(self, stage: str) -> None:
    if stage == "fit":
        self.model.fusion = torch.compile(self.model.fusion, backend="inductor")
        self.model.action_head = torch.compile(self.model.action_head, backend="inductor")
```
`setup()` is the correct PL hook — guaranteed to run after device placement.

**Issue 2 — Dynamic shapes:**
`fusion.forward()` receives dynamic `vision_features` (N varies if vision backbone changes)
and `language_features` (L varies with instruction length).
`torch.compile` with `dynamic=True` or `fullgraph=False` (default) handles this but
may retrace graphs. Explicit `mode="reduce-overhead"` is better for training.

**Issue 3 — Flash Attention 2 + compile:**
`vla/nn/attention.py` uses FA2 when available. FA2 + `torch.compile` may conflict
on PyTorch < 2.2. Recommend pinning `torch>=2.2`.

**Issue 4 — Dict returns in compiled graph:**
`vla_base.py:VLAModel.forward()` returns `Dict[str, torch.Tensor]` and uses Python
`if isinstance(logits, tuple):` control flow. Compile applies only to `fusion` and
`action_head` submodules, not the orchestrator — this is safe.

**Issue 5 — Gradient checkpointing + compile conflict (see Action 7):**
Applying BOTH compile (Action 4) AND gradient checkpointing (Action 7) to the same
`PerceiverBlock` can cause graph capture failures. These must be applied to
**different modules** or with specific `compiler_config`.

**Recommended compile targets:**
- `model.fusion` (PerceiverResampler) — pure tensor ops, no Python control flow on tensors
- `model.action_head` — simple linear layers, safe to compile
- ❌ Do NOT compile `model.vision` or `model.language` (frozen, no benefit; FA2 conflict risk)
- ❌ Do NOT compile `VLAModel.forward()` (dict return, validation utils, Python control flow)

**Expected speedup:** 1.5-2.5x on `fusion` forward, ~1.2x on `action_head`.

---

### 🟡 Action 5: Batched Frame Forward (`vla_base.py` — `TemporalVLAModel`)

**Current state (line 567):**
```python
vision_features = [self.vision(img) for img in image_sequence]
```
T separate GPU kernel launches for T frames.

**Risk: LOW-MEDIUM — Isolated to TemporalVLAModel**

**Fix:**
```python
B = image_sequence[0].size(0)
T = len(image_sequence)
all_frames = torch.cat(image_sequence, dim=0)  # [B*T, 3, H, W]
all_features = self.vision(all_frames)         # [B*T, N, D_v]
vision_features_concat = all_features.view(B, T * all_features.size(1), -1)  # [B, T*N, D]
```

**VRAM trade-off:**
- Old: B images peak VRAM per call (but T calls total)
- New: B×T images peak VRAM for one call
- For B=4, T=6: 24 images simultaneously — may OOM on RTX 4070Ti (12GB)
- For B=2, T=6: 12 images — feasible

**Recommendation:** Add configurable `batch_temporal` flag. Default True only if
`B * T * 3 * 224 * 224 * 4 bytes < 0.5 * available_vram`.

**No PL impact** — purely model-level change.

---

### 🟢 Action 6: `torch.no_grad()` for Frozen Backbones (`vla_base.py`)

**Current state (lines 347-356):**
```python
vision_features = self.vision(images)        # no no_grad
language_features = self.language(texts=texts)  # no no_grad
```

**Risk: VERY LOW — No-brainer optimization**

Even though backbones are frozen (`requires_grad=False`), PyTorch still builds
partial computation graphs for non-leaf tensors. `torch.no_grad()` skips this
entirely, saving ~10-20% activation memory.

**PL + AMP compatibility:**
`torch.no_grad()` nests cleanly inside PL's `autocast` context.
The AMP manager wraps the entire `training_step`; `no_grad` for sub-regions
is fully supported.

**Implementation:**
```python
with torch.no_grad():
    vision_features = self.vision(images)
    if texts is not None:
        language_features = self.language(texts=texts)
    else:
        language_features = self.language(input_ids=input_ids, attention_mask=attention_mask)
```

**Dependency:** Independent. Implement immediately after reading this report.

---

### 🟢 Action 7: Gradient Checkpointing for PerceiverBlock (`perceiver.py`)

**Current state (line 130):**
```python
for layer in self.layers:
    latents = layer(latents, context)
```

**Risk: MEDIUM — Compile conflict; reentrant mode issue**

**Fix:**
```python
from torch.utils.checkpoint import checkpoint

for layer in self.layers:
    if self.training:
        latents = checkpoint(layer, latents, context, use_reentrant=False)
    else:
        latents = layer(latents, context)
```

**Critical: `use_reentrant=False`:**
- `use_reentrant=True` (old default) has memory leaks with DDP and AMP in PL
- `use_reentrant=False` is required for PyTorch >= 2.0 with PL

**PL + AMP compatibility:**
PL applies `torch.autocast` in `training_step`. `checkpoint` with `use_reentrant=False`
correctly preserves the autocast context. ✅ Compatible.

**Conflict with Action 4 (compile):**
If `torch.compile(model.fusion)` is applied (Action 4), then checkpointing inside
`PerceiverResampler.forward()` may cause graph capture failures with Inductor.
**Resolution:** Choose one per module:
- If compile is applied to `fusion`: SKIP gradient checkpointing on `PerceiverBlock`
- If checkpointing is needed: compile only `action_head`, skip `fusion` compile

**VRAM savings:** ~40-50% reduction in peak activation memory for fusion module
(trades compute for memory — each block's activations are recomputed on backward).

**Training-only guard (`if self.training`) is mandatory** — inference doesn't
need backward, so checkpointing doubles compute for zero VRAM benefit during eval.

---

## Dependency & Execution Order

```
Action 6 (no_grad)           → No deps → Implement FIRST
Action 1 (normalization)     → No deps → Implement SECOND
Action 5 (batched frames)    → No deps → Implement THIRD
Action 2 (augmentation)      → Needs TransformWrapper → FOURTH
Action 3 (tokenization)      → Cascading changes → FIFTH (validate tests pass)
Action 7 (grad checkpoint)   → Conflicts with Action 4 → SIXTH
Action 4 (torch.compile)     → Apply last, test stability → SEVENTH
```

**Rule:** Actions 4 and 7 on the same module (fusion) are **mutually exclusive**.
Choose compile OR checkpoint for `PerceiverResampler`, not both.

---

## PyTorch Lightning Compatibility Summary

| Action | PL Risk | Key Concern | Resolution |
|---|---|---|---|
| 1 (normalization) | None | — | Safe |
| 2 (augmentation) | None | TransformWrapper after random_split | Use wrapper dataset |
| 3 (tokenization) | Low | batch format change | Update `_shared_step()` to accept `input_ids` |
| 4 (compile) | Medium | Apply in `setup()` not `__init__` | Use `setup(stage="fit")` hook |
| 5 (batched frames) | None | VRAM with large B*T | Add configurable flag |
| 6 (no_grad) | None | AMP nesting | Fully compatible |
| 7 (grad checkpoint) | Medium | `use_reentrant=False` required; conflicts with compile | Mutually exclusive with compile on same module |

---

## Files To Modify

| File | Actions | Estimated LOC delta |
|---|---|---|
| `src/vla/data/lerobot_dataset.py` | 1 | +5 |
| `src/vla/data/dummy_vla_dataset.py` | 1 | +5 |
| `src/vla/data/datamodule_lightning.py` | 2, 3 | +30 |
| `src/vla/data/collate_batch_samples.py` | 3 | +25 |
| `src/vla/data/transform_wrapper.py` (NEW) | 2 | +15 |
| `src/vla/models/vla_base.py` | 5, 6 | +15 |
| `src/vla/training/lightning_module.py` | 3, 4 | +20 |
| `src/vla/fusion/perceiver.py` | 7 | +10 |

Total: ~125 LOC changed/added across 8 files.

---

## Success Criteria

- [ ] All existing tests pass after each action item
- [ ] `ruff check` + `mypy` pass
- [ ] Training throughput on RTX 4070Ti improves ≥ 20% (steps/sec)
- [ ] VRAM usage drops ≤ 10GB for default config (B=32, fusion 4-layer)
- [ ] Val loss is numerically comparable (normalization shifts absolute values; relative ordering preserved)
- [ ] No regressions in `test_step` MSE/MAE metrics

---

## Unresolved Questions

1. **Checkpoint-vs-compile choice**: Which gives more benefit for RTX 4070Ti —
   gradient checkpointing on Perceiver (VRAM savings) or compiling fusion (speed)?
   Depends on bottleneck: if VRAM-bound, use checkpoint; if compute-bound, use compile.
   Needs empirical measurement.

2. **Tokenizer decoupling for Action 3**: Should `VLADataModule` instantiate a
   standalone tokenizer (requires knowing language model name), or should the
   tokenizer be exported from the language backbone as a public attribute?

3. **Batched temporal frames (Action 5)**: For typical training config (B=4, T=6,
   RTX 4070Ti 12GB), will B*T=24 frames exceed VRAM? Needs profiling.

4. **`antialias=True` in resize**: PyTorch < 1.11 doesn't support this kwarg.
   Confirm PyTorch version requirement in `pyproject.toml`.
