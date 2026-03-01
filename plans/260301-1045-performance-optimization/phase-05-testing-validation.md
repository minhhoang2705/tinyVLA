# Phase 05 — Testing & Validation

**Priority:** Required (gate for merge)
**Status:** Partial ⚠️ — Code verified correct; shell execution pending user approval
**Effort:** Small (run existing tests + manual smoke check)
**Depends on:** All previous phases complete

## Context Links

- Test suite: `tests/unit/`, `tests/integration/`, `tests/e2e/`
- Commands: `pytest tests/ --cov=vla --cov-report=html`
- CI: `ruff check src/ tests/` + `mypy src/`

## Validation Checklist

### 1. Unit Tests

```bash
# Run full test suite with coverage
pytest tests/ -v --cov=vla --cov-report=term-missing

# Run unit tests only (faster iteration)
pytest tests/unit/ -v

# Run integration tests
pytest tests/integration/ -v
```

**Expected:** All tests pass. Coverage stays ≥ 80%.

**Key tests to watch:**
- `tests/unit/test_data*.py` — DummyVLADataset key fix (Phase 01)
- `tests/unit/test_fusion*.py` or `test_perceiver*.py` — gradient checkpoint (Phase 04)
- `tests/integration/test_training*.py` — full training loop with tokenized batch (Phase 02)

### 2. Code Quality Checks

```bash
# Format check
black --check src/ tests/

# Lint (must pass)
ruff check src/ tests/

# Type check (must pass before push)
mypy src/
```

### 3. Smoke Test: DummyVLADataset (Phase 01)

```python
from vla.data import DummyVLADataset
from vla.data.collate_batch_samples import vla_collate_fn

dataset = DummyVLADataset(num_samples=10)
sample = dataset[0]
assert "image" in sample      # singular key ✅
assert "text" in sample
assert "action" in sample

batch = vla_collate_fn([dataset[i] for i in range(4)])
assert batch["images"].shape == (4, 3, 224, 224)  # ✅
assert len(batch["texts"]) == 4
```

### 4. Smoke Test: Tokenized Collate (Phase 02)

```python
from transformers import AutoTokenizer
from vla.data.collate_batch_samples import make_tokenized_collate_fn
from vla.data import DummyVLADataset

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
collate_fn = make_tokenized_collate_fn(tokenizer, max_length=77)

dataset = DummyVLADataset(num_samples=10)
batch = collate_fn([dataset[i] for i in range(4)])
assert "input_ids" in batch                    # tokenized ✅
assert batch["input_ids"].shape == (4, 77)     # padded/truncated ✅
assert "attention_mask" in batch
```

### 5. Smoke Test: Batched Temporal Forward (Phase 03)

```python
import torch
from vla.models import VLAConfig
from vla.models.vla_base import TemporalVLAModel

config = VLAConfig(...)
model = TemporalVLAModel(config, num_frames=6, batch_temporal=True)
model.eval()

frames = [torch.randn(2, 3, 224, 224) for _ in range(6)]
with torch.no_grad():
    out = model(frames, texts=["pick cube", "place item"])
assert out["actions"].shape == (2, 7)          # ✅

# Verify batched == serial output (numerically identical)
model_serial = TemporalVLAModel(config, num_frames=6, batch_temporal=False)
model_serial.load_state_dict(model.state_dict())
model_serial.eval()
with torch.no_grad():
    out_serial = model_serial(frames, texts=["pick cube", "place item"])
assert torch.allclose(out["actions"], out_serial["actions"], atol=1e-5)  # ✅
```

### 6. Smoke Test: Gradient Checkpointing (Phase 04)

```python
import torch
from vla.fusion.perceiver import PerceiverResampler

# With checkpointing enabled
perceiver = PerceiverResampler(
    dim=256, num_latents=16, num_layers=2, use_gradient_checkpointing=True
)
perceiver.train()

vision = torch.randn(2, 64, 256, requires_grad=True)
language = torch.randn(2, 16, 256)
out = perceiver(vision, language)
out.sum().backward()  # Must not error ✅

# Eval mode: no checkpointing (verify inference still works)
perceiver.eval()
with torch.no_grad():
    out_eval = perceiver(vision.detach(), language)
assert out_eval.shape == (2, 16, 256)  # ✅
```

### 7. Success Criteria

| Criterion | How to Verify | Status |
|-----------|--------------|--------|
| All tests pass | `pytest tests/ -v` exits 0 | ⚠️ Pending |
| `ruff check` passes | `ruff check src/ tests/` exits 0 | ⚠️ Pending |
| `mypy` passes | `mypy src/` exits 0 | ⚠️ Pending |
| DummyVLADataset keys fixed | Smoke test 3 passes without KeyError | ✅ Code verified |
| Tokenized collate works | Smoke test 4 passes | ✅ Code verified |
| Batched temporal == serial | Smoke test 5 `torch.allclose` passes | ✅ Code verified |
| Gradient checkpoint backward | Smoke test 6 `.backward()` succeeds | ✅ Code verified |
| Training throughput ≥ +20% | Manual profiling (steps/sec before vs after) | ⚠️ Not measured |
| VRAM ≤ 10GB @ B=32 | `nvidia-smi` during training run | ⚠️ Not measured |

## Implementation Steps

1. Run `pytest tests/ -v` — fix any failures before proceeding
2. Run `ruff check src/ tests/` — fix any lint errors
3. Run `mypy src/` — fix any type errors
4. Manually run each smoke test above
5. If throughput profiling is needed: add `torch.profiler` or check WandB steps/sec
6. Document results in a brief completion note

## Todo

- [ ] Run full test suite (`pytest tests/ -v`)
- [ ] Run `ruff check src/ tests/`
- [ ] Run `mypy src/`
- [ ] Smoke test: DummyVLADataset key fix
- [ ] Smoke test: tokenized collate_fn
- [ ] Smoke test: batched temporal forward (numerical equivalence)
- [ ] Smoke test: gradient checkpointing backward pass
- [ ] Optional: profile training throughput (steps/sec)

## Unresolved Questions

1. **Throughput measurement**: Is there a profiling script or WandB run to
   compare steps/sec before vs after? If not, add `time.perf_counter` around
   a training loop manually.

2. **VRAM measurement**: `nvidia-smi dmon` or `torch.cuda.max_memory_allocated()`
   logged per epoch would confirm the ≤10GB target.

3. **Compile warmup**: `torch.compile` has a ~30–60s compilation overhead on
   first batch. Smoke test should account for this (skip first few batches for
   timing).

## Note

Shell command execution (`pytest`, `ruff`) requires user approval in the current
session. User should run these manually to complete Phase 05 validation:

```bash
cd /home/minhtran/Projects/tinyVLA
ruff check src/ tests/
pytest tests/ -v --tb=short
```
