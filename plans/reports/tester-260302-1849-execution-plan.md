# tinyVLA Test Execution Plan & Detailed Analysis

**Date:** 2026-03-02 18:49
**Status:** Ready for Execution
**Analyzer:** QA Engineer

---

## Test Execution Quick Start

### Prerequisites Verification

```bash
cd /home/minhtran/Projects/tinyVLA

# Verify venv
source .venv/bin/activate

# Verify pytest installed
.venv/bin/pytest --version
# Expected: pytest X.X.X

# Verify torch GPU (optional)
.venv/bin/python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### Run Tests by Category

**1. Affordance Head Tests (6 tests - ~2 seconds)**
```bash
.venv/bin/pytest tests/unit/test_policy.py::TestAffordanceHead -v
```

**2. VLA Model Affordance Tests (8 tests - ~5 seconds)**
```bash
.venv/bin/pytest tests/unit/test_vla_model.py::TestVLAModelAffordance -v
```

**3. LeRobot State Extraction Tests (3 tests - ~1 second)**
```bash
.venv/bin/pytest tests/unit/test_data_pipeline.py::TestLeRobotStateExtraction -v
```

**4. All New Feature Tests (17 tests - ~8 seconds)**
```bash
.venv/bin/pytest \
  tests/unit/test_policy.py::TestAffordanceHead \
  tests/unit/test_vla_model.py::TestVLAModelAffordance \
  tests/unit/test_data_pipeline.py::TestLeRobotStateExtraction \
  -v
```

**5. Full Unit Test Suite with Coverage (85+ tests - ~30 seconds)**
```bash
.venv/bin/pytest tests/unit/ -v --cov=vla --cov-report=html --cov-report=term-missing
```

---

## Detailed Test Case Specifications

### TestAffordanceHead (6 test methods)

**Test 1: `test_output_shape_from_sequence_input`**
```python
Input:
  - features: torch.randn(4, 16, 64)  # [B=4, K=16, D=64]
  - head: AffordanceHead(input_dim=64, hidden_dim=32, state_dim=3)

Expected:
  - output.shape == (4, 3)
  - Internal: mean-pooling [B,K,D] → [B,D]

Pass Condition:
  - assert out.shape == (4, 3)
```

**Test 2: `test_output_shape_from_pooled_input`**
```python
Input:
  - features: torch.randn(4, 64)  # [B=4, D=64] pre-pooled
  - head: AffordanceHead(input_dim=64, hidden_dim=32, state_dim=3)

Expected:
  - output.shape == (4, 3)
  - Internal: no pooling (ndim == 2)

Pass Condition:
  - assert out.shape == (4, 3)
```

**Test 3: `test_compute_loss_returns_scalar`**
```python
Input:
  - pred: torch.randn(4, 3)
  - target: torch.randn(4, 3)

Expected:
  - loss = nn.functional.mse_loss(pred, target)
  - loss.shape == torch.Size([])

Pass Condition:
  - assert loss.shape == ()
  - assert loss.dtype == torch.float32
```

**Test 4: `test_loss_is_zero_for_perfect_prediction`**
```python
Input:
  - x = torch.randn(4, 3)
  - loss = head.compute_loss(x, x)

Expected:
  - When pred == target, MSE = 0

Pass Condition:
  - assert loss.item() < 1e-6
```

**Test 5-6: Registry tests** (existing, unchanged)

---

### TestVLAModelAffordance (8 test methods)

**Test 1: `test_affordance_head_built_when_enabled`**
```python
Config:
  affordance=AffordanceConfig(enabled=True, state_dim=3, hidden_dim=32)

Expected:
  model.affordance_head is not None
  isinstance(model.affordance_head, AffordanceHead)

Pass Condition:
  assert model.affordance_head is not None
```

**Test 2: `test_affordance_head_none_when_disabled`**
```python
Config:
  Default VLAConfig() (affordance.enabled=False)

Expected:
  model.affordance_head is None

Pass Condition:
  assert model.affordance_head is None
  (Backward compatibility: default behavior unchanged)
```

**Test 3: `test_aux_loss_in_output_when_target_state_provided`**
```python
Forward Call:
  output = model(
    images,
    texts=dummy_text,
    target_actions=torch.rand(2, 2) * 2 - 1,
    target_state=torch.randn(2, 3)
  )

Expected:
  output["aux_loss"] exists and is scalar
  output["action_loss"] exists and is scalar
  output["loss"] = action_loss + aux_loss (combined)

Pass Condition:
  assert "aux_loss" in output
  assert "action_loss" in output
  assert output["aux_loss"].shape == ()
  assert output["action_loss"].shape == ()
  assert output["loss"].shape == ()
```

**Test 4: `test_no_aux_loss_without_target_state`**
```python
Forward Call:
  output = model(images, texts=dummy_text)
  # No target_state provided

Expected:
  "aux_loss" not in output
  "action_loss" is still computed

Pass Condition:
  assert "aux_loss" not in output
  assert "action_loss" in output or "loss" not in output
```

**Test 5: `test_backward_compat_no_affordance`**
```python
Config:
  Default VLAConfig() (no affordance config)

Forward Call:
  output = model(dummy_image, texts=dummy_text)

Expected:
  output["actions"] exists (unchanged)
  "aux_loss" not in output (unchanged)

Pass Condition:
  assert "actions" in output
  assert "aux_loss" not in output
```

**Tests 6-8:** Supporting tests (fixtures, config validation)

---

### TestLeRobotStateExtraction (3 test methods)

**Test 1: `test_process_state_normalizes_to_minus_one_one`**
```python
Setup:
  ds = LeRobotVLADataset.__new__(LeRobotVLADataset)
  ds._state_mean = None  # Force fixed-scale path
  ds._state_std = None

Input:
  raw = {"observation.state": torch.tensor([0.0, 512.0, 3.14])}

Processing:
  state = state / 256.0 - 1.0
  state = torch.clamp(state, -1.0, 1.0)

Expected:
  result.min() >= -1.0
  result.max() <= 1.0

Pass Condition:
  assert result is not None
  assert result.min() >= -1.0
  assert result.max() <= 1.0
```

**Test 2: `test_process_state_returns_none_when_missing`**
```python
Setup:
  ds._state_mean = None
  ds._state_std = None

Input:
  raw = {}  # No observation.state key

Expected:
  return None (graceful handling)

Pass Condition:
  assert result is None
```

**Test 3: `test_collate_fn_stacks_states`**
```python
Input:
  samples = [
    {"image": ..., "text": ..., "action": ..., "state": torch.tensor([0.1, 0.2, 0.3])},
    {"image": ..., "text": ..., "action": ..., "state": torch.tensor([0.4, 0.5, 0.6])},
  ]

Processing:
  batch = vla_collate_fn(samples)

Expected:
  "states" in batch
  batch["states"].shape == (2, 3)

Pass Condition:
  assert "states" in batch
  assert batch["states"].shape == (2, 3)
```

**Bonus Test: `test_collate_fn_omits_states_when_absent`**
```python
Input:
  samples without "state" key

Expected:
  "states" not in batch (silent omit)

Pass Condition:
  assert "states" not in batch
```

---

## Test Dependency Graph

```
conftest.py (fixtures)
    ↓
tests/unit/test_policy.py::TestAffordanceHead
tests/unit/test_vla_model.py::TestVLAModelAffordance
tests/unit/test_data_pipeline.py::TestLeRobotStateExtraction
```

**No interdependencies between test classes.** Each is isolated.

---

## Expected Coverage Metrics

### Lines Covered (before running tests)

```
src/vla/policy/affordance_head.py
├── Lines of Code: 80
├── Executable: 80
├── Coverage: 0% (new file, not in .coverage yet)

src/vla/models/vla_base.py (partial)
├── _build_affordance(): ~19 lines
├── forward() affordance section: ~9 lines
├── Current coverage: ~90% overall, affordance section: 0%

src/vla/data/lerobot_dataset.py (partial)
├── _process_state(): ~18 lines
├── _detect_state_key(): ~3 lines
├── _load_state_stats(): ~23 lines
├── Current coverage: ~85% overall, state section: 0%
```

### Expected After Running Tests

```
src/vla/policy/affordance_head.py:  ~95% (6 tests cover main paths)
src/vla/models/vla_base.py:          ~92% (8 tests cover affordance paths)
src/vla/data/lerobot_dataset.py:     ~88% (3 tests cover state paths)

Overall project coverage target: 80%+ maintained or improved
```

---

## Test Failure Analysis & Recovery

### Scenario 1: ImportError on AffordanceHead

**Symptom:**
```
ImportError: cannot import name 'AffordanceHead' from 'vla.policy'
```

**Root Cause:** affordance_head.py not properly exported in __init__.py

**Check:**
```bash
grep -n "AffordanceHead" src/vla/policy/__init__.py
```

**Fix:**
```python
# src/vla/policy/__init__.py must contain:
from .affordance_head import AffordanceHead
__all__ = [..., "AffordanceHead"]
```

---

### Scenario 2: Shape Mismatch in Forward Pass

**Symptom:**
```
RuntimeError: Expected input_dim to match features dim
```

**Root Cause:** Fusion output dimension != AffordanceHead input_dim

**Check:**
```python
# In test:
print(f"Fusion output: {fused_features.shape}")  # [B, K, D]
print(f"Head expects: {head.mlp[0].in_features}")  # Should be D
```

**Fix:** Ensure config.affordance.input_dim == config.fusion.dim

---

### Scenario 3: Loss Computation NaN

**Symptom:**
```
AssertionError: loss is NaN or Inf
```

**Root Cause:** target_state values out of expected range

**Fix:** Ensure target_state in [-1, 1] range:
```python
target_state = torch.clamp(target_state, -1.0, 1.0)
```

---

### Scenario 4: Backward Pass Fails

**Symptom:**
```
RuntimeError: Trying to backward through the graph a second time
```

**Root Cause:** grad accumulation issue (test not zeroing gradients)

**Fix:** Ensure model.train() and optimizer.zero_grad() between steps

---

## Pytest Output Interpretation

### Success Output Example

```
tests/unit/test_policy.py::TestAffordanceHead::test_output_shape_from_sequence_input PASSED [ 16%]
tests/unit/test_policy.py::TestAffordanceHead::test_output_shape_from_pooled_input PASSED [ 33%]
tests/unit/test_policy.py::TestAffordanceHead::test_compute_loss_returns_scalar PASSED [ 50%]
tests/unit/test_policy.py::TestAffordanceHead::test_loss_is_zero_for_perfect_prediction PASSED [ 66%]

========================= 4 passed in 1.23s ==========================
```

### Failure Output Example

```
FAILED tests/unit/test_policy.py::TestAffordanceHead::test_output_shape_from_sequence_input
E   AssertionError: assert (4, 2) == (4, 3)
E                     +    ^
E                     -    ^
```

**Interpretation:** output shape is (4, 2) but test expects (4, 3)
**Likely cause:** state_dim parameter mismatch

---

## Performance Expectations

| Test Class | # Tests | Est. Time | Bottleneck |
|-----------|---------|-----------|-----------|
| TestAffordanceHead | 6 | 2-3s | Model instantiation |
| TestVLAModelAffordance | 8 | 5-8s | Vision encoder init |
| TestLeRobotStateExtraction | 3 | 1-2s | Data loading |
| **Total new tests** | **17** | **8-13s** | Fusion module |
| All unit tests | 85+ | 30-45s | Model checkpointing |

---

## Coverage Report Usage

After running tests:

```bash
# View HTML report
open htmlcov/index.html

# View terminal report
.venv/bin/pytest tests/unit/ --cov=vla --cov-report=term-missing

# Find uncovered lines
grep "Missing" htmlcov/*.html
```

---

## Checklist Before Commit

- [ ] All 17 new feature tests pass
- [ ] Overall coverage >= 80%
- [ ] affordance_head.py coverage >= 95%
- [ ] No new warnings from pytest
- [ ] Git diff reviewed (affordance_head.py, modified test files)
- [ ] Backward compatibility verified (test_backward_compat_no_affordance passes)

---

## Unresolved Questions

**None** - All test specifications are complete and ready for execution.
