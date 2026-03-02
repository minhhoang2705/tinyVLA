# tinyVLA Test Execution Plan - Phases 05-07
**Date:** 2026-03-02 18:44 | **Version:** 1.0

---

## Test Execution Sequence

### Phase 1: AffordanceHead Tests (test_policy.py)
**Command:** `python -m pytest tests/unit/test_policy.py::TestAffordanceHead -v`

**Expected Results:**

| Test | Expected Status | Duration | Notes |
|------|-----------------|----------|-------|
| `test_output_shape_from_sequence_input` | ✓ PASS | <100ms | Tests [B,K,D]→[B,state_dim] mean-pool |
| `test_output_shape_from_pooled_input` | ✓ PASS | <100ms | Tests [B,D]→[B,state_dim] direct |
| `test_compute_loss_returns_scalar` | ✓ PASS | <100ms | Validates MSE loss shape |
| `test_loss_is_zero_for_perfect_prediction` | ✓ PASS | <100ms | Zero loss on identical inputs |

**Summary:** 4/4 tests should PASS
**Total Duration:** ~400ms

---

### Phase 2: VLAModel Affordance Tests (test_vla_model.py)
**Command:** `python -m pytest tests/unit/test_vla_model.py::TestVLAModelAffordance -v`

**Expected Results:**

| Test | Expected Status | Duration | Notes |
|------|-----------------|----------|-------|
| `test_affordance_head_built_when_enabled` | ✓ PASS | ~1-2s | Model instantiation with affordance=enabled |
| `test_affordance_head_none_when_disabled` | ✓ PASS | ~1-2s | Default config (affordance disabled) |
| `test_aux_loss_in_output_when_target_state_provided` | ✓ PASS | ~3-4s | Forward pass with target_state |
| `test_no_aux_loss_without_target_state` | ✓ PASS | ~1-2s | Forward without target_state |
| `test_backward_compat_no_affordance` | ✓ PASS | ~1-2s | Backward compatibility check |

**Summary:** 5/5 tests should PASS
**Total Duration:** ~7-12s

---

### Phase 3: LeRobot State Extraction Tests (test_data_pipeline.py)
**Command:** `python -m pytest tests/unit/test_data_pipeline.py::TestLeRobotStateExtraction -v`

**Expected Results:**

| Test | Expected Status | Duration | Notes |
|------|-----------------|----------|-------|
| `test_process_state_normalizes_to_minus_one_one` | ✓ PASS | <100ms | Fixed-scale norm [0,512]→[-1,1] |
| `test_process_state_returns_none_when_missing` | ✓ PASS | <100ms | Graceful handling of missing key |
| `test_collate_fn_stacks_states` | ✓ PASS | <100ms | Batch stacking [B,state_dim] |
| `test_collate_fn_omits_states_when_absent` | ✓ PASS | <100ms | Silent omission on partial batch |

**Summary:** 4/4 tests should PASS
**Total Duration:** ~400ms

---

### Phase 4: Full Unit Test Suite
**Command:** `python -m pytest tests/unit/ -v --tb=short`

**Expected Results:**
- All 13+ Phase 05-07 tests should PASS
- All existing tests should still PASS
- No regressions expected
- Coverage report generated

---

## Test Data Validation

### AffordanceHead Test Data

**Test 1: Sequence Input Shape**
```python
Input:  features [4, 16, 64]  # [B=4, K=16, D=64]
Expected: output [4, 3]       # [B=4, state_dim=3]
Operation: mean(dim=1) then MLP
```

**Test 2: Pooled Input Shape**
```python
Input:  features [4, 64]      # [B=4, D=64]
Expected: output [4, 3]       # [B=4, state_dim=3]
Operation: MLP only (no pooling)
```

**Test 3: Loss Computation**
```python
pred [4, 3], target [4, 3]
MSE loss = mean((pred - target)^2)
Expected: scalar tensor with .shape == torch.Size([])
```

**Test 4: Perfect Prediction**
```python
pred == target (both same tensor)
MSE loss = 0
Expected: loss.item() < 1e-6
```

### VLAModel Affordance Test Data

**Test 1: Enabled Configuration**
```python
config = VLAConfig(affordance=AffordanceConfig(enabled=True))
model = VLAModel(config)
Expected: model.affordance_head is not None
```

**Test 2: Disabled Configuration**
```python
config = VLAConfig()  # default affordance.enabled=False
model = VLAModel(config)
Expected: model.affordance_head is None
```

**Test 3: Loss with Target State**
```python
images [2, 3, 224, 224]
texts ["pick cube", "move block"]
target_state [2, 3]
target_actions [2, 2]  # action_dim from config
Output keys: "actions", "action_loss", "aux_loss", "loss", "state_pred"
```

**Test 4: No Loss without Target State**
```python
images [2, 3, 224, 224]
texts ["pick cube", "move block"]
# No target_state provided
Output keys: "actions" only
```

**Test 5: Backward Compatibility**
```python
config = VLAConfig()  # Default, no affordance
model = VLAModel(config)
Output keys: "actions" only (no aux_loss)
```

### LeRobot State Extraction Test Data

**Test 1: Fixed-Scale Normalization**
```python
Input state: [0.0, 512.0, 3.14]
_state_mean = None, _state_std = None
Normalization: state / 256.0 - 1.0
Expected output: [-1.0, 1.0, -0.988]
After clamp: [-1.0, 1.0, -0.988]
All in range [-1, 1]: ✓
```

**Test 2: Missing State Key**
```python
Input dict: {} (no "observation.state" key)
Expected: None returned
```

**Test 3: Batch Stacking with State**
```python
samples = [
    {..., "state": [0.1, 0.2, 0.3]},
    {..., "state": [0.4, 0.5, 0.6]},
]
Expected: batch["states"].shape == (2, 3)
```

**Test 4: Batch Omission without State**
```python
samples = [
    {...},  # no "state" key
    {...},  # no "state" key
]
Expected: "states" not in batch
```

---

## Assertion Validation

### AffordanceHead Assertions

| Assertion | Type | Expected |
|-----------|------|----------|
| `out.shape == (4, 3)` | Shape | torch.Size([4, 3]) |
| `loss.shape == ()` | Shape | torch.Size([]) |
| `loss.item() < 1e-6` | Value | True |

### VLAModel Affordance Assertions

| Assertion | Type | Expected |
|-----------|------|----------|
| `model.affordance_head is not None` | Type | True |
| `model.affordance_head is None` | Type | True |
| `"aux_loss" in output` | Key | True |
| `"aux_loss" not in output` | Key | True |
| `output["aux_loss"].shape == ()` | Shape | torch.Size([]) |

### LeRobot State Assertions

| Assertion | Type | Expected |
|-----------|------|----------|
| `result.min() >= -1.0` | Value | True |
| `result.max() <= 1.0` | Value | True |
| `result is None` | Type | True |
| `batch["states"].shape == (2, 3)` | Shape | torch.Size([2, 3]) |
| `"states" not in batch` | Key | True |

---

## Potential Issues & Mitigations

### Issue 1: Missing Dependencies
**Scenario:** lerobot not installed
**Mitigation:** Check conftest.py imports; installation verified in pyproject.toml
**Expected:** No issues (already installed in environment)

### Issue 2: GPU/CPU Device Mismatch
**Scenario:** Tests fail due to device placement
**Mitigation:** conftest.py has device fixture that handles both cuda/cpu
**Expected:** No issues (tests device-agnostic)

### Issue 3: Random Seed Reproducibility
**Scenario:** Tests fail due to randomness
**Mitigation:** conftest.py sets seed=42
**Expected:** No issues (deterministic with seed)

### Issue 4: Shape Mismatches
**Scenario:** Dimension checks fail
**Mitigation:** Carefully validated in implementation
**Expected:** No issues (shapes verified in code review)

---

## Success Criteria

### Primary Criteria
- [ ] TestAffordanceHead: 4/4 tests PASS
- [ ] TestVLAModelAffordance: 5/5 tests PASS
- [ ] TestLeRobotStateExtraction: 4/4 tests PASS
- [ ] No regressions in existing tests

### Secondary Criteria
- [ ] Code coverage >= 80%
- [ ] No type checking errors (mypy)
- [ ] No linting errors (ruff)
- [ ] All docstrings present

### Coverage Expectations

**Target Coverage by Module:**
- `vla/policy/affordance_head.py`: 100% (simple MLP)
- `vla/models/vla_configs.py`: 95%+ (config dataclasses)
- `vla/models/vla_base.py`: 90%+ (integration logic)
- `vla/data/lerobot_dataset.py`: 85%+ (state methods)
- `vla/data/collate_batch_samples.py`: 90%+ (collate functions)

---

## Test Execution Timing

### Individual Test Timing
```
AffordanceHead unit tests:      ~400ms  (4 tests)
VLAModel integration tests:     ~12s    (5 tests, includes model instantiation)
LeRobot state tests:            ~400ms  (4 tests)
──────────────────────────────────────
Subtotal Phase 05-07:           ~13s
```

### Full Unit Test Suite
```
Phase 05-07 new tests:          ~13s
Existing unit tests:            ~30-60s
────────────────────────────────────────
Total unit tests:               ~45-75s
```

---

## Post-Execution Report Structure

After test execution, report should contain:

### 1. Test Results Summary
- Total tests run: X
- Passed: X
- Failed: 0 (expected)
- Skipped: 0 (expected)
- Duration: X.XXs

### 2. By-Test Breakdown
Each test listed with:
- Status (PASS/FAIL)
- Duration
- Error message (if failed)

### 3. Coverage Report
- Line coverage: X%
- Branch coverage: X%
- Function coverage: X%
- Uncovered files/lines (if any)

### 4. Regression Analysis
- No regressions expected
- All existing tests should pass

### 5. Recommendations
- Any failing tests should include:
  - Error message
  - Stack trace
  - Suggested fix

---

## Example Passing Test Output

```
tests/unit/test_policy.py::TestAffordanceHead::test_output_shape_from_sequence_input PASSED [100%]
tests/unit/test_policy.py::TestAffordanceHead::test_output_shape_from_pooled_input PASSED [100%]
tests/unit/test_policy.py::TestAffordanceHead::test_compute_loss_returns_scalar PASSED [100%]
tests/unit/test_policy.py::TestAffordanceHead::test_loss_is_zero_for_perfect_prediction PASSED [100%]

tests/unit/test_vla_model.py::TestVLAModelAffordance::test_affordance_head_built_when_enabled PASSED
tests/unit/test_vla_model.py::TestVLAModelAffordance::test_affordance_head_none_when_disabled PASSED
tests/unit/test_vla_model.py::TestVLAModelAffordance::test_aux_loss_in_output_when_target_state_provided PASSED
tests/unit/test_vla_model.py::TestVLAModelAffordance::test_no_aux_loss_without_target_state PASSED
tests/unit/test_vla_model.py::TestVLAModelAffordance::test_backward_compat_no_affordance PASSED

tests/unit/test_data_pipeline.py::TestLeRobotStateExtraction::test_process_state_normalizes_to_minus_one_one PASSED
tests/unit/test_data_pipeline.py::TestLeRobotStateExtraction::test_process_state_returns_none_when_missing PASSED
tests/unit/test_data_pipeline.py::TestLeRobotStateExtraction::test_collate_fn_stacks_states PASSED
tests/unit/test_data_pipeline.py::TestLeRobotStateExtraction::test_collate_fn_omits_states_when_absent PASSED

======================== 13 passed in 13.45s =========================
```

---

## Contingency Plans

### If AffordanceHead Tests Fail
1. Check: Is AffordanceHead class imported correctly?
2. Check: Does __init__ properly initialize MLP?
3. Check: Does forward handle both 2D and 3D inputs?
4. Check: Does compute_loss return scalar?

### If VLAModel Affordance Tests Fail
1. Check: Is _build_affordance() creating head correctly?
2. Check: Is target_state parameter in forward()?
3. Check: Is aux_loss weighted by auxiliary_loss_weight?
4. Check: Is aux_loss combined with action_loss?

### If LeRobot State Tests Fail
1. Check: Is _process_state handling missing key?
2. Check: Is normalization using correct formula?
3. Check: Is clamping to [-1, 1] working?
4. Check: Is vla_collate_fn stacking states correctly?

---

## Notes

- Tests use fixtures from conftest.py
- All tests are unit-level (isolated)
- No external API calls or network access
- Deterministic with seed=42
- Device-agnostic (CPU/GPU)
- Quick execution (<2min for full suite)

**Confidence Level:** 95% all tests will pass on first run
