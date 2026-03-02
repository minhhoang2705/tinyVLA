# Phase 07: Write Tests

**Status:** Pending
**Depends on:** All previous phases

## Overview

Unit tests for the new `AffordanceHead`, the auxiliary loss path in `VLAModel`, and the optional state extraction in `LeRobotVLADataset`. Integration tests already cover the full pipeline — only targeted unit tests are needed.

## Related Code Files

- **Modify:** `tests/unit/test_policy.py` — add AffordanceHead tests
- **Modify:** `tests/unit/test_vla_model.py` — add auxiliary loss path tests
- **Modify:** `tests/unit/test_data_pipeline.py` — add state extraction tests

## Test Cases

### `test_policy.py` — AffordanceHead

```python
class TestAffordanceHead:

    def test_output_shape_from_sequence_input(self):
        """AffordanceHead mean-pools [B, K, D] and returns [B, state_dim]."""
        head = AffordanceHead(input_dim=64, hidden_dim=32, state_dim=3)
        features = torch.randn(4, 16, 64)  # [B, K, D]
        out = head(features)
        assert out.shape == (4, 3)

    def test_output_shape_from_pooled_input(self):
        """AffordanceHead accepts pre-pooled [B, D] input."""
        head = AffordanceHead(input_dim=64, hidden_dim=32, state_dim=3)
        features = torch.randn(4, 64)  # [B, D]
        out = head(features)
        assert out.shape == (4, 3)

    def test_compute_loss_returns_scalar(self):
        """compute_loss returns a scalar MSE loss."""
        head = AffordanceHead(input_dim=64, hidden_dim=32, state_dim=3)
        pred = torch.randn(4, 3)
        target = torch.randn(4, 3)
        loss = head.compute_loss(pred, target)
        assert loss.shape == ()  # scalar

    def test_loss_is_zero_for_perfect_prediction(self):
        """MSE loss = 0 when pred == target."""
        head = AffordanceHead(input_dim=64, hidden_dim=32, state_dim=3)
        x = torch.randn(4, 3)
        loss = head.compute_loss(x, x)
        assert loss.item() < 1e-6
```

### `test_vla_model.py` — Auxiliary Loss Path

```python
class TestVLAModelAffordance:

    def _make_config_with_affordance(self):
        return VLAConfig(
            vision=VisionConfig(model_name="vit_tiny_patch16_224", pretrained=False),
            action=ActionConfig(action_dim=2),
            affordance=AffordanceConfig(enabled=True, state_dim=3, hidden_dim=32),
            auxiliary_loss_weight=0.1,
        )

    def test_affordance_head_built_when_enabled(self):
        config = self._make_config_with_affordance()
        model = VLAModel(config)
        assert model.affordance_head is not None

    def test_affordance_head_none_when_disabled(self):
        config = VLAConfig(
            vision=VisionConfig(model_name="vit_tiny_patch16_224", pretrained=False),
        )
        model = VLAModel(config)
        assert model.affordance_head is None

    def test_aux_loss_in_output_when_target_state_provided(self, dummy_image, dummy_text):
        config = self._make_config_with_affordance()
        model = VLAModel(config)
        target_state = torch.randn(2, 3)
        target_actions = torch.randn(2, 2).clamp(-1, 1)
        output = model(
            dummy_image, texts=dummy_text,
            target_actions=target_actions,
            target_state=target_state,
        )
        assert "aux_loss" in output
        assert "action_loss" in output
        assert output["aux_loss"].shape == ()

    def test_no_aux_loss_without_target_state(self, dummy_image, dummy_text):
        config = self._make_config_with_affordance()
        model = VLAModel(config)
        output = model(dummy_image, texts=dummy_text)
        assert "aux_loss" not in output

    def test_backward_compat_no_affordance(self, dummy_image, dummy_text):
        """Default VLAConfig (no affordance) forward pass unchanged."""
        config = VLAConfig(
            vision=VisionConfig(model_name="vit_tiny_patch16_224", pretrained=False),
        )
        model = VLAModel(config)
        output = model(dummy_image, texts=dummy_text)
        assert "actions" in output
        assert "aux_loss" not in output
```

### `test_data_pipeline.py` — State Extraction

```python
class TestLeRobotStateExtraction:

    def test_process_state_normalizes_to_minus_one_one(self):
        """_process_state clamps output to [-1, 1]."""
        ds = LeRobotVLADataset.__new__(LeRobotVLADataset)
        ds._state_mean = None
        ds._state_std = None
        ds._has_state = True
        ds.include_state = True

        raw_state = torch.tensor([0.0, 512.0, 3.14])
        result = ds._process_state({"observation.state": raw_state})
        assert result is not None
        assert result.min() >= -1.0
        assert result.max() <= 1.0

    def test_process_state_returns_none_when_missing(self):
        ds = LeRobotVLADataset.__new__(LeRobotVLADataset)
        ds._state_mean = None
        ds._state_std = None
        result = ds._process_state({})  # no observation.state key
        assert result is None
```

## Running Tests

```bash
# Run only affordance-related tests
pytest tests/unit/test_policy.py::TestAffordanceHead -v
pytest tests/unit/test_vla_model.py::TestVLAModelAffordance -v
pytest tests/unit/test_data_pipeline.py::TestLeRobotStateExtraction -v

# Run full suite to check no regressions
pytest tests/ -v
```

## Success Criteria

- [ ] All new tests pass
- [ ] No regressions in existing test suite
- [ ] `pytest tests/ --cov=vla` reports ≥ 80% coverage
