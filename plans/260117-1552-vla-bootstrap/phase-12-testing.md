# Phase 12: Testing Suite

## Context Links
- [Tech Stack](../../docs/tech-stack.md) - Testing section
- [PyTorch VLA Research](../reports/researcher-260118-0228-pytorch-vla.md)

## Overview
| Field | Value |
|-------|-------|
| Priority | P2 - Quality Assurance |
| Status | Pending |
| Effort | 2h |
| Dependencies | Phases 1-11 |

Implement comprehensive testing suite covering unit tests, integration tests, and end-to-end tests for all VLA components.

## Key Insights
- Unit tests per module (registry, nn, backbones, etc.)
- Integration tests for component composition
- End-to-end test with dummy data through full pipeline
- Coverage target: 80%+

## Requirements

### Functional
- FR-01: Unit tests for all public APIs
- FR-02: Integration tests for component interactions
- FR-03: End-to-end training smoke test
- FR-04: Config validation tests
- FR-05: Data loading tests

### Non-Functional
- NFR-01: Tests run <5 minutes on CPU
- NFR-02: 80%+ code coverage

## Architecture

```
tests/
├── conftest.py              # Shared fixtures
├── unit/
│   ├── test_registry.py
│   ├── test_nn.py
│   ├── test_vision.py
│   ├── test_language.py
│   ├── test_fusion.py
│   ├── test_policy.py
│   ├── test_vla_model.py
│   └── test_data.py
├── integration/
│   ├── test_model_composition.py
│   ├── test_config_loading.py
│   └── test_training_loop.py
└── e2e/
    └── test_full_pipeline.py
```

## Related Code Files

### Files to Create
| Path | Purpose | Lines |
|------|---------|-------|
| `tests/conftest.py` | Shared fixtures | ~80 |
| `tests/integration/test_model_composition.py` | Integration | ~100 |
| `tests/integration/test_config_loading.py` | Config tests | ~80 |
| `tests/e2e/test_full_pipeline.py` | E2E tests | ~80 |

## Implementation Steps

### Step 1: Enhance conftest.py (30 min)
```python
"""Shared pytest fixtures for VLA testing."""
import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path
from omegaconf import OmegaConf


# ============= Device Fixtures =============

@pytest.fixture
def device():
    """Get available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(params=["cpu", "cuda"])
def any_device(request):
    """Parametrized device fixture."""
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)


# ============= Data Fixtures =============

@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def dummy_image(batch_size):
    """Generate dummy image batch."""
    return torch.randn(batch_size, 3, 224, 224)


@pytest.fixture
def dummy_texts(batch_size):
    """Generate dummy text batch."""
    return ["pick up the red block", "place on table"][:batch_size]


@pytest.fixture
def dummy_actions(batch_size):
    """Generate dummy action batch."""
    return torch.rand(batch_size, 7) * 2 - 1


@pytest.fixture
def dummy_batch(dummy_image, dummy_texts, dummy_actions):
    """Generate complete dummy batch."""
    return {
        "images": dummy_image,
        "texts": dummy_texts,
        "actions": dummy_actions,
    }


# ============= Config Fixtures =============

@pytest.fixture
def base_config():
    """Minimal valid config for testing."""
    return OmegaConf.create({
        "seed": 42,
        "output_dir": "/tmp/vla_test",
        "model": {
            "freeze_vision": True,
            "freeze_language": True,
        },
        "vision": {
            "name": "timm_vit",
            "model_name": "vit_tiny_patch16_224",
            "pretrained": False,
            "frozen": True,
            "output_mode": "spatial",
            "proj_dim": None,
        },
        "language": {
            "name": "gpt2",
            "model_name": "gpt2",
            "frozen": True,
            "output_mode": "mean",
            "max_length": 32,
            "proj_dim": None,
        },
        "fusion": {
            "name": "perceiver_resampler",
            "dim": 256,
            "num_latents": 16,
            "num_layers": 1,
            "num_heads": 4,
        },
        "action": {
            "name": "discrete_action",
            "action_dim": 7,
            "num_bins": 256,
            "hidden_dim": None,
        },
        "train": {
            "max_epochs": 1,
            "batch_size": 2,
            "gradient_clip_val": 1.0,
            "gradient_accumulation_steps": 1,
            "checkpoint": {"save_top_k": 1, "monitor": "val/loss", "mode": "min", "save_last": True},
            "logging": {"log_every_n_steps": 1, "val_check_interval": 1.0},
        },
        "data": {
            "name": "dummy",
            "num_samples": 20,
            "image_size": 224,
            "num_workers": 0,
        },
    })


@pytest.fixture
def small_vision_config():
    """Small vision config for fast testing."""
    return {
        "name": "timm_vit",
        "model_name": "vit_tiny_patch16_224",
        "pretrained": False,
        "frozen": False,
        "output_mode": "spatial",
        "proj_dim": None,
    }


# ============= Model Fixtures =============

@pytest.fixture
def small_vla_model(base_config):
    """Create small VLA model for testing."""
    from vla.models import VLAModel, VLAConfig

    config = VLAConfig.from_dict(OmegaConf.to_container(base_config))
    return VLAModel(config)


# ============= Temp Directory Fixtures =============

@pytest.fixture
def tmp_dir():
    """Create temporary directory."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# ============= Seeding =============

@pytest.fixture(autouse=True)
def seed_everything():
    """Seed random number generators for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
```

### Step 2: Implement integration tests (45 min)
```python
# tests/integration/test_model_composition.py
"""Integration tests for VLA model composition."""
import pytest
import torch
from vla.models import VLAModel, VLAConfig
from vla.registry import VISION_REGISTRY, LANGUAGE_REGISTRY, FUSION_REGISTRY, ACTION_REGISTRY


class TestModelComposition:
    """Test that components compose correctly."""

    def test_vision_to_fusion_dimensions(self, base_config, dummy_image):
        """Test vision output matches fusion input."""
        from vla.backbones import VisionBackbone
        from vla.fusion import PerceiverResampler

        # Build vision
        vision = VisionBackbone(
            model_name="vit_tiny_patch16_224",
            pretrained=False,
            output_mode="spatial",
        )
        vision_out = vision(dummy_image)

        # Build fusion expecting vision dimensions
        fusion = PerceiverResampler(
            dim=256,
            num_latents=16,
            vision_dim=vision.embed_dim,
        )

        # Should work without dimension errors
        fused = fusion(vision_out)
        assert fused.shape[0] == dummy_image.shape[0]
        assert fused.shape[1] == 16  # num_latents

    def test_fusion_to_action_dimensions(self, base_config):
        """Test fusion output matches action input."""
        from vla.fusion import PerceiverResampler
        from vla.policy import DiscreteActionHead

        fusion_dim = 256
        num_latents = 16
        action_dim = 7

        # Dummy fusion output
        fused = torch.randn(2, num_latents, fusion_dim)

        # Action head should accept fusion output
        action_head = DiscreteActionHead(
            input_dim=fusion_dim,
            action_dim=action_dim,
        )
        actions, _ = action_head(fused)
        assert actions.shape == (2, action_dim)

    def test_full_model_forward(self, small_vla_model, dummy_batch):
        """Test full model forward pass."""
        output = small_vla_model(
            images=dummy_batch["images"],
            texts=dummy_batch["texts"],
            target_actions=dummy_batch["actions"],
        )
        assert "actions" in output
        assert "loss" in output
        assert output["actions"].shape == dummy_batch["actions"].shape

    def test_gradient_flow(self, small_vla_model, dummy_batch):
        """Test gradients flow to trainable parameters."""
        output = small_vla_model(
            images=dummy_batch["images"],
            texts=dummy_batch["texts"],
            target_actions=dummy_batch["actions"],
        )
        loss = output["loss"]
        loss.backward()

        # Check fusion has gradients (should be trainable)
        has_grad = False
        for param in small_vla_model.fusion.parameters():
            if param.grad is not None:
                has_grad = True
                break
        assert has_grad, "Fusion should have gradients"


# tests/integration/test_config_loading.py
"""Integration tests for Hydra config loading."""
import pytest
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
from pathlib import Path


class TestConfigLoading:
    """Test Hydra configuration system."""

    @pytest.fixture
    def config_dir(self):
        """Get configs directory path."""
        # Adjust path based on project structure
        return Path(__file__).parent.parent.parent / "configs"

    def test_base_config_valid(self, base_config):
        """Test base config has all required fields."""
        required = ["vision", "language", "fusion", "action", "train"]
        for field in required:
            assert field in base_config, f"Missing {field}"

    def test_config_overrides(self, base_config):
        """Test config can be overridden."""
        base_config.vision.model_name = "vit_base_patch16_224"
        assert base_config.vision.model_name == "vit_base_patch16_224"

    def test_config_to_model(self, base_config):
        """Test config can instantiate model."""
        from vla.models import VLAModel, VLAConfig

        config = VLAConfig.from_dict(OmegaConf.to_container(base_config))
        model = VLAModel(config)
        assert model is not None
```

### Step 3: Implement E2E tests (45 min)
```python
# tests/e2e/test_full_pipeline.py
"""End-to-end tests for full VLA pipeline."""
import pytest
import torch
import pytorch_lightning as pl
from pathlib import Path


class TestFullPipeline:
    """End-to-end tests for complete training pipeline."""

    def test_training_smoke_test(self, base_config, tmp_dir):
        """Test full training loop runs without errors."""
        from vla.training import VLALightningModule, VLADataModule

        base_config.output_dir = str(tmp_dir)
        base_config.train.max_epochs = 1
        base_config.data.num_samples = 16

        # Create modules
        model = VLALightningModule(base_config)
        data_module = VLADataModule(base_config)

        # Create trainer
        trainer = pl.Trainer(
            max_epochs=1,
            fast_dev_run=True,
            accelerator="cpu",
            enable_progress_bar=False,
            logger=False,
        )

        # Should complete without errors
        trainer.fit(model, datamodule=data_module)

    def test_checkpoint_save_load(self, small_vla_model, dummy_batch, tmp_dir):
        """Test checkpoint save and load."""
        ckpt_path = tmp_dir / "model.pt"

        # Get initial predictions
        output1 = small_vla_model(
            images=dummy_batch["images"],
            texts=dummy_batch["texts"],
        )

        # Save
        small_vla_model.save_checkpoint(str(ckpt_path))

        # Load
        from vla.models import VLAModel
        loaded = VLAModel.load_checkpoint(str(ckpt_path))

        # Check predictions match
        output2 = loaded(
            images=dummy_batch["images"],
            texts=dummy_batch["texts"],
        )

        assert torch.allclose(
            output1["actions"], output2["actions"], atol=1e-5
        )

    def test_inference_mode(self, small_vla_model, dummy_batch):
        """Test model in inference mode."""
        actions = small_vla_model.predict(
            images=dummy_batch["images"],
            texts=dummy_batch["texts"],
        )
        assert actions.shape == (2, 7)
        assert actions.requires_grad == False

    def test_data_to_model_pipeline(self, base_config):
        """Test data flows correctly through model."""
        from vla.data import DummyVLADataset, collate_vla_batch
        from vla.models import VLAModel, VLAConfig

        # Create dataset
        dataset = DummyVLADataset(
            num_samples=10,
            action_dim=base_config.action.action_dim,
        )

        # Get batch
        batch = collate_vla_batch([dataset[i] for i in range(2)])

        # Create model
        config = VLAConfig.from_dict(base_config)
        model = VLAModel(config)

        # Forward pass
        output = model(
            images=batch["images"],
            texts=batch["texts"],
            target_actions=batch["actions"],
        )

        assert output["actions"].shape == batch["actions"].shape
        assert output["loss"].item() > 0


class TestRegistryIntegration:
    """Test registry-based component loading."""

    def test_all_registries_populated(self):
        """Test all registries have expected components."""
        from vla.registry import (
            VISION_REGISTRY,
            LANGUAGE_REGISTRY,
            FUSION_REGISTRY,
            ACTION_REGISTRY,
            MODEL_REGISTRY,
        )

        # Import modules to trigger registration
        import vla.backbones
        import vla.fusion
        import vla.policy
        import vla.models

        assert len(VISION_REGISTRY.list_available()) > 0
        assert len(LANGUAGE_REGISTRY.list_available()) > 0
        assert len(FUSION_REGISTRY.list_available()) > 0
        assert len(ACTION_REGISTRY.list_available()) > 0
        assert len(MODEL_REGISTRY.list_available()) > 0

    def test_build_from_registry(self):
        """Test building components from registry."""
        from vla.registry import VISION_REGISTRY

        # Import to register
        import vla.backbones

        # Build from registry
        vision = VISION_REGISTRY.get(
            "timm_vit",
            model_name="vit_tiny_patch16_224",
            pretrained=False,
        )

        assert vision is not None
```

### Step 4: Create pytest.ini or pyproject.toml test config (10 min)
```toml
# Add to pyproject.toml

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
addopts = [
    "-v",
    "--cov=vla",
    "--cov-report=term-missing",
    "--cov-report=html:coverage_html",
    "--cov-fail-under=80",
]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "gpu: marks tests that require GPU",
]
filterwarnings = [
    "ignore::DeprecationWarning",
    "ignore::UserWarning",
]
```

### Step 5: Create test running script (15 min)
```bash
#!/bin/bash
# scripts/run_tests.sh

set -e

echo "Running VLA test suite..."

# Unit tests first (fast)
echo "=== Unit Tests ==="
pytest tests/unit -v --tb=short

# Integration tests
echo "=== Integration Tests ==="
pytest tests/integration -v --tb=short

# E2E tests (slower)
echo "=== End-to-End Tests ==="
pytest tests/e2e -v --tb=short

# Coverage report
echo "=== Coverage Report ==="
pytest tests/ --cov=vla --cov-report=term-missing

echo "All tests passed!"
```

## Todo List
- [ ] Enhance conftest.py with comprehensive fixtures
- [ ] Complete unit tests for all modules
- [ ] Implement integration tests
- [ ] Implement E2E tests
- [ ] Configure pytest in pyproject.toml
- [ ] Create test running script
- [ ] Achieve 80%+ coverage
- [ ] Add CI workflow (optional)

## Success Criteria
1. All unit tests pass
2. All integration tests pass
3. E2E smoke test completes
4. Coverage >= 80%
5. Tests run in <5 minutes on CPU

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Flaky tests | Medium | Use fixed seeds, mock external |
| Slow tests | Low | Mark slow tests, use fast_dev_run |
| GPU-only tests fail on CPU | Medium | Skip GPU tests when unavailable |

## Security Considerations
- No external network calls in tests
- No sensitive data in test fixtures

## Next Steps
After Phase 12, the framework is ready for:
1. Real dataset integration
2. Training experiments
3. Evaluation benchmarks
4. Documentation
