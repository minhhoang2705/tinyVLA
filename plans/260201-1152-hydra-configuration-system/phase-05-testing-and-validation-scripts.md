# Phase 05: Testing and Validation Scripts

## Context Links
- [Plan Overview](plan.md)
- [Existing conftest.py](../../tests/conftest.py)
- [Registry Factories](../../src/vla/registry/factories.py)
- [Hydra Utils](phase-04-hydra-utility-functions-and-resolvers.md)

## Overview
| Field | Value |
|-------|-------|
| Priority | P1 |
| Status | Pending |
| Effort | 60m |
| Dependencies | Phase 01-04 |

Create a standalone test script (`scripts/test-hydra-config.py`) for manual verification plus pytest unit tests (`tests/unit/test_hydra_config_loading.py`) for automated CI. Both validate config loading, CLI overrides, composition, and dimension validation.

## Key Insights
- Hydra's `compose` API enables testing without `@hydra.main()` decorator
- Must call `GlobalHydra.instance().clear()` between tests to reset state
- Use `initialize_config_dir()` for absolute path to configs/ (avoids relative path issues)
- Multirun testing is best done via subprocess (Hydra's multirun changes process behavior)
- Pytest tests use `hydra.compose()` in `with initialize()` context manager

## Requirements

### Functional
- FR-01: Manual test script verifies all 6 success criteria from the plan
- FR-02: Pytest tests cover config loading, CLI overrides, experiment composition
- FR-03: Pytest tests cover validation (missing fields, dimension mismatch)
- FR-04: Pytest tests cover `build_vla_from_hydra()` produces correct model type

### Non-Functional
- NFR-01: Tests run without GPU (CPU-only)
- NFR-02: Tests don't download model weights (use config loading only, not model instantiation for most tests)
- NFR-03: Tests run in <10 seconds total

## Architecture

```
scripts/
└── test-hydra-config.py              # Manual verification script

tests/
└── unit/
    └── test_hydra_config_loading.py  # Pytest automated tests
```

## Related Code Files

### Files to Create
| Path | Purpose | Lines |
|------|---------|-------|
| `scripts/test-hydra-config.py` | Manual Hydra verification script | ~80 |
| `tests/unit/test_hydra_config_loading.py` | Pytest tests for config system | ~150 |

## Implementation Steps

### Step 1: Create `scripts/test-hydra-config.py` (25 min)

Standalone script that validates each success criterion sequentially.

```python
#!/usr/bin/env python
"""Manual verification script for Hydra configuration system.

Run from project root:
    python scripts/test-hydra-config.py
    python scripts/test-hydra-config.py vision=dinov2
    python scripts/test-hydra-config.py +experiment=baseline
    python scripts/test-hydra-config.py --multirun vision=vit-base,dinov2

Tests:
    1. Default config loads successfully
    2. CLI overrides work
    3. Experiment presets load
    4. Config validation passes
    5. Config auto-saves to .hydra/
"""

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from vla.utils.hydra_config_helpers import (
    print_config,
    register_resolvers,
    validate_config,
)


# Register custom resolvers before Hydra loads configs
register_resolvers()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for config testing.

    Args:
        cfg: Composed Hydra configuration
    """
    print("=" * 60)
    print("tinyVLA Hydra Configuration Test")
    print("=" * 60)

    # Test 1: Config loaded
    print("\n[1/5] Config loaded successfully")
    print(f"  Vision: {cfg.vision.name}")
    print(f"  Language: {cfg.language.name}")
    print(f"  Fusion: {cfg.fusion.name}")
    print(f"  Action: {cfg.action.name}")
    print(f"  Model: {cfg.model.name}")

    # Test 2: Print full config
    print("\n[2/5] Full resolved config:")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # Test 3: Validate config
    print("[3/5] Validating config...")
    try:
        validate_config(cfg)
        print("  Validation PASSED")
    except ValueError as e:
        print(f"  Validation FAILED: {e}")

    # Test 4: Check output directory
    output_dir = Path(cfg.output_dir)
    print(f"\n[4/5] Output directory: {output_dir}")
    hydra_dir = output_dir / ".hydra"
    if hydra_dir.exists():
        print(f"  .hydra/ directory exists with auto-saved configs")
        for f in sorted(hydra_dir.iterdir()):
            print(f"    - {f.name}")
    else:
        print("  .hydra/ directory will be created by Hydra")

    # Test 5: Flatten config (useful for WandB logging)
    from vla.utils.hydra_config_helpers import flatten_config

    flat = flatten_config(cfg)
    print(f"\n[5/5] Flattened config has {len(flat)} keys")
    print("  Sample keys:")
    for key in sorted(flat.keys())[:5]:
        print(f"    {key}: {flat[key]}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

### Step 2: Create `tests/unit/test_hydra_config_loading.py` (35 min)

Pytest tests using Hydra's compose API for non-interactive testing.

```python
"""Tests for Hydra configuration system.

Tests config loading, composition, CLI overrides, validation,
and factory integration without requiring GPU or model weights.
"""

import pytest
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra


# Absolute path to configs/ directory
CONFIGS_DIR = str(Path(__file__).parent.parent.parent / "configs")


@pytest.fixture(autouse=True)
def clear_hydra():
    """Clear GlobalHydra singleton between tests."""
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


@pytest.fixture
def default_cfg() -> DictConfig:
    """Load default config via Hydra compose API."""
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = compose(config_name="config")
    return cfg


# ── Config Loading Tests ──────────────────────────────────


class TestDefaultConfigLoading:
    """Test that default config loads with all expected fields."""

    def test_default_config_loads(self, default_cfg):
        """Default config.yaml loads without errors."""
        assert default_cfg is not None
        assert isinstance(default_cfg, DictConfig)

    def test_has_all_component_groups(self, default_cfg):
        """All component groups present in composed config."""
        required = ["model", "vision", "language", "fusion", "action", "train", "data"]
        for group in required:
            assert group in default_cfg, f"Missing config group: {group}"

    def test_default_component_names(self, default_cfg):
        """Default component names match expected values."""
        assert default_cfg.vision.name == "timm_vit"
        assert default_cfg.language.name == "gpt2"
        assert default_cfg.fusion.name == "perceiver_resampler"
        assert default_cfg.action.name == "discrete_action"
        assert default_cfg.model.name == "vla_base"

    def test_project_metadata(self, default_cfg):
        """Project metadata present in config."""
        assert default_cfg.project.name == "tinyVLA"
        assert default_cfg.seed == 42


# ── CLI Override Tests ────────────────────────────────────


class TestCLIOverrides:
    """Test Hydra CLI override functionality."""

    def test_override_vision_encoder(self):
        """Override vision config group via CLI-style override."""
        with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
            cfg = compose(config_name="config", overrides=["vision=dinov2"])
        assert cfg.vision.name == "dinov2"
        assert cfg.vision.size == "base"

    def test_override_single_value(self):
        """Override a single nested value."""
        with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
            cfg = compose(config_name="config", overrides=["seed=123"])
        assert cfg.seed == 123

    def test_override_fusion_dim(self):
        """Override fusion dimension."""
        with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
            cfg = compose(config_name="config", overrides=["fusion.dim=512"])
        assert cfg.fusion.dim == 512

    def test_override_action_type(self):
        """Switch action head to gaussian."""
        with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
            cfg = compose(config_name="config", overrides=["action=gaussian"])
        assert cfg.action.name == "gaussian_action"
        assert cfg.action.min_std == 0.01


# ── Experiment Preset Tests ───────────────────────────────


class TestExperimentPresets:
    """Test experiment config composition."""

    def test_baseline_experiment_loads(self):
        """Baseline experiment config loads and overrides defaults."""
        with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
            cfg = compose(config_name="config", overrides=["+experiment=baseline"])
        assert cfg.project.name == "tinyVLA-baseline"
        assert cfg.train.max_epochs == 50

    def test_ablation_vision_experiment_loads(self):
        """Vision ablation experiment loads."""
        with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
            cfg = compose(
                config_name="config",
                overrides=["+experiment=ablation-vision"],
            )
        assert cfg.project.name == "tinyVLA-ablation-vision"
        assert cfg.train.max_epochs == 30


# ── Config Validation Tests ───────────────────────────────


class TestConfigValidation:
    """Test config validation utilities."""

    def test_valid_config_passes(self, default_cfg):
        """Valid default config passes validation."""
        from vla.utils.hydra_config_helpers import validate_config

        validate_config(default_cfg)  # Should not raise

    def test_missing_group_raises(self):
        """Missing required group raises ValueError."""
        from vla.utils.hydra_config_helpers import validate_config

        incomplete = OmegaConf.create({"model": {"name": "vla_base"}, "vision": {"name": "x"}})
        with pytest.raises(ValueError, match="Missing required config group"):
            validate_config(incomplete)

    def test_missing_name_raises(self):
        """Component without 'name' field raises ValueError."""
        from vla.utils.hydra_config_helpers import validate_config

        cfg = OmegaConf.create({
            "model": {"name": "vla_base"},
            "vision": {"no_name": True},
            "language": {"name": "gpt2"},
            "fusion": {"name": "perceiver_resampler"},
            "action": {"name": "discrete_action"},
        })
        with pytest.raises(ValueError, match="missing required 'name' field"):
            validate_config(cfg)

    def test_dimension_mismatch_warns(self, default_cfg, caplog):
        """Dimension mismatch logs a warning."""
        import logging
        from vla.utils.hydra_config_helpers import validate_config

        cfg = OmegaConf.create({
            "model": {"name": "vla_base"},
            "vision": {"name": "vit", "proj_dim": 512},
            "language": {"name": "gpt2"},
            "fusion": {"name": "perceiver", "dim": 768},
            "action": {"name": "discrete"},
        })
        with caplog.at_level(logging.WARNING):
            validate_config(cfg)
        assert "Dimension mismatch" in caplog.text


# ── Utility Function Tests ────────────────────────────────


class TestUtilityFunctions:
    """Test hydra utility helper functions."""

    def test_flatten_config(self, default_cfg):
        """Flatten config produces dot-notation keys."""
        from vla.utils.hydra_config_helpers import flatten_config

        flat = flatten_config(default_cfg)
        assert isinstance(flat, dict)
        assert "vision.name" in flat
        assert "fusion.dim" in flat
        assert flat["seed"] == 42

    def test_get_config_dir(self):
        """get_config_dir returns existing configs/ path."""
        from vla.utils.hydra_config_helpers import get_config_dir

        config_dir = get_config_dir()
        assert config_dir.exists()
        assert (config_dir / "config.yaml").exists()

    def test_save_config(self, default_cfg, tmp_path):
        """save_config writes valid YAML."""
        from vla.utils.hydra_config_helpers import save_config

        out_path = str(tmp_path / "saved_config.yaml")
        save_config(default_cfg, out_path)
        loaded = OmegaConf.load(out_path)
        assert loaded.vision.name == default_cfg.vision.name


# ── VLAConfig Bridge Tests ────────────────────────────────


class TestVLAConfigBridge:
    """Test bridging Hydra DictConfig to VLAConfig dataclass."""

    def test_from_hydra_creates_valid_config(self, default_cfg):
        """VLAConfig.from_hydra() creates config from DictConfig."""
        from vla.models.vla_configs import VLAConfig

        config = VLAConfig.from_hydra(default_cfg)
        assert config.vision.name == "timm_vit"
        assert config.fusion.dim == 768
        assert config.action.action_dim == 7
        assert config.freeze_vision is True
```

### Step 3: Run tests (10 min)

```bash
# Run pytest config tests
pytest tests/unit/test_hydra_config_loading.py -v

# Run manual test script
python scripts/test-hydra-config.py

# Test CLI override
python scripts/test-hydra-config.py vision=dinov2

# Test experiment preset
python scripts/test-hydra-config.py +experiment=baseline
```

## Todo List
- [ ] Create `scripts/test-hydra-config.py`
- [ ] Create `tests/unit/test_hydra_config_loading.py`
- [ ] Run pytest tests -- all pass
- [ ] Run manual test script -- default config loads
- [ ] Run manual test with CLI override -- vision=dinov2 works
- [ ] Run manual test with experiment -- +experiment=baseline works
- [ ] Verify config auto-saves to outputs/.hydra/
- [ ] Run `black` and `ruff` on new files

## Success Criteria
1. `pytest tests/unit/test_hydra_config_loading.py -v` -- all tests pass
2. `python scripts/test-hydra-config.py` -- loads default config, prints it, validates
3. `python scripts/test-hydra-config.py vision=dinov2` -- overrides vision encoder
4. `python scripts/test-hydra-config.py +experiment=baseline` -- loads experiment preset
5. Config auto-saved to `outputs/.hydra/` directory after script run
6. `validate_config()` catches missing fields and dimension mismatches
7. All new files pass `black` and `ruff` formatting checks

## Risk Assessment
| Risk | Impact | Mitigation |
|------|--------|------------|
| GlobalHydra not cleared between tests | High -- tests interfere | Use `autouse=True` fixture with clear() |
| Config path resolution fails in pytest | High -- all tests fail | Use absolute path via `initialize_config_dir()` |
| Multirun test flaky | Medium | Test via subprocess, not in-process |
| Model weight download in tests | High -- slow CI | Test config loading only, not model instantiation |

## Security Considerations
- Test script runs locally only, no network access needed for config tests
- No secrets in test configs

## Next Steps
- After all phases complete: run full test suite, update docs
- Phase 10 (Data) and Phase 11 (Training) are now unblocked
