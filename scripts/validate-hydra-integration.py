#!/usr/bin/env python3
"""Validation script for Phase 03: Factory Integration with Hydra DictConfig.

Tests the new from_hydra() and build_vla_from_hydra() functions.
"""

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vla.models.vla_configs import VLAConfig
from vla.registry import build_vla_from_hydra


def test_vla_config_from_hydra():
    """Test VLAConfig.from_hydra() with default config."""
    print("\n" + "=" * 60)
    print("TEST 1: VLAConfig.from_hydra() with default config")
    print("=" * 60)

    # Initialize Hydra with absolute config path
    config_dir = os.path.abspath("configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        # Compose default config
        cfg = compose(config_name="config")

        # Extract component configs (skip runtime interpolations like ${hydra:...})
        component_cfg = DictConfig({
            "vision": cfg.vision,
            "language": cfg.language,
            "fusion": cfg.fusion,
            "action": cfg.action,
            "freeze_vision": cfg.model.get("freeze_vision", True),
            "freeze_language": cfg.model.get("freeze_language", True),
            "action_loss_weight": cfg.model.get("action_loss_weight", 1.0),
            "auxiliary_loss_weight": cfg.model.get("auxiliary_loss_weight", 0.0),
        })

        # Create VLAConfig from component config
        vla_config = VLAConfig.from_hydra(component_cfg)

        print(f"✓ Created VLAConfig from Hydra DictConfig")
        print(f"  Vision model: {vla_config.vision.name} / {vla_config.vision.model_name}")
        print(f"  Language model: {vla_config.language.name} / {vla_config.language.model_name}")
        print(f"  Fusion: {vla_config.fusion.name} (latents={vla_config.fusion.num_latents})")
        print(f"  Action: {vla_config.action.name} (dim={vla_config.action.action_dim})")
        print(f"  Freeze vision: {vla_config.freeze_vision}")
        print(f"  Freeze language: {vla_config.freeze_language}")

        # Verify types
        assert isinstance(vla_config, VLAConfig)
        assert vla_config.vision.name == "timm_vit"
        assert vla_config.fusion.num_latents == 64

        print("✓ All assertions passed!")


def test_vla_config_with_overrides():
    """Test VLAConfig.from_hydra() with CLI overrides."""
    print("\n" + "=" * 60)
    print("TEST 2: VLAConfig.from_hydra() with overrides")
    print("=" * 60)

    config_dir = os.path.abspath("configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        # Compose with overrides
        cfg = compose(config_name="config", overrides=["vision=dinov2", "fusion=cross-attention"])

        # Extract component configs
        component_cfg = DictConfig({
            "vision": cfg.vision,
            "language": cfg.language,
            "fusion": cfg.fusion,
            "action": cfg.action,
            "freeze_vision": cfg.model.get("freeze_vision", True),
            "freeze_language": cfg.model.get("freeze_language", True),
        })

        vla_config = VLAConfig.from_hydra(component_cfg)

        print(f"✓ Created VLAConfig with overrides")
        print(f"  Vision: {vla_config.vision.name}")
        print(f"  Fusion: {vla_config.fusion.name}")

        # After override vision=dinov2, fusion=cross-attention
        assert vla_config.vision.name == "dinov2"
        assert vla_config.fusion.name == "cross_attention_fusion"

        print("✓ Overrides applied correctly!")


def test_build_vla_from_hydra():
    """Test build_vla_from_hydra() builds complete model."""
    print("\n" + "=" * 60)
    print("TEST 3: build_vla_from_hydra() - Build VLAModel")
    print("=" * 60)

    config_dir = os.path.abspath("configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config")

        # Build complete model
        model = build_vla_from_hydra(cfg)

        print(f"✓ Built VLA model from Hydra config")
        print(f"  Model type: {type(model).__name__}")
        print(f"  Vision: {type(model.vision).__name__}")
        print(f"  Language: {type(model.language).__name__}")
        print(f"  Fusion: {type(model.fusion).__name__}")
        print(f"  Action head: {type(model.action_head).__name__}")

        # Check model is callable
        import torch
        images = torch.rand(2, 3, 224, 224)  # Use rand() not randn() - must be in [0, 1]
        texts = ["pick up cube", "move to target"]

        output = model(images, texts)
        print(f"✓ Model forward pass successful")
        print(f"  Output keys: {list(output.keys())}")
        print(f"  Actions shape: {output['actions'].shape}")

        # Forward returns decoded actions [B, action_dim], not logits [B, action_dim * num_bins]
        expected_shape = (2, 7)
        assert output["actions"].shape == expected_shape, f"Expected {expected_shape}, got {output['actions'].shape}"

        print("✓ Output shape correct!")


def test_build_temporal_vla_from_hydra():
    """Test build_vla_from_hydra() with temporal model."""
    print("\n" + "=" * 60)
    print("TEST 4: build_vla_from_hydra() - Build TemporalVLAModel")
    print("=" * 60)

    config_dir = os.path.abspath("configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        # Use temporal model config
        cfg = compose(config_name="config", overrides=["model=vla-temporal"])

        model = build_vla_from_hydra(cfg)

        print(f"✓ Built Temporal VLA model from Hydra config")
        print(f"  Model type: {type(model).__name__}")
        print(f"  Num frames: {model.num_frames}")

        # Check temporal forward pass
        import torch
        # Temporal model expects List[Tensor], not a single 5D tensor
        images = [torch.rand(2, 3, 224, 224) for _ in range(6)]  # List of 6 [B, C, H, W] frames
        texts = ["pick up cube", "move to target"]

        output = model(images, texts)
        print(f"✓ Temporal model forward pass successful")
        print(f"  Actions shape: {output['actions'].shape}")

        # Temporal model also returns decoded actions [B, action_dim]
        assert output["actions"].shape == (2, 7)
        print("✓ Output shape correct!")


def test_experiment_preset():
    """Test building model from experiment preset."""
    print("\n" + "=" * 60)
    print("TEST 5: build_vla_from_hydra() - Experiment preset")
    print("=" * 60)

    config_dir = os.path.abspath("configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        # Load baseline experiment
        cfg = compose(config_name="config", overrides=["+experiment=baseline"])

        model = build_vla_from_hydra(cfg)

        print(f"✓ Built model from 'baseline' experiment preset")
        print(f"  Project name: {cfg.project.name}")
        print(f"  Max epochs: {cfg.train.max_epochs}")
        print(f"  Batch size: {cfg.train.batch_size}")

        assert cfg.project.name == "tinyVLA-baseline"
        assert cfg.train.max_epochs == 50

        print("✓ Experiment preset loaded correctly!")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PHASE 03 VALIDATION: Hydra Integration Tests")
    print("=" * 60)

    try:
        test_vla_config_from_hydra()
        test_vla_config_with_overrides()
        test_build_vla_from_hydra()
        test_build_temporal_vla_from_hydra()
        test_experiment_preset()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nPhase 03 factory integration is complete and validated!")

    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ TEST FAILED!")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
