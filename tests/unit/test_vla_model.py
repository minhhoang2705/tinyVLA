"""Tests for VLA model orchestration.

Tests the complete VLA model assembly, forward pass, loss computation,
checkpoint save/load, and component freezing.
"""

import pytest
import torch

from vla.models import (
    ActionConfig,
    AffordanceConfig,
    FusionConfig,
    LanguageConfig,
    TemporalVLAModel,
    VisionConfig,
    VLAConfig,
    VLAModel,
)


class TestVLAConfig:
    """Test VLAConfig dataclass and construction."""

    def test_default_config(self):
        """Test VLAConfig initializes with sensible defaults."""
        config = VLAConfig()
        assert config.vision.name == "timm_vit"
        assert config.language.name == "gpt2"
        assert config.fusion.name == "perceiver_resampler"
        assert config.action.name == "discrete_action"
        assert config.freeze_vision is True
        assert config.freeze_language is True

    def test_config_from_dict(self):
        """Test creating VLAConfig from nested dictionary."""
        config_dict = {
            "vision": {"model_name": "vit_tiny_patch16_224", "pretrained": False},
            "language": {"model_name": "gpt2"},
            "fusion": {"num_latents": 32, "dim": 512},
            "action": {"action_dim": 6, "num_bins": 128},
            "freeze_vision": False,
            "action_loss_weight": 2.0,
        }
        config = VLAConfig.from_dict(config_dict)

        assert config.vision.model_name == "vit_tiny_patch16_224"
        assert config.vision.pretrained is False
        assert config.fusion.num_latents == 32
        assert config.fusion.dim == 512
        assert config.action.action_dim == 6
        assert config.action.num_bins == 128
        assert config.freeze_vision is False
        assert config.action_loss_weight == 2.0

    def test_config_partial_dict(self):
        """Test from_dict with partial config (uses defaults)."""
        config_dict = {
            "vision": {"model_name": "vit_tiny_patch16_224"},
            "action": {"action_dim": 10},
        }
        config = VLAConfig.from_dict(config_dict)

        # Specified values
        assert config.vision.model_name == "vit_tiny_patch16_224"
        assert config.action.action_dim == 10

        # Defaults
        assert config.vision.pretrained is True  # default
        assert config.language.model_name == "gpt2"  # default
        assert config.fusion.num_latents == 64  # default


class TestVLAModel:
    """Test VLAModel forward pass, loss, and checkpointing."""

    @pytest.fixture
    def small_config(self):
        """Config for fast tests with small models."""
        return VLAConfig(
            vision=VisionConfig(
                name="timm_vit",
                model_name="vit_tiny_patch16_224",
                pretrained=False,
                frozen=False,  # Unfreeze for gradient tests
            ),
            language=LanguageConfig(
                name="gpt2",
                model_name="gpt2",
                frozen=False,  # Unfreeze for gradient tests
            ),
            fusion=FusionConfig(
                num_latents=16,  # Small for speed
                dim=192,  # Match vit_tiny
            ),
            action=ActionConfig(
                action_dim=7,
                num_bins=256,
            ),
        )

    @pytest.fixture
    def dummy_batch(self):
        """Create dummy input batch."""
        return {
            "images": torch.rand(2, 3, 224, 224),  # Valid images [0, 1]
            "texts": ["pick up the red block", "place it on the table"],
            "target_actions": torch.rand(2, 7) * 2 - 1,  # Valid actions [-1, 1]
        }

    def test_model_instantiation(self, small_config):
        """Test VLAModel can be instantiated from config."""
        model = VLAModel(small_config)

        assert hasattr(model, "vision")
        assert hasattr(model, "language")
        assert hasattr(model, "fusion")
        assert hasattr(model, "action_head")
        assert hasattr(model, "config")

    def test_forward_shape(self, small_config, dummy_batch):
        """Test forward pass produces correct action shape."""
        model = VLAModel(small_config)
        output = model(
            images=dummy_batch["images"],
            texts=dummy_batch["texts"],
        )

        assert "actions" in output
        assert output["actions"].shape == (2, 7)
        assert output["actions"].dtype == torch.float32

    def test_forward_with_loss(self, small_config, dummy_batch):
        """Test forward pass computes loss when targets provided."""
        model = VLAModel(small_config)
        output = model(
            images=dummy_batch["images"],
            texts=dummy_batch["texts"],
            target_actions=dummy_batch["target_actions"],
        )

        assert "loss" in output
        assert "actions" in output
        assert output["loss"].ndim == 0  # scalar
        assert output["loss"].dtype == torch.float32
        assert output["loss"].item() >= 0  # loss should be non-negative

    def test_predict_inference_mode(self, small_config, dummy_batch):
        """Test predict() convenience method."""
        model = VLAModel(small_config)
        actions = model.predict(
            images=dummy_batch["images"],
            texts=dummy_batch["texts"],
        )

        assert actions.shape == (2, 7)
        assert actions.dtype == torch.float32
        # Check actions are in valid range [-1, 1]
        assert actions.min() >= -1.0
        assert actions.max() <= 1.0

    def test_freezing_vision(self, small_config):
        """Test vision backbone is frozen when freeze_vision=True."""
        small_config.freeze_vision = True
        small_config.freeze_language = False
        model = VLAModel(small_config)

        # Vision should be frozen
        for name, param in model.vision.named_parameters():
            assert not param.requires_grad, f"Vision param {name} should be frozen"

        # Fusion and action should be trainable
        assert any(p.requires_grad for p in model.fusion.parameters())
        assert any(p.requires_grad for p in model.action_head.parameters())

    def test_freezing_language(self, small_config):
        """Test language backbone is frozen when freeze_language=True."""
        small_config.freeze_vision = False
        small_config.freeze_language = True
        model = VLAModel(small_config)

        # Language should be frozen
        for name, param in model.language.named_parameters():
            assert not param.requires_grad, f"Language param {name} should be frozen"

        # Fusion and action should be trainable
        assert any(p.requires_grad for p in model.fusion.parameters())
        assert any(p.requires_grad for p in model.action_head.parameters())

    def test_get_trainable_params(self, small_config):
        """Test get_trainable_params returns only unfrozen parameters."""
        small_config.freeze_vision = True
        small_config.freeze_language = True
        model = VLAModel(small_config)

        trainable = model.get_trainable_params()

        # Should have trainable params from fusion + action
        assert len(trainable) > 0

        # All returned params should have requires_grad=True
        for param in trainable:
            assert param.requires_grad

        # No vision/language params should be in trainable list
        vision_params = set(model.vision.parameters())
        language_params = set(model.language.parameters())
        trainable_set = set(trainable)

        assert trainable_set.isdisjoint(vision_params)
        assert trainable_set.isdisjoint(language_params)

    def test_checkpoint_roundtrip(self, small_config, dummy_batch, tmp_path, seed):
        """Test saving and loading checkpoint preserves model state."""
        model = VLAModel(small_config)
        model.eval()  # Use deterministic mode for checkpoint comparison

        # Get initial prediction
        with torch.no_grad():
            output1 = model(dummy_batch["images"], texts=dummy_batch["texts"])

        # Save checkpoint
        ckpt_path = tmp_path / "model.pt"
        model.save_checkpoint(str(ckpt_path))

        # Load checkpoint - use no_grad context for inference
        loaded_model = VLAModel.load_checkpoint(str(ckpt_path))
        loaded_model.eval()

        with torch.no_grad():
            output2 = loaded_model(dummy_batch["images"], texts=dummy_batch["texts"])

        # Verify that state dict was correctly loaded (check a few params)
        for (name1, param1), (name2, param2) in zip(
            model.named_parameters(), loaded_model.named_parameters()
        ):
            assert name1 == name2, f"Parameter name mismatch: {name1} vs {name2}"
            assert torch.allclose(param1, param2, atol=1e-6), (
                f"Parameter values differ for {name1}"
            )

    def test_checkpoint_preserves_config(self, small_config, tmp_path):
        """Test checkpoint preserves configuration."""
        model = VLAModel(small_config)

        # Save and load
        ckpt_path = tmp_path / "model.pt"
        model.save_checkpoint(str(ckpt_path))
        loaded_model = VLAModel.load_checkpoint(str(ckpt_path))

        # Compare configs
        assert loaded_model.config.vision.model_name == small_config.vision.model_name
        assert loaded_model.config.fusion.num_latents == small_config.fusion.num_latents
        assert loaded_model.config.action.action_dim == small_config.action.action_dim

    def test_backward_pass(self, small_config, dummy_batch):
        """Test backward pass runs without errors."""
        model = VLAModel(small_config)
        output = model(
            images=dummy_batch["images"],
            texts=dummy_batch["texts"],
            target_actions=dummy_batch["target_actions"],
        )

        loss = output["loss"]
        loss.backward()

        # Check gradients exist for trainable params
        trainable = model.get_trainable_params()
        for param in trainable:
            assert param.grad is not None

    def test_dict_config_instantiation(self, dummy_batch):
        """Test VLAModel can be instantiated from dict config."""
        config_dict = {
            "vision": {"model_name": "vit_tiny_patch16_224", "pretrained": False},
            "action": {"action_dim": 7},
        }
        model = VLAModel(config_dict)

        output = model(dummy_batch["images"], texts=dummy_batch["texts"])
        assert output["actions"].shape == (2, 7)


class TestTemporalVLAModel:
    """Test TemporalVLAModel for multi-frame inputs."""

    @pytest.fixture
    def temporal_config(self):
        """Config for temporal model."""
        return VLAConfig(
            vision=VisionConfig(
                model_name="vit_tiny_patch16_224",
                pretrained=False,
            ),
            fusion=FusionConfig(num_latents=16, dim=192),
        )

    @pytest.fixture
    def temporal_batch(self):
        """Create temporal input batch with 4 frames."""
        return {
            "image_sequence": [torch.rand(2, 3, 224, 224) for _ in range(4)],  # Valid frames [0, 1]
            "texts": ["track the moving ball", "intercept it"],
            "target_actions": torch.rand(2, 7) * 2 - 1,  # Valid actions [-1, 1]
        }

    def test_temporal_forward(self, temporal_config, temporal_batch):
        """Test temporal model forward pass."""
        model = TemporalVLAModel(temporal_config, num_frames=4)

        output = model(
            image_sequence=temporal_batch["image_sequence"],
            texts=temporal_batch["texts"],
        )

        assert "actions" in output
        assert output["actions"].shape == (2, 7)

    def test_temporal_with_loss(self, temporal_config, temporal_batch):
        """Test temporal model computes loss."""
        model = TemporalVLAModel(temporal_config, num_frames=4)

        output = model(
            image_sequence=temporal_batch["image_sequence"],
            texts=temporal_batch["texts"],
            target_actions=temporal_batch["target_actions"],
        )

        assert "loss" in output
        assert output["loss"].ndim == 0

    def test_temporal_predict(self, temporal_config, temporal_batch):
        """Test temporal predict() method."""
        model = TemporalVLAModel(temporal_config, num_frames=4)

        actions = model.predict(
            image_sequence=temporal_batch["image_sequence"],
            texts=temporal_batch["texts"],
        )

        assert actions.shape == (2, 7)
        assert actions.min() >= -1.0
        assert actions.max() <= 1.0


class TestRegistryIntegration:
    """Test VLA models are properly registered."""

    def test_vla_base_registered(self):
        """Test vla_base is in MODEL_REGISTRY."""
        from vla.registry import MODEL_REGISTRY

        assert "vla_base" in MODEL_REGISTRY

    def test_vla_temporal_registered(self):
        """Test vla_temporal is in MODEL_REGISTRY."""
        from vla.registry import MODEL_REGISTRY

        assert "vla_temporal" in MODEL_REGISTRY

    def test_build_from_registry(self):
        """Test building VLA model from registry."""
        from vla.registry import MODEL_REGISTRY

        config = VLAConfig(
            vision=VisionConfig(model_name="vit_tiny_patch16_224", pretrained=False),
        )

        # Build via registry
        model = MODEL_REGISTRY.get("vla_base", config=config)

        assert isinstance(model, VLAModel)
        assert hasattr(model, "vision")
        assert hasattr(model, "action_head")


class TestVLAModelAffordance:
    """Tests for VLAModel auxiliary affordance head integration."""

    def _make_affordance_config(self):
        from vla.models import AffordanceConfig
        return VLAConfig(
            vision=VisionConfig(
                name="timm_vit",
                model_name="vit_tiny_patch16_224",
                pretrained=False,
                frozen=False,
            ),
            language=LanguageConfig(model_name="gpt2", frozen=False),
            fusion=FusionConfig(num_latents=16, dim=192),
            action=ActionConfig(action_dim=2),
            affordance=AffordanceConfig(enabled=True, state_dim=3, hidden_dim=32),
            auxiliary_loss_weight=0.1,
        )

    def test_affordance_head_built_when_enabled(self):
        """Model has affordance_head when config.affordance.enabled=True."""
        model = VLAModel(self._make_affordance_config())
        assert model.affordance_head is not None

    def test_affordance_head_none_when_disabled(self):
        """Model has no affordance_head when config.affordance.enabled=False (default)."""
        config = VLAConfig(
            vision=VisionConfig(model_name="vit_tiny_patch16_224", pretrained=False),
        )
        model = VLAModel(config)
        assert model.affordance_head is None

    def test_aux_loss_in_output_when_target_state_provided(self, dummy_image, dummy_text):
        """Forward returns aux_loss and action_loss when target_state given."""
        model = VLAModel(self._make_affordance_config())
        target_state = torch.randn(2, 3)
        target_actions = torch.rand(2, 2) * 2 - 1  # valid [-1, 1]
        output = model(
            dummy_image,
            texts=dummy_text,
            target_actions=target_actions,
            target_state=target_state,
        )
        assert "aux_loss" in output
        assert "action_loss" in output
        assert output["aux_loss"].shape == ()

    def test_no_aux_loss_without_target_state(self, dummy_image, dummy_text):
        """Forward has no aux_loss when target_state=None."""
        model = VLAModel(self._make_affordance_config())
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
