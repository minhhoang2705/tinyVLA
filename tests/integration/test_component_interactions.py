"""Integration tests for VLA component interactions.

Tests verify that vision, language, fusion, and action components
work together correctly with proper data flow and shape compatibility.
"""

import pytest
import torch

from vla.backbones.language import GPT2Backbone
from vla.backbones.vision import DINOv2Backbone, SigLIPBackbone
from vla.fusion.perceiver import PerceiverResampler
from vla.models import VLAConfig, VLAModel
from vla.policy.action_heads import DiscreteActionHead, GaussianActionHead


class TestVisionLanguageFusion:
    """Test vision and language encoding with fusion."""

    @pytest.fixture
    def vision_backbone(self):
        """Create vision backbone."""
        return DINOv2Backbone(
            model_name="vit_tiny_patch16_224",
            pretrained=False,
        )

    @pytest.fixture
    def language_backbone(self):
        """Create language backbone."""
        return GPT2Backbone(
            model_name="gpt2",
            max_length=77,
        )

    @pytest.fixture
    def fusion_module(self, vision_backbone, language_backbone):
        """Create fusion module."""
        return PerceiverResampler(
            dim=768,
            num_latents=64,
            num_layers=2,
            num_heads=8,
            vision_dim=vision_backbone.embed_dim,
            language_dim=language_backbone.embed_dim,
        )

    def test_vision_language_shape_compatibility(
        self,
        vision_backbone,
        language_backbone,
        fusion_module,
    ):
        """Test vision and language features are compatible for fusion."""
        images = torch.rand(4, 3, 224, 224)
        texts = ["pick cube", "place cup", "move left", "grasp object"]

        # Encode modalities
        vision_features = vision_backbone(images)
        language_features = language_backbone(texts=texts)

        # Verify shapes
        assert vision_features.shape[0] == 4  # Batch size
        assert vision_features.ndim == 3  # [B, N, D]
        assert language_features.shape[0] == 4
        assert language_features.ndim == 3  # [B, L, D]

        # Fusion should accept both
        fused = fusion_module(vision_features, language_features)
        assert fused.shape == (4, 64, 768)

    def test_different_batch_sizes(
        self,
        vision_backbone,
        language_backbone,
        fusion_module,
    ):
        """Test components handle different batch sizes."""
        for batch_size in [1, 2, 8, 16]:
            images = torch.rand(batch_size, 3, 224, 224)
            texts = [f"instruction {i}" for i in range(batch_size)]

            vision_features = vision_backbone(images)
            language_features = language_backbone(texts=texts)
            fused = fusion_module(vision_features, language_features)

            assert fused.shape[0] == batch_size

    def test_frozen_backbones_no_gradients(
        self,
        vision_backbone,
        language_backbone,
    ):
        """Test frozen backbones don't compute gradients."""
        # Freeze backbones
        for param in vision_backbone.parameters():
            param.requires_grad = False
        for param in language_backbone.parameters():
            param.requires_grad = False

        images = torch.rand(2, 3, 224, 224)
        texts = ["pick", "place"]

        vision_features = vision_backbone(images)
        language_features = language_backbone(texts=texts)

        # Compute some loss and backprop
        loss = vision_features.mean() + language_features.mean()
        loss.backward()

        # Verify no gradients for frozen params
        for param in vision_backbone.parameters():
            assert param.grad is None

        for param in language_backbone.parameters():
            assert param.grad is None


class TestFusionActionHead:
    """Test fusion module with action heads."""

    @pytest.fixture
    def fusion_output(self):
        """Create dummy fusion output."""
        return torch.randn(4, 64, 768)

    def test_discrete_action_head_with_fusion(self, fusion_output):
        """Test discrete action head accepts fusion output."""
        action_head = DiscreteActionHead(
            input_dim=768,
            action_dim=7,
            num_bins=256,
        )

        actions, logits = action_head(fusion_output, return_logits=True)

        assert actions.shape == (4, 7)
        assert logits.shape == (4, 7, 256)
        assert actions.min() >= -1.0
        assert actions.max() <= 1.0

    def test_gaussian_action_head_with_fusion(self, fusion_output):
        """Test Gaussian action head accepts fusion output."""
        action_head = GaussianActionHead(
            input_dim=768,
            action_dim=7,
        )

        actions, logits = action_head(fusion_output, return_logits=True)

        assert actions.shape == (4, 7)
        assert logits is not None
        assert isinstance(logits, tuple)
        mean, std = logits
        assert mean.shape == (4, 7)
        assert std.shape == (4, 7)

    def test_loss_computation_discrete(self, fusion_output):
        """Test discrete action head loss computation."""
        action_head = DiscreteActionHead(
            input_dim=768,
            action_dim=7,
            num_bins=256,
        )

        actions, logits = action_head(fusion_output, return_logits=True)
        target_actions = torch.rand(4, 7) * 2 - 1

        loss = action_head.compute_loss(logits, target_actions)

        assert loss.ndim == 0  # Scalar
        assert loss.item() >= 0
        assert torch.isfinite(loss)

    def test_loss_computation_gaussian(self, fusion_output):
        """Test Gaussian action head loss computation."""
        action_head = GaussianActionHead(
            input_dim=768,
            action_dim=7,
        )

        actions, logits = action_head(fusion_output, return_logits=True)
        mean, std = logits
        target_actions = torch.rand(4, 7) * 2 - 1

        loss = action_head.compute_loss(mean, std, target_actions)

        assert loss.ndim == 0
        assert torch.isfinite(loss)


class TestFullPipelineDataFlow:
    """Test complete data flow through all components."""

    @pytest.fixture
    def model(self):
        """Create complete VLA model."""
        config = VLAConfig()
        return VLAModel(config)

    def test_forward_pass_data_flow(self, model):
        """Test data flows correctly through all components."""
        batch_size = 4
        images = torch.rand(batch_size, 3, 224, 224)
        texts = [f"instruction {i}" for i in range(batch_size)]
        target_actions = torch.rand(batch_size, 7) * 2 - 1

        # Track intermediate shapes
        with torch.no_grad():
            # Vision encoding
            vision_features = model.vision(images)
            assert vision_features.shape[0] == batch_size
            assert vision_features.ndim == 3

            # Language encoding
            language_features = model.language(texts=texts)
            assert language_features.shape[0] == batch_size
            assert language_features.ndim in [2, 3]  # Pooled or sequence

            # Fusion
            fused = model.fusion(vision_features, language_features)
            assert fused.shape[0] == batch_size
            assert fused.shape[1] == model.config.fusion.num_latents

            # Action prediction
            actions, logits = model.action_head(fused, return_logits=True)
            assert actions.shape == (batch_size, 7)
            assert logits is not None

    def test_gradient_flow_through_pipeline(self, model):
        """Test gradients flow through trainable components."""
        images = torch.rand(2, 3, 224, 224)
        texts = ["pick", "place"]
        target_actions = torch.rand(2, 7) * 2 - 1

        model.train()
        output = model(images, texts=texts, target_actions=target_actions)

        # Backprop
        output["loss"].backward()

        # Verify gradients
        # Fusion should have gradients
        fusion_has_grad = False
        for param in model.fusion.parameters():
            if param.requires_grad and param.grad is not None:
                fusion_has_grad = True
                break
        assert fusion_has_grad, "Fusion module has no gradients"

        # Action head should have gradients
        action_has_grad = False
        for param in model.action_head.parameters():
            if param.requires_grad and param.grad is not None:
                action_has_grad = True
                break
        assert action_has_grad, "Action head has no gradients"

    def test_different_sequence_lengths(self, model):
        """Test model handles different text lengths."""
        images = torch.rand(3, 3, 224, 224)

        # Different length texts
        texts = [
            "pick",
            "pick up the cube",
            "move the object to the target location carefully",
        ]

        model.eval()
        with torch.no_grad():
            actions = model.predict(images, texts)

        assert actions.shape == (3, 7)
        assert actions.min() >= -1.0
        assert actions.max() <= 1.0


class TestBatchSizeConsistency:
    """Test batch size consistency across components."""

    @pytest.fixture
    def model(self):
        """Create VLA model."""
        config = VLAConfig()
        return VLAModel(config)

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16])
    def test_various_batch_sizes(self, model, batch_size):
        """Test model handles various batch sizes."""
        images = torch.rand(batch_size, 3, 224, 224)
        texts = [f"instruction {i}" for i in range(batch_size)]
        target_actions = torch.rand(batch_size, 7) * 2 - 1

        output = model(images, texts=texts, target_actions=target_actions)

        assert output["actions"].shape == (batch_size, 7)
        assert "loss" in output
        assert torch.isfinite(output["loss"])

    def test_single_sample_batch(self, model):
        """Test model handles single sample (batch_size=1)."""
        images = torch.rand(1, 3, 224, 224)
        texts = ["pick cube"]
        target_actions = torch.rand(1, 7) * 2 - 1

        output = model(images, texts=texts, target_actions=target_actions)

        assert output["actions"].shape == (1, 7)
        assert torch.isfinite(output["loss"])


class TestErrorPropagation:
    """Test error handling across component boundaries."""

    @pytest.fixture
    def model(self):
        """Create VLA model."""
        config = VLAConfig()
        return VLAModel(config)

    def test_invalid_image_shape_caught_early(self, model):
        """Test invalid images caught by validation."""
        images = torch.rand(2, 4, 224, 224)  # Wrong channels
        texts = ["pick", "place"]

        with pytest.raises(ValueError, match="must have 3 channels"):
            model(images, texts=texts)

    def test_batch_size_mismatch_caught(self, model):
        """Test batch size mismatch caught by validation."""
        images = torch.rand(4, 3, 224, 224)
        texts = ["pick", "place"]  # Only 2 texts

        with pytest.raises(ValueError, match="Batch size mismatch"):
            model(images, texts=texts)

    def test_action_dimension_mismatch_caught(self, model):
        """Test action dimension mismatch caught."""
        images = torch.rand(2, 3, 224, 224)
        texts = ["pick", "place"]
        target_actions = torch.rand(2, 10)  # Wrong action_dim

        with pytest.raises(ValueError, match="must have action_dim=7"):
            model(images, texts=texts, target_actions=target_actions)
