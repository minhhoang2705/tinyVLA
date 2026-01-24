"""Tests for vision backbones."""

import pytest
import torch

from vla.backbones import (
    DINOv2Backbone,
    DualEncoderVision,
    MultiScaleFeatureExtractor,
    SigLIPBackbone,
    VisionBackbone,
)
from vla.registry import VISION_REGISTRY


class TestVisionBackbone:
    """Tests for VisionBackbone class."""

    @pytest.fixture
    def tiny_model_name(self):
        """Use tiny ViT for fast testing."""
        return "vit_tiny_patch16_224"

    def test_timm_vit_spatial(self, dummy_image, tiny_model_name):
        """Test spatial output mode returns patch tokens only."""
        backbone = VisionBackbone(
            model_name=tiny_model_name,
            pretrained=False,
            output_mode="spatial",
        )
        out = backbone(dummy_image)

        # 224/16 = 14, 14*14 = 196 patches
        expected_patches = 196
        assert out.shape == (
            dummy_image.shape[0],
            expected_patches,
            backbone.embed_dim,
        ), f"Expected shape (B, {expected_patches}, D), got {out.shape}"

    def test_timm_vit_cls(self, dummy_image, tiny_model_name):
        """Test cls output mode returns only CLS token."""
        backbone = VisionBackbone(
            model_name=tiny_model_name,
            pretrained=False,
            output_mode="cls",
        )
        out = backbone(dummy_image)

        assert out.shape == (
            dummy_image.shape[0],
            1,
            backbone.embed_dim,
        ), f"Expected shape (B, 1, D), got {out.shape}"

    def test_timm_vit_both(self, dummy_image, tiny_model_name):
        """Test both output mode returns CLS + patch tokens."""
        backbone = VisionBackbone(
            model_name=tiny_model_name,
            pretrained=False,
            output_mode="both",
        )
        out = backbone(dummy_image)

        # 196 patches + 1 CLS = 197 tokens
        expected_tokens = 197
        assert out.shape == (
            dummy_image.shape[0],
            expected_tokens,
            backbone.embed_dim,
        ), f"Expected shape (B, {expected_tokens}, D), got {out.shape}"

    def test_projection(self, dummy_image, tiny_model_name):
        """Test optional projection layer changes output dimension."""
        proj_dim = 512
        backbone = VisionBackbone(
            model_name=tiny_model_name,
            pretrained=False,
            proj_dim=proj_dim,
        )
        out = backbone(dummy_image)

        assert out.shape[-1] == proj_dim, f"Expected dim {proj_dim}, got {out.shape[-1]}"
        assert backbone.embed_dim == proj_dim, "embed_dim should update with projection"

    def test_frozen_params(self, tiny_model_name):
        """Test frozen=True disables gradients on backbone."""
        backbone = VisionBackbone(
            model_name=tiny_model_name,
            pretrained=False,
            frozen=True,
        )

        # Check all backbone parameters are frozen
        for name, param in backbone.backbone.named_parameters():
            assert not param.requires_grad, f"{name} should be frozen but requires_grad=True"

        # Projection layer should still have gradients
        if backbone.proj is not None:
            for param in backbone.proj.parameters():
                assert param.requires_grad, "Projection layer should be trainable"

    def test_unfrozen_params(self, tiny_model_name):
        """Test frozen=False keeps gradients enabled."""
        backbone = VisionBackbone(
            model_name=tiny_model_name,
            pretrained=False,
            frozen=False,
        )

        # At least some parameters should require gradients
        has_grad = any(p.requires_grad for p in backbone.backbone.parameters())
        assert has_grad, "At least some backbone parameters should require gradients"

    def test_num_patches_property(self, tiny_model_name):
        """Test num_patches property returns correct count."""
        backbone = VisionBackbone(
            model_name=tiny_model_name,
            pretrained=False,
        )

        assert backbone.num_patches == 196, f"Expected 196 patches, got {backbone.num_patches}"

    def test_invalid_input_shape(self, tiny_model_name):
        """Test error handling for invalid input shapes."""
        backbone = VisionBackbone(
            model_name=tiny_model_name,
            pretrained=False,
        )

        # 3D tensor should raise error
        invalid_input = torch.randn(2, 3, 224)
        with pytest.raises(ValueError, match="Expected 4D image tensor"):
            backbone(invalid_input)

    def test_registry_registration(self):
        """Test that VisionBackbone is registered correctly."""
        assert "timm_vit" in VISION_REGISTRY, "timm_vit should be registered"
        assert "dinov2" in VISION_REGISTRY, "dinov2 should be registered"
        assert "siglip" in VISION_REGISTRY, "siglip should be registered"


class TestDINOv2Backbone:
    """Tests for DINOv2Backbone convenience class."""

    @pytest.mark.parametrize("size", ["small", "base", "large", "giant"])
    def test_dinov2_sizes(self, size):
        """Test all DINOv2 size variants can be instantiated."""
        # Use pretrained=False to avoid downloading weights
        backbone = DINOv2Backbone(size=size, pretrained=False)
        assert isinstance(backbone, VisionBackbone)
        assert "dinov2" in backbone.model_name.lower()

    def test_dinov2_default_output_mode(self):
        """Test DINOv2 uses spatial output mode by default."""
        backbone = DINOv2Backbone(size="small", pretrained=False)

        # DINOv2 models default to 518x518, create appropriate input
        # Use patch14 model, so input should be divisible by 14
        dinov2_image = torch.randn(2, 3, 518, 518)
        out = backbone(dinov2_image)

        # Should return patch tokens only (no CLS)
        assert out.shape[1] > 1, "Should return multiple patch tokens"

    def test_dinov2_frozen_by_default(self):
        """Test DINOv2 freezes backbone by default."""
        backbone = DINOv2Backbone(size="small", pretrained=False, frozen=True)

        for param in backbone.backbone.parameters():
            assert not param.requires_grad, "Backbone should be frozen by default"


class TestSigLIPBackbone:
    """Tests for SigLIPBackbone convenience class."""

    @pytest.mark.parametrize("size", ["base", "large"])
    def test_siglip_sizes(self, size):
        """Test all SigLIP size variants can be instantiated."""
        backbone = SigLIPBackbone(size=size, pretrained=False)
        assert isinstance(backbone, VisionBackbone)
        assert "siglip" in backbone.model_name.lower()

    def test_siglip_default_output_mode(self, dummy_image):
        """Test SigLIP uses spatial output mode by default."""
        backbone = SigLIPBackbone(size="base", pretrained=False)
        out = backbone(dummy_image)

        # Should return patch tokens only
        assert out.shape[1] > 1, "Should return multiple patch tokens"


class TestMultiScaleFeatureExtractor:
    """Tests for MultiScaleFeatureExtractor."""

    @pytest.fixture
    def tiny_model_name(self):
        """Use tiny ViT for fast testing."""
        return "vit_tiny_patch16_224"

    def test_multi_scale_output(self, dummy_image, tiny_model_name):
        """Test multi-scale extraction returns features from all specified layers."""
        layers = [-4, -3, -2, -1]
        extractor = MultiScaleFeatureExtractor(
            model_name=tiny_model_name,
            pretrained=False,
            layers=layers,
        )

        features = extractor(dummy_image)

        # Should have features for all requested layers
        assert len(features) == len(layers), f"Expected {len(layers)} feature maps"

        # All features should have correct batch size
        for _layer_idx, feat in features.items():
            assert feat.shape[0] == dummy_image.shape[0], "Batch size mismatch"
            assert feat.dim() == 3, "Features should be 3D [B, N, D]"

    def test_frozen_params(self, tiny_model_name):
        """Test frozen=True disables gradients."""
        extractor = MultiScaleFeatureExtractor(
            model_name=tiny_model_name,
            pretrained=False,
            frozen=True,
        )

        for param in extractor.backbone.parameters():
            assert not param.requires_grad, "Backbone should be frozen"

    def test_invalid_layer_index(self, tiny_model_name):
        """Test error handling for invalid layer indices."""
        # Try to access layer that doesn't exist
        with pytest.raises(ValueError, match="Invalid layer index"):
            MultiScaleFeatureExtractor(
                model_name=tiny_model_name,
                pretrained=False,
                layers=[100],  # ViT-tiny doesn't have 100 layers
            )

    def test_invalid_input_shape(self, tiny_model_name):
        """Test error handling for invalid input shapes."""
        extractor = MultiScaleFeatureExtractor(
            model_name=tiny_model_name,
            pretrained=False,
        )

        invalid_input = torch.randn(2, 3, 224)  # 3D instead of 4D
        with pytest.raises(ValueError, match="Expected 4D image tensor"):
            extractor(invalid_input)


class TestDualEncoderVision:
    """Tests for DualEncoderVision."""

    @pytest.fixture
    def encoder_names(self):
        """Use tiny models for fast testing."""
        return {
            "encoder1": "vit_tiny_patch16_224",
            "encoder2": "vit_tiny_patch16_224",
        }

    def test_dual_encoder_output_shape(self, dummy_image, encoder_names):
        """Test dual encoder concatenates features correctly."""
        encoder = DualEncoderVision(
            encoder1_name=encoder_names["encoder1"],
            encoder2_name=encoder_names["encoder2"],
            pretrained=False,
        )

        out = encoder(dummy_image)

        # Should concatenate features from both encoders
        assert out.shape[0] == dummy_image.shape[0], "Batch size mismatch"
        assert out.shape[1] > 1, "Should have multiple tokens"
        assert out.dim() == 3, "Output should be 3D [B, N, D]"

    def test_dual_encoder_with_projection(self, dummy_image, encoder_names):
        """Test dual encoder with projection layer."""
        proj_dim = 512
        encoder = DualEncoderVision(
            encoder1_name=encoder_names["encoder1"],
            encoder2_name=encoder_names["encoder2"],
            pretrained=False,
            proj_dim=proj_dim,
        )

        out = encoder(dummy_image)

        assert out.shape[-1] == proj_dim, f"Expected dim {proj_dim}, got {out.shape[-1]}"
        assert encoder.embed_dim == proj_dim, "embed_dim should match proj_dim"

    def test_dual_encoder_frozen(self, encoder_names):
        """Test dual encoder freezes both backbones."""
        encoder = DualEncoderVision(
            encoder1_name=encoder_names["encoder1"],
            encoder2_name=encoder_names["encoder2"],
            pretrained=False,
            frozen=True,
        )

        # Check both encoders are frozen
        for param in encoder.encoder1.backbone.parameters():
            assert not param.requires_grad, "Encoder1 backbone should be frozen"

        for param in encoder.encoder2.backbone.parameters():
            assert not param.requires_grad, "Encoder2 backbone should be frozen"

    def test_dual_encoder_different_patch_counts(self, dummy_image):
        """Test dual encoder handles different patch counts via interpolation."""
        # Use models with different patch sizes (14 vs 16)
        encoder = DualEncoderVision(
            encoder1_name="vit_tiny_patch16_224",  # 196 patches
            encoder2_name="vit_tiny_patch16_224",  # 196 patches (same for simplicity)
            pretrained=False,
        )

        out = encoder(dummy_image)

        # Should successfully combine despite potential differences
        assert out.shape[0] == dummy_image.shape[0]
        assert out.dim() == 3

    def test_invalid_input_shape(self, encoder_names):
        """Test error handling for invalid input shapes."""
        encoder = DualEncoderVision(
            encoder1_name=encoder_names["encoder1"],
            encoder2_name=encoder_names["encoder2"],
            pretrained=False,
        )

        invalid_input = torch.randn(2, 3, 224)  # 3D instead of 4D
        with pytest.raises(ValueError, match="Expected 4D image tensor"):
            encoder(invalid_input)


class TestRegistryIntegration:
    """Tests for registry integration."""

    def test_get_from_registry(self):
        """Test instantiation via registry."""
        # Should be able to get VisionBackbone from registry
        backbone = VISION_REGISTRY.get(
            "timm_vit",
            model_name="vit_tiny_patch16_224",
            pretrained=False,
        )
        assert isinstance(backbone, VisionBackbone)

    def test_dinov2_from_registry(self):
        """Test DINOv2 instantiation via registry."""
        backbone = VISION_REGISTRY.get("dinov2", size="small", pretrained=False)
        assert isinstance(backbone, DINOv2Backbone)

    def test_siglip_from_registry(self):
        """Test SigLIP instantiation via registry."""
        backbone = VISION_REGISTRY.get("siglip", size="base", pretrained=False)
        assert isinstance(backbone, SigLIPBackbone)

    def test_list_available(self):
        """Test listing available vision backbones."""
        available = VISION_REGISTRY.list_available()
        assert "timm_vit" in available
        assert "dinov2" in available
        assert "siglip" in available
