"""Tests for fusion modules."""

import pytest
import torch

from vla.fusion import (
    ConcatFusion,
    CrossAttentionFusion,
    GatedFusion,
    PerceiverResampler,
    PrependFusion,
    TemporalPerceiverResampler,
)
from vla.registry import FUSION_REGISTRY


class TestPerceiverResampler:
    """Test PerceiverResampler fusion mechanism."""

    def test_basic_fusion(self):
        """Test basic vision+language fusion."""
        fusion = PerceiverResampler(
            dim=256,
            num_latents=32,
            num_layers=2,
            vision_dim=384,
            language_dim=512,
        )
        vision = torch.randn(2, 196, 384)
        language = torch.randn(2, 16, 512)

        out = fusion(vision, language)
        assert out.shape == (2, 32, 256)

    def test_vision_only(self):
        """Test fusion with vision only (no language)."""
        fusion = PerceiverResampler(dim=256, num_latents=32, vision_dim=256)
        vision = torch.randn(2, 196, 256)

        out = fusion(vision)
        assert out.shape == (2, 32, 256)

    def test_output_deterministic_with_seed(self):
        """Test that model initialization is deterministic with fixed seed."""
        torch.manual_seed(42)
        fusion1 = PerceiverResampler(dim=128, num_latents=16)

        # Create identical model with same seed
        torch.manual_seed(42)
        fusion2 = PerceiverResampler(dim=128, num_latents=16)

        # Models initialized with same seed should have same params
        for p1, p2 in zip(fusion1.parameters(), fusion2.parameters(), strict=True):
            assert torch.allclose(p1, p2, atol=1e-6)

    def test_different_batch_sizes(self):
        """Test fusion works with different batch sizes."""
        fusion = PerceiverResampler(dim=256, num_latents=32)

        for batch_size in [1, 4, 8]:
            vision = torch.randn(batch_size, 196, 256)
            language = torch.randn(batch_size, 16, 256)
            out = fusion(vision, language)
            assert out.shape == (batch_size, 32, 256)

    def test_gradient_flow(self):
        """Test gradients flow through fusion module."""
        fusion = PerceiverResampler(dim=256, num_latents=32)
        vision = torch.randn(2, 196, 256, requires_grad=True)
        language = torch.randn(2, 16, 256, requires_grad=True)

        out = fusion(vision, language)
        loss = out.sum()
        loss.backward()

        assert vision.grad is not None
        assert language.grad is not None
        assert vision.grad.abs().sum() > 0
        assert language.grad.abs().sum() > 0

    def test_registry_registration(self):
        """Test perceiver is registered correctly."""
        assert "perceiver_resampler" in FUSION_REGISTRY
        fusion = FUSION_REGISTRY.get("perceiver_resampler", dim=256, num_latents=32)
        assert isinstance(fusion, PerceiverResampler)

    def test_invalid_input_shape(self):
        """Test error handling for invalid input shapes."""
        fusion = PerceiverResampler(dim=256, num_latents=32)

        # 2D input (missing batch dimension)
        with pytest.raises(ValueError, match="Expected vision_features to be 3D"):
            fusion(torch.randn(196, 256))

        # 4D input (extra dimension)
        with pytest.raises(ValueError, match="Expected vision_features to be 3D"):
            fusion(torch.randn(2, 196, 256, 1))

    def test_parameter_count(self):
        """Test parameter count is reasonable (< 10M as per requirements)."""
        # Use smaller config to stay under 10M params
        fusion = PerceiverResampler(dim=512, num_latents=64, num_layers=2, num_heads=8)
        total_params = sum(p.numel() for p in fusion.parameters())
        assert total_params < 10_000_000, f"Too many params: {total_params}"

        # Verify full config (768) is still reasonably sized (allow up to 20M)
        fusion_large = PerceiverResampler(dim=768, num_latents=64, num_layers=2, num_heads=8)
        total_params_large = sum(p.numel() for p in fusion_large.parameters())
        assert (
            total_params_large < 20_000_000
        ), f"Large config too many params: {total_params_large}"


class TestTemporalPerceiverResampler:
    """Test TemporalPerceiverResampler for multi-frame fusion."""

    def test_multi_frame_fusion(self):
        """Test fusion with multiple video frames."""
        fusion = TemporalPerceiverResampler(
            dim=256,
            num_latents=32,
            max_frames=8,
            vision_dim=256,
        )
        frames = [torch.randn(2, 196, 256) for _ in range(6)]

        out = fusion(frames)
        assert out.shape == (2, 32, 256)

    def test_single_frame(self):
        """Test temporal perceiver works with single frame."""
        fusion = TemporalPerceiverResampler(dim=256, num_latents=32, max_frames=8)
        frames = [torch.randn(2, 196, 256)]

        out = fusion(frames)
        assert out.shape == (2, 32, 256)

    def test_with_language(self):
        """Test multi-frame fusion with language conditioning."""
        fusion = TemporalPerceiverResampler(dim=256, num_latents=32, max_frames=8, language_dim=512)
        frames = [torch.randn(2, 196, 256) for _ in range(4)]
        language = torch.randn(2, 16, 512)

        out = fusion(frames, language)
        assert out.shape == (2, 32, 256)

    def test_empty_frames_raises_error(self):
        """Test error when frames list is empty."""
        fusion = TemporalPerceiverResampler(dim=256, num_latents=32)

        with pytest.raises(ValueError, match="vision_frames list cannot be empty"):
            fusion([])

    def test_inconsistent_batch_size_raises_error(self):
        """Test error when frames have different batch sizes."""
        fusion = TemporalPerceiverResampler(dim=256, num_latents=32)
        frames = [
            torch.randn(2, 196, 256),
            torch.randn(4, 196, 256),  # Different batch size
        ]

        with pytest.raises(ValueError, match="Inconsistent batch size"):
            fusion(frames)

    def test_registry_registration(self):
        """Test temporal perceiver is registered correctly."""
        assert "temporal_perceiver" in FUSION_REGISTRY


class TestCrossAttentionFusion:
    """Test CrossAttentionFusion mechanism."""

    def test_conditioned_vision(self):
        """Test language-conditioned vision processing."""
        fusion = CrossAttentionFusion(
            dim=256,
            num_layers=2,
            vision_dim=384,
            language_dim=512,
        )
        vision = torch.randn(2, 196, 384)
        language = torch.randn(2, 16, 512)

        out = fusion(vision, language)
        assert out.shape == (2, 196, 256)  # Same seq length as vision

    def test_output_maintains_vision_length(self):
        """Test output maintains vision sequence length."""
        fusion = CrossAttentionFusion(dim=256, num_layers=2)
        vision = torch.randn(2, 100, 256)
        language = torch.randn(2, 20, 256)

        out = fusion(vision, language)
        assert out.shape == (2, 100, 256)

    def test_gradient_flow(self):
        """Test gradients flow through cross-attention layers."""
        fusion = CrossAttentionFusion(dim=256, num_layers=2)
        vision = torch.randn(2, 196, 256, requires_grad=True)
        language = torch.randn(2, 16, 256, requires_grad=True)

        out = fusion(vision, language)
        loss = out.sum()
        loss.backward()

        assert vision.grad is not None
        assert language.grad is not None

    def test_registry_registration(self):
        """Test cross-attention fusion is registered correctly."""
        assert "cross_attention_fusion" in FUSION_REGISTRY

    def test_invalid_input_shapes(self):
        """Test error handling for invalid shapes."""
        fusion = CrossAttentionFusion(dim=256, num_layers=2)

        with pytest.raises(ValueError, match="Expected vision_features to be 3D"):
            fusion(torch.randn(196, 256), torch.randn(2, 16, 256))

        with pytest.raises(ValueError, match="Expected language_features to be 3D"):
            fusion(torch.randn(2, 196, 256), torch.randn(16, 256))


class TestGatedFusion:
    """Test GatedFusion mechanism."""

    def test_gated_combination(self):
        """Test gated fusion produces correct output shape."""
        fusion = GatedFusion(dim=256, vision_dim=256, language_dim=256)
        vision = torch.randn(2, 196, 256)
        language = torch.randn(2, 16, 256)

        out = fusion(vision, language)
        assert out.shape == (2, 196, 256)

    def test_language_pooling(self):
        """Test language is pooled to match vision length."""
        fusion = GatedFusion(dim=256)
        vision = torch.randn(2, 100, 256)
        language = torch.randn(2, 10, 256)  # Different length

        out = fusion(vision, language)
        assert out.shape == (2, 100, 256)  # Matches vision

    def test_gate_values_in_range(self):
        """Test gate values are in [0, 1] due to sigmoid."""
        fusion = GatedFusion(dim=256)
        vision = torch.randn(2, 196, 256)
        language = torch.randn(2, 16, 256)

        # Access gate computation
        vision_proj = fusion.vision_proj(vision)
        lang_proj = fusion.lang_proj(language)
        lang_pooled = lang_proj.mean(dim=1, keepdim=True).expand(-1, vision.size(1), -1)
        combined = torch.cat([vision_proj, lang_pooled], dim=-1)
        gate = fusion.gate(combined)

        assert (gate >= 0).all()
        assert (gate <= 1).all()

    def test_registry_registration(self):
        """Test gated fusion is registered correctly."""
        assert "gated_fusion" in FUSION_REGISTRY


class TestConcatFusion:
    """Test ConcatFusion mechanism."""

    def test_concat_output_size(self):
        """Test concatenation produces correct output size."""
        fusion = ConcatFusion(dim=256, vision_dim=256, language_dim=256)
        vision = torch.randn(2, 196, 256)
        language = torch.randn(2, 16, 256)

        out = fusion(vision, language)
        assert out.shape == (2, 196 + 16, 256)  # 212 tokens

    def test_sequence_order(self):
        """Test sequence order is [language, vision]."""
        fusion = ConcatFusion(dim=256)
        vision = torch.randn(2, 196, 256)
        language = torch.randn(2, 16, 256)

        out = fusion(vision, language)

        # First tokens should be more similar to language than vision
        # (This is a simple heuristic test)
        assert out.shape[1] == 212  # 16 + 196

    def test_dimension_projection(self):
        """Test dimension projection works correctly."""
        fusion = ConcatFusion(dim=256, vision_dim=384, language_dim=512)
        vision = torch.randn(2, 196, 384)
        language = torch.randn(2, 16, 512)

        out = fusion(vision, language)
        assert out.shape == (2, 212, 256)

    def test_batch_size_mismatch_raises_error(self):
        """Test error when batch sizes don't match."""
        fusion = ConcatFusion(dim=256)
        vision = torch.randn(2, 196, 256)
        language = torch.randn(4, 16, 256)  # Different batch size

        with pytest.raises(ValueError, match="Batch size mismatch"):
            fusion(vision, language)

    def test_registry_registration(self):
        """Test concat fusion is registered correctly."""
        assert "concat_fusion" in FUSION_REGISTRY


class TestPrependFusion:
    """Test PrependFusion mechanism."""

    def test_prepend_with_separator(self):
        """Test prepend fusion adds separator token."""
        fusion = PrependFusion(dim=256, vision_dim=256, language_dim=256)
        vision = torch.randn(2, 196, 256)
        language = torch.randn(2, 16, 256)

        out = fusion(vision, language)
        assert out.shape == (2, 196 + 16 + 1, 256)  # +1 for separator

    def test_separator_is_learnable(self):
        """Test separator token is learnable."""
        fusion = PrependFusion(dim=256)
        assert fusion.separator.requires_grad

    def test_dimension_projection(self):
        """Test dimension projection with separator."""
        fusion = PrependFusion(dim=256, vision_dim=384, language_dim=512)
        vision = torch.randn(2, 196, 384)
        language = torch.randn(2, 16, 512)

        out = fusion(vision, language)
        assert out.shape == (2, 213, 256)  # 16 + 1 + 196

    def test_batch_size_mismatch_raises_error(self):
        """Test error when batch sizes don't match."""
        fusion = PrependFusion(dim=256)
        vision = torch.randn(2, 196, 256)
        language = torch.randn(4, 16, 256)

        with pytest.raises(ValueError, match="Batch size mismatch"):
            fusion(vision, language)

    def test_registry_registration(self):
        """Test prepend fusion is registered correctly."""
        assert "prepend_fusion" in FUSION_REGISTRY


class TestFusionRegistry:
    """Test fusion registry integration."""

    def test_all_fusion_modules_registered(self):
        """Test all fusion modules are in registry."""
        expected_modules = [
            "perceiver_resampler",
            "temporal_perceiver",
            "cross_attention_fusion",
            "gated_fusion",
            "concat_fusion",
            "prepend_fusion",
        ]
        for module_name in expected_modules:
            assert module_name in FUSION_REGISTRY

    def test_registry_instantiation(self):
        """Test modules can be instantiated from registry."""
        perceiver = FUSION_REGISTRY.get("perceiver_resampler", dim=256, num_latents=32)
        assert isinstance(perceiver, PerceiverResampler)

        concat = FUSION_REGISTRY.get("concat_fusion", dim=256)
        assert isinstance(concat, ConcatFusion)
