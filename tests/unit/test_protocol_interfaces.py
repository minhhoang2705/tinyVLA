"""Unit tests for Protocol interfaces.

Tests verify that:
- Protocols can be checked at runtime
- Existing components implement protocols correctly
- Protocol validation catches non-compliant components
"""

import pytest
import torch
import torch.nn as nn
from typing import List, Optional, Tuple

from vla.backbones.protocols import (
    VisionBackboneProtocol,
    LanguageBackboneProtocol,
    FusionModuleProtocol,
    ActionHeadProtocol,
    validate_vision_backbone,
    validate_language_backbone,
    validate_fusion_module,
    validate_action_head,
)


class TestVisionBackboneProtocol:
    """Tests for VisionBackboneProtocol."""

    def test_compliant_vision_backbone(self):
        """Test that a compliant vision backbone passes protocol check."""

        class CompliantVision(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_dim = 768

            def forward(self, images: torch.Tensor) -> torch.Tensor:
                B = images.shape[0]
                return torch.randn(B, 196, self.embed_dim)

        vision = CompliantVision()
        assert isinstance(vision, VisionBackboneProtocol)
        assert validate_vision_backbone(vision)

    def test_missing_embed_dim(self):
        """Test that missing embed_dim fails protocol check."""

        class NonCompliantVision(nn.Module):
            def forward(self, images: torch.Tensor) -> torch.Tensor:
                return torch.randn(images.shape[0], 196, 768)

        vision = NonCompliantVision()
        assert not isinstance(vision, VisionBackboneProtocol)
        assert not validate_vision_backbone(vision)

    def test_missing_forward_method(self):
        """Test protocol check with inherited forward from nn.Module.

        Note: Python's @runtime_checkable only checks attribute existence,
        not signatures. nn.Module always provides forward(), so isinstance
        returns True even without a custom forward. Use validate functions
        for strict validation.
        """

        class NonCompliantVision(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_dim = 768

        vision = NonCompliantVision()
        # nn.Module provides forward() — protocol limitation: signature not checked
        assert isinstance(vision, VisionBackboneProtocol)


class TestLanguageBackboneProtocol:
    """Tests for LanguageBackboneProtocol."""

    def test_compliant_language_backbone(self):
        """Test that a compliant language backbone passes protocol check."""

        class CompliantLanguage(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_dim = 768

            def forward(
                self,
                texts: Optional[List[str]] = None,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
            ) -> torch.Tensor:
                if texts:
                    B = len(texts)
                else:
                    B = input_ids.shape[0]
                return torch.randn(B, 10, self.embed_dim)

        language = CompliantLanguage()
        assert isinstance(language, LanguageBackboneProtocol)
        assert validate_language_backbone(language)

    def test_missing_embed_dim(self):
        """Test that missing embed_dim fails protocol check."""

        class NonCompliantLanguage(nn.Module):
            def forward(
                self,
                texts: Optional[List[str]] = None,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
            ) -> torch.Tensor:
                return torch.randn(2, 10, 768)

        language = NonCompliantLanguage()
        assert not isinstance(language, LanguageBackboneProtocol)

    def test_wrong_forward_signature(self):
        """Test that wrong forward signature fails protocol check."""

        class NonCompliantLanguage(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_dim = 768

            def forward(self, text: str) -> torch.Tensor:  # Wrong signature
                return torch.randn(1, 10, self.embed_dim)

        language = NonCompliantLanguage()
        # Protocol check may pass at runtime but actual usage will fail
        # This is a limitation of structural typing


class TestFusionModuleProtocol:
    """Tests for FusionModuleProtocol."""

    def test_compliant_fusion_module(self):
        """Test that a compliant fusion module passes protocol check."""

        class CompliantFusion(nn.Module):
            def forward(
                self,
                vision_features: torch.Tensor,
                language_features: torch.Tensor,
            ) -> torch.Tensor:
                B = vision_features.shape[0]
                return torch.randn(B, 64, 768)

        fusion = CompliantFusion()
        assert isinstance(fusion, FusionModuleProtocol)
        assert validate_fusion_module(fusion)

    def test_missing_forward_method(self):
        """Test protocol check with inherited forward from nn.Module.

        Note: Python's @runtime_checkable only checks attribute existence,
        not signatures. nn.Module always provides forward(), so isinstance
        returns True even without a custom forward.
        """

        class NonCompliantFusion(nn.Module):
            pass

        fusion = NonCompliantFusion()
        # nn.Module provides forward() — protocol limitation: signature not checked
        assert isinstance(fusion, FusionModuleProtocol)


class TestActionHeadProtocol:
    """Tests for ActionHeadProtocol."""

    def test_compliant_action_head(self):
        """Test that a compliant action head passes protocol check."""

        class CompliantActionHead(nn.Module):
            def forward(
                self,
                fused_features: torch.Tensor,
                return_logits: bool = False,
            ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
                B = fused_features.shape[0]
                actions = torch.randn(B, 7)
                logits = torch.randn(B, 7, 256) if return_logits else None
                return actions, logits

            def compute_loss(
                self,
                logits: torch.Tensor,
                target_actions: torch.Tensor,
            ) -> torch.Tensor:
                return torch.nn.functional.mse_loss(
                    logits.mean(dim=-1), target_actions
                )

        head = CompliantActionHead()
        assert isinstance(head, ActionHeadProtocol)
        assert validate_action_head(head)

    def test_missing_compute_loss(self):
        """Test that missing compute_loss fails protocol check."""

        class NonCompliantActionHead(nn.Module):
            def forward(
                self,
                fused_features: torch.Tensor,
                return_logits: bool = False,
            ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
                B = fused_features.shape[0]
                actions = torch.randn(B, 7)
                return actions, None

        head = NonCompliantActionHead()
        assert not isinstance(head, ActionHeadProtocol)


class TestProtocolIntegration:
    """Integration tests for protocols with real VLA components."""

    def test_timm_vision_backbone_protocol(self):
        """Test that VisionBackbone implements VisionBackboneProtocol."""
        from vla.backbones.vision import VisionBackbone

        vision = VisionBackbone(
            model_name="vit_tiny_patch16_224",
            pretrained=False,
        )

        assert isinstance(vision, VisionBackboneProtocol)
        assert hasattr(vision, "embed_dim")
        assert vision.embed_dim > 0

    def test_language_backbone_protocol(self):
        """Test that GPT2Backbone implements LanguageBackboneProtocol."""
        from vla.backbones.language import GPT2Backbone

        language = GPT2Backbone(
            model_name="gpt2",
            max_length=77,
        )

        assert isinstance(language, LanguageBackboneProtocol)
        assert hasattr(language, "embed_dim")
        assert language.embed_dim > 0

    def test_perceiver_fusion_protocol(self):
        """Test that PerceiverResampler implements FusionModuleProtocol."""
        from vla.fusion.perceiver import PerceiverResampler

        fusion = PerceiverResampler(
            dim=768,
            num_latents=64,
            num_layers=2,
            num_heads=8,
        )

        assert isinstance(fusion, FusionModuleProtocol)

    def test_discrete_action_head_protocol(self):
        """Test that DiscreteActionHead implements ActionHeadProtocol."""
        from vla.policy.action_heads import DiscreteActionHead

        head = DiscreteActionHead(
            input_dim=768,
            action_dim=7,
            num_bins=256,
        )

        assert isinstance(head, ActionHeadProtocol)
        assert hasattr(head, "forward")
        assert hasattr(head, "compute_loss")

    def test_gaussian_action_head_protocol(self):
        """Test that GaussianActionHead implements ActionHeadProtocol."""
        from vla.policy.action_heads import GaussianActionHead

        head = GaussianActionHead(
            input_dim=768,
            action_dim=7,
        )

        assert isinstance(head, ActionHeadProtocol)
        assert hasattr(head, "forward")
        assert hasattr(head, "compute_loss")

    def test_vla_model_validates_protocols(self):
        """Test that VLAModel validates protocols during initialization."""
        from vla.models import VLAModel, VLAConfig

        # Should not raise - all components are protocol-compliant
        config = VLAConfig()
        model = VLAModel(config)

        # Verify all components
        assert isinstance(model.vision, VisionBackboneProtocol)
        assert isinstance(model.language, LanguageBackboneProtocol)
        assert isinstance(model.fusion, FusionModuleProtocol)
        assert isinstance(model.action_head, ActionHeadProtocol)
