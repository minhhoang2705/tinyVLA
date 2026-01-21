# Phase 04: Vision Backbone Builder

## Context Links
- [Tech Stack](../../docs/tech-stack.md) - timm section
- [VLA Architectures](../reports/researcher-260118-vla-architectures.md) - Vision encoders
- [PyTorch VLA Research](../reports/researcher-260118-0228-pytorch-vla.md) - timm integration

## Overview
| Field | Value |
|-------|-------|
| Priority | P1 - Core Component |
| Status | Pending |
| Effort | 3h |
| Dependencies | Phases 2, 3 |

Build vision encoder wrapper around timm models with VLA-specific features: feature extraction modes, multi-scale outputs, and frame processing.

## Key Insights
- timm provides 500+ pretrained models with consistent API
- DINOv2 + SigLIP fusion is SOTA for VLA (OpenVLA pattern)
- Use `features_only=True` for intermediate layer access
- Freeze backbone, train projection head (transfer learning)

## Requirements

### Functional
- FR-01: Load any timm model by name
- FR-02: Extract CLS token, spatial tokens, or both
- FR-03: Support multi-scale feature extraction
- FR-04: Optional backbone freezing
- FR-05: Configurable input resolution

### Non-Functional
- NFR-01: <100ms forward pass on RTX 3090 (224x224)
- NFR-02: Memory <4GB for ViT-Base

## Architecture

```
src/vla/backbones/
├── __init__.py
├── vision.py           # Vision backbone wrapper
└── feature_extractor.py # Multi-scale extraction
```

**Vision Pipeline:**
```
Image [B, 3, H, W]
    ↓
timm.create_model (DINOv2/SigLIP/ViT)
    ↓
VisionBackbone wrapper
    ↓
Features [B, N, D] (N=HW/patch² + CLS)
```

## Related Code Files

### Files to Create
| Path | Purpose | Lines |
|------|---------|-------|
| `src/vla/backbones/__init__.py` | Exports | ~15 |
| `src/vla/backbones/vision.py` | Vision wrapper | ~120 |
| `src/vla/backbones/feature_extractor.py` | Multi-scale | ~80 |
| `tests/unit/test_vision.py` | Vision tests | ~80 |

## Implementation Steps

### Step 1: Implement vision.py (60 min)
```python
"""Vision backbone wrapper for timm models."""
import torch
import torch.nn as nn
import timm
from typing import Optional, Literal
from einops import rearrange

from vla.registry import VISION_REGISTRY


@VISION_REGISTRY.register("timm_vit")
class VisionBackbone(nn.Module):
    """Wrapper around timm vision models for VLA.

    Args:
        model_name: timm model name (e.g., "vit_base_patch16_224")
        pretrained: Load pretrained weights
        frozen: Freeze backbone parameters
        output_mode: "cls" (CLS token), "spatial" (patch tokens), "both"
        proj_dim: Optional projection dimension
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        frozen: bool = True,
        output_mode: Literal["cls", "spatial", "both"] = "spatial",
        proj_dim: Optional[int] = None,
    ):
        super().__init__()
        self.output_mode = output_mode

        # Load timm model
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
        )
        self.embed_dim = self.backbone.embed_dim

        # Freeze if requested
        if frozen:
            self._freeze_backbone()

        # Optional projection
        self.proj = None
        if proj_dim is not None:
            self.proj = nn.Linear(self.embed_dim, proj_dim)
            self.embed_dim = proj_dim

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Image tensor [B, C, H, W]
        Returns:
            Features [B, N, D] where N depends on output_mode
        """
        # Get all tokens including CLS
        features = self.backbone.forward_features(x)

        if self.output_mode == "cls":
            # Just CLS token [B, D]
            if hasattr(self.backbone, "fc_norm"):
                features = self.backbone.fc_norm(features[:, 0])
            else:
                features = features[:, 0]
            features = features.unsqueeze(1)  # [B, 1, D]
        elif self.output_mode == "spatial":
            # Patch tokens only [B, N-1, D]
            features = features[:, 1:]
        # else "both": keep all tokens [B, N, D]

        if self.proj is not None:
            features = self.proj(features)

        return features

    @property
    def num_patches(self) -> int:
        """Number of spatial patches (excluding CLS)."""
        return self.backbone.patch_embed.num_patches


@VISION_REGISTRY.register("dinov2")
class DINOv2Backbone(VisionBackbone):
    """DINOv2 backbone with sensible defaults."""

    def __init__(
        self,
        size: Literal["small", "base", "large", "giant"] = "base",
        pretrained: bool = True,
        frozen: bool = True,
        proj_dim: Optional[int] = None,
    ):
        model_map = {
            "small": "vit_small_patch14_dinov2.lvd142m",
            "base": "vit_base_patch14_dinov2.lvd142m",
            "large": "vit_large_patch14_dinov2.lvd142m",
            "giant": "vit_giant_patch14_dinov2.lvd142m",
        }
        super().__init__(
            model_name=model_map[size],
            pretrained=pretrained,
            frozen=frozen,
            output_mode="spatial",
            proj_dim=proj_dim,
        )


@VISION_REGISTRY.register("siglip")
class SigLIPBackbone(VisionBackbone):
    """SigLIP backbone for vision-language alignment."""

    def __init__(
        self,
        size: Literal["base", "large"] = "base",
        pretrained: bool = True,
        frozen: bool = True,
        proj_dim: Optional[int] = None,
    ):
        model_map = {
            "base": "vit_base_patch16_siglip_224",
            "large": "vit_large_patch16_siglip_256",
        }
        super().__init__(
            model_name=model_map[size],
            pretrained=pretrained,
            frozen=frozen,
            output_mode="spatial",
            proj_dim=proj_dim,
        )
```

### Step 2: Implement feature_extractor.py (45 min)
```python
"""Multi-scale feature extraction from vision backbones."""
import torch
import torch.nn as nn
import timm
from typing import List


class MultiScaleFeatureExtractor(nn.Module):
    """Extract features from multiple backbone layers.

    Useful for dense prediction tasks or multi-resolution fusion.
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        frozen: bool = True,
        layers: List[int] = [-4, -3, -2, -1],
    ):
        super().__init__()
        self.layers = layers

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=False,
        )

        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Hook storage
        self._features = {}
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks on target layers."""
        blocks = self.backbone.blocks
        for idx in self.layers:
            layer_idx = idx if idx >= 0 else len(blocks) + idx
            blocks[layer_idx].register_forward_hook(
                self._make_hook(layer_idx)
            )

    def _make_hook(self, layer_idx: int):
        def hook(module, input, output):
            self._features[layer_idx] = output
        return hook

    def forward(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        """Extract multi-scale features.

        Args:
            x: Image tensor [B, C, H, W]
        Returns:
            Dict mapping layer index to features [B, N, D]
        """
        self._features.clear()
        _ = self.backbone.forward_features(x)
        return dict(self._features)


class DualEncoderVision(nn.Module):
    """Dual vision encoder (e.g., DINOv2 + SigLIP) following OpenVLA.

    Concatenates features from two complementary vision backbones.
    """

    def __init__(
        self,
        encoder1_name: str = "vit_base_patch14_dinov2.lvd142m",
        encoder2_name: str = "vit_base_patch16_siglip_224",
        pretrained: bool = True,
        frozen: bool = True,
        proj_dim: Optional[int] = None,
    ):
        super().__init__()
        from .vision import VisionBackbone

        self.encoder1 = VisionBackbone(
            encoder1_name, pretrained, frozen, output_mode="spatial"
        )
        self.encoder2 = VisionBackbone(
            encoder2_name, pretrained, frozen, output_mode="spatial"
        )

        combined_dim = self.encoder1.embed_dim + self.encoder2.embed_dim
        self.proj = None
        if proj_dim:
            self.proj = nn.Linear(combined_dim, proj_dim)
            self.embed_dim = proj_dim
        else:
            self.embed_dim = combined_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat1 = self.encoder1(x)
        feat2 = self.encoder2(x)

        # Interpolate if patch counts differ
        if feat1.size(1) != feat2.size(1):
            feat2 = nn.functional.interpolate(
                feat2.transpose(1, 2),
                size=feat1.size(1),
                mode="linear",
            ).transpose(1, 2)

        combined = torch.cat([feat1, feat2], dim=-1)

        if self.proj:
            combined = self.proj(combined)

        return combined
```

### Step 3: Create __init__.py (10 min)
```python
"""Vision backbone modules."""
from .vision import VisionBackbone, DINOv2Backbone, SigLIPBackbone
from .feature_extractor import MultiScaleFeatureExtractor, DualEncoderVision

__all__ = [
    "VisionBackbone",
    "DINOv2Backbone",
    "SigLIPBackbone",
    "MultiScaleFeatureExtractor",
    "DualEncoderVision",
]
```

### Step 4: Write tests (45 min)
```python
"""Tests for vision backbones."""
import pytest
import torch
from vla.backbones import VisionBackbone, DINOv2Backbone
from vla.registry import VISION_REGISTRY


class TestVisionBackbone:
    @pytest.fixture
    def dummy_image(self):
        return torch.randn(2, 3, 224, 224)

    def test_timm_vit_spatial(self, dummy_image):
        backbone = VisionBackbone(
            model_name="vit_tiny_patch16_224",
            pretrained=False,
            output_mode="spatial",
        )
        out = backbone(dummy_image)
        # 224/16 = 14, 14*14 = 196 patches
        assert out.shape == (2, 196, backbone.embed_dim)

    def test_timm_vit_cls(self, dummy_image):
        backbone = VisionBackbone(
            model_name="vit_tiny_patch16_224",
            pretrained=False,
            output_mode="cls",
        )
        out = backbone(dummy_image)
        assert out.shape == (2, 1, backbone.embed_dim)

    def test_timm_vit_both(self, dummy_image):
        backbone = VisionBackbone(
            model_name="vit_tiny_patch16_224",
            pretrained=False,
            output_mode="both",
        )
        out = backbone(dummy_image)
        assert out.shape == (2, 197, backbone.embed_dim)  # 196 + CLS

    def test_projection(self, dummy_image):
        backbone = VisionBackbone(
            model_name="vit_tiny_patch16_224",
            pretrained=False,
            proj_dim=512,
        )
        out = backbone(dummy_image)
        assert out.shape[-1] == 512

    def test_frozen_params(self, dummy_image):
        backbone = VisionBackbone(
            model_name="vit_tiny_patch16_224",
            frozen=True,
        )
        for name, param in backbone.backbone.named_parameters():
            assert not param.requires_grad, f"{name} should be frozen"

    def test_registry_lookup(self):
        assert "timm_vit" in VISION_REGISTRY
        assert "dinov2" in VISION_REGISTRY
        assert "siglip" in VISION_REGISTRY


class TestDualEncoder:
    def test_dual_encoder_shape(self):
        from vla.backbones import DualEncoderVision

        encoder = DualEncoderVision(
            encoder1_name="vit_tiny_patch16_224",
            encoder2_name="vit_tiny_patch16_224",
            pretrained=False,
            proj_dim=768,
        )
        x = torch.randn(2, 3, 224, 224)
        out = encoder(x)
        assert out.shape == (2, 196, 768)
```

## Todo List
- [ ] Implement VisionBackbone with output modes
- [ ] Implement DINOv2Backbone convenience class
- [ ] Implement SigLIPBackbone convenience class
- [ ] Implement MultiScaleFeatureExtractor
- [ ] Implement DualEncoderVision
- [ ] Register all backbones in VISION_REGISTRY
- [ ] Write unit tests
- [ ] Test with real pretrained weights

## Success Criteria
1. VisionBackbone loads any timm model
2. Output modes (cls/spatial/both) work correctly
3. Freezing prevents gradient flow
4. Projection layer applied correctly
5. All tests pass

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| timm model not found | Medium | Validate model names, provide fallbacks |
| Memory OOM on large models | High | Default to smaller models, doc requirements |
| Patch count mismatch in dual encoder | Medium | Interpolation handling |

## Security Considerations
- Pretrained weights from trusted sources (timm hub)
- No arbitrary code execution from model loading

## Next Steps
- Phase 5: Language backbone for instruction encoding
