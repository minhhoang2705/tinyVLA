# Phase 08: VLA Model Orchestration

## Context Links
- [VLA Architectures](../reports/researcher-260118-vla-architectures.md) - Full model patterns
- [PyTorch VLA Research](../reports/researcher-260118-0228-pytorch-vla.md) - Composition patterns

## Overview
| Field | Value |
|-------|-------|
| Priority | P1 - Integration |
| Status | Pending |
| Effort | 4h |
| Dependencies | Phases 2-7 |

Compose vision backbone, language encoder, fusion module, and action head into complete VLA model. Provide unified forward pass and loss computation.

## Key Insights
- Composition over inheritance for flexibility
- Config-driven component selection via registry
- Forward pass: image + text → fused features → actions
- Loss combines action prediction + optional auxiliary losses

## Requirements

### Functional
- FR-01: Unified VLA model class composing all components
- FR-02: Config-driven component instantiation
- FR-03: Forward pass for training (returns loss) and inference (returns actions)
- FR-04: Support frozen backbone training
- FR-05: Checkpoint save/load with component isolation

### Non-Functional
- NFR-01: End-to-end forward <200ms on RTX 3090
- NFR-02: Memory <16GB for full model (frozen backbones)

## Architecture

```
src/vla/models/
├── __init__.py
├── vla_base.py          # Base VLA model
└── vla_configs.py       # Model configuration dataclasses
```

**VLA Forward Flow:**
```
Image [B, C, H, W]  +  Text [B,]
        ↓                  ↓
   VisionBackbone    LanguageBackbone
        ↓                  ↓
    [B, N, D_v]        [B, L, D_l]
              ↘      ↙
            FusionModule
                 ↓
           [B, K, D_fused]
                 ↓
            ActionHead
                 ↓
        Actions [B, action_dim]
```

## Related Code Files

### Files to Create
| Path | Purpose | Lines |
|------|---------|-------|
| `src/vla/models/__init__.py` | Exports | ~10 |
| `src/vla/models/vla_base.py` | VLA model | ~150 |
| `src/vla/models/vla_configs.py` | Config dataclasses | ~80 |
| `tests/unit/test_vla_model.py` | Model tests | ~100 |

## Implementation Steps

### Step 1: Implement vla_configs.py (30 min)
```python
"""Configuration dataclasses for VLA models."""
from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class VisionConfig:
    """Vision backbone configuration."""
    name: str = "timm_vit"
    model_name: str = "vit_base_patch16_224"
    pretrained: bool = True
    frozen: bool = True
    output_mode: Literal["cls", "spatial", "both"] = "spatial"
    proj_dim: Optional[int] = None


@dataclass
class LanguageConfig:
    """Language backbone configuration."""
    name: str = "gpt2"
    model_name: str = "gpt2"
    frozen: bool = True
    output_mode: Literal["last", "mean", "all"] = "mean"
    max_length: int = 77
    proj_dim: Optional[int] = None


@dataclass
class FusionConfig:
    """Fusion module configuration."""
    name: str = "perceiver_resampler"
    dim: int = 768
    num_latents: int = 64
    num_layers: int = 2
    num_heads: int = 8


@dataclass
class ActionConfig:
    """Action head configuration."""
    name: str = "discrete_action"
    action_dim: int = 7
    num_bins: int = 256
    hidden_dim: Optional[int] = None


@dataclass
class VLAConfig:
    """Complete VLA model configuration."""
    vision: VisionConfig = field(default_factory=VisionConfig)
    language: LanguageConfig = field(default_factory=LanguageConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    action: ActionConfig = field(default_factory=ActionConfig)

    # Training options
    freeze_vision: bool = True
    freeze_language: bool = True

    # Loss weights
    action_loss_weight: float = 1.0
    auxiliary_loss_weight: float = 0.0

    @classmethod
    def from_dict(cls, config_dict: dict) -> "VLAConfig":
        """Create config from dictionary."""
        return cls(
            vision=VisionConfig(**config_dict.get("vision", {})),
            language=LanguageConfig(**config_dict.get("language", {})),
            fusion=FusionConfig(**config_dict.get("fusion", {})),
            action=ActionConfig(**config_dict.get("action", {})),
            freeze_vision=config_dict.get("freeze_vision", True),
            freeze_language=config_dict.get("freeze_language", True),
        )
```

### Step 2: Implement vla_base.py (90 min)
```python
"""Base VLA model composing vision, language, fusion, and action components."""
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import asdict

from vla.registry import (
    VISION_REGISTRY,
    LANGUAGE_REGISTRY,
    FUSION_REGISTRY,
    ACTION_REGISTRY,
    MODEL_REGISTRY,
)
from vla.policy.action_utils import compute_action_loss
from .vla_configs import VLAConfig


@MODEL_REGISTRY.register("vla_base")
class VLAModel(nn.Module):
    """Vision-Language-Action model.

    Composes:
    - Vision backbone (timm/DINOv2/SigLIP)
    - Language encoder (GPT-2)
    - Fusion module (Perceiver/Cross-attention)
    - Action head (Discrete/Gaussian)

    Args:
        config: VLAConfig or dict with component configurations
    """

    def __init__(self, config: VLAConfig | dict):
        super().__init__()

        if isinstance(config, dict):
            config = VLAConfig.from_dict(config)
        self.config = config

        # Build components from config
        self.vision = self._build_vision(config.vision)
        self.language = self._build_language(config.language)
        self.fusion = self._build_fusion(config)
        self.action_head = self._build_action(config.action, config.fusion.dim)

        # Apply freezing
        if config.freeze_vision:
            self._freeze_module(self.vision)
        if config.freeze_language:
            self._freeze_module(self.language)

    def _build_vision(self, cfg) -> nn.Module:
        """Build vision backbone from config."""
        return VISION_REGISTRY.get(
            cfg.name,
            model_name=cfg.model_name,
            pretrained=cfg.pretrained,
            frozen=cfg.frozen,
            output_mode=cfg.output_mode,
            proj_dim=cfg.proj_dim,
        )

    def _build_language(self, cfg) -> nn.Module:
        """Build language encoder from config."""
        return LANGUAGE_REGISTRY.get(
            cfg.name,
            model_name=cfg.model_name,
            frozen=cfg.frozen,
            output_mode=cfg.output_mode,
            max_length=cfg.max_length,
            proj_dim=cfg.proj_dim,
        )

    def _build_fusion(self, cfg: VLAConfig) -> nn.Module:
        """Build fusion module from config."""
        vision_dim = cfg.vision.proj_dim or self.vision.embed_dim
        language_dim = cfg.language.proj_dim or self.language.embed_dim

        return FUSION_REGISTRY.get(
            cfg.fusion.name,
            dim=cfg.fusion.dim,
            num_latents=cfg.fusion.num_latents,
            num_layers=cfg.fusion.num_layers,
            num_heads=cfg.fusion.num_heads,
            vision_dim=vision_dim,
            language_dim=language_dim,
        )

    def _build_action(self, cfg, fusion_dim: int) -> nn.Module:
        """Build action head from config."""
        return ACTION_REGISTRY.get(
            cfg.name,
            input_dim=fusion_dim,
            action_dim=cfg.action_dim,
            num_bins=cfg.num_bins,
            hidden_dim=cfg.hidden_dim,
        )

    def _freeze_module(self, module: nn.Module):
        """Freeze all parameters in module."""
        for param in module.parameters():
            param.requires_grad = False

    def forward(
        self,
        images: torch.Tensor,
        texts: Optional[List[str]] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        target_actions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            images: [B, C, H, W] input images
            texts: List of instruction strings
            input_ids: Pre-tokenized input IDs [B, L]
            attention_mask: Attention mask [B, L]
            target_actions: [B, action_dim] for training
        Returns:
            Dictionary with "actions" and optionally "loss"
        """
        # Encode vision
        vision_features = self.vision(images)

        # Encode language
        if texts is not None:
            language_features = self.language(texts=texts)
        else:
            language_features = self.language(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # Fuse modalities
        fused_features = self.fusion(vision_features, language_features)

        # Predict actions
        actions, logits = self.action_head(fused_features, return_logits=True)

        output = {"actions": actions}

        # Compute loss if targets provided
        if target_actions is not None and logits is not None:
            loss = compute_action_loss(
                logits, target_actions,
                num_bins=self.config.action.num_bins,
            )
            output["loss"] = loss * self.config.action_loss_weight

        return output

    def predict(
        self,
        images: torch.Tensor,
        texts: List[str],
    ) -> torch.Tensor:
        """Inference-only forward pass.

        Args:
            images: [B, C, H, W] input images
            texts: List of instruction strings
        Returns:
            actions: [B, action_dim] predicted actions
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(images, texts=texts)
        return output["actions"]

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get list of trainable parameters."""
        return [p for p in self.parameters() if p.requires_grad]

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        state = {
            "config": asdict(self.config),
            "state_dict": self.state_dict(),
        }
        torch.save(state, path)

    @classmethod
    def load_checkpoint(cls, path: str, map_location: str = "cpu") -> "VLAModel":
        """Load model from checkpoint."""
        state = torch.load(path, map_location=map_location)
        model = cls(state["config"])
        model.load_state_dict(state["state_dict"])
        return model


@MODEL_REGISTRY.register("vla_temporal")
class TemporalVLAModel(VLAModel):
    """VLA model with multi-frame temporal processing."""

    def __init__(self, config: VLAConfig | dict, num_frames: int = 6):
        super().__init__(config)
        self.num_frames = num_frames

    def forward(
        self,
        image_sequence: List[torch.Tensor],
        texts: Optional[List[str]] = None,
        target_actions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with frame sequence.

        Args:
            image_sequence: List of [B, C, H, W] tensors
            texts: List of instruction strings
            target_actions: [B, action_dim] for training
        """
        # Encode each frame
        vision_features = [self.vision(img) for img in image_sequence]

        # Encode language once
        language_features = self.language(texts=texts) if texts else None

        # Temporal fusion
        fused = self.fusion(vision_features, language_features)

        # Predict actions
        actions, logits = self.action_head(fused, return_logits=True)

        output = {"actions": actions}
        if target_actions is not None and logits is not None:
            loss = compute_action_loss(logits, target_actions)
            output["loss"] = loss

        return output
```

### Step 3: Create __init__.py (10 min)
```python
"""VLA model implementations."""
from .vla_configs import (
    VisionConfig,
    LanguageConfig,
    FusionConfig,
    ActionConfig,
    VLAConfig,
)
from .vla_base import VLAModel, TemporalVLAModel

__all__ = [
    "VisionConfig",
    "LanguageConfig",
    "FusionConfig",
    "ActionConfig",
    "VLAConfig",
    "VLAModel",
    "TemporalVLAModel",
]
```

### Step 4: Write tests (60 min)
```python
"""Tests for VLA model."""
import pytest
import torch
from vla.models import VLAModel, VLAConfig, VisionConfig, LanguageConfig


class TestVLAModel:
    @pytest.fixture
    def small_config(self):
        return VLAConfig(
            vision=VisionConfig(
                name="timm_vit",
                model_name="vit_tiny_patch16_224",
                pretrained=False,
                frozen=False,
            ),
            language=LanguageConfig(
                name="gpt2",
                model_name="gpt2",
                frozen=False,
            ),
        )

    @pytest.fixture
    def dummy_batch(self):
        return {
            "images": torch.randn(2, 3, 224, 224),
            "texts": ["pick up block", "place on table"],
            "target_actions": torch.rand(2, 7) * 2 - 1,
        }

    def test_forward_shape(self, small_config, dummy_batch):
        model = VLAModel(small_config)
        output = model(
            images=dummy_batch["images"],
            texts=dummy_batch["texts"],
        )
        assert output["actions"].shape == (2, 7)

    def test_forward_with_loss(self, small_config, dummy_batch):
        model = VLAModel(small_config)
        output = model(
            images=dummy_batch["images"],
            texts=dummy_batch["texts"],
            target_actions=dummy_batch["target_actions"],
        )
        assert "loss" in output
        assert output["loss"].ndim == 0  # scalar

    def test_predict(self, small_config, dummy_batch):
        model = VLAModel(small_config)
        actions = model.predict(
            images=dummy_batch["images"],
            texts=dummy_batch["texts"],
        )
        assert actions.shape == (2, 7)

    def test_freezing(self, small_config):
        small_config.freeze_vision = True
        small_config.freeze_language = True
        model = VLAModel(small_config)

        for name, param in model.vision.named_parameters():
            assert not param.requires_grad, f"Vision {name} should be frozen"
        for name, param in model.language.named_parameters():
            assert not param.requires_grad, f"Language {name} should be frozen"

        # Fusion and action should be trainable
        fusion_trainable = any(p.requires_grad for p in model.fusion.parameters())
        action_trainable = any(p.requires_grad for p in model.action_head.parameters())
        assert fusion_trainable
        assert action_trainable

    def test_checkpoint_roundtrip(self, small_config, dummy_batch, tmp_path):
        model = VLAModel(small_config)
        output1 = model(dummy_batch["images"], texts=dummy_batch["texts"])

        # Save and load
        ckpt_path = tmp_path / "model.pt"
        model.save_checkpoint(str(ckpt_path))
        loaded_model = VLAModel.load_checkpoint(str(ckpt_path))

        output2 = loaded_model(dummy_batch["images"], texts=dummy_batch["texts"])
        assert torch.allclose(output1["actions"], output2["actions"])

    def test_config_from_dict(self):
        config_dict = {
            "vision": {"model_name": "vit_tiny_patch16_224", "pretrained": False},
            "language": {"model_name": "gpt2"},
            "fusion": {"num_latents": 32},
            "action": {"action_dim": 6},
        }
        config = VLAConfig.from_dict(config_dict)
        assert config.vision.model_name == "vit_tiny_patch16_224"
        assert config.fusion.num_latents == 32


class TestRegistry:
    def test_vla_registered(self):
        from vla.registry import MODEL_REGISTRY
        assert "vla_base" in MODEL_REGISTRY
        assert "vla_temporal" in MODEL_REGISTRY
```

## Todo List
- [ ] Implement VLAConfig and component configs
- [ ] Implement VLAModel with component composition
- [ ] Implement forward pass with loss computation
- [ ] Implement checkpoint save/load
- [ ] Implement TemporalVLAModel for multi-frame
- [ ] Register in MODEL_REGISTRY
- [ ] Write integration tests
- [ ] Test with real pretrained weights

## Success Criteria
1. VLAModel instantiates from config
2. End-to-end forward pass produces actions
3. Loss computed when targets provided
4. Checkpoint roundtrip preserves weights
5. Freezing properly disables gradients
6. All tests pass

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Component dimension mismatch | High | Validate dims in _build_fusion |
| OOM on full model | High | Default frozen backbones |
| Slow forward pass | Medium | Profile and optimize bottlenecks |

## Security Considerations
- Checkpoints contain model weights only, no code execution
- Config validation prevents invalid model instantiation

## Next Steps
- Phase 9: Hydra configuration for all components
