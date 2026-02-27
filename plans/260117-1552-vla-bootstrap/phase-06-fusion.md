# Phase 06: Fusion Mechanisms

## Context Links
- [VLA Architectures](../reports/researcher-260118-vla-architectures.md) - Perceiver Resampler
- [Tech Stack](../../docs/tech-stack.md) - Fusion mechanism section

## Overview
| Field | Value |
|-------|-------|
| Priority | P1 - Core Component |
| Status | ✅ Complete |
| Effort | 4h |
| Dependencies | Phases 2, 3, 4, 5 |

Implement fusion modules to combine vision and language features. Perceiver Resampler (Flamingo/RT-2 style) as primary approach with cross-attention alternative.

## Key Insights
- Perceiver Resampler: Fixed-size latent bottleneck (e.g., 64 tokens)
- Reduces compute for multi-frame inputs
- Cross-attention enables language-conditioned vision processing
- Prepend/concat fusion simpler but less expressive

## Requirements

### Functional
- FR-01: Perceiver Resampler with configurable latent size
- FR-02: Cross-attention fusion (vision queries, language keys/values)
- FR-03: Simple concatenation fusion (baseline)
- FR-04: Support multi-frame vision inputs
- FR-05: Temporal attention for frame sequences

### Non-Functional
- NFR-01: Perceiver adds <10M params
- NFR-02: Linear scaling with input sequence length

## Architecture

```
src/vla/fusion/
├── __init__.py
├── perceiver.py         # Perceiver Resampler
├── cross_attention.py   # Cross-attention fusion
└── simple.py            # Concat/prepend fusion
```

**Perceiver Resampler Flow:**
```
Vision [B, N, D_v]  +  Language [B, L, D_l]
              ↓
         Project to D
              ↓
    Learnable Queries [1, K, D]
              ↓
    Cross-Attention (Q=queries, KV=vision+language)
              ↓
    Latent Features [B, K, D]
```

## Related Code Files

### Files to Create
| Path | Purpose | Lines |
|------|---------|-------|
| `src/vla/fusion/__init__.py` | Exports | ~15 |
| `src/vla/fusion/perceiver.py` | Perceiver Resampler | ~100 |
| `src/vla/fusion/cross_attention.py` | Cross-attention | ~80 |
| `src/vla/fusion/simple.py` | Simple fusion | ~60 |
| `tests/unit/test_fusion.py` | Fusion tests | ~100 |

## Implementation Steps

### Step 1: Implement perceiver.py (60 min)
```python
"""Perceiver Resampler for multimodal fusion."""
import torch
import torch.nn as nn
from typing import Optional

from vla.nn import MultiHeadAttention, CrossAttention, MLP, RMSNorm
from vla.registry import FUSION_REGISTRY


@FUSION_REGISTRY.register("perceiver_resampler")
class PerceiverResampler(nn.Module):
    """Perceiver Resampler for vision-language fusion.

    Compresses variable-length vision+language inputs into fixed-size latents.

    Args:
        dim: Latent dimension
        num_latents: Number of latent queries
        num_layers: Number of cross-attention layers
        num_heads: Attention heads per layer
        vision_dim: Vision input dimension
        language_dim: Language input dimension
        dropout: Dropout rate
    """

    def __init__(
        self,
        dim: int = 768,
        num_latents: int = 64,
        num_layers: int = 2,
        num_heads: int = 8,
        vision_dim: Optional[int] = None,
        language_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_latents = num_latents

        # Learnable latent queries
        self.latents = nn.Parameter(torch.randn(1, num_latents, dim) * 0.02)

        # Input projections
        self.vision_proj = nn.Linear(vision_dim, dim) if vision_dim != dim else nn.Identity()
        self.lang_proj = nn.Linear(language_dim, dim) if language_dim != dim else nn.Identity()

        # Cross-attention layers
        self.layers = nn.ModuleList([
            PerceiverBlock(dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.norm = RMSNorm(dim)

    def forward(
        self,
        vision_features: torch.Tensor,
        language_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fuse vision and language into latent representation.

        Args:
            vision_features: [B, N_v, D_v]
            language_features: Optional [B, N_l, D_l]
        Returns:
            Latent features [B, num_latents, dim]
        """
        B = vision_features.size(0)

        # Project inputs
        vision = self.vision_proj(vision_features)

        # Combine vision and language as context
        if language_features is not None:
            language = self.lang_proj(language_features)
            context = torch.cat([vision, language], dim=1)
        else:
            context = vision

        # Expand latents for batch
        latents = self.latents.expand(B, -1, -1)

        # Cross-attention layers
        for layer in self.layers:
            latents = layer(latents, context)

        return self.norm(latents)


class PerceiverBlock(nn.Module):
    """Single Perceiver cross-attention block."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.cross_attn = CrossAttention(dim, dim, num_heads, dropout)
        self.norm2 = RMSNorm(dim)
        self.mlp = MLP(dim, dropout=dropout)

    def forward(self, latents: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        latents = latents + self.cross_attn(self.norm1(latents), context)
        latents = latents + self.mlp(self.norm2(latents))
        return latents


@FUSION_REGISTRY.register("temporal_perceiver")
class TemporalPerceiverResampler(nn.Module):
    """Perceiver Resampler with temporal awareness for multi-frame inputs.

    Adds temporal embeddings to distinguish frames before fusion.
    """

    def __init__(
        self,
        dim: int = 768,
        num_latents: int = 64,
        num_layers: int = 2,
        num_heads: int = 8,
        max_frames: int = 16,
        vision_dim: Optional[int] = None,
        language_dim: Optional[int] = None,
    ):
        super().__init__()
        self.perceiver = PerceiverResampler(
            dim, num_latents, num_layers, num_heads, vision_dim, language_dim
        )
        self.temporal_embed = nn.Embedding(max_frames, dim)

    def forward(
        self,
        vision_frames: list[torch.Tensor],
        language_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Process multiple vision frames with temporal embeddings.

        Args:
            vision_frames: List of [B, N, D] tensors, one per frame
            language_features: Optional [B, L, D]
        Returns:
            Fused latents [B, num_latents, dim]
        """
        B = vision_frames[0].size(0)
        device = vision_frames[0].device

        # Add temporal embeddings
        temporal_vision = []
        for t, frame in enumerate(vision_frames):
            temp_emb = self.temporal_embed.weight[t:t+1].unsqueeze(0).expand(B, frame.size(1), -1)
            # Project frame first if needed
            frame_proj = self.perceiver.vision_proj(frame)
            temporal_vision.append(frame_proj + temp_emb)

        # Concatenate all frames
        vision_combined = torch.cat(temporal_vision, dim=1)

        return self.perceiver(vision_combined, language_features)
```

### Step 2: Implement cross_attention.py (45 min)
```python
"""Cross-attention based fusion modules."""
import torch
import torch.nn as nn
from typing import Optional

from vla.nn import CrossAttention, MLP, RMSNorm
from vla.registry import FUSION_REGISTRY


@FUSION_REGISTRY.register("cross_attention_fusion")
class CrossAttentionFusion(nn.Module):
    """Language-conditioned vision processing via cross-attention.

    Vision features attend to language features to create
    instruction-aware visual representations.
    """

    def __init__(
        self,
        dim: int = 768,
        num_layers: int = 4,
        num_heads: int = 8,
        vision_dim: Optional[int] = None,
        language_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Input projections
        self.vision_proj = nn.Linear(vision_dim, dim) if vision_dim and vision_dim != dim else nn.Identity()
        self.lang_proj = nn.Linear(language_dim, dim) if language_dim and language_dim != dim else nn.Identity()

        # Interleaved self-attention and cross-attention
        self.layers = nn.ModuleList([
            CrossAttentionLayer(dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.norm = RMSNorm(dim)

    def forward(
        self,
        vision_features: torch.Tensor,
        language_features: torch.Tensor,
    ) -> torch.Tensor:
        """Apply language-conditioned processing to vision.

        Args:
            vision_features: [B, N_v, D_v]
            language_features: [B, N_l, D_l]
        Returns:
            Conditioned vision features [B, N_v, dim]
        """
        vision = self.vision_proj(vision_features)
        language = self.lang_proj(language_features)

        for layer in self.layers:
            vision = layer(vision, language)

        return self.norm(vision)


class CrossAttentionLayer(nn.Module):
    """Self-attention + cross-attention block."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        from vla.nn import MultiHeadAttention

        self.norm1 = RMSNorm(dim)
        self.self_attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm2 = RMSNorm(dim)
        self.cross_attn = CrossAttention(dim, dim, num_heads, dropout)
        self.norm3 = RMSNorm(dim)
        self.mlp = MLP(dim, dropout=dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), context)
        x = x + self.mlp(self.norm3(x))
        return x


@FUSION_REGISTRY.register("gated_fusion")
class GatedFusion(nn.Module):
    """Gated fusion with learnable modality weighting."""

    def __init__(
        self,
        dim: int = 768,
        vision_dim: Optional[int] = None,
        language_dim: Optional[int] = None,
    ):
        super().__init__()

        self.vision_proj = nn.Linear(vision_dim, dim) if vision_dim and vision_dim != dim else nn.Identity()
        self.lang_proj = nn.Linear(language_dim, dim) if language_dim and language_dim != dim else nn.Identity()

        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )
        self.norm = RMSNorm(dim)

    def forward(
        self,
        vision_features: torch.Tensor,
        language_features: torch.Tensor,
    ) -> torch.Tensor:
        vision = self.vision_proj(vision_features)

        # Pool language to match vision sequence length
        lang = self.lang_proj(language_features)
        if lang.size(1) != vision.size(1):
            lang = lang.mean(dim=1, keepdim=True).expand(-1, vision.size(1), -1)

        # Gated combination
        combined = torch.cat([vision, lang], dim=-1)
        gate = self.gate(combined)

        fused = gate * vision + (1 - gate) * lang
        return self.norm(fused)
```

### Step 3: Implement simple.py (30 min)
```python
"""Simple fusion baselines."""
import torch
import torch.nn as nn
from typing import Optional

from vla.nn import RMSNorm
from vla.registry import FUSION_REGISTRY


@FUSION_REGISTRY.register("concat_fusion")
class ConcatFusion(nn.Module):
    """Simple concatenation fusion."""

    def __init__(
        self,
        dim: int = 768,
        vision_dim: Optional[int] = None,
        language_dim: Optional[int] = None,
    ):
        super().__init__()

        self.vision_proj = nn.Linear(vision_dim, dim) if vision_dim and vision_dim != dim else nn.Identity()
        self.lang_proj = nn.Linear(language_dim, dim) if language_dim and language_dim != dim else nn.Identity()
        self.norm = RMSNorm(dim)

    def forward(
        self,
        vision_features: torch.Tensor,
        language_features: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate vision and language tokens."""
        vision = self.vision_proj(vision_features)
        language = self.lang_proj(language_features)
        return self.norm(torch.cat([language, vision], dim=1))


@FUSION_REGISTRY.register("prepend_fusion")
class PrependFusion(nn.Module):
    """Prepend language tokens to vision tokens (GPT-4V style)."""

    def __init__(
        self,
        dim: int = 768,
        vision_dim: Optional[int] = None,
        language_dim: Optional[int] = None,
    ):
        super().__init__()

        self.vision_proj = nn.Linear(vision_dim, dim) if vision_dim and vision_dim != dim else nn.Identity()
        self.lang_proj = nn.Linear(language_dim, dim) if language_dim and language_dim != dim else nn.Identity()

        # Learnable separator token
        self.separator = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.norm = RMSNorm(dim)

    def forward(
        self,
        vision_features: torch.Tensor,
        language_features: torch.Tensor,
    ) -> torch.Tensor:
        """Prepend language tokens before vision tokens."""
        B = vision_features.size(0)

        vision = self.vision_proj(vision_features)
        language = self.lang_proj(language_features)
        separator = self.separator.expand(B, -1, -1)

        # [language, sep, vision]
        return self.norm(torch.cat([language, separator, vision], dim=1))
```

### Step 4: Create __init__.py (10 min)
```python
"""Fusion modules for multimodal combination."""
from .perceiver import PerceiverResampler, TemporalPerceiverResampler
from .cross_attention import CrossAttentionFusion, GatedFusion
from .simple import ConcatFusion, PrependFusion

__all__ = [
    "PerceiverResampler",
    "TemporalPerceiverResampler",
    "CrossAttentionFusion",
    "GatedFusion",
    "ConcatFusion",
    "PrependFusion",
]
```

### Step 5: Write tests (45 min)
```python
"""Tests for fusion modules."""
import pytest
import torch
from vla.fusion import (
    PerceiverResampler,
    TemporalPerceiverResampler,
    CrossAttentionFusion,
    ConcatFusion,
)
from vla.registry import FUSION_REGISTRY


class TestPerceiverResampler:
    def test_basic_fusion(self):
        fusion = PerceiverResampler(
            dim=256,
            num_latents=32,
            vision_dim=384,
            language_dim=512,
        )
        vision = torch.randn(2, 196, 384)
        language = torch.randn(2, 16, 512)

        out = fusion(vision, language)
        assert out.shape == (2, 32, 256)

    def test_vision_only(self):
        fusion = PerceiverResampler(dim=256, num_latents=32, vision_dim=256)
        vision = torch.randn(2, 196, 256)

        out = fusion(vision)
        assert out.shape == (2, 32, 256)

    def test_registry(self):
        assert "perceiver_resampler" in FUSION_REGISTRY


class TestTemporalPerceiver:
    def test_multi_frame(self):
        fusion = TemporalPerceiverResampler(
            dim=256,
            num_latents=32,
            max_frames=8,
            vision_dim=256,
        )
        frames = [torch.randn(2, 196, 256) for _ in range(6)]

        out = fusion(frames)
        assert out.shape == (2, 32, 256)


class TestCrossAttentionFusion:
    def test_conditioned_vision(self):
        fusion = CrossAttentionFusion(
            dim=256,
            num_layers=2,
            vision_dim=384,
            language_dim=512,
        )
        vision = torch.randn(2, 196, 384)
        language = torch.randn(2, 16, 512)

        out = fusion(vision, language)
        assert out.shape == (2, 196, 256)


class TestConcatFusion:
    def test_concat_output_size(self):
        fusion = ConcatFusion(dim=256, vision_dim=256, language_dim=256)
        vision = torch.randn(2, 196, 256)
        language = torch.randn(2, 16, 256)

        out = fusion(vision, language)
        assert out.shape == (2, 196 + 16, 256)
```

## Todo List
- [ ] Implement PerceiverResampler with latent queries
- [ ] Implement TemporalPerceiverResampler for multi-frame
- [ ] Implement CrossAttentionFusion
- [ ] Implement GatedFusion
- [ ] Implement ConcatFusion and PrependFusion
- [ ] Register all in FUSION_REGISTRY
- [ ] Write comprehensive tests
- [ ] Verify gradient flow through fusion

## Success Criteria
1. PerceiverResampler reduces to fixed latent size
2. Temporal fusion handles variable frame counts
3. Cross-attention properly conditions vision on language
4. All fusion types produce correct output shapes
5. Tests pass with coverage >90%

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Memory growth with frames | High | Use gradient checkpointing |
| Attention over long sequences | Medium | Perceiver limits sequence length |
| Dimension mismatches | Medium | Projection layers handle mismatches |

## Security Considerations
- No external data dependencies
- Deterministic with fixed seed

## Next Steps
- Phase 7: Action heads for policy output
