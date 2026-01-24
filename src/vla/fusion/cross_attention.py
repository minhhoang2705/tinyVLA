"""Cross-attention based fusion modules.

This module implements fusion mechanisms that use cross-attention to combine
vision and language modalities. The primary approach conditions vision features
on language instructions through interleaved self-attention and cross-attention.

Fusion Strategies:
    1. CrossAttentionFusion: Vision attends to language (language-conditioned vision)
    2. GatedFusion: Learnable gating mechanism to blend modalities

Example:
    >>> fusion = CrossAttentionFusion(dim=768, num_layers=4)
    >>> vision = torch.randn(2, 196, 768)
    >>> language = torch.randn(2, 16, 768)
    >>> conditioned_vision = fusion(vision, language)  # [2, 196, 768]
"""

from typing import Optional

import torch
import torch.nn as nn

from vla.nn import MLP, CrossAttention, MultiHeadAttention, RMSNorm
from vla.registry import FUSION_REGISTRY


@FUSION_REGISTRY.register("cross_attention_fusion")
class CrossAttentionFusion(nn.Module):
    """Language-conditioned vision processing via cross-attention.

    Vision features attend to language features to create instruction-aware
    visual representations. Uses interleaved self-attention and cross-attention
    layers similar to Perceiver IO and Flamingo.

    Args:
        dim: Feature dimension
        num_layers: Number of cross-attention layers
        num_heads: Attention heads per layer
        vision_dim: Vision input dimension (if None, assumes same as dim)
        language_dim: Language input dimension (if None, assumes same as dim)
        dropout: Dropout rate

    Example:
        >>> fusion = CrossAttentionFusion(
        ...     dim=768, num_layers=4, vision_dim=1024, language_dim=512
        ... )
        >>> vision = torch.randn(2, 196, 1024)
        >>> language = torch.randn(2, 16, 512)
        >>> out = fusion(vision, language)
        >>> print(out.shape)
        torch.Size([2, 196, 768])
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

        # Input projections to align dimensions
        vision_dim = vision_dim or dim
        language_dim = language_dim or dim
        self.vision_proj = nn.Linear(vision_dim, dim) if vision_dim != dim else nn.Identity()
        self.lang_proj = nn.Linear(language_dim, dim) if language_dim != dim else nn.Identity()

        # Interleaved self-attention and cross-attention layers
        self.layers = nn.ModuleList(
            [CrossAttentionLayer(dim, num_heads, dropout) for _ in range(num_layers)]
        )

        self.norm = RMSNorm(dim)

    def forward(
        self,
        vision_features: torch.Tensor,
        language_features: torch.Tensor,
    ) -> torch.Tensor:
        """Apply language-conditioned processing to vision.

        Args:
            vision_features: Vision tokens [B, N_v, D_v]
            language_features: Language tokens [B, N_l, D_l]

        Returns:
            Conditioned vision features [B, N_v, dim]

        Raises:
            ValueError: If input tensors have invalid shapes
        """
        if vision_features.ndim != 3:
            raise ValueError(
                f"Expected vision_features to be 3D [B, N, D], got {vision_features.shape}"
            )
        if language_features.ndim != 3:
            raise ValueError(
                f"Expected language_features to be 3D [B, L, D], got {language_features.shape}"
            )

        # Project to common dimension
        vision = self.vision_proj(vision_features)
        language = self.lang_proj(language_features)

        # Apply cross-attention layers
        for layer in self.layers:
            vision = layer(vision, language)

        return self.norm(vision)  # type: ignore[no-any-return]


class CrossAttentionLayer(nn.Module):
    """Self-attention + cross-attention block.

    Architecture:
        1. Self-attention on vision features
        2. Cross-attention: vision attends to language
        3. Feed-forward network
        All with residual connections and pre-normalization.

    Args:
        dim: Feature dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        # Self-attention on vision
        self.norm1 = RMSNorm(dim)
        self.self_attn = MultiHeadAttention(dim, num_heads, dropout)

        # Cross-attention: vision queries, language keys/values
        self.norm2 = RMSNorm(dim)
        self.cross_attn = CrossAttention(dim, dim, num_heads, dropout)

        # Feed-forward
        self.norm3 = RMSNorm(dim)
        self.mlp = MLP(dim, dropout=dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Apply self-attention, cross-attention, and MLP.

        Args:
            x: Vision tokens [B, N, dim]
            context: Language tokens [B, L, dim]

        Returns:
            Updated vision tokens [B, N, dim]
        """
        # Self-attention
        x = x + self.self_attn(self.norm1(x))
        # Cross-attention: vision attends to language
        x = x + self.cross_attn(self.norm2(x), context)
        # Feed-forward
        x = x + self.mlp(self.norm3(x))
        return x


@FUSION_REGISTRY.register("gated_fusion")
class GatedFusion(nn.Module):
    """Gated fusion with learnable modality weighting.

    Uses a gating mechanism to dynamically balance vision and language features.
    The gate is computed from concatenated features and applied element-wise:
        output = gate * vision + (1 - gate) * language

    Args:
        dim: Feature dimension
        vision_dim: Vision input dimension (if None, assumes same as dim)
        language_dim: Language input dimension (if None, assumes same as dim)

    Example:
        >>> fusion = GatedFusion(dim=768, vision_dim=1024, language_dim=512)
        >>> vision = torch.randn(2, 196, 1024)
        >>> language = torch.randn(2, 16, 512)
        >>> out = fusion(vision, language)
        >>> print(out.shape)
        torch.Size([2, 196, 768])
    """

    def __init__(
        self,
        dim: int = 768,
        vision_dim: Optional[int] = None,
        language_dim: Optional[int] = None,
    ):
        super().__init__()

        vision_dim = vision_dim or dim
        language_dim = language_dim or dim

        self.vision_proj = nn.Linear(vision_dim, dim) if vision_dim != dim else nn.Identity()
        self.lang_proj = nn.Linear(language_dim, dim) if language_dim != dim else nn.Identity()

        # Gating mechanism: learns to weight vision vs language
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
        """Fuse vision and language with learnable gating.

        Args:
            vision_features: Vision tokens [B, N_v, D_v]
            language_features: Language tokens [B, N_l, D_l]

        Returns:
            Fused features [B, N_v, dim]

        Raises:
            ValueError: If input tensors have invalid shapes
        """
        if vision_features.ndim != 3:
            raise ValueError(
                f"Expected vision_features to be 3D [B, N, D], got {vision_features.shape}"
            )
        if language_features.ndim != 3:
            raise ValueError(
                f"Expected language_features to be 3D [B, L, D], got {language_features.shape}"
            )

        # Project to common dimension
        vision = self.vision_proj(vision_features)

        # Pool language to match vision sequence length
        lang = self.lang_proj(language_features)
        if lang.size(1) != vision.size(1):
            # Mean pooling across language sequence, then expand to match vision
            lang = lang.mean(dim=1, keepdim=True).expand(-1, vision.size(1), -1)

        # Compute gate from concatenated features
        combined = torch.cat([vision, lang], dim=-1)  # [B, N, 2*dim]
        gate = self.gate(combined)  # [B, N, dim]

        # Gated combination: blend vision and language
        fused = gate * vision + (1 - gate) * lang
        return self.norm(fused)  # type: ignore[no-any-return]
