"""Perceiver Resampler for multimodal fusion.

This module implements Perceiver-based fusion mechanisms as used in Flamingo and RT-2.
The Perceiver Resampler compresses variable-length vision and language inputs into
a fixed-size latent representation through learnable query vectors.

Architecture:
    1. Learnable latent queries [num_latents, dim]
    2. Cross-attention: queries attend to concatenated vision+language features
    3. Multiple cross-attention layers with residual connections
    4. Output: Fixed-size latent representation [B, num_latents, dim]

Example:
    >>> perceiver = PerceiverResampler(dim=768, num_latents=64)
    >>> vision = torch.randn(2, 196, 768)  # [B, N_patches, D]
    >>> language = torch.randn(2, 16, 768)  # [B, L, D]
    >>> latents = perceiver(vision, language)  # [2, 64, 768]
"""

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from vla.nn import MLP, CrossAttention, RMSNorm
from vla.registry import FUSION_REGISTRY


@FUSION_REGISTRY.register("perceiver_resampler")
class PerceiverResampler(nn.Module):
    """Perceiver Resampler for vision-language fusion.

    Compresses variable-length vision+language inputs into fixed-size latents
    using learnable query vectors and cross-attention.

    Args:
        dim: Latent dimension
        num_latents: Number of latent queries (output sequence length)
        num_layers: Number of cross-attention layers
        num_heads: Attention heads per layer
        vision_dim: Vision input dimension (if None, assumes same as dim)
        language_dim: Language input dimension (if None, assumes same as dim)
        dropout: Dropout rate
        use_gradient_checkpointing: If True, recompute activations during backward
            to reduce peak VRAM (~40-50% savings). Uses use_reentrant=False for
            compatibility with PyTorch >= 2.0, AMP, and PyTorch Lightning.
            Only active during training; eval mode uses normal forward pass.

    Example:
        >>> perceiver = PerceiverResampler(
        ...     dim=768, num_latents=64, num_layers=2, num_heads=8,
        ...     vision_dim=1024, language_dim=512
        ... )
        >>> vision = torch.randn(2, 196, 1024)
        >>> language = torch.randn(2, 16, 512)
        >>> latents = perceiver(vision, language)
        >>> print(latents.shape)
        torch.Size([2, 64, 768])
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
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_latents = num_latents
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Learnable latent queries [1, num_latents, dim]
        self.latents = nn.Parameter(torch.randn(1, num_latents, dim) * 0.02)

        # Input projections to align dimensions
        vision_dim = vision_dim or dim
        language_dim = language_dim or dim
        self.vision_proj = nn.Linear(vision_dim, dim) if vision_dim != dim else nn.Identity()
        self.lang_proj = nn.Linear(language_dim, dim) if language_dim != dim else nn.Identity()

        # Cross-attention layers
        self.layers = nn.ModuleList(
            [PerceiverBlock(dim, num_heads, dropout) for _ in range(num_layers)]
        )

        self.norm = RMSNorm(dim)

    def forward(
        self,
        vision_features: torch.Tensor,
        language_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fuse vision and language into latent representation.

        Args:
            vision_features: Vision tokens [B, N_v, D_v]
            language_features: Optional language tokens [B, N_l, D_l]

        Returns:
            Latent features [B, num_latents, dim]

        Raises:
            ValueError: If vision_features has invalid shape
        """
        if vision_features.ndim != 3:
            raise ValueError(
                f"Expected vision_features to be 3D [B, N, D], got {vision_features.shape}"
            )

        B = vision_features.size(0)

        # Project inputs to common dimension
        vision = self.vision_proj(vision_features)

        # Combine vision and language as context for cross-attention
        if language_features is not None:
            if language_features.ndim != 3:
                raise ValueError(
                    f"Expected language_features to be 3D [B, L, D], got {language_features.shape}"
                )
            language = self.lang_proj(language_features)
            context = torch.cat([vision, language], dim=1)  # [B, N_v+N_l, dim]
        else:
            context = vision

        # Expand latents for batch [B, num_latents, dim]
        latents = self.latents.expand(B, -1, -1)

        # Cross-attention layers: latents attend to context
        # use_reentrant=False: required for PyTorch >= 2.0 + AMP + Lightning
        # self.training guard: inference doesn't need backward, skip checkpointing
        for layer in self.layers:
            if self.use_gradient_checkpointing and self.training:
                latents = grad_checkpoint(layer, latents, context, use_reentrant=False)
            else:
                latents = layer(latents, context)

        return self.norm(latents)  # type: ignore[no-any-return]


class PerceiverBlock(nn.Module):
    """Single Perceiver cross-attention block.

    Architecture:
        1. Cross-attention: latents (queries) attend to context (keys/values)
        2. Residual connection
        3. MLP with residual connection

    Args:
        dim: Feature dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.cross_attn = CrossAttention(dim, dim, num_heads, dropout)
        self.norm2 = RMSNorm(dim)
        self.mlp = MLP(dim, dropout=dropout)

    def forward(self, latents: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Apply cross-attention and MLP with residuals.

        Args:
            latents: Query tokens [B, num_latents, dim]
            context: Context tokens (vision+language) [B, N, dim]

        Returns:
            Updated latents [B, num_latents, dim]
        """
        # Cross-attention: latents attend to context
        latents = latents + self.cross_attn(self.norm1(latents), context)
        # Feed-forward
        latents = latents + self.mlp(self.norm2(latents))
        return latents


@FUSION_REGISTRY.register("temporal_perceiver")
class TemporalPerceiverResampler(nn.Module):
    """Perceiver Resampler with temporal awareness for multi-frame inputs.

    Extends PerceiverResampler to handle video sequences by adding temporal
    embeddings to distinguish different frames before fusion.

    Args:
        dim: Latent dimension
        num_latents: Number of latent queries
        num_layers: Number of cross-attention layers
        num_heads: Attention heads per layer
        max_frames: Maximum number of frames (for temporal embeddings)
        vision_dim: Vision input dimension
        language_dim: Language input dimension
        dropout: Dropout rate
        use_gradient_checkpointing: Forwarded to inner PerceiverResampler.

    Example:
        >>> temporal_perceiver = TemporalPerceiverResampler(
        ...     dim=768, num_latents=64, max_frames=16
        ... )
        >>> frames = [torch.randn(2, 196, 768) for _ in range(8)]
        >>> language = torch.randn(2, 16, 768)
        >>> latents = temporal_perceiver(frames, language)
        >>> print(latents.shape)
        torch.Size([2, 64, 768])
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
        dropout: float = 0.1,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.perceiver = PerceiverResampler(
            dim=dim,
            num_latents=num_latents,
            num_layers=num_layers,
            num_heads=num_heads,
            vision_dim=vision_dim,
            language_dim=language_dim,
            dropout=dropout,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
        # Temporal embeddings to distinguish frames
        self.temporal_embed = nn.Embedding(max_frames, dim)

    def forward(
        self,
        vision_frames: list[torch.Tensor],
        language_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Process multiple vision frames with temporal embeddings.

        Args:
            vision_frames: List of vision tensors [B, N, D], one per frame
            language_features: Optional language tokens [B, L, D]

        Returns:
            Fused latents [B, num_latents, dim]

        Raises:
            ValueError: If vision_frames list is empty or frames have inconsistent shapes
        """
        if not vision_frames:
            raise ValueError("vision_frames list cannot be empty")

        B = vision_frames[0].size(0)

        # Validate all frames have same batch size
        for i, frame in enumerate(vision_frames):
            if frame.size(0) != B:
                raise ValueError(
                    f"Inconsistent batch size: frame 0 has {B}, frame {i} has {frame.size(0)}"
                )

        # Add temporal embeddings to each frame
        temporal_vision = []
        for t, frame in enumerate(vision_frames):
            # Get temporal embedding for frame t
            temp_emb = (
                self.temporal_embed.weight[t : t + 1].unsqueeze(0).expand(B, frame.size(1), -1)
            )
            # Project frame if needed
            frame_proj = self.perceiver.vision_proj(frame)
            # Add temporal information
            temporal_vision.append(frame_proj + temp_emb)

        # Concatenate all frames along sequence dimension
        vision_combined = torch.cat(temporal_vision, dim=1)  # [B, T*N, dim]

        # Process through perceiver (bypassing vision_proj since already projected)
        if language_features is not None:
            language = self.perceiver.lang_proj(language_features)
            context = torch.cat([vision_combined, language], dim=1)
        else:
            context = vision_combined

        # Expand latents and process (respects use_gradient_checkpointing from inner perceiver)
        latents = self.perceiver.latents.expand(B, -1, -1)
        for layer in self.perceiver.layers:
            if self.perceiver.use_gradient_checkpointing and self.perceiver.training:
                latents = grad_checkpoint(layer, latents, context, use_reentrant=False)
            else:
                latents = layer(latents, context)

        return self.perceiver.norm(latents)  # type: ignore[no-any-return]
