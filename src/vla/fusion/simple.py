"""Simple fusion baselines.

This module implements basic concatenation-based fusion strategies that serve
as baselines for comparison with more sophisticated fusion mechanisms.

Fusion Strategies:
    1. ConcatFusion: Concatenate language and vision tokens
    2. PrependFusion: Prepend language tokens before vision (GPT-4V style)

These approaches are simpler than cross-attention but can be effective when
followed by transformer layers that process the combined sequence.

Example:
    >>> concat_fusion = ConcatFusion(dim=768)
    >>> vision = torch.randn(2, 196, 768)
    >>> language = torch.randn(2, 16, 768)
    >>> fused = concat_fusion(vision, language)  # [2, 212, 768]
"""

from typing import Optional

import torch
import torch.nn as nn

from vla.nn import RMSNorm
from vla.registry import FUSION_REGISTRY


@FUSION_REGISTRY.register("concat_fusion")
class ConcatFusion(nn.Module):
    """Simple concatenation fusion.

    Concatenates language and vision tokens along the sequence dimension.
    Output can be processed by downstream transformer layers.

    Output sequence: [language_tokens, vision_tokens]

    Args:
        dim: Feature dimension
        vision_dim: Vision input dimension (if None, assumes same as dim)
        language_dim: Language input dimension (if None, assumes same as dim)

    Example:
        >>> fusion = ConcatFusion(dim=768, vision_dim=1024, language_dim=512)
        >>> vision = torch.randn(2, 196, 1024)
        >>> language = torch.randn(2, 16, 512)
        >>> out = fusion(vision, language)
        >>> print(out.shape)
        torch.Size([2, 212, 768])  # 16 + 196 tokens
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
        self.norm = RMSNorm(dim)

    def forward(
        self,
        vision_features: torch.Tensor,
        language_features: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate vision and language tokens.

        Args:
            vision_features: Vision tokens [B, N_v, D_v]
            language_features: Language tokens [B, N_l, D_l]

        Returns:
            Concatenated features [B, N_v + N_l, dim]

        Raises:
            ValueError: If input tensors have invalid shapes or mismatched batch sizes
        """
        if vision_features.ndim != 3:
            raise ValueError(
                f"Expected vision_features to be 3D [B, N, D], got {vision_features.shape}"
            )
        if language_features.ndim != 3:
            raise ValueError(
                f"Expected language_features to be 3D [B, L, D], got {language_features.shape}"
            )
        if vision_features.size(0) != language_features.size(0):
            raise ValueError(
                f"Batch size mismatch: vision {vision_features.size(0)} vs language {language_features.size(0)}"
            )

        # Project to common dimension
        vision = self.vision_proj(vision_features)
        language = self.lang_proj(language_features)

        # Concatenate: [language, vision]
        fused = torch.cat([language, vision], dim=1)
        return self.norm(fused)  # type: ignore[no-any-return]


@FUSION_REGISTRY.register("prepend_fusion")
class PrependFusion(nn.Module):
    """Prepend language tokens to vision tokens (GPT-4V style).

    Similar to ConcatFusion but adds a learnable separator token between
    language and vision sequences. This can help the model distinguish
    modality boundaries.

    Output sequence: [language_tokens, separator_token, vision_tokens]

    Args:
        dim: Feature dimension
        vision_dim: Vision input dimension (if None, assumes same as dim)
        language_dim: Language input dimension (if None, assumes same as dim)

    Example:
        >>> fusion = PrependFusion(dim=768, vision_dim=1024, language_dim=512)
        >>> vision = torch.randn(2, 196, 1024)
        >>> language = torch.randn(2, 16, 512)
        >>> out = fusion(vision, language)
        >>> print(out.shape)
        torch.Size([2, 213, 768])  # 16 + 1 (separator) + 196 tokens
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

        # Learnable separator token between language and vision
        self.separator = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.norm = RMSNorm(dim)

    def forward(
        self,
        vision_features: torch.Tensor,
        language_features: torch.Tensor,
    ) -> torch.Tensor:
        """Prepend language tokens before vision tokens with separator.

        Args:
            vision_features: Vision tokens [B, N_v, D_v]
            language_features: Language tokens [B, N_l, D_l]

        Returns:
            Concatenated features [B, N_v + N_l + 1, dim]

        Raises:
            ValueError: If input tensors have invalid shapes or mismatched batch sizes
        """
        if vision_features.ndim != 3:
            raise ValueError(
                f"Expected vision_features to be 3D [B, N, D], got {vision_features.shape}"
            )
        if language_features.ndim != 3:
            raise ValueError(
                f"Expected language_features to be 3D [B, L, D], got {language_features.shape}"
            )
        if vision_features.size(0) != language_features.size(0):
            raise ValueError(
                f"Batch size mismatch: vision {vision_features.size(0)} vs language {language_features.size(0)}"
            )

        B = vision_features.size(0)

        # Project to common dimension
        vision = self.vision_proj(vision_features)
        language = self.lang_proj(language_features)

        # Expand separator for batch
        separator = self.separator.expand(B, -1, -1)

        # Concatenate: [language, separator, vision]
        fused = torch.cat([language, separator, vision], dim=1)
        return self.norm(fused)  # type: ignore[no-any-return]
