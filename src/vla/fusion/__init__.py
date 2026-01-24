"""Fusion modules for multimodal combination.

This package provides various fusion mechanisms for combining vision and language
features in Vision-Language-Action models:

Perceiver-based Fusion:
    - PerceiverResampler: Fixed-size latent bottleneck (Flamingo/RT-2 style)
    - TemporalPerceiverResampler: Multi-frame fusion with temporal embeddings

Cross-attention Fusion:
    - CrossAttentionFusion: Language-conditioned vision processing
    - GatedFusion: Learnable gating for modality weighting

Simple Fusion (Baselines):
    - ConcatFusion: Concatenate language and vision tokens
    - PrependFusion: Prepend language tokens with separator

Example:
    >>> from vla.fusion import PerceiverResampler
    >>> perceiver = PerceiverResampler(dim=768, num_latents=64)
    >>> vision = torch.randn(2, 196, 768)
    >>> language = torch.randn(2, 16, 768)
    >>> latents = perceiver(vision, language)  # [2, 64, 768]
"""

from .cross_attention import CrossAttentionFusion, GatedFusion
from .perceiver import PerceiverResampler, TemporalPerceiverResampler
from .simple import ConcatFusion, PrependFusion

__all__ = [
    # Perceiver-based
    "PerceiverResampler",
    "TemporalPerceiverResampler",
    # Cross-attention based
    "CrossAttentionFusion",
    "GatedFusion",
    # Simple baselines
    "ConcatFusion",
    "PrependFusion",
]
