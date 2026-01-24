"""Neural network primitives for VLA models.

This module provides reusable building blocks for transformer-based VLA architectures:
- Attention mechanisms (self-attention and cross-attention)
- Feed-forward networks (standard and gated variants)
- Normalization layers (LayerNorm, RMSNorm)
- Positional encodings (sinusoidal, learnable, RoPE)
- Temporal modeling utilities (frame stacking, causal operations)

Example:
    >>> from vla.nn import MultiHeadAttention, MLP, RMSNorm
    >>> # Build a simple transformer block
    >>> attn = MultiHeadAttention(dim=768, num_heads=12)
    >>> mlp = MLP(dim=768)
    >>> norm = RMSNorm(768)
"""

from .attention import CrossAttention, MultiHeadAttention
from .mlp import MLP, GatedMLP
from .norm import RMSNorm, get_norm
from .pos_encoding import (
    LearnablePositionEncoding,
    RotaryPositionEncoding,
    SinusoidalPositionEncoding,
)
from .temporal import CausalConv1d, FrameStacker, TemporalBlock

__all__ = [
    # Attention
    "MultiHeadAttention",
    "CrossAttention",
    # MLP
    "MLP",
    "GatedMLP",
    # Normalization
    "RMSNorm",
    "get_norm",
    # Position encoding
    "SinusoidalPositionEncoding",
    "LearnablePositionEncoding",
    "RotaryPositionEncoding",
    # Temporal
    "FrameStacker",
    "CausalConv1d",
    "TemporalBlock",
]
