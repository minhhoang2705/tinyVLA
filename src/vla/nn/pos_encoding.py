"""Positional encoding implementations.

Transformers have no notion of position by default. Positional encodings add
information about token positions so the model can understand sequence order.

Why these implementations:
- Sinusoidal: Original Transformer, fixed patterns, extrapolates well
- Learnable: BERT-style, can adapt to data but fixed max length
- RoPE: Used in modern LLMs (LLaMA, GPT-NeoX), rotates embeddings in head space
"""

import math
from typing import Tuple

import torch
import torch.nn as nn


class SinusoidalPositionEncoding(nn.Module):
    """Fixed sinusoidal position encoding (Transformer original).

    Uses sine and cosine functions at different frequencies to encode positions.
    This allows the model to learn to attend by relative positions and can
    extrapolate to longer sequences than seen during training.

    Formula:
        PE(pos, 2i) = sin(pos / 10000^(2i/dim))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/dim))

    Args:
        dim: Embedding dimension (must be even)
        max_len: Maximum sequence length to pre-compute

    Example:
        >>> pe = SinusoidalPositionEncoding(768, max_len=1000)
        >>> x = torch.randn(2, 100, 768)
        >>> x_with_pos = pe(x)
        >>> print(x_with_pos.shape)
        torch.Size([2, 100, 768])
    """

    def __init__(self, dim: int, max_len: int = 5000):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}")

        # Pre-compute positional encodings
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        # Compute the division term: 10000^(2i/dim)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))

        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter, but part of state_dict)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: Input tensor [batch, seq_len, dim]

        Returns:
            Input with added positional encoding [batch, seq_len, dim]
        """
        return x + self.pe[:, : x.size(1)]  # type: ignore


class LearnablePositionEncoding(nn.Module):
    """Learnable position embeddings.

    Instead of fixed patterns, learns optimal position embeddings from data.
    Used in BERT and many vision models. Cannot extrapolate beyond max_len.

    Args:
        dim: Embedding dimension
        max_len: Maximum sequence length (fixed during training)

    Example:
        >>> pe = LearnablePositionEncoding(768, max_len=512)
        >>> x = torch.randn(2, 100, 768)
        >>> x_with_pos = pe(x)
        >>> print(x_with_pos.shape)
        torch.Size([2, 100, 768])
    """

    def __init__(self, dim: int, max_len: int = 5000):
        super().__init__()
        # Learnable embedding table for positions
        self.pe = nn.Embedding(max_len, dim)
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add learnable positional encoding to input.

        Args:
            x: Input tensor [batch, seq_len, dim]

        Returns:
            Input with added positional encoding [batch, seq_len, dim]

        Raises:
            ValueError: If seq_len > max_len
        """
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.max_len}")

        # Create position indices [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=x.device)
        return x + self.pe(positions)  # type: ignore


class RotaryPositionEncoding(nn.Module):
    """Rotary Position Embedding (RoPE).

    RoPE applies rotations to query and key embeddings in attention heads.
    This encodes relative positions and extrapolates better to longer sequences
    than absolute position encodings.

    Used in: GPT-NeoX, LLaMA, PaLM

    Args:
        dim: Dimension of each attention head (not full model dim)
        max_len: Maximum sequence length
        base: Base for the geometric progression (10000 in original paper)

    Example:
        >>> rope = RotaryPositionEncoding(dim=64)  # head_dim
        >>> q = torch.randn(2, 8, 100, 64)  # [B, heads, seq, head_dim]
        >>> k = torch.randn(2, 8, 100, 64)
        >>> q_rot, k_rot = rope(q, k)
        >>> print(q_rot.shape)
        torch.Size([2, 8, 100, 64])
    """

    def __init__(self, dim: int, max_len: int = 5000, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim must be even for RoPE, got {dim}")

        # Compute inverse frequencies: 1 / (base^(2i/dim))
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_len = max_len

        # Pre-compute rotations for efficiency
        self._build_cache(max_len)

    def _build_cache(self, seq_len: int):
        """Pre-compute cos and sin for rotations.

        Args:
            seq_len: Sequence length to cache
        """
        # Position indices [0, 1, 2, ..., seq_len-1]
        t = torch.arange(seq_len, device=self.inv_freq.device)  # type: ignore
        # Outer product: [seq_len, dim/2]
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Concatenate for even/odd indices: [seq_len, dim]
        emb = torch.cat([freqs, freqs], dim=-1)

        # Cache cos and sin with proper shape for broadcasting
        # [1, 1, seq_len, dim] for [B, heads, seq, dim]
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position encoding to queries and keys.

        Args:
            q: Query tensor [batch, num_heads, seq_len, head_dim]
            k: Key tensor [batch, num_heads, seq_len, head_dim]

        Returns:
            Tuple of (rotated_q, rotated_k) with same shapes as input
        """
        seq_len = q.size(2)

        # Rebuild cache if sequence is longer than cached
        if seq_len > self.max_len:
            self._build_cache(seq_len)

        # Get cached cos/sin for current sequence length
        cos = self.cos_cached[:, :, :seq_len, :]  # type: ignore
        sin = self.sin_cached[:, :, :seq_len, :]  # type: ignore

        # Apply rotation to both q and k
        return self._apply_rotary(q, cos, sin), self._apply_rotary(k, cos, sin)

    def _apply_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotary transformation.

        Splits features into pairs and rotates them by position-dependent angles.

        Args:
            x: Input tensor [..., dim]
            cos: Cosine values [..., dim]
            sin: Sine values [..., dim]

        Returns:
            Rotated tensor [..., dim]
        """
        # Split into even and odd indices
        x1, x2 = x[..., ::2], x[..., 1::2]

        # Apply rotation matrix: [cos -sin; sin cos]
        # x_rot = [x1*cos - x2*sin, x1*sin + x2*cos]
        return torch.cat(
            [x1 * cos[..., ::2] - x2 * sin[..., ::2], x1 * sin[..., 1::2] + x2 * cos[..., 1::2]],
            dim=-1,
        )
