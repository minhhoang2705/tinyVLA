"""Multi-head attention with optional Flash Attention support.

This module provides self-attention and cross-attention mechanisms that are core
building blocks for transformer architectures. Flash Attention 2 is automatically
used when available for 2-4x speedup and reduced memory usage.

Why these implementations:
- MultiHeadAttention: Standard self-attention for within-modality processing
- CrossAttention: Attends from one modality to another (e.g., language to vision)
- Flash Attention: Uses PyTorch's scaled_dot_product_attention when available
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from vla.utils import setup_logger

logger = setup_logger(__name__)


class MultiHeadAttention(nn.Module):
    """Multi-head attention with Flash Attention 2 support.

    This is the core attention mechanism used in transformers. It allows the model
    to attend to different parts of the input sequence in parallel.

    Args:
        dim: Model dimension (must be divisible by num_heads)
        num_heads: Number of parallel attention heads
        dropout: Dropout rate for attention weights
        use_flash: Use Flash Attention if available (recommended for training)
        bias: Whether to use bias in linear projections

    Example:
        >>> attn = MultiHeadAttention(dim=768, num_heads=12)
        >>> x = torch.randn(2, 196, 768)  # [batch, seq_len, dim]
        >>> out = attn(x)
        >>> print(out.shape)
        torch.Size([2, 196, 768])
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        use_flash: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5  # Scaling factor for attention scores

        # Check if Flash Attention is available
        self.use_flash = use_flash and hasattr(F, "scaled_dot_product_attention")
        if use_flash and not self.use_flash:
            logger.warning("Flash Attention requested but not available, using standard attention")

        # Single linear layer for Q, K, V projections (more efficient)
        self.qkv = nn.Linear(dim, 3 * dim, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Forward pass through multi-head attention.

        Args:
            x: Input tensor [batch, seq_len, dim]
            attn_mask: Optional attention mask [batch, num_heads, seq_len, seq_len]
            is_causal: Whether to apply causal masking (for autoregressive models)

        Returns:
            Output tensor [batch, seq_len, dim]
        """
        B, N, C = x.shape

        # Project to Q, K, V and split into heads
        # Shape: [3, batch, num_heads, seq_len, head_dim]
        qkv = self.qkv(x)
        qkv = rearrange(qkv, "b n (three h d) -> three b h n d", three=3, h=self.num_heads)
        q, k, v = qkv.unbind(0)

        if self.use_flash:
            # Use Flash Attention (memory-efficient, 2-4x faster)
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=is_causal,
            )
        else:
            # Standard attention computation (fallback)
            attn = (q @ k.transpose(-2, -1)) * self.scale

            # Apply attention mask if provided
            if attn_mask is not None:
                attn = attn.masked_fill(~attn_mask, float("-inf"))

            # Apply causal mask for autoregressive models
            if is_causal:
                causal_mask = torch.triu(
                    torch.ones(N, N, dtype=torch.bool, device=x.device), diagonal=1
                )
                attn = attn.masked_fill(causal_mask, float("-inf"))

            attn = attn.softmax(dim=-1)
            attn = self.dropout(attn)
            x = attn @ v

        # Merge heads and project output
        x = rearrange(x, "b h n d -> b n (h d)")
        return self.proj(x)  # type: ignore


class CrossAttention(nn.Module):
    """Cross-attention for multimodal fusion.

    Unlike self-attention, cross-attention attends from one modality (query) to
    another modality (key/value). This is essential for fusing vision and language.

    Args:
        dim: Query dimension
        context_dim: Key/Value dimension (can differ from query)
        num_heads: Number of attention heads
        dropout: Dropout rate

    Example:
        >>> # Language queries attend to vision features
        >>> ca = CrossAttention(dim=768, context_dim=768, num_heads=12)
        >>> lang_features = torch.randn(2, 10, 768)  # [batch, lang_len, dim]
        >>> vision_features = torch.randn(2, 196, 768)  # [batch, vision_len, dim]
        >>> fused = ca(lang_features, vision_features)
        >>> print(fused.shape)
        torch.Size([2, 10, 768])
    """

    def __init__(
        self,
        dim: int,
        context_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        # Separate projections for query (from x) and key/value (from context)
        self.to_q = nn.Linear(dim, dim)
        self.to_kv = nn.Linear(context_dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Cross-attend from x to context.

        Args:
            x: Query tensor [batch, query_len, dim]
            context: Key/Value tensor [batch, context_len, context_dim]

        Returns:
            Output tensor [batch, query_len, dim]
        """
        B, N, _ = x.shape

        # Project queries from x
        q = rearrange(self.to_q(x), "b n (h d) -> b h n d", h=self.num_heads)

        # Project keys and values from context
        kv = rearrange(self.to_kv(context), "b m (two h d) -> two b h m d", two=2, h=self.num_heads)
        k, v = kv.unbind(0)

        # Use Flash Attention for cross-attention
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b h n d -> b n (h d)")
        return self.proj(x)  # type: ignore
