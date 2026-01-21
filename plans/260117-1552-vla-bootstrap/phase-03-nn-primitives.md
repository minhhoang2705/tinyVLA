# Phase 03: Neural Network Primitives

## Context Links
- [PyTorch VLA Research](../reports/researcher-260118-0228-pytorch-vla.md) - Flash Attention section
- [VLA Architectures](../reports/researcher-260118-vla-architectures.md) - Temporal modeling

## Overview
| Field | Value |
|-------|-------|
| Priority | P1 - Foundation |
| Status | Pending |
| Effort | 4h |
| Dependencies | Phase 2 |

Implement reusable neural network building blocks: multi-head attention, MLP, normalization layers, and positional encodings. These primitives compose into vision/language/fusion modules.

## Key Insights
- Flash Attention 2 reduces memory O(N²) → O(N), 2-4x speedup
- Pre-norm (LayerNorm before attention) more stable than post-norm
- Rotary Position Embeddings (RoPE) better than absolute for variable sequences
- einops makes tensor ops readable and less error-prone

## Requirements

### Functional
- FR-01: Multi-head attention with optional Flash Attention
- FR-02: MLP with configurable activation (GELU, SiLU)
- FR-03: LayerNorm and RMSNorm options
- FR-04: Sinusoidal, learnable, and RoPE position encodings
- FR-05: Causal masking for temporal models

### Non-Functional
- NFR-01: Each primitive <100 lines
- NFR-02: Support torch.compile

## Architecture

```
src/vla/nn/
├── __init__.py          # Public exports
├── attention.py         # Multi-head attention
├── mlp.py               # Feed-forward networks
├── norm.py              # Normalization layers
├── pos_encoding.py      # Position embeddings
└── temporal.py          # Frame stacking, causal conv
```

## Related Code Files

### Files to Create
| Path | Purpose | Lines |
|------|---------|-------|
| `src/vla/nn/__init__.py` | Exports | ~20 |
| `src/vla/nn/attention.py` | MHA + Flash | ~90 |
| `src/vla/nn/mlp.py` | FFN blocks | ~50 |
| `src/vla/nn/norm.py` | LayerNorm, RMSNorm | ~50 |
| `src/vla/nn/pos_encoding.py` | Position encodings | ~80 |
| `src/vla/nn/temporal.py` | Temporal ops | ~60 |
| `tests/unit/test_nn.py` | Primitive tests | ~120 |

## Implementation Steps

### Step 1: Implement attention.py (60 min)
```python
"""Multi-head attention with optional Flash Attention support."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional


class MultiHeadAttention(nn.Module):
    """Multi-head attention with Flash Attention 2 support.

    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        dropout: Attention dropout rate
        use_flash: Use Flash Attention if available
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
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_flash = use_flash and hasattr(F, "scaled_dot_product_attention")

        self.qkv = nn.Linear(dim, 3 * dim, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x)
        qkv = rearrange(qkv, "b n (three h d) -> three b h n d",
                        three=3, h=self.num_heads)
        q, k, v = qkv.unbind(0)

        if self.use_flash:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=is_causal,
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if attn_mask is not None:
                attn = attn.masked_fill(~attn_mask, float("-inf"))
            if is_causal:
                causal_mask = torch.triu(
                    torch.ones(N, N, dtype=torch.bool, device=x.device), diagonal=1
                )
                attn = attn.masked_fill(causal_mask, float("-inf"))
            attn = attn.softmax(dim=-1)
            attn = self.dropout(attn)
            x = attn @ v

        x = rearrange(x, "b h n d -> b n (h d)")
        return self.proj(x)


class CrossAttention(nn.Module):
    """Cross-attention for multimodal fusion."""

    def __init__(
        self,
        dim: int,
        context_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(dim, dim)
        self.to_kv = nn.Linear(context_dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape

        q = rearrange(self.to_q(x), "b n (h d) -> b h n d", h=self.num_heads)
        kv = rearrange(self.to_kv(context), "b m (two h d) -> two b h m d",
                       two=2, h=self.num_heads)
        k, v = kv.unbind(0)

        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b h n d -> b n (h d)")
        return self.proj(x)
```

### Step 2: Implement mlp.py (30 min)
```python
"""Feed-forward network blocks."""
import torch
import torch.nn as nn
from typing import Optional


class MLP(nn.Module):
    """Standard MLP with configurable activation.

    Args:
        dim: Input/output dimension
        hidden_dim: Hidden layer dimension (default: 4x dim)
        dropout: Dropout rate
        activation: Activation function name
    """

    ACTIVATIONS = {
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "relu": nn.ReLU,
    }

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = self.ACTIVATIONS[activation]()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.dropout(x)


class GatedMLP(nn.Module):
    """SwiGLU-style gated MLP (used in LLaMA)."""

    def __init__(self, dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 4 * 2 / 3)

        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = nn.functional.silu(self.gate_proj(x))
        x = gate * self.up_proj(x)
        x = self.down_proj(x)
        return self.dropout(x)
```

### Step 3: Implement norm.py (30 min)
```python
"""Normalization layers."""
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (LLaMA style).

    More efficient than LayerNorm, no mean computation.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


def get_norm(norm_type: str, dim: int, eps: float = 1e-6) -> nn.Module:
    """Factory for normalization layers."""
    if norm_type == "layer":
        return nn.LayerNorm(dim, eps=eps)
    elif norm_type == "rms":
        return RMSNorm(dim, eps=eps)
    elif norm_type == "batch":
        return nn.BatchNorm1d(dim, eps=eps)
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")
```

### Step 4: Implement pos_encoding.py (45 min)
```python
"""Positional encoding implementations."""
import torch
import torch.nn as nn
import math
from einops import rearrange


class SinusoidalPositionEncoding(nn.Module):
    """Fixed sinusoidal position encoding (Transformer original)."""

    def __init__(self, dim: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class LearnablePositionEncoding(nn.Module):
    """Learnable position embeddings."""

    def __init__(self, dim: int, max_len: int = 5000):
        super().__init__()
        self.pe = nn.Embedding(max_len, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device)
        return x + self.pe(positions)


class RotaryPositionEncoding(nn.Module):
    """Rotary Position Embedding (RoPE).

    Better extrapolation to longer sequences than absolute positions.
    """

    def __init__(self, dim: int, max_len: int = 5000, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_len = max_len
        self._build_cache(max_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple:
        seq_len = q.size(2)
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        return self._apply_rotary(q, cos, sin), self._apply_rotary(k, cos, sin)

    def _apply_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.cat([x1 * cos[..., ::2] - x2 * sin[..., ::2],
                          x1 * sin[..., 1::2] + x2 * cos[..., 1::2]], dim=-1)
```

### Step 5: Implement temporal.py (40 min)
```python
"""Temporal modeling for video/sequence inputs."""
import torch
import torch.nn as nn
from einops import rearrange


class FrameStacker(nn.Module):
    """Stack multiple frames into temporal sequence.

    Args:
        num_frames: Number of frames to stack
        dim: Feature dimension per frame
    """

    def __init__(self, num_frames: int = 6, dim: int = 768):
        super().__init__()
        self.num_frames = num_frames
        self.temporal_embed = nn.Embedding(num_frames, dim)

    def forward(self, frames: list[torch.Tensor]) -> torch.Tensor:
        """Stack frames with temporal embeddings.

        Args:
            frames: List of [B, N, D] tensors
        Returns:
            [B, T*N, D] stacked tensor
        """
        B = frames[0].size(0)
        T = len(frames)

        stacked = []
        for t, frame in enumerate(frames):
            pos = self.temporal_embed.weight[t:t+1].expand(B, -1, -1)
            stacked.append(frame + pos)

        return torch.cat(stacked, dim=1)


class CausalConv1d(nn.Module):
    """Causal 1D convolution for temporal modeling."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.padding = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        x = nn.functional.pad(x, (self.padding, 0))
        return self.conv(x)


class TemporalBlock(nn.Module):
    """Transformer block with causal masking for temporal data."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        from .attention import MultiHeadAttention
        from .mlp import MLP
        from .norm import RMSNorm

        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm2 = RMSNorm(dim)
        self.mlp = MLP(dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), is_causal=True)
        x = x + self.mlp(self.norm2(x))
        return x
```

### Step 6: Write tests (45 min)
```python
"""Tests for nn primitives."""
import pytest
import torch
from vla.nn import MultiHeadAttention, CrossAttention, MLP, RMSNorm
from vla.nn import SinusoidalPositionEncoding, RotaryPositionEncoding
from vla.nn import FrameStacker, CausalConv1d


class TestAttention:
    def test_mha_shape(self):
        mha = MultiHeadAttention(dim=256, num_heads=8)
        x = torch.randn(2, 16, 256)
        out = mha(x)
        assert out.shape == (2, 16, 256)

    def test_mha_causal(self):
        mha = MultiHeadAttention(dim=256, num_heads=8)
        x = torch.randn(2, 16, 256)
        out = mha(x, is_causal=True)
        assert out.shape == (2, 16, 256)

    def test_cross_attention_shape(self):
        ca = CrossAttention(dim=256, context_dim=512, num_heads=8)
        x = torch.randn(2, 16, 256)
        context = torch.randn(2, 32, 512)
        out = ca(x, context)
        assert out.shape == (2, 16, 256)


class TestMLP:
    def test_mlp_shape(self):
        mlp = MLP(dim=256, hidden_dim=1024)
        x = torch.randn(2, 16, 256)
        assert mlp(x).shape == x.shape


class TestNorm:
    def test_rmsnorm(self):
        norm = RMSNorm(256)
        x = torch.randn(2, 16, 256)
        out = norm(x)
        assert out.shape == x.shape


class TestPositionEncoding:
    def test_sinusoidal(self):
        pe = SinusoidalPositionEncoding(256)
        x = torch.randn(2, 100, 256)
        out = pe(x)
        assert out.shape == x.shape

    def test_rope(self):
        rope = RotaryPositionEncoding(64)
        q = torch.randn(2, 8, 100, 64)
        k = torch.randn(2, 8, 100, 64)
        q_rot, k_rot = rope(q, k)
        assert q_rot.shape == q.shape


class TestTemporal:
    def test_frame_stacker(self):
        stacker = FrameStacker(num_frames=6, dim=256)
        frames = [torch.randn(2, 196, 256) for _ in range(6)]
        out = stacker(frames)
        assert out.shape == (2, 6 * 196, 256)

    def test_causal_conv(self):
        conv = CausalConv1d(256, 256, kernel_size=3)
        x = torch.randn(2, 256, 100)
        out = conv(x)
        assert out.shape == x.shape
```

## Todo List
- [ ] Implement MultiHeadAttention with Flash Attention
- [ ] Implement CrossAttention for fusion
- [ ] Implement MLP and GatedMLP
- [ ] Implement RMSNorm and get_norm factory
- [ ] Implement position encodings (Sinusoidal, Learnable, RoPE)
- [ ] Implement temporal ops (FrameStacker, CausalConv1d)
- [ ] Write comprehensive tests
- [ ] Verify torch.compile compatibility

## Success Criteria
1. All primitives support variable batch/sequence sizes
2. Flash Attention used when available
3. Causal masking works correctly
4. All tests pass with 90%+ coverage on nn/
5. torch.compile works without graph breaks

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Flash Attention unavailable | Medium | Fallback to standard attention |
| einops import issues | Low | Pin version, test on install |
| Gradient issues with RoPE | Medium | Test backward pass explicitly |

## Security Considerations
- No external data loaded in primitives
- Deterministic operations when seeded

## Next Steps
- Phase 4: Build vision backbone using these primitives
