"""Temporal modeling for video/sequence inputs.

For robotics VLAs, we often have sequences of observations (video frames) rather
than single images. These modules help the model understand temporal relationships.

Why these implementations:
- FrameStacker: Combines multiple frames with temporal embeddings
- CausalConv1d: 1D convolution that doesn't look into the future
- TemporalBlock: Transformer block with causal masking for autoregressive modeling
"""
import torch
import torch.nn as nn
from typing import List


class FrameStacker(nn.Module):
    """Stack multiple frames into temporal sequence.

    When we have a video clip with multiple frames, we need to combine them
    while preserving temporal order. This module adds learnable temporal
    embeddings to distinguish frames at different time steps.

    Args:
        num_frames: Number of frames to stack
        dim: Feature dimension per frame

    Example:
        >>> stacker = FrameStacker(num_frames=6, dim=768)
        >>> # Six frames, each with 196 patches of 768 dims
        >>> frames = [torch.randn(2, 196, 768) for _ in range(6)]
        >>> stacked = stacker(frames)
        >>> print(stacked.shape)
        torch.Size([2, 1176, 768])  # 6 * 196 = 1176
    """

    def __init__(self, num_frames: int = 6, dim: int = 768):
        super().__init__()
        self.num_frames = num_frames
        # Learnable temporal embeddings (one per frame)
        self.temporal_embed = nn.Embedding(num_frames, dim)

    def forward(self, frames: List[torch.Tensor]) -> torch.Tensor:
        """Stack frames with temporal embeddings.

        Args:
            frames: List of [batch, num_patches, dim] tensors

        Returns:
            Stacked tensor [batch, num_frames * num_patches, dim]

        Raises:
            ValueError: If number of frames doesn't match num_frames
        """
        if len(frames) != self.num_frames:
            raise ValueError(
                f"Expected {self.num_frames} frames, got {len(frames)}"
            )

        B = frames[0].size(0)
        stacked = []

        # Add temporal embedding to each frame
        for t, frame in enumerate(frames):
            # Get temporal embedding for this time step
            # Shape: [1, 1, dim] -> broadcast to [batch, num_patches, dim]
            temporal_pos = self.temporal_embed.weight[t:t+1].unsqueeze(0).expand(B, -1, -1)
            # Add temporal embedding to frame features
            stacked.append(frame + temporal_pos)

        # Concatenate along sequence dimension
        return torch.cat(stacked, dim=1)


class CausalConv1d(nn.Module):
    """Causal 1D convolution for temporal modeling.

    Causal convolutions ensure the model can't look into the future when
    processing sequences. This is important for autoregressive tasks where
    we predict future states from past observations.

    The key is left-padding the input so that each output only depends on
    current and previous time steps.

    Args:
        in_channels: Input channel dimension
        out_channels: Output channel dimension
        kernel_size: Size of convolution kernel

    Example:
        >>> conv = CausalConv1d(in_channels=768, out_channels=768, kernel_size=3)
        >>> x = torch.randn(2, 768, 100)  # [batch, channels, time]
        >>> out = conv(x)
        >>> print(out.shape)
        torch.Size([2, 768, 100])
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        # Padding = kernel_size - 1 ensures output[t] only sees input[:t+1]
        self.padding = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal convolution.

        Args:
            x: Input tensor [batch, channels, time]

        Returns:
            Output tensor [batch, channels, time]
        """
        # Pad only on the left (past)
        x = nn.functional.pad(x, (self.padding, 0))
        return self.conv(x)


class TemporalBlock(nn.Module):
    """Transformer block with causal masking for temporal data.

    This is a standard transformer block but with causal attention masking,
    making it suitable for autoregressive temporal modeling.

    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        dropout: Dropout rate

    Example:
        >>> block = TemporalBlock(dim=768, num_heads=12)
        >>> x = torch.randn(2, 100, 768)  # [batch, time, dim]
        >>> out = block(x)
        >>> print(out.shape)
        torch.Size([2, 100, 768])
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        # Import here to avoid circular dependency
        from .attention import MultiHeadAttention
        from .mlp import MLP
        from .norm import RMSNorm

        # Pre-norm architecture (norm before attention/MLP)
        # More stable training than post-norm
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm2 = RMSNorm(dim)
        self.mlp = MLP(dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply temporal transformer block.

        Args:
            x: Input tensor [batch, time, dim]

        Returns:
            Output tensor [batch, time, dim]
        """
        # Pre-norm + attention with causal masking + residual
        x = x + self.attn(self.norm1(x), is_causal=True)
        # Pre-norm + MLP + residual
        x = x + self.mlp(self.norm2(x))
        return x
