"""Normalization layers.

Normalization is critical for training stability in deep networks. Different
normalization methods have tradeoffs in speed, memory, and effectiveness.

Why these implementations:
- LayerNorm: Standard in transformers, normalizes across features
- RMSNorm: Faster variant used in LLaMA (no mean computation)
- Factory function: Easy switching via configuration
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (LLaMA style).

    RMSNorm is simpler and faster than LayerNorm because it only normalizes
    by the root mean square (no mean subtraction). This gives ~10% speedup
    with similar performance.

    Formula: x * rsqrt(mean(x²) + eps) * learned_scale

    Args:
        dim: Feature dimension to normalize
        eps: Small constant for numerical stability

    Example:
        >>> norm = RMSNorm(768)
        >>> x = torch.randn(2, 196, 768)
        >>> out = norm(x)
        >>> print(out.shape)
        torch.Size([2, 196, 768])
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Learnable scale parameter (one per feature)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization.

        Args:
            x: Input tensor [..., dim]

        Returns:
            Normalized tensor [..., dim]
        """
        # Compute RMS: sqrt(mean(x²) + eps)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # Normalize and scale
        return x * rms * self.weight


def get_norm(norm_type: str, dim: int, eps: float = 1e-6) -> nn.Module:
    """Factory for normalization layers.

    This function makes it easy to switch normalization types via configuration
    without changing code.

    Args:
        norm_type: Type of normalization ("layer", "rms", "batch")
        dim: Feature dimension
        eps: Epsilon for numerical stability

    Returns:
        Normalization layer module

    Raises:
        ValueError: If norm_type is not recognized

    Example:
        >>> # Easy to switch via config
        >>> norm = get_norm("rms", dim=768)
        >>> norm = get_norm("layer", dim=768)
    """
    if norm_type == "layer":
        return nn.LayerNorm(dim, eps=eps)
    elif norm_type == "rms":
        return RMSNorm(dim, eps=eps)
    elif norm_type == "batch":
        return nn.BatchNorm1d(dim, eps=eps)
    else:
        raise ValueError(f"Unknown norm type: {norm_type}. Choose from: layer, rms, batch")
