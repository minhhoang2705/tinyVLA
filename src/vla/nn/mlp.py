"""Feed-forward network blocks.

MLPs (Multi-Layer Perceptrons) are the other half of transformer blocks.
After attention mixes information across positions, MLPs process each position
independently with non-linear transformations.

Why these implementations:
- Standard MLP: Used in original Transformer, ViT, BERT
- GatedMLP (SwiGLU): Used in modern LLMs like LLaMA - better performance
"""

from typing import Optional

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Standard MLP with configurable activation.

    This is the classic feed-forward network used in transformers. The hidden
    dimension is typically 4x the input dimension for good capacity.

    Args:
        dim: Input/output dimension
        hidden_dim: Hidden layer dimension (default: 4x input)
        dropout: Dropout rate applied after each layer
        activation: Activation function ("gelu", "silu", "relu")

    Example:
        >>> mlp = MLP(dim=768, hidden_dim=3072)
        >>> x = torch.randn(2, 196, 768)
        >>> out = mlp(x)
        >>> print(out.shape)
        torch.Size([2, 196, 768])
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
        # Default hidden_dim is 4x input (standard transformer ratio)
        hidden_dim = hidden_dim or dim * 4

        if activation not in self.ACTIVATIONS:
            raise ValueError(
                f"Unknown activation: {activation}. Choose from {list(self.ACTIVATIONS.keys())}"
            )

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = self.ACTIVATIONS[activation]()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply two-layer MLP with activation and dropout.

        Args:
            x: Input tensor [batch, seq_len, dim]

        Returns:
            Output tensor [batch, seq_len, dim]
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.dropout(x)


class GatedMLP(nn.Module):
    """SwiGLU-style gated MLP (used in LLaMA).

    Gated MLPs multiply the output of two parallel projections, one with
    activation. This provides better expressivity than standard MLPs.

    The formula is: output = SiLU(gate_proj(x)) * up_proj(x) -> down_proj

    Args:
        dim: Input/output dimension
        hidden_dim: Hidden dimension (default: ~2.67x input for param parity)
        dropout: Dropout rate

    Example:
        >>> mlp = GatedMLP(dim=768)
        >>> x = torch.randn(2, 196, 768)
        >>> out = mlp(x)
        >>> print(out.shape)
        torch.Size([2, 196, 768])
    """

    def __init__(self, dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        # Default hidden_dim for SwiGLU (keeps params similar to standard MLP)
        hidden_dim = hidden_dim or int(dim * 4 * 2 / 3)

        # No bias in gated MLPs (LLaMA convention)
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gated MLP with SiLU activation.

        Args:
            x: Input tensor [batch, seq_len, dim]

        Returns:
            Output tensor [batch, seq_len, dim]
        """
        # Gate path with SiLU activation
        gate = nn.functional.silu(self.gate_proj(x))
        # Element-wise multiply gate with up projection
        x = gate * self.up_proj(x)
        # Project back to original dimension
        x = self.down_proj(x)
        return self.dropout(x)
