"""Action prediction heads for VLA policies."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from vla.registry import ACTION_REGISTRY

from .action_utils import bins_to_continuous


@ACTION_REGISTRY.register("discrete_action")
class DiscreteActionHead(nn.Module):
    """Discrete action head with binned outputs.

    Predicts logits over bins for each action dimension, following the
    RT-2 and OpenVLA approach for robust action prediction.

    Args:
        input_dim: Input feature dimension
        action_dim: Number of action dimensions (e.g., 7 for 7-DOF arm)
        num_bins: Number of discrete bins per dimension
        hidden_dim: Hidden layer dimension (defaults to input_dim)

    Example:
        >>> head = DiscreteActionHead(input_dim=768, action_dim=7)
        >>> features = torch.randn(4, 768)
        >>> actions, logits = head(features, return_logits=True)
        >>> actions.shape, logits.shape
        (torch.Size([4, 7]), torch.Size([4, 7, 256]))
    """

    def __init__(
        self,
        input_dim: int = 768,
        action_dim: int = 7,
        num_bins: int = 256,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_bins = num_bins

        hidden_dim = hidden_dim or input_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim * num_bins),
        )

    def forward(
        self,
        features: torch.Tensor,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Predict discrete actions.

        Args:
            features: [B, D] pooled features or [B, K, D] sequence
            return_logits: Whether to return raw logits

        Returns:
            actions: [B, action_dim] continuous actions in [-1, 1]
            logits: Optional [B, action_dim, num_bins] if return_logits=True
        """
        # Pool if sequence input
        if features.ndim == 3:
            features = features.mean(dim=1)

        logits = self.mlp(features)
        logits = logits.view(-1, self.action_dim, self.num_bins)

        # Get most likely bin per dimension
        bins = logits.argmax(dim=-1)
        actions = bins_to_continuous(bins, self.num_bins)

        if return_logits:
            return actions, logits
        return actions, None


@ACTION_REGISTRY.register("gaussian_action")
class GaussianActionHead(nn.Module):
    """Gaussian action head for continuous control.

    Predicts mean and log standard deviation for each action dimension,
    enabling stochastic policies and uncertainty estimation.

    Args:
        input_dim: Input feature dimension
        action_dim: Number of action dimensions
        hidden_dim: Hidden layer dimension (defaults to input_dim)
        min_std: Minimum standard deviation
        max_std: Maximum standard deviation

    Example:
        >>> head = GaussianActionHead(input_dim=768, action_dim=7)
        >>> features = torch.randn(4, 768)
        >>> actions, std = head(features, sample=True)
        >>> actions.shape, std.shape
        (torch.Size([4, 7]), torch.Size([4, 7]))
    """

    def __init__(
        self,
        input_dim: int = 768,
        action_dim: int = 7,
        hidden_dim: Optional[int] = None,
        min_std: float = 0.01,
        max_std: float = 1.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.min_std = min_std
        self.max_std = max_std

        hidden_dim = hidden_dim or input_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim * 2),  # mean + log_std
        )

    def forward(
        self,
        features: torch.Tensor,
        sample: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict Gaussian actions.

        Args:
            features: [B, D] pooled features or [B, K, D] sequence
            sample: Whether to sample from distribution (True for training)

        Returns:
            actions: [B, action_dim] in range [-1, 1]
            std: [B, action_dim] standard deviations
        """
        if features.ndim == 3:
            features = features.mean(dim=1)

        output = self.mlp(features)
        mean, log_std = output.chunk(2, dim=-1)

        # Clamp std to valid range
        std = torch.clamp(log_std.exp(), self.min_std, self.max_std)

        if sample:
            actions = mean + std * torch.randn_like(std)
        else:
            actions = mean

        return actions.clamp(-1, 1), std


@ACTION_REGISTRY.register("hybrid_action")
class HybridActionHead(nn.Module):
    """Hybrid action head: discrete for arm, continuous for gripper.

    Combines discrete binned prediction for most DOFs with continuous
    prediction for others. Useful when different action dimensions have
    different characteristics.

    Args:
        input_dim: Input feature dimension
        discrete_dims: Number of discrete action dimensions
        continuous_dims: Number of continuous action dimensions
        num_bins: Number of bins for discrete dimensions

    Example:
        >>> head = HybridActionHead(input_dim=768, discrete_dims=6, continuous_dims=1)
        >>> features = torch.randn(4, 768)
        >>> actions, info = head(features, return_logits=True)
        >>> actions.shape
        torch.Size([4, 7])
    """

    def __init__(
        self,
        input_dim: int = 768,
        discrete_dims: int = 6,
        continuous_dims: int = 1,
        num_bins: int = 256,
    ):
        super().__init__()
        self.discrete_head = DiscreteActionHead(input_dim, discrete_dims, num_bins)
        self.continuous_head = GaussianActionHead(input_dim, continuous_dims)

    def forward(
        self,
        features: torch.Tensor,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Predict hybrid actions.

        Args:
            features: [B, D] or [B, K, D] input features
            return_logits: Whether to return additional info

        Returns:
            actions: [B, discrete_dims + continuous_dims]
            info: Optional dict with discrete_logits and continuous_std
        """
        discrete_actions, logits = self.discrete_head(features, return_logits)
        continuous_actions, std = self.continuous_head(features)

        actions = torch.cat([discrete_actions, continuous_actions], dim=-1)

        if return_logits:
            return actions, {"discrete_logits": logits, "continuous_std": std}
        return actions, None
