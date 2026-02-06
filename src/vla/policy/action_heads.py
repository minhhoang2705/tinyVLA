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

    def compute_loss(
        self,
        logits: torch.Tensor,
        target_actions: torch.Tensor,
        label_smoothing: float = 0.0,
    ) -> torch.Tensor:
        """Compute cross-entropy loss for discrete action prediction.

        Args:
            logits: Predicted logits [B, action_dim, num_bins]
            target_actions: Ground truth actions [B, action_dim] in range [-1, 1]
            label_smoothing: Label smoothing factor (0.0 to 1.0)

        Returns:
            Scalar loss tensor

        Example:
            >>> head = DiscreteActionHead(input_dim=768, action_dim=7)
            >>> logits = torch.randn(4, 7, 256)
            >>> target_actions = torch.rand(4, 7) * 2 - 1
            >>> loss = head.compute_loss(logits, target_actions)
            >>> loss.shape
            torch.Size([])
        """
        from .action_utils import continuous_to_bins

        # Convert continuous targets to bin indices
        target_bins = continuous_to_bins(target_actions, self.num_bins)

        # Reshape for cross entropy: [B * action_dim, num_bins] vs [B * action_dim]
        B, D, _ = logits.shape
        logits_flat = logits.reshape(B * D, self.num_bins)
        targets_flat = target_bins.reshape(B * D)

        loss = nn.functional.cross_entropy(
            logits_flat,
            targets_flat,
            label_smoothing=label_smoothing,
        )
        return loss


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
        return_logits: bool = False,
        sample: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Predict Gaussian actions.

        Args:
            features: [B, D] pooled features or [B, K, D] sequence
            return_logits: Whether to return (mean, std) for loss computation
            sample: Whether to sample from distribution (True for training)

        Returns:
            actions: [B, action_dim] in range [-1, 1]
            logits: Optional (mean, std) tuple if return_logits=True
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

        actions = actions.clamp(-1, 1)

        if return_logits:
            return actions, (mean, std)
        return actions, None

    def compute_loss(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        target_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Gaussian negative log-likelihood loss.

        Args:
            mean: Predicted mean actions [B, action_dim]
            std: Predicted standard deviations [B, action_dim]
            target_actions: Ground truth actions [B, action_dim] in range [-1, 1]

        Returns:
            Scalar loss tensor

        Example:
            >>> head = GaussianActionHead(input_dim=768, action_dim=7)
            >>> mean = torch.randn(4, 7).clamp(-1, 1)
            >>> std = torch.ones(4, 7) * 0.1
            >>> target_actions = torch.rand(4, 7) * 2 - 1
            >>> loss = head.compute_loss(mean, std, target_actions)
            >>> loss.shape
            torch.Size([])
        """
        # Gaussian negative log-likelihood
        loss = nn.functional.gaussian_nll_loss(
            mean,
            target_actions,
            std ** 2,  # variance
            reduction="mean",
        )
        return loss


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
