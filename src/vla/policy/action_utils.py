"""Action normalization and bin conversion utilities."""

import torch


class ActionNormalizer:
    """Normalize actions to [-1, 1] range.

    This class handles normalization and denormalization of action values,
    which is useful for standardizing action ranges across different tasks.

    Args:
        action_min: Minimum action values [action_dim]
        action_max: Maximum action values [action_dim]

    Example:
        >>> action_min = torch.tensor([0.0, -1.0, -2.0])
        >>> action_max = torch.tensor([1.0, 1.0, 2.0])
        >>> normalizer = ActionNormalizer(action_min, action_max)
        >>> actions = torch.tensor([[0.5, 0.0, 0.0]])
        >>> normalized = normalizer.normalize(actions)
        >>> recovered = normalizer.denormalize(normalized)
        >>> torch.allclose(actions, recovered)
        True
    """

    def __init__(
        self,
        action_min: torch.Tensor,
        action_max: torch.Tensor,
    ):
        self.action_min = action_min
        self.action_max = action_max
        self.action_range = action_max - action_min

    def normalize(self, actions: torch.Tensor) -> torch.Tensor:
        """Normalize actions to [-1, 1].

        Args:
            actions: Raw action values [..., action_dim]

        Returns:
            Normalized actions [..., action_dim] in range [-1, 1]
        """
        return 2 * (actions - self.action_min) / self.action_range - 1

    def denormalize(self, actions: torch.Tensor) -> torch.Tensor:
        """Denormalize actions from [-1, 1] to original range.

        Args:
            actions: Normalized actions [..., action_dim] in range [-1, 1]

        Returns:
            Denormalized actions [..., action_dim] in original range
        """
        return (actions + 1) / 2 * self.action_range + self.action_min


def continuous_to_bins(
    actions: torch.Tensor,
    num_bins: int = 256,
    action_min: float = -1.0,
    action_max: float = 1.0,
) -> torch.Tensor:
    """Convert continuous actions to bin indices.

    Discretizes continuous actions into bins, useful for discrete action
    prediction heads like RT-2 and OpenVLA.

    Args:
        actions: Continuous actions [..., action_dim]
        num_bins: Number of discrete bins per dimension
        action_min: Minimum action value
        action_max: Maximum action value

    Returns:
        Bin indices [..., action_dim] as long tensor

    Example:
        >>> actions = torch.tensor([[-1.0, 0.0, 1.0]])
        >>> bins = continuous_to_bins(actions, num_bins=256)
        >>> bins.shape
        torch.Size([1, 3])
        >>> bins[0, 0].item(), bins[0, 1].item(), bins[0, 2].item()
        (0, 127, 255)
    """
    # Clamp to valid range
    actions = actions.clamp(action_min, action_max)

    # Scale to [0, num_bins-1]
    normalized = (actions - action_min) / (action_max - action_min)
    bins = (normalized * (num_bins - 1)).round().long()

    return bins


def bins_to_continuous(
    bins: torch.Tensor,
    num_bins: int = 256,
    action_min: float = -1.0,
    action_max: float = 1.0,
) -> torch.Tensor:
    """Convert bin indices to continuous actions.

    Inverse of continuous_to_bins, converting discrete bin indices back
    to continuous action values.

    Args:
        bins: Bin indices [..., action_dim] as integer tensor
        num_bins: Number of discrete bins per dimension
        action_min: Minimum action value
        action_max: Maximum action value

    Returns:
        Continuous actions [..., action_dim]

    Example:
        >>> bins = torch.tensor([[0, 127, 255]])
        >>> actions = bins_to_continuous(bins, num_bins=256)
        >>> actions.shape
        torch.Size([1, 3])
        >>> torch.allclose(actions, torch.tensor([[-1.0, 0.0, 1.0]]), atol=1e-2)
        True
    """
    normalized = bins.float() / (num_bins - 1)
    return normalized * (action_max - action_min) + action_min


def compute_action_loss(
    pred_logits: torch.Tensor,
    target_actions: torch.Tensor,
    num_bins: int = 256,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Compute cross-entropy loss for discrete action prediction.

    Converts continuous target actions to bin indices, then computes
    cross-entropy loss against predicted logits.

    Args:
        pred_logits: Predicted logits [B, action_dim, num_bins]
        target_actions: Target continuous actions [B, action_dim]
        num_bins: Number of bins per dimension
        label_smoothing: Label smoothing factor (0.0 to 1.0)

    Returns:
        Scalar loss value

    Example:
        >>> pred_logits = torch.randn(4, 7, 256)
        >>> target_actions = torch.rand(4, 7) * 2 - 1
        >>> loss = compute_action_loss(pred_logits, target_actions)
        >>> loss.shape
        torch.Size([])
    """
    target_bins = continuous_to_bins(target_actions, num_bins)

    # Reshape for cross entropy: [B * action_dim, num_bins] vs [B * action_dim]
    B, D, _ = pred_logits.shape
    logits_flat = pred_logits.reshape(B * D, num_bins)
    targets_flat = target_bins.reshape(B * D)

    loss = torch.nn.functional.cross_entropy(
        logits_flat, targets_flat, label_smoothing=label_smoothing
    )
    return loss
