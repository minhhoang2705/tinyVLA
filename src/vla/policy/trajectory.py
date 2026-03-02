"""Multi-step trajectory prediction."""

from typing import Optional

import torch
import torch.nn as nn

from vla.registry import ACTION_REGISTRY


@ACTION_REGISTRY.register("trajectory_head")
class TrajectoryHead(nn.Module):
    """Predict multiple future action steps.

    Uses either parallel prediction (all steps at once) or autoregressive
    prediction (one step at a time) for generating action trajectories.

    Args:
        input_dim: Input feature dimension
        action_dim: Number of action dimensions
        num_steps: Number of future steps to predict
        num_bins: Number of bins for discrete actions
        mode: "parallel" or "autoregressive"

    Example:
        >>> head = TrajectoryHead(input_dim=768, action_dim=7, num_steps=8)
        >>> features = torch.randn(4, 768)
        >>> actions = head(features)
        >>> actions.shape
        torch.Size([4, 8, 7])
    """

    def __init__(
        self,
        input_dim: int = 768,
        action_dim: int = 7,
        num_steps: int = 8,
        num_bins: int = 256,
        mode: str = "parallel",
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_steps = num_steps
        self.num_bins = num_bins
        self.mode = mode

        # Step embeddings
        self.step_embed = nn.Embedding(num_steps, input_dim)

        if mode == "parallel":
            # Single forward pass for all steps
            self.predictor = nn.Linear(input_dim, num_steps * action_dim * num_bins)
        else:
            # Autoregressive with causal attention
            self.action_embed = nn.Linear(action_dim, input_dim)
            self.transformer = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    input_dim, nhead=8, dim_feedforward=input_dim * 4, batch_first=True
                ),
                num_layers=2,
            )
            self.output_head = nn.Linear(input_dim, action_dim * num_bins)

    def forward(
        self,
        features: torch.Tensor,
        prev_actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict trajectory of actions.

        Args:
            features: [B, D] pooled features or [B, K, D] sequence
            prev_actions: Optional [B, T, action_dim] for autoregressive

        Returns:
            actions: [B, num_steps, action_dim]
        """
        if features.ndim == 3:
            features = features.mean(dim=1)

        B = features.size(0)

        if self.mode == "parallel":
            logits = self.predictor(features)
            logits = logits.reshape(B, self.num_steps, self.action_dim, self.num_bins)
            bins = logits.argmax(dim=-1)
            from .action_utils import bins_to_continuous

            return bins_to_continuous(bins, self.num_bins)

        else:
            # Autoregressive decoding
            actions = []
            step_tokens = self.step_embed.weight.unsqueeze(0).expand(B, -1, -1)

            for t in range(self.num_steps):
                query = step_tokens[:, : t + 1]
                out = self.transformer(query, features.unsqueeze(1))
                logits = self.output_head(out[:, -1])
                logits = logits.reshape(B, self.action_dim, self.num_bins)
                bins = logits.argmax(dim=-1)
                from .action_utils import bins_to_continuous

                actions.append(bins_to_continuous(bins, self.num_bins))

            return torch.stack(actions, dim=1)


@ACTION_REGISTRY.register("diffusion_action")
class DiffusionActionHead(nn.Module):
    """Placeholder for diffusion-based action prediction.

    Diffusion policies generate smoother trajectories but require
    multiple denoising steps at inference. This is a placeholder
    for future implementation.

    Args:
        input_dim: Input feature dimension
        action_dim: Number of action dimensions
        num_steps: Number of trajectory steps
        diffusion_steps: Number of diffusion denoising steps

    Note:
        This is a placeholder implementation. Full diffusion policy
        requires implementing DDPM scheduler and iterative denoising.

    Example:
        >>> head = DiffusionActionHead(input_dim=768, action_dim=7, num_steps=8)
        >>> features = torch.randn(4, 768)
        >>> actions = head(features)
        >>> actions.shape
        torch.Size([4, 8, 7])
    """

    def __init__(
        self,
        input_dim: int = 768,
        action_dim: int = 7,
        num_steps: int = 8,
        diffusion_steps: int = 100,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_steps = num_steps
        self.diffusion_steps = diffusion_steps

        # Placeholder - full implementation requires DDPM scheduler
        self.net = nn.Linear(input_dim + action_dim * num_steps, action_dim * num_steps)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Generate action trajectory via diffusion.

        Note:
            This is a placeholder. Full implementation requires
            iterative denoising with proper DDPM scheduler.

        Args:
            features: [B, D] or [B, K, D] input features

        Returns:
            actions: [B, num_steps, action_dim]
        """
        B = features.size(0) if features.ndim == 2 else features.size(0)
        if features.ndim == 3:
            features = features.mean(dim=1)

        # For now, just output zeros as placeholder
        return torch.zeros(B, self.num_steps, self.action_dim, device=features.device)
