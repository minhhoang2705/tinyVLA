# Phase 07: Policy and Action Heads

## Context Links
- [VLA Architectures](../reports/researcher-260118-vla-architectures.md) - Action prediction section
- [Tech Stack](../../docs/tech-stack.md) - Action head section

## Overview
| Field | Value |
|-------|-------|
| Priority | P1 - Core Component |
| Status | Pending |
| Effort | 3h |
| Dependencies | Phases 2, 3 |

Implement action prediction heads: discrete bins (RT-2/OpenVLA style) and continuous Gaussian. Support multi-step action prediction for trajectory-level control.

## Key Insights
- Discrete bins (256 per DOF): Classification loss, stable training
- Continuous Gaussian: MSE loss, better for fine control
- Multi-step prediction (8 steps ahead) improves smoothness
- Separate heads per action dimension enables modular training

## Requirements

### Functional
- FR-01: Discrete action head with configurable bins (default 256)
- FR-02: Continuous Gaussian head with mean + variance
- FR-03: Multi-step action prediction
- FR-04: Action normalization/denormalization utilities
- FR-05: Support variable action dimensions (7-DOF arm, 6-DOF pose, etc.)

### Non-Functional
- NFR-01: <1ms action decoding
- NFR-02: Differentiable for end-to-end training

## Architecture

```
src/vla/policy/
├── __init__.py
├── action_heads.py      # Discrete + Gaussian heads
├── action_utils.py      # Normalization, bin conversion
└── trajectory.py        # Multi-step prediction
```

**Discrete Action Flow:**
```
Fused Features [B, K, D]
    ↓
Linear(D, action_dim * num_bins)
    ↓
Reshape to [B, action_dim, num_bins]
    ↓
Softmax per dimension
    ↓
argmax → bin index → continuous value
```

**Gaussian Action Flow:**
```
Fused Features [B, K, D]
    ↓
Linear(D, action_dim * 2)  # mean + log_std
    ↓
Reshape to [B, action_dim, 2]
    ↓
Sample or use mean for inference
```

## Related Code Files

### Files to Create
| Path | Purpose | Lines |
|------|---------|-------|
| `src/vla/policy/__init__.py` | Exports | ~15 |
| `src/vla/policy/action_heads.py` | Action heads | ~120 |
| `src/vla/policy/action_utils.py` | Utilities | ~80 |
| `src/vla/policy/trajectory.py` | Multi-step | ~80 |
| `tests/unit/test_policy.py` | Policy tests | ~100 |

## Implementation Steps

### Step 1: Implement action_utils.py (30 min)
```python
"""Action normalization and bin conversion utilities."""
import torch
from typing import Optional, Tuple


class ActionNormalizer:
    """Normalize actions to [-1, 1] range."""

    def __init__(
        self,
        action_min: torch.Tensor,
        action_max: torch.Tensor,
    ):
        """
        Args:
            action_min: Minimum action values [action_dim]
            action_max: Maximum action values [action_dim]
        """
        self.action_min = action_min
        self.action_max = action_max
        self.action_range = action_max - action_min

    def normalize(self, actions: torch.Tensor) -> torch.Tensor:
        """Normalize actions to [-1, 1]."""
        return 2 * (actions - self.action_min) / self.action_range - 1

    def denormalize(self, actions: torch.Tensor) -> torch.Tensor:
        """Denormalize actions from [-1, 1] to original range."""
        return (actions + 1) / 2 * self.action_range + self.action_min


def continuous_to_bins(
    actions: torch.Tensor,
    num_bins: int = 256,
    action_min: float = -1.0,
    action_max: float = 1.0,
) -> torch.Tensor:
    """Convert continuous actions to bin indices.

    Args:
        actions: Continuous actions [..., action_dim]
        num_bins: Number of discrete bins
        action_min: Minimum action value
        action_max: Maximum action value
    Returns:
        Bin indices [..., action_dim] as long tensor
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

    Args:
        bins: Bin indices [..., action_dim]
        num_bins: Number of discrete bins
        action_min: Minimum action value
        action_max: Maximum action value
    Returns:
        Continuous actions [..., action_dim]
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

    Args:
        pred_logits: [B, action_dim, num_bins]
        target_actions: [B, action_dim] continuous actions
        num_bins: Number of bins
        label_smoothing: Label smoothing factor
    Returns:
        Scalar loss
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
```

### Step 2: Implement action_heads.py (60 min)
```python
"""Action prediction heads for VLA policies."""
import torch
import torch.nn as nn
from typing import Optional, Tuple

from vla.nn import MLP, RMSNorm
from vla.registry import ACTION_REGISTRY
from .action_utils import bins_to_continuous, continuous_to_bins


@ACTION_REGISTRY.register("discrete_action")
class DiscreteActionHead(nn.Module):
    """Discrete action head with binned outputs.

    Predicts logits over bins for each action dimension.

    Args:
        input_dim: Input feature dimension
        action_dim: Number of action dimensions (e.g., 7 for 7-DOF arm)
        num_bins: Number of discrete bins per dimension
        hidden_dim: Hidden layer dimension
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
            actions: [B, action_dim] continuous actions
            logits: Optional [B, action_dim, num_bins] if return_logits
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

    Predicts mean and log standard deviation for each action dimension.

    Args:
        input_dim: Input feature dimension
        action_dim: Number of action dimensions
        hidden_dim: Hidden layer dimension
        min_std: Minimum standard deviation
        max_std: Maximum standard deviation
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
            sample: Whether to sample from distribution
        Returns:
            actions: [B, action_dim]
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

    Useful when some dimensions benefit from discrete bins
    and others from continuous output.
    """

    def __init__(
        self,
        input_dim: int = 768,
        discrete_dims: int = 6,
        continuous_dims: int = 1,
        num_bins: int = 256,
    ):
        super().__init__()
        self.discrete_head = DiscreteActionHead(
            input_dim, discrete_dims, num_bins
        )
        self.continuous_head = GaussianActionHead(
            input_dim, continuous_dims
        )

    def forward(
        self,
        features: torch.Tensor,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        discrete_actions, logits = self.discrete_head(features, return_logits)
        continuous_actions, std = self.continuous_head(features)

        actions = torch.cat([discrete_actions, continuous_actions], dim=-1)

        if return_logits:
            return actions, {"discrete_logits": logits, "continuous_std": std}
        return actions, None
```

### Step 3: Implement trajectory.py (45 min)
```python
"""Multi-step trajectory prediction."""
import torch
import torch.nn as nn
from typing import Optional, Tuple

from vla.nn import MultiHeadAttention, MLP, RMSNorm
from vla.registry import ACTION_REGISTRY


@ACTION_REGISTRY.register("trajectory_head")
class TrajectoryHead(nn.Module):
    """Predict multiple future action steps.

    Uses autoregressive or parallel prediction based on mode.

    Args:
        input_dim: Input feature dimension
        action_dim: Number of action dimensions
        num_steps: Number of future steps to predict
        num_bins: Bins for discrete actions
        mode: "parallel" or "autoregressive"
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
                nn.TransformerDecoderLayer(input_dim, 8, batch_first=True),
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
            features: [B, D] pooled features
            prev_actions: Optional [B, T, action_dim] for autoregressive
        Returns:
            actions: [B, num_steps, action_dim]
        """
        if features.ndim == 3:
            features = features.mean(dim=1)

        B = features.size(0)

        if self.mode == "parallel":
            logits = self.predictor(features)
            logits = logits.view(B, self.num_steps, self.action_dim, self.num_bins)
            bins = logits.argmax(dim=-1)
            from .action_utils import bins_to_continuous
            return bins_to_continuous(bins, self.num_bins)

        else:
            # Autoregressive decoding
            actions = []
            step_tokens = self.step_embed.weight.unsqueeze(0).expand(B, -1, -1)

            for t in range(self.num_steps):
                query = step_tokens[:, :t+1]
                out = self.transformer(query, features.unsqueeze(1))
                logits = self.output_head(out[:, -1])
                logits = logits.view(B, self.action_dim, self.num_bins)
                bins = logits.argmax(dim=-1)
                from .action_utils import bins_to_continuous
                actions.append(bins_to_continuous(bins, self.num_bins))

            return torch.stack(actions, dim=1)


@ACTION_REGISTRY.register("diffusion_action")
class DiffusionActionHead(nn.Module):
    """Placeholder for diffusion-based action prediction.

    Diffusion policies generate smoother trajectories but require
    multiple denoising steps at inference.

    TODO: Implement DDPM-based action generation.
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

        # Placeholder - full implementation requires DDPM scheduler
        self.net = nn.Linear(input_dim + action_dim * num_steps, action_dim * num_steps)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Generate action trajectory via diffusion.

        Note: This is a placeholder. Full implementation requires
        iterative denoising with proper scheduler.
        """
        B = features.size(0) if features.ndim == 2 else features.size(0)
        if features.ndim == 3:
            features = features.mean(dim=1)

        # For now, just output zeros as placeholder
        return torch.zeros(B, self.num_steps, self.action_dim, device=features.device)
```

### Step 4: Create __init__.py (10 min)
```python
"""Policy and action head modules."""
from .action_heads import DiscreteActionHead, GaussianActionHead, HybridActionHead
from .action_utils import (
    ActionNormalizer,
    continuous_to_bins,
    bins_to_continuous,
    compute_action_loss,
)
from .trajectory import TrajectoryHead, DiffusionActionHead

__all__ = [
    "DiscreteActionHead",
    "GaussianActionHead",
    "HybridActionHead",
    "TrajectoryHead",
    "DiffusionActionHead",
    "ActionNormalizer",
    "continuous_to_bins",
    "bins_to_continuous",
    "compute_action_loss",
]
```

### Step 5: Write tests (45 min)
```python
"""Tests for policy and action heads."""
import pytest
import torch
from vla.policy import (
    DiscreteActionHead,
    GaussianActionHead,
    TrajectoryHead,
    continuous_to_bins,
    bins_to_continuous,
)
from vla.registry import ACTION_REGISTRY


class TestActionUtils:
    def test_bin_conversion_roundtrip(self):
        actions = torch.rand(8, 7) * 2 - 1  # [-1, 1]
        bins = continuous_to_bins(actions, num_bins=256)
        recovered = bins_to_continuous(bins, num_bins=256)
        # Should be close within bin resolution
        assert torch.abs(actions - recovered).max() < 2 / 256

    def test_bin_clamping(self):
        actions = torch.tensor([[1.5, -1.5]])
        bins = continuous_to_bins(actions)
        assert bins.max() <= 255
        assert bins.min() >= 0


class TestDiscreteActionHead:
    def test_output_shape(self):
        head = DiscreteActionHead(input_dim=256, action_dim=7, num_bins=256)
        features = torch.randn(4, 256)
        actions, _ = head(features)
        assert actions.shape == (4, 7)

    def test_action_range(self):
        head = DiscreteActionHead(input_dim=256, action_dim=7)
        features = torch.randn(4, 256)
        actions, _ = head(features)
        assert actions.min() >= -1
        assert actions.max() <= 1

    def test_logits_output(self):
        head = DiscreteActionHead(input_dim=256, action_dim=7, num_bins=256)
        features = torch.randn(4, 256)
        actions, logits = head(features, return_logits=True)
        assert logits.shape == (4, 7, 256)

    def test_sequence_input(self):
        head = DiscreteActionHead(input_dim=256, action_dim=7)
        features = torch.randn(4, 64, 256)  # [B, K, D]
        actions, _ = head(features)
        assert actions.shape == (4, 7)


class TestGaussianActionHead:
    def test_output_shape(self):
        head = GaussianActionHead(input_dim=256, action_dim=7)
        features = torch.randn(4, 256)
        actions, std = head(features)
        assert actions.shape == (4, 7)
        assert std.shape == (4, 7)

    def test_std_bounds(self):
        head = GaussianActionHead(input_dim=256, action_dim=7, min_std=0.01, max_std=1.0)
        features = torch.randn(4, 256)
        _, std = head(features)
        assert std.min() >= 0.01
        assert std.max() <= 1.0

    def test_sampling(self):
        head = GaussianActionHead(input_dim=256, action_dim=7)
        features = torch.randn(4, 256)
        torch.manual_seed(42)
        a1, _ = head(features, sample=True)
        torch.manual_seed(43)
        a2, _ = head(features, sample=True)
        assert not torch.allclose(a1, a2)


class TestTrajectoryHead:
    def test_parallel_prediction(self):
        head = TrajectoryHead(
            input_dim=256, action_dim=7, num_steps=8, mode="parallel"
        )
        features = torch.randn(4, 256)
        actions = head(features)
        assert actions.shape == (4, 8, 7)


class TestRegistry:
    def test_action_heads_registered(self):
        assert "discrete_action" in ACTION_REGISTRY
        assert "gaussian_action" in ACTION_REGISTRY
        assert "trajectory_head" in ACTION_REGISTRY
```

## Todo List
- [ ] Implement action_utils.py (normalization, bin conversion)
- [ ] Implement DiscreteActionHead with bin prediction
- [ ] Implement GaussianActionHead with mean/std
- [ ] Implement HybridActionHead
- [ ] Implement TrajectoryHead for multi-step
- [ ] Create DiffusionActionHead placeholder
- [ ] Register all in ACTION_REGISTRY
- [ ] Write comprehensive tests

## Success Criteria
1. Discrete head outputs valid bin indices
2. Gaussian head outputs bounded actions
3. Trajectory head predicts multiple steps
4. Bin conversion roundtrip preserves values (within resolution)
5. All tests pass

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Action clipping artifacts | Medium | Smooth clipping, document limits |
| Gradient explosion in trajectory | Medium | Gradient clipping, proper init |
| Diffusion slow at inference | High | Optional, document latency |

## Security Considerations
- No external dependencies for action generation
- Bounded action outputs prevent dangerous motions

## Next Steps
- Phase 8: VLA model orchestrating all components
