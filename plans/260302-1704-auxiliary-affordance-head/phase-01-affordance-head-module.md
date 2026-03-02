# Phase 01: Create AffordanceHead Module

**Status:** Pending
**Priority:** High (blocks phases 02-03)

## Overview

Create a standalone `AffordanceHead` MLP that predicts block affordance state `[block_x, block_y, block_angle]` from mean-pooled fused latents. Separate file to keep `vla_base.py` focused.

## Architecture

```
fused_features [B, K, D]
       ↓ mean-pool
  pooled [B, D=768]
       ↓ Linear(768 → 256)
       ↓ GELU
       ↓ Linear(256 → 3)
  state_pred [B, 3]   ← normalized to [-1, 1]
```

## Related Code Files

- **Create:** `src/vla/policy/affordance-head.py`
- **Modify:** `src/vla/policy/__init__.py` — re-export `AffordanceHead`

## Implementation Steps

### 1. Create `src/vla/policy/affordance-head.py`

```python
"""Auxiliary affordance head for block state prediction.

Predicts block position + angle as auxiliary supervision signal.
Attached to mean-pooled fused latents from the VLAModel fusion module.
"""

import torch
import torch.nn as nn
from vla.utils import setup_logger

logger = setup_logger(__name__)


class AffordanceHead(nn.Module):
    """Small MLP predicting block affordance state (position + angle).

    Args:
        input_dim: Dimension of pooled fused latents (default: 768)
        hidden_dim: Hidden layer size (default: 256)
        state_dim: Output state dimension; 3 = [block_x, block_y, block_angle]

    Example:
        >>> head = AffordanceHead(input_dim=768, hidden_dim=256, state_dim=3)
        >>> features = torch.randn(4, 768)
        >>> state_pred = head(features)
        >>> state_pred.shape
        torch.Size([4, 3])
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        state_dim: int = 3,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )
        logger.info(
            f"AffordanceHead: ({input_dim} → {hidden_dim} → {state_dim})"
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict affordance state from fused latents.

        Args:
            features: [B, D] or [B, K, D] fused latents

        Returns:
            state_pred: [B, state_dim] predicted state (unnormalized)
        """
        # Mean-pool sequence dimension if present
        if features.ndim == 3:
            features = features.mean(dim=1)  # [B, K, D] → [B, D]
        return self.mlp(features)

    def compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """MSE loss between predicted and target state.

        Args:
            pred: Predicted state [B, state_dim]
            target: Normalized ground-truth state [B, state_dim] in [-1, 1]

        Returns:
            Scalar MSE loss
        """
        return nn.functional.mse_loss(pred, target)
```

### 2. Update `src/vla/policy/__init__.py`

Add `AffordanceHead` to exports.

## Common Pitfalls

- **Do NOT register in ACTION_REGISTRY** — affordance head is auxiliary, not an action head
- **Mean-pool inside forward()** — keeps the interface clean; caller passes raw fused features
- **No tanh/clamp on output** — MSE loss doesn't require bounded output; target is pre-normalized

## Success Criteria

- [ ] `AffordanceHead(768, 256, 3)(torch.randn(4, 64, 768)).shape == (4, 3)`
- [ ] `compute_loss()` returns scalar tensor
- [ ] Handles both `[B, D]` and `[B, K, D]` input shapes
