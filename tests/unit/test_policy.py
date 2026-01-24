"""Tests for policy and action heads."""

import torch

from vla.policy import (
    ActionNormalizer,
    DiffusionActionHead,
    DiscreteActionHead,
    GaussianActionHead,
    HybridActionHead,
    TrajectoryHead,
    bins_to_continuous,
    compute_action_loss,
    continuous_to_bins,
)
from vla.registry import ACTION_REGISTRY


class TestActionUtils:
    """Tests for action utility functions."""

    def test_bin_conversion_roundtrip(self):
        """Test that converting to bins and back preserves values."""
        actions = torch.rand(8, 7) * 2 - 1  # [-1, 1]
        bins = continuous_to_bins(actions, num_bins=256)
        recovered = bins_to_continuous(bins, num_bins=256)
        # Should be close within bin resolution
        assert torch.abs(actions - recovered).max() < 2 / 256

    def test_bin_clamping(self):
        """Test that out-of-range actions are clamped to valid bins."""
        actions = torch.tensor([[1.5, -1.5]])
        bins = continuous_to_bins(actions, num_bins=256)
        assert bins.max() <= 255
        assert bins.min() >= 0

    def test_bin_boundary_values(self):
        """Test that boundary values map to correct bins."""
        actions = torch.tensor([[-1.0, 0.0, 1.0]])
        bins = continuous_to_bins(actions, num_bins=256)
        assert bins[0, 0].item() == 0  # -1.0 -> bin 0
        assert bins[0, 1].item() == 128  # 0.0 -> bin 128 (banker's rounding: 127.5 → 128)
        assert bins[0, 2].item() == 255  # 1.0 -> bin 255

    def test_action_normalizer(self):
        """Test action normalizer normalize/denormalize roundtrip."""
        action_min = torch.tensor([0.0, -1.0, -2.0])
        action_max = torch.tensor([1.0, 1.0, 2.0])
        normalizer = ActionNormalizer(action_min, action_max)

        actions = torch.tensor([[0.5, 0.0, 0.0]])
        normalized = normalizer.normalize(actions)
        recovered = normalizer.denormalize(normalized)

        assert torch.allclose(actions, recovered, atol=1e-6)
        assert normalized.min() >= -1.0
        assert normalized.max() <= 1.0

    def test_compute_action_loss(self):
        """Test action loss computation."""
        pred_logits = torch.randn(4, 7, 256)
        target_actions = torch.rand(4, 7) * 2 - 1
        loss = compute_action_loss(pred_logits, target_actions, num_bins=256)

        assert loss.shape == torch.Size([])
        assert loss.item() >= 0
        assert torch.isfinite(loss)


class TestDiscreteActionHead:
    """Tests for discrete action head."""

    def test_output_shape(self):
        """Test output shape matches expected dimensions."""
        head = DiscreteActionHead(input_dim=256, action_dim=7, num_bins=256)
        features = torch.randn(4, 256)
        actions, _ = head(features)
        assert actions.shape == (4, 7)

    def test_action_range(self):
        """Test actions are in valid range [-1, 1]."""
        head = DiscreteActionHead(input_dim=256, action_dim=7, num_bins=256)
        features = torch.randn(4, 256)
        actions, _ = head(features)
        assert actions.min() >= -1.0
        assert actions.max() <= 1.0

    def test_logits_output(self):
        """Test logits are returned when requested."""
        head = DiscreteActionHead(input_dim=256, action_dim=7, num_bins=256)
        features = torch.randn(4, 256)
        actions, logits = head(features, return_logits=True)
        assert logits is not None
        assert logits.shape == (4, 7, 256)

    def test_sequence_input(self):
        """Test head can process sequence inputs."""
        head = DiscreteActionHead(input_dim=256, action_dim=7, num_bins=256)
        features = torch.randn(4, 64, 256)  # [B, K, D]
        actions, _ = head(features)
        assert actions.shape == (4, 7)

    def test_gradient_flow(self):
        """Test gradients flow through the head."""
        head = DiscreteActionHead(input_dim=256, action_dim=7, num_bins=256)
        features = torch.randn(4, 256, requires_grad=True)
        actions, logits = head(features, return_logits=True)

        # Create dummy loss
        target = torch.rand(4, 7) * 2 - 1
        loss = compute_action_loss(logits, target, num_bins=256)
        loss.backward()

        assert features.grad is not None
        assert torch.isfinite(features.grad).all()


class TestGaussianActionHead:
    """Tests for Gaussian action head."""

    def test_output_shape(self):
        """Test output shapes for actions and std."""
        head = GaussianActionHead(input_dim=256, action_dim=7)
        features = torch.randn(4, 256)
        actions, std = head(features)
        assert actions.shape == (4, 7)
        assert std.shape == (4, 7)

    def test_std_bounds(self):
        """Test std is within specified bounds."""
        head = GaussianActionHead(input_dim=256, action_dim=7, min_std=0.01, max_std=1.0)
        features = torch.randn(4, 256)
        _, std = head(features)
        assert std.min() >= 0.01
        assert std.max() <= 1.0

    def test_sampling(self):
        """Test sampling produces different results."""
        head = GaussianActionHead(input_dim=256, action_dim=7)
        features = torch.randn(4, 256)

        torch.manual_seed(42)
        a1, _ = head(features, sample=True)
        torch.manual_seed(43)
        a2, _ = head(features, sample=True)

        assert not torch.allclose(a1, a2)

    def test_deterministic_mode(self):
        """Test deterministic mode returns same actions."""
        head = GaussianActionHead(input_dim=256, action_dim=7)
        features = torch.randn(4, 256)

        a1, _ = head(features, sample=False)
        a2, _ = head(features, sample=False)

        assert torch.allclose(a1, a2)

    def test_action_range(self):
        """Test actions are clamped to [-1, 1]."""
        head = GaussianActionHead(input_dim=256, action_dim=7)
        features = torch.randn(4, 256)
        actions, _ = head(features)
        assert actions.min() >= -1.0
        assert actions.max() <= 1.0


class TestHybridActionHead:
    """Tests for hybrid action head."""

    def test_output_shape(self):
        """Test output combines discrete and continuous dimensions."""
        head = HybridActionHead(input_dim=256, discrete_dims=6, continuous_dims=1, num_bins=256)
        features = torch.randn(4, 256)
        actions, _ = head(features)
        assert actions.shape == (4, 7)  # 6 + 1

    def test_info_output(self):
        """Test info dict is returned when requested."""
        head = HybridActionHead(input_dim=256, discrete_dims=6, continuous_dims=1, num_bins=256)
        features = torch.randn(4, 256)
        actions, info = head(features, return_logits=True)

        assert info is not None
        assert "discrete_logits" in info
        assert "continuous_std" in info
        assert info["discrete_logits"].shape == (4, 6, 256)
        assert info["continuous_std"].shape == (4, 1)


class TestTrajectoryHead:
    """Tests for trajectory prediction head."""

    def test_parallel_prediction(self):
        """Test parallel prediction produces correct shape."""
        head = TrajectoryHead(input_dim=256, action_dim=7, num_steps=8, mode="parallel")
        features = torch.randn(4, 256)
        actions = head(features)
        assert actions.shape == (4, 8, 7)

    def test_autoregressive_prediction(self):
        """Test autoregressive prediction produces correct shape."""
        head = TrajectoryHead(input_dim=256, action_dim=7, num_steps=8, mode="autoregressive")
        features = torch.randn(4, 256)
        actions = head(features)
        assert actions.shape == (4, 8, 7)

    def test_action_range(self):
        """Test trajectory actions are in valid range."""
        head = TrajectoryHead(input_dim=256, action_dim=7, num_steps=8)
        features = torch.randn(4, 256)
        actions = head(features)
        assert actions.min() >= -1.0
        assert actions.max() <= 1.0

    def test_sequence_input(self):
        """Test trajectory head handles sequence input."""
        head = TrajectoryHead(input_dim=256, action_dim=7, num_steps=8)
        features = torch.randn(4, 64, 256)
        actions = head(features)
        assert actions.shape == (4, 8, 7)


class TestDiffusionActionHead:
    """Tests for diffusion action head (placeholder)."""

    def test_output_shape(self):
        """Test placeholder returns correct shape."""
        head = DiffusionActionHead(input_dim=256, action_dim=7, num_steps=8)
        features = torch.randn(4, 256)
        actions = head(features)
        assert actions.shape == (4, 8, 7)

    def test_placeholder_zeros(self):
        """Test placeholder returns zeros."""
        head = DiffusionActionHead(input_dim=256, action_dim=7, num_steps=8)
        features = torch.randn(4, 256)
        actions = head(features)
        assert torch.allclose(actions, torch.zeros_like(actions))


class TestRegistry:
    """Tests for action registry integration."""

    def test_action_heads_registered(self):
        """Test all action heads are registered."""
        assert "discrete_action" in ACTION_REGISTRY
        assert "gaussian_action" in ACTION_REGISTRY
        assert "hybrid_action" in ACTION_REGISTRY
        assert "trajectory_head" in ACTION_REGISTRY
        assert "diffusion_action" in ACTION_REGISTRY

    def test_registry_instantiation(self):
        """Test registry can instantiate components."""
        head = ACTION_REGISTRY.get("discrete_action", input_dim=256, action_dim=7, num_bins=256)
        assert isinstance(head, DiscreteActionHead)

        features = torch.randn(4, 256)
        actions, _ = head(features)
        assert actions.shape == (4, 7)
