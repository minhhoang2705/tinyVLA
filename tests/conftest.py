"""Pytest configuration and shared fixtures."""


import pytest
import torch

# Import component modules to trigger @REGISTRY.register() decorators
# This populates VISION_REGISTRY, LANGUAGE_REGISTRY, FUSION_REGISTRY, ACTION_REGISTRY
from vla import backbones, fusion, models, policy  # noqa: F401


@pytest.fixture
def device():
    """Return available compute device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def batch_size():
    """Default batch size for tests."""
    return 2


@pytest.fixture
def seq_length():
    """Default sequence length for tests."""
    return 10


@pytest.fixture
def dummy_image(batch_size):
    """Generate dummy image tensor (B, C, H, W) in range [0, 1]."""
    return torch.rand(batch_size, 3, 224, 224)  # Uniform [0, 1] for valid images


@pytest.fixture
def dummy_text():
    """Sample instruction texts."""
    return [
        "pick up the red block",
        "place the cup on the table",
    ]


@pytest.fixture
def dummy_actions(batch_size):
    """Generate dummy action tensor (B, action_dim) in range [-1, 1]."""
    action_dim = 7  # Common robot action dimension
    return torch.rand(batch_size, action_dim) * 2 - 1  # Uniform [-1, 1] for valid actions


@pytest.fixture
def seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    return 42
