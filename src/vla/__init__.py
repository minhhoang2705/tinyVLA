"""tinyVLA: Modular Vision-Language-Action research framework."""

__version__ = "0.1.0"

# Import all modules to trigger registry decorators
# This ensures components are registered when the package is imported
from vla import backbones  # noqa: F401
from vla import fusion  # noqa: F401
from vla import models  # noqa: F401
from vla import policy  # noqa: F401

# Import commonly used classes for convenience
from vla.models import VLAConfig, VLAModel

__all__ = [
    "VLAConfig",
    "VLAModel",
]
