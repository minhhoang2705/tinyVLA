# Code Standards & Implementation Guidelines

## 1. File Organization

### Directory Structure Conventions

**Module Naming:** Use `kebab-case` for clarity
```
src/vla/module_name/
├── __init__.py          # Public API exports
├── core.py              # Main implementation (if >50 lines)
├── utils.py             # Module-specific utilities
└── tests.py             # Module tests (optional, prefer tests/ dir)
```

**File Naming:** Descriptive kebab-case names that indicate purpose
```
GOOD:
- vision-backbone.py
- perceiver-resampler.py
- discrete-action-head.py
- oxe-dataloader.py

AVOID:
- vb.py
- fusion.py
- head.py
- loader.py
```

### Module Size Limits

**Target:** Keep individual Python files under 200 lines of code (LOC)

**Rationale:**
- Improves code readability and maintainability
- Forces single-responsibility principle
- Reduces cognitive load during review
- Easier to test and debug

**Splitting Strategy:**
If a file exceeds 200 LOC:
1. Extract related functions → new `utils.py` file
2. Extract classes → new `impl_name.py` file
3. Move tests → `tests/unit/test_module_name.py`

**Example Refactoring:**
```
❌ fusion.py (250 LOC)
├── PerceiverResampler class (80 LOC)
├── CrossAttentionFusion class (70 LOC)
├── AdapterFusion class (50 LOC)
└── Utility functions (50 LOC)

✓ fusion/ directory
├── __init__.py
├── perceiver-resampler.py (80 LOC)
├── cross-attention.py (70 LOC)
├── adapter.py (50 LOC)
└── utils.py (50 LOC)
```

## 2. Code Style

### Python Version & Features

**Target:** Python 3.10+

**Allowed Modern Features:**
- Type hints (required for all public functions)
- Match statements (`match`/`case`)
- Walrus operator (`:=`)
- f-strings (required over `%` or `.format()`)
- Dataclasses (use over `__init__` boilerplate)
- `TypedDict` / `NamedTuple` for structured data

### Formatting: Black

**Configuration (pyproject.toml):**
```toml
[tool.black]
line-length = 100
target-version = ["py310"]
```

**Requirements:**
- Run before every commit: `black src/ tests/`
- Max line length: 100 characters
- 4-space indentation (Black enforces)

**Black Overrides:** Rare, use `# fmt: off` / `# fmt: on` only for:
- Complex data structures requiring alignment
- DSL-like code (Hydra configs, YAML structures)
```python
# fmt: off
CONFIG_MAPPING = {
    "dinov2_small":  {"size": "s",  "patch": 14},
    "dinov2_base":   {"size": "b",  "patch": 14},
    "siglip_small":  {"size": "s",  "patch": 16},
}
# fmt: on
```

### Linting: Ruff

**Configuration (pyproject.toml):**
```toml
[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "W", "B", "C90"]
ignore = ["E501", "B008"]

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401", "F403"]
```

**Rules Enforced:**
- **E**: PEP 8 errors
- **F**: PyFlakes (undefined names, unused imports)
- **I**: isort (import ordering)
- **W**: Warnings
- **B**: flake8-bugbear (common bugs)
- **C90**: McCabe complexity (max 10)

**Run:** `ruff check src/ tests/`

### Type Hints: mypy

**Configuration (pyproject.toml):**
```toml
[tool.mypy]
python_version = "3.10"
warn_return_any = true
check_untyped_defs = true
disallow_untyped_defs = false
```

**Requirements:**
- Function signatures: Always include parameter + return types
- Class attributes: Type hint in class definition or via dataclass
- Complex types: Use `typing` module or `typing_extensions`

**Examples:**
```python
# ✓ Good
def load_model(model_name: str, device: torch.device) -> nn.Module:
    """Load pretrained model."""
    ...

# ✓ Dataclass with types
@dataclass
class VLAConfig:
    vision_size: str
    fusion_dim: int
    num_action_bins: int = 256

# ✓ Complex types
from typing import Optional, Dict, List
def process_batch(
    images: torch.Tensor,
    texts: List[str],
    config: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    ...
```

**Type Checking:** `mypy src/` (run before push)

## 3. Imports & Module Organization

### Import Ordering

Follow isort/Ruff convention:
1. Standard library imports
2. Third-party imports (alphabetical)
3. Local imports (relative)

```python
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.nn import functional as F

from vla.registry import VISION_REGISTRY
from vla.utils import setup_logger
```

### Import Practices

**DO:**
- Import modules, not individual functions (unless extremely common): `import torch` not `from torch import tensor`
- Use relative imports within package: `from .registry import Registry`
- Import types via `TYPE_CHECKING` for circular dependency avoidance

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vla.models import VLAModel
```

**DON'T:**
- Use wildcard imports (except in `__init__.py`): `from module import *`
- Import inside functions (except for expensive/optional dependencies)
- Create circular dependencies

### Public API Definition

Each module defines `__all__` to document public API:

```python
# src/vla/registry/__init__.py
from .core import Registry, register, get

__all__ = [
    "Registry",
    "register",
    "get",
    "VISION_REGISTRY",
    "LANGUAGE_REGISTRY",
]

VISION_REGISTRY = Registry("vision")
LANGUAGE_REGISTRY = Registry("language")
```

## 4. Docstring Conventions

**Format:** Numpy-style docstrings (consistent with PyTorch ecosystem)

**Required For:**
- All public functions/classes
- All module-level exports

**Optional For:**
- Simple internal utility functions
- Property getters/setters

**Template: Function**
```python
def forward_pass(
    images: torch.Tensor,
    texts: List[str],
    return_features: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Process images and texts through VLA model.

    Combines vision and language encoders with fusion mechanism
    to predict robot actions.

    Args:
        images: Image batch [B, C, H, W] normalized to [0, 1]
        texts: List of instruction strings [B]
        return_features: If True, return intermediate features

    Returns:
        actions: Predicted actions [B, action_dim] if return_features=False
        (actions, features): Tuple of actions and fused features if True

    Raises:
        ValueError: If image shape is invalid
        RuntimeError: If model not on same device as inputs

    Notes:
        - Vision encoder is frozen by default (no gradients)
        - Text is tokenized using GPT-2 tokenizer internally
        - Output actions are in range [-1, 1] for continuous,
          or [0, 255] for discrete binning

    Example:
        >>> images = torch.randn(2, 3, 224, 224)
        >>> texts = ["pick up the cube", "move to target"]
        >>> actions = model.forward_pass(images, texts)
        >>> print(actions.shape)
        torch.Size([2, 7])
    """
```

**Template: Class**
```python
class PerceiverResampler(nn.Module):
    """Fixed-size latent bottleneck fusion mechanism.

    Compresses variable-length visual sequences into fixed K tokens
    via cross-attention. Proven in Flamingo, RT-2, OpenVLA.

    Attributes:
        latent_dim: Dimension of latent tokens (default: 768)
        num_latents: Number of latent vectors K (default: 64)
        num_heads: Multi-head attention heads (default: 8)

    Args:
        latent_dim: Latent feature dimension
        num_latents: Number of latent queries
        num_heads: Number of attention heads
        num_layers: Number of transformer layers

    Example:
        >>> perceiver = PerceiverResampler(latent_dim=768, num_latents=64)
        >>> images = torch.randn(2, 256, 768)  # [B, N, D]
        >>> text = torch.randn(2, 32, 768)     # [B, L, D]
        >>> output = perceiver(images, text)
        >>> print(output.shape)
        torch.Size([2, 64, 768])
    """
```

## 5. Error Handling

### Exception Strategy

**Principle:** Be specific about error conditions; let unexpected errors propagate

```python
# ✓ Good: Specific exceptions with context
def load_checkpoint(path: Path) -> Dict[str, Any]:
    """Load model checkpoint from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if path.suffix != ".pt":
        raise ValueError(f"Expected .pt file, got {path.suffix}")

    try:
        checkpoint = torch.load(path, map_location="cpu")
    except RuntimeError as e:
        raise RuntimeError(f"Failed to load checkpoint {path}: {e}") from e

    return checkpoint

# ✗ Bad: Too broad, swallows important context
def load_checkpoint(path):
    try:
        return torch.load(path)
    except:
        return {}

# ✗ Bad: Not specific enough
def load_checkpoint(path):
    if not path.exists():
        print("Error!")
        return None
```

### Try-Except Usage

**When to use try-except:**
1. Handling expected exceptions (file not found, network error)
2. Wrapping external library calls that might fail
3. Cleanup operations (use `finally` or context managers)

**When NOT to use:**
1. Control flow (don't use exceptions for normal cases)
2. Validating inputs (use assertions or early returns)
3. Catching bare `Exception` (always specify exception type)

```python
# ✓ Good: Context manager for resource cleanup
def process_dataset(hdf5_path: Path) -> torch.Tensor:
    """Load and process HDF5 dataset."""
    with h5py.File(hdf5_path, "r") as f:
        data = torch.from_numpy(f["data"][:])
    return data

# ✗ Bad: Manual try-finally
def process_dataset(hdf5_path):
    f = h5py.File(hdf5_path, "r")
    try:
        data = torch.from_numpy(f["data"][:])
    finally:
        f.close()
    return data
```

### Logging over Print Statements

**Always use logging module; avoid print():**

```python
from vla.utils import setup_logger

logger = setup_logger(__name__)

# ✓ Good
logger.info(f"Training epoch {epoch}, loss={loss:.4f}")
logger.warning("GPU memory usage high: {memory_mb}MB")
logger.error("Failed to load checkpoint", exc_info=True)

# ✗ Bad
print(f"Training epoch {epoch}")
print("ERROR: Failed to load checkpoint")
```

## 6. Common Patterns

### Registry Pattern

```python
# src/vla/registry/core.py
class Registry:
    """Generic component registry."""

    def __init__(self, name: str):
        self._name = name
        self._components: Dict[str, Any] = {}

    def register(self, name: str):
        """Decorator to register component."""
        def decorator(cls):
            if name in self._components:
                raise ValueError(f"Already registered: {name}")
            self._components[name] = cls
            return cls
        return decorator

    def get(self, name: str, **kwargs) -> Any:
        """Instantiate registered component."""
        if name not in self._components:
            raise KeyError(f"Not found: {name}. Available: {list(self._components.keys())}")
        return self._components[name](**kwargs)

# Usage
VISION_REGISTRY = Registry("vision")

@VISION_REGISTRY.register("dinov2")
class DINOv2(nn.Module):
    ...

# Instantiate from config
encoder = VISION_REGISTRY.get("dinov2", size="base")
```

### Dataclass Configuration

```python
from dataclasses import dataclass, field

@dataclass
class VLAConfig:
    """VLA model configuration."""

    # Vision encoder
    vision_encoder: str = "dinov2"
    vision_size: str = "base"
    vision_freeze: bool = True

    # Language model
    language_model: str = "gpt2"
    language_freeze: bool = True

    # Fusion
    fusion_type: str = "perceiver"
    fusion_dim: int = 768
    latent_dim: int = 64

    # Action head
    action_type: str = "discrete"
    action_dim: int = 7
    num_action_bins: int = 256

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Hydra."""
        return asdict(self)
```

### Frozen Module Pattern (for feature extraction)

```python
class FrozenBackbone(nn.Module):
    """Wrapper that freezes backbone for transfer learning."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        # Freeze all parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass (no gradients)."""
        with torch.no_grad():
            return self.backbone(x)
```

## 7. Testing Standards

### Test File Organization

```
tests/
├── conftest.py              # Shared fixtures
├── unit/
│   ├── test_registry.py
│   ├── test_nn.py
│   ├── test_backbones.py
│   ├── test_fusion.py
│   ├── test_policy.py
│   └── test_models.py
└── integration/
    ├── test_vla_pipeline.py
    ├── test_training.py
    └── test_data_loading.py
```

### Test Naming Conventions

```python
# ✓ Good: Clear, descriptive test names
def test_perceiver_resampler_output_shape():
    """Test Perceiver produces fixed-size latent."""
    ...

def test_vision_backbone_freeze_disables_gradients():
    """Test frozen backbone has no gradients."""
    ...

def test_discrete_action_head_with_invalid_input_raises_error():
    """Test validation of input shape."""
    ...

# ✗ Bad: Vague test names
def test_perceiver():
    ...

def test_forward():
    ...

def test_error():
    ...
```

### Fixture Usage

```python
def test_vla_model_forward_pass(dummy_image, dummy_text, device):
    """Test end-to-end forward pass."""
    model = VLAModel(...)
    model.to(device)

    images = dummy_image.to(device)
    actions = model(images, dummy_text)

    assert actions.shape == (2, 7)  # batch_size=2, action_dim=7
```

## 8. Performance Considerations

### Memory Efficiency

```python
# ✓ Good: Avoid storing intermediate activations
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Use in-place operations sparingly
    x = self.conv(x)
    x = self.relu(x)  # In-place would be x.relu_()
    return x

# ✓ Gradient checkpointing for large models
class LargeModel(nn.Module):
    def forward(self, x):
        # Save memory by recomputing activations during backward
        x = torch.utils.checkpoint.checkpoint(self.layer1, x)
        x = torch.utils.checkpoint.checkpoint(self.layer2, x)
        return x
```

### Computation Efficiency

```python
# ✓ Use einops for readable tensor operations
from einops import rearrange, reduce

# Clearer than torch.reshape / torch.view
features = rearrange(features, "b (n d) -> b n d", d=768)

# ✓ Vectorize operations (avoid for-loops)
# Don't: for img in images: process(img)
# Do: process(images)  # Batch-wise processing
```

## 9. Security Considerations

### No Hardcoded Secrets

```python
# ✗ Bad: Never commit API keys
WANDB_API_KEY = "xxx-yyy-zzz"

# ✓ Good: Load from environment
import os
wandb_api_key = os.getenv("WANDB_API_KEY")
if not wandb_api_key:
    raise RuntimeError("Set WANDB_API_KEY environment variable")
```

### Validate External Input

```python
def create_model(config: Dict[str, Any]) -> nn.Module:
    """Create model from untrusted config."""

    # Validate required keys
    required_keys = {"vision_encoder", "action_dim"}
    if not required_keys.issubset(config.keys()):
        raise KeyError(f"Missing keys: {required_keys - set(config.keys())}")

    # Validate types
    if not isinstance(config["action_dim"], int) or config["action_dim"] <= 0:
        raise ValueError(f"action_dim must be positive int, got {config['action_dim']}")

    # Only allow known components
    allowed_encoders = {"dinov2", "siglip", "vit"}
    if config["vision_encoder"] not in allowed_encoders:
        raise ValueError(f"Unknown encoder: {config['vision_encoder']}")

    return VLAModel(config)
```

## 10. Commit Message Format

Use conventional commits:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:** `feat`, `fix`, `refactor`, `docs`, `test`, `chore`

**Examples:**
```
feat(registry): add generic component registry with type safety

fix(fusion): correct perceiver resampler output shape bug

docs(readme): add quick start guide and examples

test(backbones): add unit tests for vision encoder freezing

refactor(data): extract dataset mixing logic to separate module
```

## 11. Documentation Standards

**Module-level docstring:** Required at top of every `.py` file

```python
"""Vision backbone implementations for VLA models.

Provides wrappers for pretrained vision encoders (DINOv2, SigLIP, ViT)
from timm library. Supports freezing for transfer learning.

Examples:
    Load DINOv2 from timm:
    >>> from vla.backbones import DINOv2
    >>> encoder = DINOv2(size="base", pretrained=True, freeze=True)
    >>> features = encoder(images)  # [B, N, D]
"""
```

## 12. Checklists

### Before Committing Code

- [ ] `black src/ tests/` passes (code formatted)
- [ ] `ruff check src/ tests/` passes (no linting errors)
- [ ] `mypy src/` passes (no type errors)
- [ ] Tests pass: `pytest tests/ -v`
- [ ] No hardcoded secrets or credentials
- [ ] Docstrings added for public functions/classes
- [ ] No print() statements (use logger)
- [ ] No unused imports or variables

### Before Code Review

- [ ] README updated if public API changed
- [ ] Comments explain "why", not "what" (code explains "what")
- [ ] File size under 200 LOC; split if necessary
- [ ] Test coverage for new code (aim for 80%+)
- [ ] No breaking changes without migration guide
- [ ] Performance-critical code profiled

---

**Last Updated:** 2026-01-22 | **Version:** 1.0
