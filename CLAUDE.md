# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

tinyVLA is a modular Vision-Language-Action (VLA) research framework for robotics. The codebase follows a **registry-based composition pattern** where independent components (vision encoders, language models, fusion modules, action heads) are assembled via configuration.

**Key Architectural Principle:** Components are frozen backbones (vision/language) plus trained fusion/action layers. All assembly happens through registries, not hardcoded imports.

## Development Commands

### Environment Setup

```bash
# Create virtual environment
conda create -n <ENV_NAME> python=<PYTHON_VERSION> -y
conda activate <ENV_NAME>
# Install with development tools
uv pip install -e ".[dev]"
```

### Code Quality

```bash
# Format code (MUST run before commit)
black src/ tests/

# Lint code (MUST pass before commit)
ruff check src/ tests/

# Type check (MUST pass before push)
mypy src/

# Run all quality checks together
pre-commit run --all-files
```

### Testing

```bash
# Run all tests with coverage report
pytest tests/ --cov=vla --cov-report=html

# Run specific test file
pytest tests/unit/test_registry.py -v

# Run unit tests only (faster)
pytest tests/unit/ -v

# Run integration tests only
pytest tests/integration/ -v

# Run single test by name
pytest tests/unit/test_nn.py::test_attention_output_shape -v

# Run with verbose output and stop on first failure
pytest tests/ -vx
```

**IMPORTANT:** Tests MUST pass before committing. Never ignore failing tests.

### Training (when implemented)

```bash
# Train with default configuration
python scripts/train.py

# Override configuration via CLI
python scripts/train.py model.vision_encoder=dinov2 train.batch_size=32

# Run hyperparameter sweeps
python scripts/train.py --multirun \
  model.vision_encoder=dinov2,siglip \
  train.lr=1e-4,3e-4
```

## Architecture Fundamentals

### Registry Pattern (CRITICAL)

The codebase uses a **typesafe registry pattern** for all components. This is NOT optional—all components MUST be registered to be usable.

**Five Global Registries:**
```python
from vla.registry import (
    VISION_REGISTRY,      # Vision encoders (DINOv2, SigLIP, ViT)
    LANGUAGE_REGISTRY,    # Language models (GPT-2, etc.)
    FUSION_REGISTRY,      # Fusion mechanisms (Perceiver, CrossAttention)
    ACTION_REGISTRY,      # Action heads (Discrete, Gaussian)
    MODEL_REGISTRY        # Complete VLA models
)
```

**Usage Pattern:**
```python
# Register a component (in component definition file)
@VISION_REGISTRY.register("dinov2_base")
class DINOv2(nn.Module):
    def __init__(self, hidden_dim: int = 768):
        ...

# Instantiate from registry
encoder = VISION_REGISTRY.get("dinov2_base", hidden_dim=1024)

# Or via factory functions (for Hydra integration)
from vla.registry.factories import build_vision_encoder
encoder = build_vision_encoder(cfg.vision)
```

**Why This Matters:**
- Components can be swapped via configuration without code changes
- Type-safe: registered components enforce interfaces
- Factory functions bridge Hydra configs to Python objects

### Data Flow Pipeline

Understanding the forward pass is critical for debugging:

```
Images [B, 3, 224, 224] + Text Instructions [B]
        ↓
Vision Encoder (frozen DINOv2/ViT) → Vision Features [B, N=196, D=768]
Language Encoder (frozen GPT-2) → Language Features [B, L, D=768]
        ↓
Fusion Module (TRAINED) → Fused Features [B, K=64, D=768]
        ↓
Action Head (TRAINED) → Actions [B, 7]
```

**Key Constraints:**
- Vision/Language encoders are FROZEN (no gradients) for transfer learning
- Only fusion module and action head are trained
- Fusion module compresses variable-length inputs to fixed-size (K=64 latent tokens)
- Action heads support discrete (256-bin classification) or continuous (Gaussian) outputs

### Module Organization

```
src/vla/
├── registry/          # Component registry system (base.py, factories.py)
├── nn/               # Neural network primitives (attention, MLP, norms)
├── backbones/        # Vision & language encoders (timm/transformers wrappers)
├── fusion/           # Multimodal fusion (Perceiver, CrossAttention, Concat)
├── policy/           # Action heads (discrete binning, Gaussian distributions)
├── models/           # VLA model orchestration (assembles components)
├── data/             # Data loaders & preprocessing (OXE, HDF5, dummy)
├── training/         # PyTorch Lightning training modules
└── utils/            # Logging, config utilities
```

**Dependency Rules:**
- `registry/` has NO dependencies on other vla modules
- All other modules depend on `registry/`
- Circular imports are prevented via lazy imports or `TYPE_CHECKING`
- Each module should be under 200 LOC; split if larger

## Configuration System (Hydra)

**Status:** Configs directory not yet created. When implementing:

```yaml
# configs/config.yaml (main entry point)
defaults:
  - model: vla
  - vision: dinov2
  - language: gpt2
  - fusion: perceiver
  - action: discrete
  - train: default
  - data: dummy

# Override at runtime:
# python scripts/train.py vision=siglip fusion=cross_attn
```

**Hydra Integration Pattern:**
```python
# Factory functions (in registry/factories.py) handle both:
# 1. Registry-based: {"name": "dinov2_base", "hidden_dim": 768}
# 2. Hydra _target_: {"_target_": "timm.create_model", "model_name": "vit_base"}
```

## Code Standards (MANDATORY)

### File Organization

**Naming:** Use kebab-case for file names with descriptive names
- GOOD: `vision-backbone.py`, `perceiver-resampler.py`, `discrete-action-head.py`
- BAD: `vb.py`, `fusion.py`, `head.py`

**Size Limits:** Keep files under 200 LOC
- If a file exceeds 200 lines, split into multiple files
- Extract related functions → new `utils.py`
- Extract classes → new `impl_name.py`

### Type Hints (REQUIRED)

All public functions MUST have type hints:

```python
# ✓ CORRECT
def forward_pass(
    images: torch.Tensor,
    texts: List[str],
    return_features: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Process images and texts through VLA model."""
    ...

# ✗ WRONG (no type hints)
def forward_pass(images, texts, return_features=False):
    ...
```

### Docstrings (REQUIRED)

Use NumPy-style docstrings for all public functions/classes:

```python
def forward_pass(images: torch.Tensor, texts: List[str]) -> torch.Tensor:
    """Process images and texts through VLA model.

    Combines vision and language encoders with fusion mechanism
    to predict robot actions.

    Args:
        images: Image batch [B, C, H, W] normalized to [0, 1]
        texts: List of instruction strings [B]

    Returns:
        actions: Predicted actions [B, action_dim]

    Raises:
        ValueError: If image shape is invalid

    Example:
        >>> images = torch.randn(2, 3, 224, 224)
        >>> texts = ["pick up cube", "move to target"]
        >>> actions = model.forward_pass(images, texts)
        >>> print(actions.shape)
        torch.Size([2, 7])
    """
```

### Error Handling

**Be specific; let unexpected errors propagate:**

```python
# ✓ CORRECT: Specific exception with context
def load_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    if path.suffix != ".pt":
        raise ValueError(f"Expected .pt file, got {path.suffix}")
    try:
        checkpoint = torch.load(path, map_location="cpu")
    except RuntimeError as e:
        raise RuntimeError(f"Failed to load checkpoint {path}: {e}") from e
    return checkpoint

# ✗ WRONG: Too broad, swallows context
def load_checkpoint(path):
    try:
        return torch.load(path)
    except:  # Never use bare except
        return {}
```

### Logging (REQUIRED)

**Use logging module; NEVER use print():**

```python
from vla.utils import setup_logger

logger = setup_logger(__name__)

# ✓ CORRECT
logger.info(f"Training epoch {epoch}, loss={loss:.4f}")
logger.warning("GPU memory usage high: {memory_mb}MB")
logger.error("Failed to load checkpoint", exc_info=True)

# ✗ WRONG
print(f"Training epoch {epoch}")
print("ERROR: Failed to load checkpoint")
```

## Testing Requirements

### Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── unit/                    # Unit tests (test individual components)
│   ├── test_registry.py
│   ├── test_nn.py
│   ├── test_backbones.py
│   ├── test_fusion.py
│   └── test_policy.py
└── integration/             # Integration tests (test component interactions)
    ├── test_vla_pipeline.py
    └── test_training.py
```

### Shared Fixtures (from conftest.py)

```python
# Available fixtures for all tests:
# - device: torch.device (cuda or cpu)
# - batch_size: int (default 2)
# - seq_length: int (default 10)
# - dummy_image: torch.Tensor [B, 3, 224, 224]
# - dummy_text: List[str] (sample instructions)
# - dummy_actions: torch.Tensor [B, 7]
# - seed: int (for reproducibility)
```

### Test Naming Convention

```python
# ✓ CORRECT: Descriptive test names
def test_perceiver_resampler_output_shape():
    """Test Perceiver produces fixed-size latent."""
    ...

def test_vision_backbone_freeze_disables_gradients():
    """Test frozen backbone has no gradients."""
    ...

# ✗ WRONG: Vague test names
def test_perceiver():
    ...
```

### Coverage Requirements

- Aim for 80%+ test coverage
- Run `pytest tests/ --cov=vla --cov-report=html` to generate coverage report
- View report at `htmlcov/index.html`

## Performance Considerations

### Memory Efficiency

```python
# ✓ Use gradient checkpointing for large models
from torch.utils.checkpoint import checkpoint

class LargeModel(nn.Module):
    def forward(self, x):
        x = checkpoint(self.layer1, x)  # Recompute during backward
        x = checkpoint(self.layer2, x)
        return x

# ✓ Freeze backbones to save memory
for param in self.vision_encoder.parameters():
    param.requires_grad = False
```

### Computation Efficiency

```python
# ✓ Use einops for readable tensor operations
from einops import rearrange, reduce

features = rearrange(features, "b (n d) -> b n d", d=768)

# ✓ Vectorize operations (avoid for-loops over batch)
# DON'T: for img in images: process(img)
# DO: process(images)  # Batch-wise processing
```

## Common Pitfalls

1. **Forgetting to register components:** If you create a new backbone/fusion/action head, you MUST register it via `@REGISTRY.register("name")`

2. **Breaking frozen backbones:** Vision/language encoders should be frozen. Never call `.train()` on them or set `requires_grad=True`

3. **Incorrect tensor shapes:** Pay attention to batch dimensions [B, ...]. Common errors:
   - Vision features: [B, N, D] where N = num_patches (e.g., 196 for ViT)
   - Language features: [B, L, D] where L = max_seq_len
   - Fused features: [B, K, D] where K = num_latents (fixed, e.g., 64)
   - Actions: [B, action_dim] where action_dim typically = 7 for robots

4. **Using print() instead of logger:** Always use `logger.info/warning/error`

5. **Skipping type hints:** All public functions need type hints. Run `mypy src/` to check

6. **Not running tests:** Tests MUST pass before commit. Run `pytest tests/` locally

## Commit Message Format

Use conventional commits:

```
<type>(<scope>): <subject>

Examples:
feat(registry): add generic component registry with type safety
fix(fusion): correct perceiver resampler output shape bug
docs(readme): add quick start guide and examples
test(backbones): add unit tests for vision encoder freezing
refactor(data): extract dataset mixing logic to separate module
```

**Types:** `feat`, `fix`, `refactor`, `docs`, `test`, `chore`

## Pre-Commit Checklist

Before committing, ensure:
- [ ] `black src/ tests/` passes (code formatted)
- [ ] `ruff check src/ tests/` passes (no linting errors)
- [ ] `mypy src/` passes (no type errors)
- [ ] `pytest tests/` passes (all tests pass)
- [ ] No hardcoded secrets or API keys
- [ ] Docstrings added for public functions/classes
- [ ] No `print()` statements (use logger)
- [ ] Type hints on all public functions

## Documentation References

For deeper understanding, refer to:
- `docs/system-architecture.md` - Component interactions and data flow
- `docs/code-standards.md` - Detailed coding conventions
- `docs/codebase-summary.md` - Module organization
- `docs/project-overview-pdr.md` - Project requirements and scope
- `docs/tech-stack.md` - Technology choices and rationale

## Important Notes

- **Registry Pattern is Core:** Every component MUST be registered. This is NOT optional.
- **Frozen Backbones:** Vision/language encoders are ALWAYS frozen during training.
- **Fixed-Size Bottleneck:** Fusion module compresses to fixed K latent tokens (default 64).
- **Test-Driven Development:** Write tests for new components. Coverage should be 80%+.
- **No Configuration Yet:** `configs/` directory not implemented; Hydra integration is planned.
