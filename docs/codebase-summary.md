# Codebase Summary - tinyVLA

## Overview

tinyVLA is in **Phase 2 complete** (0.1.0) with 621 lines of implemented code across 14 Python modules. The registry pattern is fully operational with factory functions enabling dynamic component loading. Remaining modules await implementation.

**Current State:**
- Architecture: Complete blueprint
- Infrastructure: Fully configured (Hydra, PyTorch Lightning, testing framework)
- Implementation: Registry pattern complete (Phase 2)
- Ready for: Parallel component development (Phases 3-7)

## Directory Structure

```
src/vla/
├── __init__.py                 # Package entry point (v0.1.0)
├── registry/                   # Component registration system (IMPLEMENTED)
│   ├── base.py                # Registry class & global instances (157 LOC)
│   ├── factories.py           # Factory functions for component building (138 LOC)
│   └── __init__.py            # Public API exports (54 LOC)
├── nn/                         # Neural network primitives (EMPTY)
├── backbones/                  # Vision/language encoders (EMPTY)
├── fusion/                     # Multimodal fusion mechanisms (EMPTY)
├── policy/                     # Action prediction heads (EMPTY)
├── models/                     # VLA model orchestration (EMPTY)
├── data/                       # Data loaders & preprocessing (EMPTY)
├── training/                   # PyTorch Lightning modules (EMPTY)
└── utils/                      # Utilities
    ├── __init__.py            # Utils exports
    └── logging.py             # Logging configuration (48 lines, ACTIVE)
```

## Module Descriptions

### Core Package: `vla/__init__.py`
**Purpose:** Package initialization
**Status:** Active
**Lines:** 6
**Content:** Version specification (0.1.0), module docstring, lazy import strategy documented

**Key Code:**
```python
"""tinyVLA: Modular Vision-Language-Action research framework."""
__version__ = "0.1.0"
# Lazy imports to avoid circular dependencies
```

### Utilities Module: `vla/utils/`
**Purpose:** Shared utilities
**Status:** Partial (logging implemented)
**Files:** 2

#### `utils/__init__.py` (5 lines)
Public API exports:
```python
from vla.utils.logging import setup_logger
__all__ = ["setup_logger"]
```

#### `utils/logging.py` (48 lines, IMPLEMENTED)
**Single Function:** `setup_logger(name, level=logging.INFO, log_file=None) -> logging.Logger`

**Features:**
- Console handler with formatted output (timestamp, logger name, level, message)
- Optional file handler (creates parent directories automatically)
- Duplicate handler prevention
- Type hints: Full signature with return type
- Docstring: Comprehensive with Args/Returns sections

**Usage Example:**
```python
from vla.utils import setup_logger

logger = setup_logger(__name__)
logger = setup_logger(__name__, log_file=Path("logs/train.log"))
```

### Registry Module: `vla/registry/`
**Purpose:** Dynamic component instantiation via registry pattern
**Status:** IMPLEMENTED (Phase 2 Complete)
**LOC:** 349 total (base.py 157, factories.py 138, __init__.py 54)

**Components:**
- Generic `Registry[T]` class with type safety
- 5 global registries: VISION_REGISTRY, LANGUAGE_REGISTRY, FUSION_REGISTRY, ACTION_REGISTRY, MODEL_REGISTRY
- 5 factory functions for building components from Hydra configs
- Support for both registry lookup and direct Hydra instantiation

**Key Features:**
- O(1) lookup time via dictionary
- Type-safe using Python generics
- Helpful error messages listing available components
- Decorator-based registration pattern
- Integration with Hydra's instantiate() for _target_-based configs

**Usage Pattern:**
```python
from vla.registry import VISION_REGISTRY, build_vision_encoder

# Register a component
@VISION_REGISTRY.register("dinov2_base")
class DINOv2Base(nn.Module):
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        self.hidden_dim = hidden_dim

# Instantiate from config (registry method)
cfg = DictConfig({"name": "dinov2_base", "hidden_dim": 768})
encoder = build_vision_encoder(cfg)

# Or direct registry access
encoder = VISION_REGISTRY.get("dinov2_base", hidden_dim=768)

# List all registered components
available = VISION_REGISTRY.list_available()  # ['dinov2_base', ...]
```

### Neural Network Primitives: `vla/nn/__init__.py`
**Purpose:** Reusable neural network building blocks
**Status:** Empty stub
**Planned Components:**
- MultiHeadAttention (with Flash Attention 2 support)
- MLP (with configurable activation)
- Normalization (RMSNorm, LayerNorm)
- RoPE (Rotary Position Embeddings)
- FrameStacker (temporal frame aggregation)
- CausalConv1d (for temporal sequences)

### Backbones Module: `vla/backbones/__init__.py`
**Purpose:** Vision and language encoder implementations
**Status:** Empty stub
**Planned Components:**

**Vision Encoders:**
- VisionBackbone (wrapper interface)
- DINOv2 adapter (ViT-B/14, 86M params) - PRIMARY
- SigLIP adapter (ViT-B/16, 87M params) - ALTERNATIVE
- DualEncoder (combination of vision encoders)

**Language Encoders:**
- LanguageBackbone (wrapper interface)
- GPT2Backbone (124M-355M params)
- Tokenizer integration (HuggingFace transformers)

**Example Usage (future):**
```python
from vla.backbones import DINOv2, GPT2Backbone

vision = DINOv2(size="base", pretrained=True, freeze=True)
language = GPT2Backbone(model_name="gpt2", freeze=True)
```

### Fusion Module: `vla/fusion/__init__.py`
**Purpose:** Multimodal fusion mechanisms
**Status:** Empty stub
**Planned Components:**
- PerceiverResampler (primary: 64-token bottleneck)
- CrossAttentionFusion (multi-head attention)
- ConcatFusion (simple concatenation baseline)
- AdapterFusion (low-rank adapter pattern)

**Expected Input/Output:**
```
Vision Features [B, N_v, D_v] + Language Features [B, N_l, D_l]
                           ↓
                  Fusion Module
                           ↓
           Fused Features [B, K, D_fused]
```

### Policy Module: `vla/policy/__init__.py`
**Purpose:** Action prediction heads
**Status:** Empty stub
**Planned Components:**
- ActionHead (base interface)
- DiscreteActionHead (256 bins per DOF, classification loss)
- GaussianActionHead (continuous, MSE loss with uncertainty)
- TrajectoryHead (multi-step action sequence)
- HybridHead (discrete arm + continuous gripper)

### Models Module: `vla/models/__init__.py`
**Purpose:** VLA model orchestration and composition
**Status:** Empty stub
**Planned Components:**
- VLAConfig (dataclass for model configuration)
- VLAModel (main model class)
- Model forward pass: images + text → actions
- Checkpoint save/load utilities
- Inference wrapper

**Architecture:**
```python
class VLAModel(nn.Module):
    def __init__(self, vision, language, fusion, action_head):
        self.vision = vision
        self.language = language
        self.fusion = fusion
        self.action_head = action_head

    def forward(self, images, texts):
        v_feats = self.vision(images)      # [B, N, D_v]
        l_feats = self.language(texts)     # [B, L, D_l]
        fused = self.fusion(v_feats, l_feats)  # [B, K, D]
        actions = self.action_head(fused)  # [B, action_dim]
        return actions
```

### Data Module: `vla/data/__init__.py`
**Purpose:** Data loading and preprocessing
**Status:** Empty stub
**Planned Components:**
- DummyDataset (testing utility)
- HDF5Dataset (local cached data)
- WebDataset (cloud streaming)
- DatasetMixture (multi-dataset mixing with weights)
- Preprocessors (image normalization, action discretization)

**Supported Datasets:**
- Open X-Embodiment (RLDS format, 1M+ trajectories)
- Custom HDF5 datasets
- Synthetic dummy data

### Training Module: `vla/training/__init__.py`
**Purpose:** PyTorch Lightning training infrastructure
**Status:** Empty stub
**Planned Components:**
- VLALightningModule (LightningModule subclass)
- Callbacks (WandB logging, checkpointing, early stopping)
- Distributed training setup (FSDP, DDP)
- Loss functions (CrossEntropy for discrete, Gaussian NLL for continuous)
- Metrics computation

## Code Organization Patterns

### Type Hints
- **Coverage:** 100% in implemented code (logging.py)
- **Standard:** Function signatures with return types
- **Tools:** mypy configured for static checking

### Docstrings
- **Format:** Numpy style (Args, Returns sections)
- **Coverage:** All functions documented
- **Example:**
```python
def setup_logger(name: str, level: int = logging.INFO, log_file: Optional[Path] = None) -> logging.Logger:
    """Configure module logger with console and optional file handler.

    Args:
        name: Logger name (usually __name__)
        level: Logging level (default: INFO)
        log_file: Optional path to log file

    Returns:
        Configured logger instance
    """
```

### Error Handling
- **File Operations:** Uses `mkdir(parents=True, exist_ok=True)` for safety
- **Deduplication:** Checks existing handlers before adding
- **Pattern:** Defensive programming without try-except for common paths

## Test Infrastructure

### Framework
- **Tool:** pytest 9.0.2+
- **Fixtures:** 7 predefined (device, batch_size, seq_length, dummy_image, dummy_text, dummy_actions, seed)
- **Coverage:** Configured for automated reporting

### Available Fixtures
```python
@pytest.fixture
def device():
    """CUDA if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def dummy_image(batch_size):
    """Generate [B, 3, 224, 224] tensor."""
    return torch.randn(batch_size, 3, 224, 224)

@pytest.fixture
def dummy_actions(batch_size):
    """Generate [B, 7] action tensor."""
    action_dim = 7  # Standard robot arm
    return torch.randn(batch_size, action_dim)

# ... and more (see tests/conftest.py)
```

### Current Test Status
- **Test Files:** 0 (directories created but empty)
- **Test Functions:** 0
- **Coverage:** 0% (no tests written)
- **Next Step:** Implement tests for each module as components are built

## Configuration System

### Hydra Setup
- **Version:** 1.3.0+
- **Structure:** Hierarchical configs in `configs/` directory
- **Features:** CLI overrides, multirun sweeps, auto-saved configs

### Planned Config Hierarchy
```
configs/
├── config.yaml          # Main entry point
├── model/               # Model component configs
│   ├── vision/         # Vision encoder variants
│   ├── language/       # Language model variants
│   ├── fusion/         # Fusion strategy variants
│   └── action/         # Action head variants
├── train/              # Training hyperparameters
├── data/               # Dataset configurations
└── experiment/         # Pre-built experiment presets
```

### Usage Pattern
```bash
python scripts/train.py                    # Default config
python scripts/train.py model.vision_encoder=siglip  # Override
python scripts/train.py --multirun train.lr=1e-4,3e-4  # Sweep
```

## Dependencies

### Core Runtime (19 packages)
```
torch >= 2.5.0              # Deep learning
torchvision >= 0.20.0       # Vision utilities
pytorch-lightning >= 2.2.0  # Training abstraction
hydra-core >= 1.3.0         # Config management
timm >= 1.0.0               # Vision backbones
transformers >= 4.40.0      # Language models
wandb >= 0.16.0             # Experiment tracking
einops >= 0.7.0             # Tensor operations
numpy >= 1.26.0             # Numerical computing
pillow >= 10.0.0            # Image processing
h5py >= 3.10.0              # HDF5 I/O
webdataset >= 0.2.86        # Stream datasets
tensorboard >= 2.15.0       # Visualization
```

### Development (8 packages)
```
pytest >= 8.0.0             # Testing
pytest-cov >= 4.0.0         # Coverage
black >= 24.0.0             # Formatting
ruff >= 0.3.0               # Linting
mypy >= 1.8.0               # Type checking
pre-commit >= 3.6.0         # Git hooks
ipython >= 8.20.0           # Interactive shell
```

## Code Quality Configuration

### Black (Formatter)
- Line length: 100 characters
- Target: Python 3.10+
- Excludes: .git, .venv, build, dist

### Ruff (Linter)
- Rules: E, F, I, W, B, C90
- Exceptions: F401/F403 in `__init__.py` (unused imports)
- Line length: 100 characters

### mypy (Type Checker)
- Strictness: Moderate (checks untyped defs but allows missing imports)
- Python version: 3.10
- Allows: Missing imports for unannotated dependencies

## Implementation Status Summary

| Component | Status | Purpose | LOC |
|-----------|--------|---------|-----|
| vla | Active | Package init | 6 |
| utils.logging | IMPLEMENTED | Logger setup | 48 |
| registry | IMPLEMENTED | Component registry | 349 |
| nn | Empty | NN primitives | 0 |
| backbones | Empty | Vision/language | 0 |
| fusion | Empty | Fusion mechanisms | 0 |
| policy | Empty | Action heads | 0 |
| models | Empty | VLA orchestration | 0 |
| data | Empty | Data loaders | 0 |
| training | Empty | Lightning modules | 0 |
| tests.unit.test_registry | IMPLEMENTED | Registry unit tests | 272 |

**Total:** 621 LOC implemented (54 LOC added in tests), 8 modules awaiting implementation, Phase 2 COMPLETE

## Entry Points

### Package Import
```python
from vla import __version__
from vla.utils import setup_logger
```

### Training (future)
```bash
python scripts/train.py --config config.yaml
```

### Evaluation (future)
```bash
python scripts/eval.py --checkpoint checkpoints/model.pt --data test_data.hdf5
```

## Next Steps

1. **Phase 1 (Setup):** Complete (project scaffolding done)
2. **Phase 2 (Registries):** Complete (registry pattern + factories + 20 unit tests passing)
3. **Phases 3-7 (Components):** Ready to start in parallel (NN primitives, backbones, fusion, policy)
4. **Phase 8 (Model):** Orchestrate components into VLA model
5. **Phases 9-11 (Config/Data/Training):** Hydra configs, data pipeline, Lightning training
6. **Phase 12 (Testing):** Comprehensive test suite with 80%+ coverage

See [Project Roadmap](./project-roadmap.md) for detailed timeline and phases.

**Phase 2 Completion Date:** 2026-01-22 (2.5h actual vs 2h estimated)
