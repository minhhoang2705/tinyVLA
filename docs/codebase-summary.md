# Codebase Summary - tinyVLA

## Overview

tinyVLA is in **Phases 4-7 complete** (0.2.0) with ~3,300 lines of production code across 30+ Python modules plus 2,525 lines of tests. Registry pattern, NN primitives, vision/language backbones, fusion mechanisms, and action heads are fully operational.

**Current State:**
- Architecture: Complete blueprint with major components implemented
- Infrastructure: Fully configured (Hydra, PyTorch Lightning, testing framework)
- Implementation: Phases 2-8 complete (registry, NN primitives, backbones, fusion, action heads, VLA orchestration)
- Ready for: Configuration system (Phase 9) and training infrastructure (Phases 10-11)

## Directory Structure

```
src/vla/
├── __init__.py                 # Package entry point (v0.2.0)
├── registry/                   # Component registration system (IMPLEMENTED - Phase 2)
│   ├── base.py                # Registry class & global instances (157 LOC)
│   ├── factories.py           # Factory functions for component building (138 LOC)
│   └── __init__.py            # Public API exports (54 LOC)
├── nn/                         # Neural network primitives (IMPLEMENTED - Phase 3)
│   ├── attention.py           # MultiHeadAttention, CrossAttention (194 LOC)
│   ├── mlp.py                 # MLP, GatedMLP (124 LOC)
│   ├── norm.py                # RMSNorm, get_norm factory (88 LOC)
│   ├── pos_encoding.py        # Sinusoidal, Learnable, RoPE (218 LOC)
│   ├── temporal.py            # FrameStacker, CausalConv1d, TemporalBlock (163 LOC)
│   └── __init__.py            # Public API exports (45 LOC)
├── backbones/                  # Vision/language encoders (IMPLEMENTED - Phase 4-5)
│   ├── vision.py              # VisionBackbone base + DINOv2, SigLIP, ViT (648 LOC)
│   ├── language.py            # LanguageBackbone base + GPT-2, LLaMA (442 LOC)
│   └── __init__.py            # Public API exports (47 LOC)
├── fusion/                     # Multimodal fusion mechanisms (IMPLEMENTED - Phase 6)
│   ├── fusion.py              # FusionModule base class (98 LOC)
│   ├── perceiver.py           # Perceiver Resampler (287 LOC)
│   ├── cross_attn.py          # Cross-attention fusion (194 LOC)
│   ├── concat.py              # Concatenation baseline (156 LOC)
│   ├── adapter.py             # Low-rank adapter fusion (168 LOC)
│   └── __init__.py            # Public API exports (52 LOC)
├── policy/                     # Action prediction heads (IMPLEMENTED - Phase 7)
│   ├── head.py                # ActionHead base class (95 LOC)
│   ├── discrete.py            # Discrete binning head (212 LOC)
│   ├── continuous.py          # Gaussian head (178 LOC)
│   ├── hybrid.py              # Hybrid arm+gripper head (186 LOC)
│   └── __init__.py            # Public API exports (48 LOC)
├── models/                     # VLA model orchestration (IMPLEMENTED - Phase 8)
│   ├── vla_base.py            # VLAModel main class (509 LOC)
│   ├── vla_configs.py         # Config dataclasses (186 LOC)
│   └── __init__.py            # Public API exports (51 LOC)
├── data/                       # Data loaders & preprocessing (EMPTY - Phase 10)
├── training/                   # PyTorch Lightning modules (EMPTY - Phase 11)
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

### Neural Network Primitives: `vla/nn/`
**Purpose:** Reusable neural network building blocks for transformer-based VLA architectures
**Status:** IMPLEMENTED (Phase 3 Complete)
**LOC:** 832 total (attention 194, mlp 124, norm 88, pos_encoding 218, temporal 163, __init__ 45)
**Coverage:** 99.5% (70 unit tests)

**Implemented Components:**

#### Attention (`attention.py` - 194 LOC)
- **MultiHeadAttention:** Self-attention with optional Flash Attention 2 support
  - Args: `dim`, `num_heads`, `dropout`, `use_flash`, `bias`
  - Returns: `[B, seq_len, dim]` tensor
  - Features: Scaled dot-product attention, O(n²) complexity, 2-4x faster with Flash
- **CrossAttention:** Attends from query to key/value (different sequences)
  - Args: `query_dim`, `context_dim`, `num_heads`, `dropout`, `use_flash`
  - Usage: Language queries attend to vision features, or vice versa
  - Fused features generation in Perceiver resampler

#### MLP (`mlp.py` - 124 LOC)
- **MLP:** Standard feed-forward network with configurable activation
  - Args: `dim`, `hidden_dim` (expansion factor), `dropout`, `activation`
  - Supports: ReLU, GELU, SiLU, Mish, GLU variants
  - Pattern: Linear → Activation → Dropout → Linear
- **GatedMLP:** Gated feed-forward network (Gating Mechanism)
  - Args: `dim`, `hidden_dim`, `dropout`
  - Pattern: Linear(3x) → Split(value, gate) → value * sigmoid(gate)
  - Alternative to attention for reduced complexity

#### Normalization (`norm.py` - 88 LOC)
- **RMSNorm:** Root Mean Square Layer Normalization (from T5, more stable)
  - Args: `dim`, `eps` (default 1e-6)
  - Computes: `x * (dim / RMS(x))` instead of `(x - mean) / sqrt(var)`
  - Benefits: Faster, no bias needed, better for large models
- **get_norm():** Factory function to select norm layer
  - Supports: "layer_norm", "rms_norm", "batch_norm"
  - Enables config-driven architecture selection

#### Position Encoding (`pos_encoding.py` - 218 LOC)
- **SinusoidalPositionEncoding:** Fixed sinusoidal encodings (original Transformer)
  - Args: `dim`, `max_seq_len`
  - Pattern: PE(pos, 2i) = sin(pos/10000^(2i/dim))
  - No learnable parameters; deterministic and extrapolates well
- **LearnablePositionEncoding:** Learnable embeddings per position
  - Args: `dim`, `max_seq_len`
  - Trained during model training, better if data distribution is known
  - Trade-off: Smaller extrapolation range
- **RotaryPositionEncoding (RoPE):** Rotary embeddings (modern alternative)
  - Args: `dim`, `base` (default 10000)
  - Pattern: Rotates Q,K vectors in 2D subspaces by angle θ = m*θ₀
  - Benefits: Length extrapolation, better long-context performance
  - Used in: LLaMA, GPT-3.5+, modern VLA models

#### Temporal Modeling (`temporal.py` - 163 LOC)
- **FrameStacker:** Stack temporal frames for multi-frame visual input
  - Args: `num_frames`, `stack_mode` (concat/mean/attention)
  - Input: Multiple images [B, num_frames, 3, H, W]
  - Output: [B, 3*num_frames, H, W] (concat) or [B, 3, H, W] (aggregated)
  - Use case: Optical flow, motion information
- **CausalConv1d:** 1D convolution with causal padding (no future leakage)
  - Args: `in_channels`, `out_channels`, `kernel_size`, `dilation`
  - Padding: `(kernel_size - 1) * dilation` on left, 0 on right
  - For: Temporal sequence processing without looking ahead
- **TemporalBlock:** Residual block with causal conv + normalization + activation
  - Args: `channels`, `kernel_size`, `dilation`, `dropout`
  - Pattern: Conv → Norm → ReLU → Dropout → Residual connection
  - Stacked for multi-layer temporal modeling

**Public API (vla.nn exports):**
```python
from vla.nn import (
    # Attention
    MultiHeadAttention,
    CrossAttention,
    # MLP
    MLP,
    GatedMLP,
    # Normalization
    RMSNorm,
    get_norm,
    # Position encoding
    SinusoidalPositionEncoding,
    LearnablePositionEncoding,
    RotaryPositionEncoding,
    # Temporal
    FrameStacker,
    CausalConv1d,
    TemporalBlock,
)
```

### Backbones Module: `vla/backbones/`
**Purpose:** Vision and language encoder implementations
**Status:** IMPLEMENTED (Phases 4-5 Complete)
**LOC:** 1,137 total (vision 648, language 442, __init__ 47)
**Coverage:** 95%+ (31 vision tests + 23 language tests)

**Vision Encoders (vision.py - 648 LOC):**
- **VisionBackbone:** Base interface for all vision encoders
  - Args: `model_name`, `pretrained`, `freeze`
  - Returns: Features [B, num_patches, feature_dim]
- **DINOv2Backbone:** Self-supervised ViT from Meta (PRIMARY)
  - Sizes: base (86M), large (300M)
  - Output: [B, 196, 768] for ViT-B/14
- **SigLIPBackbone:** Vision-language aligned ViT (ALTERNATIVE)
  - Sizes: small, base, large
  - Better instruction following than pure vision
- **ViTBackbone:** Generic Vision Transformer wrapper
  - Flexible model names via timm

**Language Encoders (language.py - 442 LOC):**
- **LanguageBackbone:** Base interface for language models
  - Args: `model_name`, `pretrained`, `freeze`
  - Returns: Features [B, seq_len, feature_dim]
- **GPT2Backbone:** HuggingFace transformers (PRIMARY)
  - Sizes: small (124M), base (355M)
  - Integrated tokenizer and padding
- **LlamaBackbone:** Modern LLaMA support (ALTERNATIVE)
  - Better long-context modeling
  - Improved instruction following

**Example Usage:**
```python
from vla.backbones import DINOv2Backbone, GPT2Backbone

vision = DINOv2Backbone(size="base", pretrained=True, freeze=True)
language = GPT2Backbone(model_name="gpt2", freeze=True)

images = torch.randn(2, 3, 224, 224)
texts = ["pick up red cube", "move to target"]

v_features = vision(images)  # [2, 196, 768]
l_features = language(texts)  # [2, seq_len, 768]
```

**Key Features:**
- All frozen during training (no gradients for transfer learning)
- Automatic padding and sequence handling
- Type-safe with full type hints
- Registered via VISION_REGISTRY and LANGUAGE_REGISTRY

### Fusion Module: `vla/fusion/`
**Purpose:** Multimodal fusion mechanisms
**Status:** IMPLEMENTED (Phase 6 Complete)
**LOC:** 955 total (perceiver 287, cross_attn 194, concat 156, adapter 168, head 98, __init__ 52)
**Coverage:** 94%+ (35 tests)

**Fusion Mechanisms:**
- **PerceiverResampler:** Fixed-size latent bottleneck (PRIMARY)
  - Learnable K latent tokens [num_latents=64]
  - Cross-attention to vision + language
  - Output: [B, K, D] (efficient O(K) complexity)
  - Used in: Flamingo, RT-2, OpenVLA

- **CrossAttentionFusion:** Direct multimodal attention
  - Vision queries attend to language
  - Lower memory than Perceiver, higher compute
  - Output: [B, N_v, D]

- **ConcatFusion:** Simple baseline
  - Concatenate vision + language features
  - Output: [B, N_v + N_l, D]
  - For ablation studies

- **AdapterFusion:** Low-rank parameter-efficient
  - Adapter networks (~1% parameters vs full fusion)
  - Output: [B, K, D]
  - For resource-constrained settings

**Input/Output:**
```
Vision Features [B, N_v=196, D_v=768]
Language Features [B, N_l=64, D_l=768]
                           ↓
                  Fusion Module
                           ↓
           Fused Features [B, K=64, D=768]
```

**Usage:**
```python
from vla.fusion import PerceiverResampler
from vla.registry import FUSION_REGISTRY

# Direct instantiation
fusion = PerceiverResampler(latent_dim=768, num_latents=64, num_layers=4)
fused = fusion(v_features, l_features)  # [B, 64, 768]

# Via registry
fusion = FUSION_REGISTRY.get("perceiver", num_latents=64)
```

### Policy Module: `vla/policy/`
**Purpose:** Action prediction heads
**Status:** IMPLEMENTED (Phase 7 Complete)
**LOC:** 719 total (discrete 212, continuous 178, hybrid 186, head 95, __init__ 48)
**Coverage:** 93%+ (25 tests)

**Action Heads:**
- **ActionHead:** Base interface for all action heads
  - Args: `feature_dim`, `action_dim`, `loss_type`
  - Methods: `forward()`, `compute_loss()`

- **DiscreteActionHead:** 256-bin classification (PRIMARY - RT-2 style)
  - Per-DOF 256-bin softmax classification
  - Output: [B, action_dim, num_bins]
  - Loss: CrossEntropyLoss
  - Inference: Argmax or sample from distribution

- **ContinuousActionHead:** Gaussian distribution
  - Output mean + log_var
  - Output: [B, action_dim, 2]
  - Loss: GaussianNLLLoss
  - Captures aleatoric uncertainty

- **HybridActionHead:** Mixed action types
  - Discrete for arm joints (e.g., 6 DOF)
  - Continuous for gripper (1 DOF)
  - Useful for real robot control

**Usage:**
```python
from vla.policy import DiscreteActionHead
from vla.registry import ACTION_REGISTRY

# Direct instantiation
head = DiscreteActionHead(feature_dim=768, action_dim=7, num_bins=256)
logits = head(fused_features)  # [B, 7, 256]
loss = head.compute_loss(logits, target_actions)

# Via registry
head = ACTION_REGISTRY.get("discrete", action_dim=7)

# Inference
actions = torch.argmax(logits, dim=-1)  # [B, 7]
actions = (actions / 255.0) * 2 - 1  # Normalize to [-1, 1]
```

### Models Module: `vla/models/` (IMPLEMENTED - Phase 8)
**Purpose:** VLA model orchestration and composition
**Status:** IMPLEMENTED (746 LOC total)
**Files:** 3

#### `models/vla_base.py` (509 LOC)
**VLAModel class:** Registry-based component composition
- Frozen vision/language backbones (99% params frozen)
- Trainable fusion module (Perceiver Resampler)
- Trainable action head (discrete/Gaussian/hybrid)
- Forward pass: images + text → actions
- Training mode with loss computation
- Inference mode with action prediction
- Checkpoint save/load with state preservation
- Input validation and enhanced logging
- Support for temporal multi-frame processing via FrameStacker

**Key Methods:**
```python
def __init__(self, config: VLAConfig | dict)
def forward(images, texts, target_actions=None) -> Dict[str, Tensor]
def predict(images, texts) -> Tensor  # Inference mode
def save_checkpoint(path: Path) -> None
def load_checkpoint(path: Path) -> None
def compute_trainable_params() -> float
```

**Architecture:**
```
Images [B,3,H,W]              Text [B]
        │                        │
        ▼                        ▼
Vision Backbone (frozen)  Language Backbone (frozen)
[B, 196, 768]                [B, L, 768]
        │                        │
        └────────┬───────────────┘
                 ▼
        Fusion Module (trainable)
             [B, 64, 768]
                 │
                 ▼
        Action Head (trainable)
            [B, action_dim]
```

#### `models/vla_configs.py` (186 LOC)
**Config dataclasses:**
- `VisionConfig`: Vision backbone configuration
  - `model_name`, `pretrained`, `hidden_dim`, `freeze`
- `LanguageConfig`: Language encoder configuration
  - `model_name`, `pretrained`, `hidden_dim`, `freeze`
- `FusionConfig`: Fusion module configuration
  - `type` (perceiver/cross_attn/concat/adapter)
  - `num_latents`, `latent_dim`, `num_layers`
- `ActionConfig`: Action head configuration
  - `type` (discrete/continuous/hybrid)
  - `action_dim`, `num_bins`, `pooling_type`
- `VLAConfig`: Master configuration
  - Composition of all above configs
  - Methods: `from_dict()`, `to_dict()`, `asdict()`

**Usage:**
```python
config = VLAConfig(
    vision=VisionConfig(model_name="vit_tiny_patch16_224"),
    language=LanguageConfig(model_name="gpt2"),
    action=ActionConfig(action_dim=7),
)
model = VLAModel(config)
```

#### `models/__init__.py` (51 LOC)
Public API exports:
```python
from .vla_base import VLAModel
from .vla_configs import (
    VLAConfig,
    VisionConfig,
    LanguageConfig,
    FusionConfig,
    ActionConfig,
)
```

**Testing:**
- 20 comprehensive unit tests (98% coverage)
- Test categories:
  - Model instantiation (registry-based, dict-based)
  - Forward pass (training mode, inference mode)
  - Loss computation (discrete, continuous)
  - Checkpoint save/load (complete state preservation)
  - Parameter freezing (99% frozen validation)
  - Input validation (shape checking, error handling)
  - Temporal processing (multi-frame support)
  - Config serialization (dict conversion roundtrip)

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

| Component | Status | Purpose | LOC | Tests | Coverage |
|-----------|--------|---------|-----|-------|----------|
| vla | Active | Package init | 6 | - | - |
| utils.logging | IMPLEMENTED | Logger setup | 48 | - | 100% |
| registry | IMPLEMENTED | Component registry | 349 | 20 | 92% |
| nn | IMPLEMENTED | NN primitives | 832 | 70 | 99.5% |
| backbones | IMPLEMENTED | Vision/language | 1,137 | 54 | 95%+ |
| fusion | IMPLEMENTED | Fusion mechanisms | 955 | 35 | 94%+ |
| policy | IMPLEMENTED | Action heads | 719 | 25 | 93%+ |
| models | IMPLEMENTED | VLA orchestration | 746 | 20 | 98% |
| data | Empty | Data loaders | 0 | - | - |
| training | Empty | Lightning modules | 0 | - | - |

**Total:** 4,792 LOC production code + 2,899 LOC tests across 11 modules, 224 unit tests (avg 95% coverage), Phases 2-8 COMPLETE

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

1. **Phase 1 (Setup):** ✓ Complete (project scaffolding done)
2. **Phase 2 (Registries):** ✓ Complete (registry pattern + factories + 20 unit tests passing)
3. **Phase 3 (NN Primitives):** ✓ Complete (832 LOC, 70 unit tests, 99.5% coverage)
4. **Phases 4-5 (Backbones):** ✓ Complete (vision + language encoders, 54 tests)
5. **Phase 6 (Fusion):** ✓ Complete (Perceiver + alternatives, 35 tests)
6. **Phase 7 (Action Heads):** ✓ Complete (discrete/continuous/hybrid, 25 tests)
7. **Phase 8 (VLA Model):** ✓ Complete (orchestration + configs, 20 tests, 98% coverage)
8. **Phase 9 (Hydra Config):** Next - Configuration system for model variants
9. **Phase 10 (Data Pipeline):** Data loaders (dummy, HDF5, WebDataset)
10. **Phase 11 (Training):** PyTorch Lightning module with WandB + FSDP
11. **Phase 12 (Testing):** Integration tests and CI/CD validation

See [Project Roadmap](./project-roadmap.md) for detailed timeline and phases.

**Phase 3 Completion Date:** 2026-01-23 (approx 3h actual vs 3h estimated)
**Phases 4-7 Completion Date:** 2026-01-24 (approx 13h actual vs 13h estimated)
**Phase 8 Completion Date:** 2026-01-26 (approx 4h actual vs 4h estimated)
**Overall Progress:** 67% complete (8 of 12 phases)
