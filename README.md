# tinyVLA

A modular Vision-Language-Action (VLA) research framework with composable block primitives, backbone builders, and Hydra-based configuration.

## Features

- **Composable Architecture**: Build VLA models from reusable primitives (attention, MLP, norms)
- **Flexible Backbones**: Vision encoders (DINOv2, ViT) and language models (GPT-2) via timm and transformers
- **Multiple Fusion Strategies**: Perceiver Resampler, cross-attention
- **Action Heads**: Discrete binning (RT-2 style) and continuous Gaussian distributions
- **Temporal Modeling**: Frame stacking, causal convolutions for sequence handling
- **Hydra Configuration**: Hierarchical configs with experiment composition and multi-run sweeps
- **PyTorch Lightning**: Production-ready training with FSDP, mixed precision, WandB integration

## Installation

### Prerequisites
- Python 3.10+
- CUDA 11.8+ (for GPU support)
- 16GB+ RAM
- 24GB+ VRAM (recommended for full models)

### Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd tinyVLA

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package with dependencies
pip install -e ".[dev]"
```

## Quick Start

### Training a VLA Model

```bash
# Train with default config
python scripts/train.py

# Override config via CLI
python scripts/train.py model.vision_encoder=dinov2 train.batch_size=32

# Run experiment sweep
python scripts/train.py --multirun model.fusion=perceiver,cross_attn train.lr=1e-4,3e-4
```

### Project Structure

```
tinyVLA/
├── configs/              # Hydra configuration files
│   ├── experiment/       # Full experiment configs
│   ├── model/            # Model architecture configs
│   ├── vision/           # Vision encoder configs
│   ├── language/         # Language model configs
│   ├── fusion/           # Fusion mechanism configs
│   ├── action/           # Action head configs
│   ├── train/            # Training configs
│   └── data/             # Dataset configs
├── src/vla/              # Source code
│   ├── registry/         # Component registries
│   ├── nn/               # Neural network primitives
│   ├── backbones/        # Vision/language backbones
│   ├── fusion/           # Fusion mechanisms
│   ├── policy/           # Action prediction heads
│   ├── models/           # VLA model orchestration
│   ├── training/         # Lightning modules
│   ├── data/             # Data loaders
│   └── utils/            # Utilities
├── scripts/              # Training/evaluation scripts
└── tests/                # Unit and integration tests
```

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=vla --cov-report=html

# Run specific test file
pytest tests/unit/test_nn.py -v
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/
```

## Tech Stack

- **PyTorch 2.5+**: Deep learning framework
- **Hydra 1.3+**: Configuration management
- **PyTorch Lightning 2.2+**: Training abstraction
- **timm 1.0+**: Vision model library
- **transformers 4.40+**: Language models
- **WandB**: Experiment tracking

## Documentation

- [Tech Stack](docs/tech-stack.md) - Detailed technology choices and rationale
- [Implementation Plan](plans/260117-1552-vla-bootstrap/plan.md) - Phase-by-phase development plan
- Research Reports:
  - [VLA Architectures](plans/reports/researcher-260118-vla-architectures.md)
  - [PyTorch VLA Patterns](plans/reports/researcher-260118-0228-pytorch-vla.md)
  - [Hydra Configuration](plans/reports/researcher-260118-hydra-ml-config.md)
  - [OXE Data Loading](plans/reports/researcher-260118-0228-oxe-data-loading.md)

## License

MIT

## Citation

If you use tinyVLA in your research, please cite:

```bibtex
@software{tinyvla2026,
  title={tinyVLA: Modular Vision-Language-Action Framework},
  author={minh-ub},
  year={2026},
  url={https://github.com/yourusername/tinyVLA}
}
```
