# tinyVLA: Modular Vision-Language-Action Framework

A lightweight, modular Vision-Language-Action (VLA) research framework for robotics and embodied AI applications. Built with PyTorch 2.5+, Hydra configuration, and PyTorch Lightning for production-ready training.

## Overview

tinyVLA enables rapid prototyping and research of vision-language-action models for robotic control. The framework provides composable building blocks (attention, MLP, normalization), multiple backbone choices (DINOv2, ViT, GPT-2), diverse fusion strategies (Perceiver Resampler, cross-attention), and flexible action heads (discrete binning, continuous Gaussian).

**Key Differentiators:**
- **Modular Architecture**: Swap vision encoders, fusion mechanisms, and action heads via simple configuration changes
- **Registry-Based Composition**: Dynamic component loading without code modifications
- **Hydra-Driven Configuration**: Hierarchical configs with CLI overrides and multirun sweeps
- **Research-Optimized**: Supports behavioral cloning, offline RL, and diffusion policies
- **Production Infrastructure**: PyTorch Lightning with FSDP, mixed precision, and WandB tracking

## Quick Start

### Prerequisites
- Python 3.10+
- CUDA 11.8+ (for GPU support)
- 16GB+ RAM (24GB+ VRAM recommended)

### Installation

```bash
# Clone and setup
git clone <repo-url>
cd tinyVLA

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install with development tools
pip install -e ".[dev]"
```

### Basic Usage

```bash
# Train with default configuration
python scripts/train.py

# Override hyperparameters via CLI
python scripts/train.py model.vision_encoder=dinov2 train.batch_size=32

# Run hyperparameter sweeps
python scripts/train.py --multirun model.fusion=perceiver,cross_attn train.lr=1e-4,3e-4
```

## Features

- **Composable Architecture**: Build models from reusable primitives (attention, MLP, norms)
- **Flexible Backbones**: Vision encoders (DINOv2, ViT) and language models (GPT-2) via timm/transformers
- **Multiple Fusion Strategies**: Perceiver Resampler, cross-attention, concatenation fusion
- **Action Heads**: Discrete binning (RT-2 style), continuous Gaussian distributions, diffusion policies
- **Temporal Modeling**: Frame stacking, causal convolutions for sequence handling
- **Hydra Configuration**: Hierarchical configs with experiment composition and multi-run sweeps
- **PyTorch Lightning**: Production-ready training with FSDP, mixed precision, WandB integration
- **Data Pipeline**: Support for Open X-Embodiment dataset with HDF5/WebDataset backends

## Project Structure

```
tinyVLA/
├── README.md                 # This file
├── pyproject.toml            # Project metadata & dependencies
├── configs/                  # Hydra configuration hierarchy
│   ├── config.yaml          # Main config (composed via defaults)
│   ├── model/               # Vision/language/fusion/action configs
│   ├── train/               # Training hyperparameters
│   ├── data/                # Dataset configurations
│   └── experiment/          # Pre-built experiment templates
├── src/vla/                 # Source code
│   ├── registry/            # Component registry system
│   ├── nn/                  # Neural network primitives
│   ├── backbones/           # Vision & language encoders
│   ├── fusion/              # Multimodal fusion mechanisms
│   ├── policy/              # Action prediction heads
│   ├── models/              # VLA model orchestration
│   ├── data/                # Data loaders & preprocessing
│   ├── training/            # PyTorch Lightning modules
│   └── utils/               # Utilities & logging
├── scripts/
│   ├── train.py             # Training entry point
│   └── eval.py              # Evaluation script
├── tests/                   # Unit & integration tests
│   ├── conftest.py          # Pytest configuration & fixtures
│   ├── unit/                # Unit tests
│   └── integration/         # Integration tests
└── docs/                    # Documentation
    ├── codebase-summary.md
    ├── project-overview-pdr.md
    ├── code-standards.md
    ├── system-architecture.md
    ├── project-roadmap.md
    └── tech-stack.md
```

## Development

### Code Quality

```bash
# Format code with Black
black src/ tests/

# Lint with Ruff
ruff check src/ tests/

# Type check with mypy
mypy src/

# All checks together
pre-commit run --all-files
```

### Testing

```bash
# Run all tests with coverage
pytest tests/ --cov=vla --cov-report=html

# Run specific test file
pytest tests/unit/test_nn.py -v

# Run only unit tests
pytest tests/unit/
```

## Tech Stack

| Component | Technology | Version | Rationale |
|-----------|-----------|---------|-----------|
| Framework | PyTorch | 2.5+ | FSDP2, torch.compile, strong RL ecosystem |
| Configuration | Hydra | 1.3+ | Hierarchical composition, CLI overrides, reproducibility |
| Training | PyTorch Lightning | 2.2+ | Distributed training, callbacks, checkpoint management |
| Vision | timm | 1.0+ | 500+ pretrained models, PyTorch 2.x optimized |
| Language | transformers | 4.40+ | GPT-2, LLaMA support, HuggingFace integration |
| Tracking | WandB | 0.16+ | Real-time metrics, hyperparameter tracking, integration with Hydra |
| Data | WebDataset + HDF5 | Latest | Cloud streaming + local fast I/O |

See [Tech Stack Documentation](docs/tech-stack.md) for detailed rationale and alternatives.

## Architecture Highlights

### Component Pipeline

```
Raw Images [B, 3, H, W]           Text Instructions [B,]
        ↓                                ↓
Vision Encoder (timm)        Language Encoder (transformers)
        ↓                                ↓
Vision Features [B, N, D_v]  Language Features [B, L, D_l]
        ↘                                ↙
            Fusion Module (Perceiver)
                    ↓
        Fused Features [B, K, D_fused]
                    ↓
            Action Head
                    ↓
            Actions [B, action_dim]
```

### Modular Design

All components registered via typesafe registry:

```python
# Instantiate from Hydra config
model = VLAModel(
    vision=VISION_REGISTRY.get("dinov2", size="base"),
    language=LANGUAGE_REGISTRY.get("gpt2"),
    fusion=FUSION_REGISTRY.get("perceiver_resampler"),
    action_head=ACTION_REGISTRY.get("discrete_action", num_bins=256)
)

# Or with configuration composition
python train.py model.vision_encoder=siglip model.fusion=cross_attention
```

## Documentation

- **[Codebase Summary](docs/codebase-summary.md)** - Module organization and code structure
- **[Project Overview & PDR](docs/project-overview-pdr.md)** - Requirements, scope, and success criteria
- **[Code Standards](docs/code-standards.md)** - Coding conventions, file organization, error handling
- **[System Architecture](docs/system-architecture.md)** - Component design and data flow
- **[Project Roadmap](docs/project-roadmap.md)** - Implementation phases and milestones
- **[Tech Stack](docs/tech-stack.md)** - Technology choices and rationale
- **[Deployment Guide](docs/deployment-guide.md)** - Setup, training, and inference

## Configuration Examples

### Train on Dummy Data

```bash
# Default dummy dataset, single GPU
python scripts/train.py data=dummy train.batch_size=16
```

### Train with Open X-Embodiment

```bash
# Real OXE dataset (requires preprocessing)
python scripts/train.py data=oxe train.batch_size=32 train.epochs=100
```

### Hyperparameter Sweep

```bash
# Grid search over vision encoders and learning rates
python scripts/train.py --multirun \
  model.vision_encoder=dinov2,siglip \
  train.lr=1e-4,3e-4,1e-3
```

## Research Foundation

tinyVLA builds on research from:
- **OpenVLA**: LLM-as-decoder architecture for VLA
- **Octo**: Efficient diffusion-based policies
- **π0**: Fast transformer-based action prediction
- **Open X-Embodiment**: Large-scale multimodal robotics dataset

See [Scout Reports](plans/reports/) for detailed research analysis.

## Contributing

We welcome contributions! Please follow the [Code Standards](docs/code-standards.md) and ensure:
- Code passes `black`, `ruff`, and `mypy` checks
- Tests pass with 80%+ coverage
- Documentation is updated for API changes
- Commit messages follow conventional commit format

## License

MIT License. See LICENSE file for details.

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

## Support

For issues, questions, or feature requests, please open a GitHub issue or discussion.
