# Phase 09: Hydra Configuration System

## Context Links
- [Hydra Config Research](../reports/researcher-260118-hydra-ml-config.md)
- [Tech Stack](../../docs/tech-stack.md) - Hydra section

## Overview
| Field | Value |
|-------|-------|
| Priority | P1 - Infrastructure |
| Status | ✅ Complete |
| Effort | 4h |
| Dependencies | Phase 1 |

Implement hierarchical Hydra configuration for all VLA components. Enable experiment composition, CLI overrides, and reproducibility.

## Key Insights
- Config groups per component type (vision/, language/, fusion/, action/)
- Structured configs with dataclasses for type safety
- Experiment configs compose base configs with overrides
- Auto-saved configs ensure reproducibility

## Requirements

### Functional
- FR-01: Config group per component type
- FR-02: Experiment configs for common setups
- FR-03: CLI overrides for any parameter
- FR-04: WandB logging integration
- FR-05: Multirun sweep support

### Non-Functional
- NFR-01: Config loading <100ms
- NFR-02: Clear error messages for invalid configs

## Architecture

```
configs/
├── config.yaml              # Default entry point
├── model/
│   ├── vla_base.yaml
│   └── vla_temporal.yaml
├── vision/
│   ├── vit_base.yaml
│   ├── dinov2.yaml
│   └── siglip.yaml
├── language/
│   ├── gpt2.yaml
│   └── gpt2_medium.yaml
├── fusion/
│   ├── perceiver.yaml
│   └── cross_attention.yaml
├── action/
│   ├── discrete.yaml
│   └── gaussian.yaml
├── train/
│   ├── default.yaml
│   └── distributed.yaml
├── data/
│   ├── oxe.yaml
│   └── dummy.yaml
└── experiment/
    ├── baseline.yaml
    └── ablation_vision.yaml
```

## Related Code Files

### Files to Create
| Path | Purpose | Lines |
|------|---------|-------|
| `configs/config.yaml` | Main config | ~30 |
| `configs/model/vla_base.yaml` | VLA model config | ~15 |
| `configs/vision/*.yaml` | Vision configs | ~10 each |
| `configs/language/*.yaml` | Language configs | ~10 each |
| `configs/fusion/*.yaml` | Fusion configs | ~10 each |
| `configs/action/*.yaml` | Action configs | ~10 each |
| `configs/train/default.yaml` | Training config | ~30 |
| `configs/data/oxe.yaml` | Data config | ~20 |
| `configs/experiment/*.yaml` | Experiment presets | ~20 each |
| `src/vla/utils/hydra_utils.py` | Hydra helpers | ~60 |

## Implementation Steps

### Step 1: Create config.yaml (15 min)
```yaml
# configs/config.yaml
# Main configuration entry point

defaults:
  - model: vla_base
  - vision: vit_base
  - language: gpt2
  - fusion: perceiver
  - action: discrete
  - train: default
  - data: dummy
  - _self_

# Override with experiment config
# python train.py +experiment=baseline

# Project info
project:
  name: tinyVLA
  version: "0.1.0"

# Seed for reproducibility
seed: 42

# Output directory (auto-set by Hydra)
output_dir: ${hydra:runtime.output_dir}
```

### Step 2: Create model configs (20 min)
```yaml
# configs/model/vla_base.yaml
_target_: vla.models.VLAModel

# Component selection (resolved from other config groups)
freeze_vision: true
freeze_language: true
action_loss_weight: 1.0
auxiliary_loss_weight: 0.0
```

```yaml
# configs/model/vla_temporal.yaml
_target_: vla.models.TemporalVLAModel

freeze_vision: true
freeze_language: true
num_frames: 6
action_loss_weight: 1.0
```

### Step 3: Create vision configs (20 min)
```yaml
# configs/vision/vit_base.yaml
name: timm_vit
model_name: vit_base_patch16_224
pretrained: true
frozen: true
output_mode: spatial
proj_dim: null
```

```yaml
# configs/vision/dinov2.yaml
name: dinov2
size: base
pretrained: true
frozen: true
proj_dim: 768
```

```yaml
# configs/vision/siglip.yaml
name: siglip
size: base
pretrained: true
frozen: true
proj_dim: 768
```

### Step 4: Create language configs (15 min)
```yaml
# configs/language/gpt2.yaml
name: gpt2
model_name: gpt2
frozen: true
output_mode: mean
max_length: 77
proj_dim: null
```

```yaml
# configs/language/gpt2_medium.yaml
name: gpt2
model_name: gpt2-medium
frozen: true
output_mode: mean
max_length: 77
proj_dim: null
```

### Step 5: Create fusion configs (15 min)
```yaml
# configs/fusion/perceiver.yaml
name: perceiver_resampler
dim: 768
num_latents: 64
num_layers: 2
num_heads: 8
```

```yaml
# configs/fusion/cross_attention.yaml
name: cross_attention_fusion
dim: 768
num_layers: 4
num_heads: 8
dropout: 0.1
```

### Step 6: Create action configs (15 min)
```yaml
# configs/action/discrete.yaml
name: discrete_action
action_dim: 7
num_bins: 256
hidden_dim: null
```

```yaml
# configs/action/gaussian.yaml
name: gaussian_action
action_dim: 7
hidden_dim: null
min_std: 0.01
max_std: 1.0
```

### Step 7: Create train configs (25 min)
```yaml
# configs/train/default.yaml
# Training hyperparameters

# Optimizer
optimizer:
  _target_: torch.optim.AdamW
  lr: 1e-4
  weight_decay: 0.01
  betas: [0.9, 0.999]

# Scheduler
scheduler:
  _target_: torch.optim.lr_scheduler.CosineAnnealingLR
  T_max: ${train.max_epochs}
  eta_min: 1e-6

# Training settings
max_epochs: 100
batch_size: 32
gradient_accumulation_steps: 1
gradient_clip_val: 1.0

# Mixed precision
precision: "bf16-mixed"

# Checkpointing
checkpoint:
  save_top_k: 3
  monitor: "val/loss"
  mode: "min"
  save_last: true

# Logging
logging:
  log_every_n_steps: 50
  val_check_interval: 0.25

# Early stopping
early_stopping:
  patience: 10
  min_delta: 0.001
```

```yaml
# configs/train/distributed.yaml
# @package _global_
defaults:
  - default

# Override for multi-GPU
trainer:
  strategy: "ddp"
  devices: -1  # Use all available GPUs
  accelerator: "gpu"

# Larger batch size with gradient accumulation
batch_size: 16
gradient_accumulation_steps: 4
```

### Step 8: Create data configs (20 min)
```yaml
# configs/data/dummy.yaml
# Dummy data for testing

name: dummy
batch_size: ${train.batch_size}
num_workers: 4
pin_memory: true

# Dummy dataset params
num_samples: 1000
image_size: 224
action_dim: 7
```

```yaml
# configs/data/oxe.yaml
# Open X-Embodiment dataset

name: oxe
dataset_path: /data/oxe
batch_size: ${train.batch_size}
num_workers: 8
pin_memory: true
prefetch_factor: 2

# Dataset mixing
datasets:
  - name: "bridge_v2"
    weight: 0.4
  - name: "rt_1_robot"
    weight: 0.3
  - name: "kuka"
    weight: 0.3

# Preprocessing
image_size: 224
normalize: true
augment: false

# Sequence settings
num_frames: 1
chunk_size: 8
```

### Step 9: Create experiment configs (30 min)
```yaml
# configs/experiment/baseline.yaml
# @package _global_
# Baseline VLA configuration

defaults:
  - override /model: vla_base
  - override /vision: vit_base
  - override /language: gpt2
  - override /fusion: perceiver
  - override /action: discrete
  - override /train: default
  - override /data: dummy

# Experiment-specific overrides
project:
  name: "tinyVLA-baseline"

train:
  max_epochs: 50
  batch_size: 32

logging:
  wandb:
    project: "tinyVLA"
    name: "baseline-${now:%Y%m%d-%H%M%S}"
    tags: ["baseline", "discrete"]
```

```yaml
# configs/experiment/ablation_vision.yaml
# @package _global_
# Vision encoder ablation

defaults:
  - override /model: vla_base
  - override /train: default

# This will be used with multirun
# python train.py +experiment=ablation_vision --multirun vision=vit_base,dinov2,siglip

project:
  name: "tinyVLA-ablation-vision"

logging:
  wandb:
    project: "tinyVLA"
    name: "ablation-vision-${vision.name}"
    tags: ["ablation", "vision"]
```

### Step 10: Implement hydra_utils.py (30 min)
```python
"""Hydra configuration utilities."""
import os
from pathlib import Path
from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
import hydra


def get_config_dir() -> Path:
    """Get the configs directory path."""
    return Path(__file__).parent.parent.parent.parent / "configs"


def print_config(cfg: DictConfig, resolve: bool = True) -> None:
    """Pretty print configuration."""
    print(OmegaConf.to_yaml(cfg, resolve=resolve))


def save_config(cfg: DictConfig, path: str) -> None:
    """Save configuration to YAML file."""
    with open(path, "w") as f:
        OmegaConf.save(cfg, f)


def flatten_config(cfg: DictConfig) -> Dict[str, Any]:
    """Flatten nested config to dot-notation dict."""
    return dict(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))


def get_run_dir() -> Path:
    """Get the current run output directory."""
    if HydraConfig.initialized():
        return Path(HydraConfig.get().runtime.output_dir)
    return Path("outputs")


def register_resolvers():
    """Register custom OmegaConf resolvers."""
    # Resolver for getting environment variables with default
    OmegaConf.register_new_resolver(
        "env",
        lambda key, default="": os.environ.get(key, default),
        replace=True,
    )

    # Resolver for computing derived values
    OmegaConf.register_new_resolver(
        "mult",
        lambda x, y: x * y,
        replace=True,
    )


def init_hydra(config_path: str = "../configs", config_name: str = "config"):
    """Initialize Hydra for testing/notebooks."""
    GlobalHydra.instance().clear()
    register_resolvers()

    @hydra.main(version_base=None, config_path=config_path, config_name=config_name)
    def _init(cfg: DictConfig) -> DictConfig:
        return cfg

    return _init


def validate_config(cfg: DictConfig) -> None:
    """Validate configuration has required fields."""
    required = ["model", "vision", "language", "fusion", "action", "train"]
    for field in required:
        if field not in cfg:
            raise ValueError(f"Missing required config field: {field}")

    # Validate dimensions are consistent
    if cfg.vision.proj_dim and cfg.fusion.dim:
        if cfg.vision.proj_dim != cfg.fusion.dim:
            print(f"Warning: vision.proj_dim ({cfg.vision.proj_dim}) != fusion.dim ({cfg.fusion.dim})")
```

## Todo List
- [ ] Create configs/ directory structure
- [ ] Create config.yaml main entry point
- [ ] Create model/ config group
- [ ] Create vision/ config group
- [ ] Create language/ config group
- [ ] Create fusion/ config group
- [ ] Create action/ config group
- [ ] Create train/ config group
- [ ] Create data/ config group
- [ ] Create experiment/ presets
- [ ] Implement hydra_utils.py helpers
- [ ] Test CLI overrides
- [ ] Test multirun sweeps

## Success Criteria
1. `python train.py` loads default config
2. `python train.py vision=dinov2` overrides vision
3. `python train.py +experiment=baseline` loads experiment
4. `python train.py --multirun vision=vit_base,dinov2` runs sweep
5. Config auto-saved to outputs/.hydra/

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Config path issues | Medium | Use absolute paths, test on clean env |
| Missing defaults | Medium | Validate config on load |
| Type mismatches | Low | Use structured configs |

## Security Considerations
- No secrets in config files
- Use environment variables for sensitive values

## Next Steps
- Phase 10: Data loading pipeline
