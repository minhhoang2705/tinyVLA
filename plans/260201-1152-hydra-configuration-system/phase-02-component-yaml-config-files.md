# Phase 02: Component YAML Config Files

## Context Links
- [Plan Overview](plan.md)
- [Phase 01](phase-01-directory-structure-and-main-config.md)
- [VLA Configs Dataclasses](../../src/vla/models/vla_configs.py) -- source of truth for defaults
- [Vision Backbones](../../src/vla/backbones/vision.py) -- constructor signatures
- [Language Backbones](../../src/vla/backbones/language.py) -- constructor signatures
- [Perceiver Fusion](../../src/vla/fusion/perceiver.py) -- constructor signatures
- [Cross-Attention Fusion](../../src/vla/fusion/cross_attention.py) -- constructor signatures
- [Action Heads](../../src/vla/policy/action_heads.py) -- constructor signatures
- [VLA Base Model](../../src/vla/models/vla_base.py) -- VLAModel constructor

## Overview
| Field | Value |
|-------|-------|
| Priority | P1 |
| Status | Pending |
| Effort | 60m |
| Dependencies | Phase 01 |

Write all YAML config files for each config group. Every YAML `name` field must match an exact registry key from the codebase. Constructor kwargs must match actual `__init__` signatures.

## Key Insights
- YAML values must match Python dataclass defaults in `vla_configs.py` for consistency
- `name` field = registry lookup key (used by factories.py)
- Remaining YAML keys = constructor kwargs passed to `REGISTRY.get(name, **kwargs)`
- Vision configs: `timm_vit` uses `model_name`, `dinov2`/`siglip` use `size`
- Action heads use `input_dim` (set at runtime from fusion.dim) -- omit from YAML, inject in factory
- `null` in YAML = `None` in Python

## Requirements

### Functional
- FR-01: One YAML per registered component variant
- FR-02: All YAML keys match constructor param names exactly
- FR-03: Experiment configs use `@package _global_` and `override` syntax
- FR-04: Train config has optimizer, scheduler, checkpoint, early stopping sections

### Non-Functional
- NFR-01: Comments explaining non-obvious parameters
- NFR-02: File count: 17 YAML files total (listed below)

## Architecture

### Complete File Manifest (17 files)

```
configs/
├── config.yaml                    # (Phase 01)
├── model/
│   ├── vla-base.yaml             # VLAModel defaults
│   └── vla-temporal.yaml         # TemporalVLAModel
├── vision/
│   ├── vit-base.yaml             # timm ViT base (default)
│   ├── dinov2.yaml               # DINOv2 base
│   └── siglip.yaml               # SigLIP base
├── language/
│   ├── gpt2.yaml                 # GPT-2 small (default)
│   └── gpt2-medium.yaml          # GPT-2 medium
├── fusion/
│   ├── perceiver.yaml            # Perceiver Resampler (default)
│   └── cross-attention.yaml      # Cross-Attention Fusion
├── action/
│   ├── discrete.yaml             # Discrete binned (default)
│   └── gaussian.yaml             # Gaussian continuous
├── train/
│   ├── default.yaml              # Standard training
│   └── distributed.yaml          # Multi-GPU DDP
├── data/
│   ├── dummy.yaml                # Synthetic data (default)
│   └── oxe.yaml                  # Open X-Embodiment
└── experiment/
    ├── baseline.yaml             # Baseline VLA experiment
    └── ablation-vision.yaml      # Vision encoder ablation
```

## Related Code Files

### Files to Create
| Path | Purpose | Lines |
|------|---------|-------|
| `configs/model/vla-base.yaml` | Base VLA model config | ~12 |
| `configs/model/vla-temporal.yaml` | Temporal VLA model config | ~14 |
| `configs/vision/vit-base.yaml` | ViT base via timm | ~10 |
| `configs/vision/dinov2.yaml` | DINOv2 backbone | ~9 |
| `configs/vision/siglip.yaml` | SigLIP backbone | ~9 |
| `configs/language/gpt2.yaml` | GPT-2 small | ~9 |
| `configs/language/gpt2-medium.yaml` | GPT-2 medium | ~9 |
| `configs/fusion/perceiver.yaml` | Perceiver Resampler | ~9 |
| `configs/fusion/cross-attention.yaml` | Cross-Attention Fusion | ~10 |
| `configs/action/discrete.yaml` | Discrete action head | ~8 |
| `configs/action/gaussian.yaml` | Gaussian action head | ~9 |
| `configs/train/default.yaml` | Default training config | ~40 |
| `configs/train/distributed.yaml` | Multi-GPU DDP config | ~18 |
| `configs/data/dummy.yaml` | Dummy dataset config | ~12 |
| `configs/data/oxe.yaml` | OXE dataset config | ~25 |
| `configs/experiment/baseline.yaml` | Baseline experiment | ~20 |
| `configs/experiment/ablation-vision.yaml` | Vision ablation | ~15 |

### Key Reference: Constructor Signatures

**VisionBackbone** (`timm_vit`):
```python
def __init__(self, model_name="vit_base_patch16_224", pretrained=True, frozen=True,
             output_mode="spatial", proj_dim=None)
```

**DINOv2Backbone** (`dinov2`):
```python
def __init__(self, size="base", pretrained=True, frozen=True, proj_dim=None)
```

**SigLIPBackbone** (`siglip`):
```python
def __init__(self, size="base", pretrained=True, frozen=True, proj_dim=None)
```

**GPT2Backbone** (`gpt2`):
```python
def __init__(self, model_name="gpt2", frozen=True, output_mode="mean",
             max_length=77, proj_dim=None)
```

**PerceiverResampler** (`perceiver_resampler`):
```python
def __init__(self, dim=768, num_latents=64, num_layers=2, num_heads=8,
             vision_dim=None, language_dim=None, dropout=0.1)
```

**CrossAttentionFusion** (`cross_attention_fusion`):
```python
def __init__(self, dim=768, num_layers=4, num_heads=8,
             vision_dim=None, language_dim=None, dropout=0.1)
```

**DiscreteActionHead** (`discrete_action`):
```python
def __init__(self, input_dim=768, action_dim=7, num_bins=256, hidden_dim=None)
```

**GaussianActionHead** (`gaussian_action`):
```python
def __init__(self, input_dim=768, action_dim=7, hidden_dim=None,
             min_std=0.01, max_std=1.0)
```

## Implementation Steps

### Step 1: Model Configs (10 min)

**`configs/model/vla-base.yaml`**
```yaml
# Base VLA model configuration
# Matches VLAConfig dataclass defaults in src/vla/models/vla_configs.py

name: vla_base
freeze_vision: true
freeze_language: true
action_loss_weight: 1.0
auxiliary_loss_weight: 0.0
```

**`configs/model/vla-temporal.yaml`**
```yaml
# Temporal VLA model for multi-frame input
# Matches TemporalVLAModel constructor

name: vla_temporal
freeze_vision: true
freeze_language: true
action_loss_weight: 1.0
auxiliary_loss_weight: 0.0
num_frames: 6
```

### Step 2: Vision Configs (10 min)

**`configs/vision/vit-base.yaml`**
```yaml
# ViT-Base via timm (default vision backbone)
# Registry: timm_vit | Constructor: VisionBackbone

name: timm_vit
model_name: vit_base_patch16_224
pretrained: true
frozen: true
output_mode: spatial
proj_dim: null
```

**`configs/vision/dinov2.yaml`**
```yaml
# DINOv2 self-supervised ViT (base variant)
# Registry: dinov2 | Constructor: DINOv2Backbone

name: dinov2
size: base
pretrained: true
frozen: true
proj_dim: null
```

**`configs/vision/siglip.yaml`**
```yaml
# SigLIP vision-language aligned ViT (base variant)
# Registry: siglip | Constructor: SigLIPBackbone

name: siglip
size: base
pretrained: true
frozen: true
proj_dim: null
```

### Step 3: Language Configs (5 min)

**`configs/language/gpt2.yaml`**
```yaml
# GPT-2 small language encoder (default)
# Registry: gpt2 | Constructor: GPT2Backbone

name: gpt2
model_name: gpt2
frozen: true
output_mode: mean
max_length: 77
proj_dim: null
```

**`configs/language/gpt2-medium.yaml`**
```yaml
# GPT-2 medium language encoder (larger capacity)
# Registry: gpt2 | Constructor: GPT2Backbone

name: gpt2
model_name: gpt2-medium
frozen: true
output_mode: mean
max_length: 77
proj_dim: null
```

### Step 4: Fusion Configs (5 min)

**`configs/fusion/perceiver.yaml`**
```yaml
# Perceiver Resampler fusion (default)
# Registry: perceiver_resampler | Constructor: PerceiverResampler
# vision_dim/language_dim set at runtime from backbone embed_dim

name: perceiver_resampler
dim: 768
num_latents: 64
num_layers: 2
num_heads: 8
dropout: 0.1
```

**`configs/fusion/cross-attention.yaml`**
```yaml
# Cross-Attention fusion (language-conditioned vision)
# Registry: cross_attention_fusion | Constructor: CrossAttentionFusion
# vision_dim/language_dim set at runtime from backbone embed_dim

name: cross_attention_fusion
dim: 768
num_layers: 4
num_heads: 8
dropout: 0.1
```

### Step 5: Action Configs (5 min)

**`configs/action/discrete.yaml`**
```yaml
# Discrete binned action head (RT-2 style, default)
# Registry: discrete_action | Constructor: DiscreteActionHead
# input_dim set at runtime from fusion.dim

name: discrete_action
action_dim: 7
num_bins: 256
hidden_dim: null
```

**`configs/action/gaussian.yaml`**
```yaml
# Gaussian continuous action head (stochastic policy)
# Registry: gaussian_action | Constructor: GaussianActionHead
# input_dim set at runtime from fusion.dim

name: gaussian_action
action_dim: 7
hidden_dim: null
min_std: 0.01
max_std: 1.0
```

### Step 6: Training Configs (10 min)

**`configs/train/default.yaml`**
```yaml
# Default training configuration
# Used by Phase 11 training loop

# Optimizer
optimizer:
  name: AdamW
  lr: 1.0e-4
  weight_decay: 0.01
  betas: [0.9, 0.999]

# Scheduler
scheduler:
  name: CosineAnnealingLR
  T_max: ${train.max_epochs}
  eta_min: 1.0e-6

# Training loop
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
log_every_n_steps: 50
val_check_interval: 0.25

# Early stopping
early_stopping:
  patience: 10
  min_delta: 0.001
```

**`configs/train/distributed.yaml`**
```yaml
# Multi-GPU distributed training via DDP
# Inherits from default, overrides for multi-GPU

defaults:
  - default

# DDP strategy
strategy: ddp
devices: -1  # All available GPUs
accelerator: gpu

# Adjusted for data parallelism
batch_size: 16
gradient_accumulation_steps: 4
```

### Step 7: Data Configs (5 min)

**`configs/data/dummy.yaml`**
```yaml
# Dummy synthetic dataset for testing (default)
# Uses DummyVLADataset from src/vla/data/

name: dummy
num_samples: 1000
image_size: 224
action_dim: ${action.action_dim}
num_workers: 4
pin_memory: true
```

**`configs/data/oxe.yaml`**
```yaml
# Open X-Embodiment dataset configuration
# Placeholder for Phase 10 data pipeline

name: oxe
dataset_path: /data/oxe
num_workers: 8
pin_memory: true
prefetch_factor: 2

# Dataset mixing weights
datasets:
  - name: bridge_v2
    weight: 0.4
  - name: rt_1_robot
    weight: 0.3
  - name: kuka
    weight: 0.3

# Preprocessing
image_size: 224
normalize: true
augment: false

# Sequence settings
num_frames: 1
chunk_size: 8
```

### Step 8: Experiment Configs (10 min)

**`configs/experiment/baseline.yaml`**
```yaml
# @package _global_
# Baseline VLA experiment preset
#
# Usage: python scripts/train.py +experiment=baseline

defaults:
  - override /model: vla-base
  - override /vision: vit-base
  - override /language: gpt2
  - override /fusion: perceiver
  - override /action: discrete
  - override /train: default
  - override /data: dummy

project:
  name: tinyVLA-baseline

seed: 42

train:
  max_epochs: 50
  batch_size: 32
```

**`configs/experiment/ablation-vision.yaml`**
```yaml
# @package _global_
# Vision encoder ablation study
#
# Usage: python scripts/train.py +experiment=ablation-vision \
#   --multirun vision=vit-base,dinov2,siglip

defaults:
  - override /model: vla-base
  - override /train: default

project:
  name: tinyVLA-ablation-vision

train:
  max_epochs: 30
  batch_size: 16
```

**Important note on experiment config `override` syntax:**
- Experiment configs reference YAML file names (kebab-case), NOT registry `name` values
- e.g., `override /vision: vit-base` loads `configs/vision/vit-base.yaml`
- The `name: timm_vit` inside that YAML is the registry key

## Todo List
- [ ] Create `configs/model/vla-base.yaml`
- [ ] Create `configs/model/vla-temporal.yaml`
- [ ] Create `configs/vision/vit-base.yaml`
- [ ] Create `configs/vision/dinov2.yaml`
- [ ] Create `configs/vision/siglip.yaml`
- [ ] Create `configs/language/gpt2.yaml`
- [ ] Create `configs/language/gpt2-medium.yaml`
- [ ] Create `configs/fusion/perceiver.yaml`
- [ ] Create `configs/fusion/cross-attention.yaml`
- [ ] Create `configs/action/discrete.yaml`
- [ ] Create `configs/action/gaussian.yaml`
- [ ] Create `configs/train/default.yaml`
- [ ] Create `configs/train/distributed.yaml`
- [ ] Create `configs/data/dummy.yaml`
- [ ] Create `configs/data/oxe.yaml`
- [ ] Create `configs/experiment/baseline.yaml`
- [ ] Create `configs/experiment/ablation-vision.yaml`

## Success Criteria
1. All 17 YAML files exist with valid syntax
2. Every `name` field matches an exact registry key from the codebase
3. Every non-`name` YAML key matches a constructor parameter name
4. `python -c "from omegaconf import OmegaConf; OmegaConf.load('configs/vision/vit-base.yaml')"` passes for each file

## Risk Assessment
| Risk | Impact | Mitigation |
|------|--------|------------|
| YAML key name mismatch with constructor | High -- runtime KeyError | Cross-referenced every constructor signature above |
| Wrong registry name | High -- KeyError at instantiation | Verified against `@REGISTRY.register()` decorators |
| Circular interpolation in ${} refs | Medium -- OmegaConf error | Only use downward refs (data refs action, not vice versa) |
| Hydra file-name vs registry-name confusion | Medium -- wrong config loaded | Documented clearly in experiment config section |

## Security Considerations
- No secrets or API keys in any config file
- `dataset_path` in oxe.yaml is a placeholder; real paths set via CLI or env var

## Next Steps
- Phase 03: Wire these configs into factory functions
