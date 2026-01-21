# Hydra Configuration Framework for ML/RL Experiments

**Date:** 2026-01-18 | **Status:** Research Complete | **Target:** tinyVLA VLA Configuration Management

---

## Executive Summary

Hydra (Facebook Research) elegantly solves ML configuration complexity through **hierarchical composition** + **runtime overrides**. Best for vision models: structured dataclass configs, multirun sweeps, WandB integration, reproducible experiment tracking.

---

## Core Concepts (Beginner-Friendly)

### 1. What is Hydra?
Configuration framework that:
- **Composable configs**: Merge YAML files hierarchically (base + model + dataset + training)
- **CLI overrides**: `python train.py model.lr=1e-4 trainer.batch_size=32` without code changes
- **Multirun**: Launch 100+ experiments in parallel with sweep syntax
- **Reproducibility**: Auto-saves all configs used per run to `.hydra/` folder

### 2. Key Abstractions

**Config Groups** → Directories of YAML configs
```
conf/
├── model/          # model.yaml, resnet50.yaml, vit.yaml
├── dataset/        # cifar10.yaml, imagenet.yaml
├── trainer/        # gpu.yaml, distributed.yaml
└── experiment/     # exp_baseline.yaml, exp_augmented.yaml
```

**Defaults List** → Execution order (hierarchical merge)
```yaml
defaults:
  - model: resnet50
  - dataset: imagenet
  - trainer: gpu
  - +experiment: baseline
```

**Overrides** → CLI changes applied last (highest priority)
```bash
python train.py model=vit dataset=cifar10 trainer.lr=5e-4
```

---

## Best Practices for ML/VLA

### 1. Structured Configs (Type Safety)
**Why**: Catch typos at runtime, IDE autocomplete, static type checking

```python
from dataclasses import dataclass, field
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore

@dataclass
class ModelConfig:
    name: str = "resnet50"
    pretrained: bool = True
    num_classes: int = MISSING  # Required field

@dataclass
class TrainerConfig:
    max_epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

cs = ConfigStore.instance()
cs.store(name="config", node=Config)
```

**Usage**:
```python
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: DictConfig) -> None:
    print(f"Model: {cfg.model.name}, LR: {cfg.trainer.lr}")
```

### 2. Experiment Tracking Pattern
**VLA-specific**: Vision models require tracking vision loss + action loss

```yaml
# conf/experiment/vla_baseline.yaml
# @package _global_
defaults:
  - override /model: vit_base
  - override /dataset: robotic_grasping
  - override /trainer: multi_gpu

model:
  hidden_dim: 768
  num_attention_heads: 12

trainer:
  max_epochs: 200
  batch_size: 64
  lr: 5e-4
  warmup_steps: 1000

logging:
  wandb:
    project: "tinyVLA"
    tags: ["baseline", "vision-only"]
```

**Run**: `python train.py +experiment=vla_baseline`

### 3. Hyperparameter Sweeps (Multirun)
**Grid search**: Test all combinations
```bash
python train.py --multirun \
  model.lr=1e-3,1e-4,1e-5 \
  trainer.batch_size=32,64 \
  trainer.warmup_steps=500,1000
# Launches 3 × 2 × 2 = 12 parallel experiments
```

**Random search**: Via Hydra + Optuna plugin
```bash
python train.py --multirun \
  hydra/launcher=joblib \
  'trainer.lr=range(1e-5, 1e-2)' \
  'trainer.dropout=choice(0.1,0.2,0.3)'
```

---

## Integration Patterns

### WandB Logging
```python
import wandb
from hydra.utils import instantiate

@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: DictConfig) -> None:
    wandb.init(project=cfg.logging.wandb.project, config=OmegaConf.to_container(cfg))

    # Training loop
    for epoch in range(cfg.trainer.max_epochs):
        loss = train_step()
        wandb.log({"loss": loss})

    wandb.finish()
```

### Checkpoint Management
```python
from pathlib import Path

@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg: DictConfig) -> None:
    ckpt_dir = Path(HydraConfig.get().runtime.output_dir) / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # Auto-saved to: outputs/2026-01-18/12-30-45/checkpoints/
    # .hydra/config.yaml contains exact config used
```

### Reproducibility
```bash
# Reproduce exact run
python train.py --config-path outputs/2026-01-18/12-30-45/.hydra --config-name config.yaml
```

---

## Common Pitfalls & Solutions

| Pitfall | Solution |
|---------|----------|
| Config bloat (100+ params) | Use `experiment/` group, only override changed values |
| Typos in overrides | Use structured configs + mypy for static checking |
| Lost experiment metadata | Hydra auto-saves `.hydra/config.yaml` per run |
| Multirun parallelization issues | Use `hydra/launcher=joblib` for local, `submitit` for SLURM |
| Config defaults order wrong | Defaults list order matters; test with `--cfg job` |
| MISSING fields forgotten | Dataclass validation catches these before training starts |

---

## VLA-Specific Architecture

**Recommended for tinyVLA**:

```yaml
# conf/config.yaml
defaults:
  - model: vit_base
  - dataset: robonet
  - vision_encoder: clip_vit
  - action_decoder: linear
  - trainer: gpu
  - +experiment: null  # Override with +experiment=baseline

model:
  name: "vla_joint"
  vision_dim: 768
  action_dim: 7

training:
  batch_size: 32
  vision_loss_weight: 1.0
  action_loss_weight: 0.5
```

**Launch vision + action experiments**:
```bash
python train.py \
  --multirun \
  vision_encoder=clip_vit,dinov2,siglip \
  training.vision_loss_weight=0.5,1.0,2.0 \
  +experiment=joint_training
```

---

## Reference Implementations

| Project | Pattern | Use Case |
|---------|---------|----------|
| [fairseq](https://github.com/facebookresearch/fairseq/tree/main/fairseq/config) | YAML + dataclass composition | NLP experiments |
| [detectron2](https://detectron2.readthedocs.io/en/v0.6/) | Lazy instantiation via LazyCall | Vision models (detection) |
| [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template) | PyTorch Lightning integration | General ML workflows |

---

## Learning Path

1. **Day 1**: Tutorial on [hydra.cc](https://hydra.cc/docs/intro/) (30 min)
2. **Day 2**: Implement structured config for tinyVLA model (1 hr)
3. **Day 3**: Add WandB logging + multirun sweeps (1 hr)
4. **Day 4**: Test reproducibility with checkpoint reload (30 min)

---

## Sources

- [Hydra Official Documentation](https://hydra.cc/)
- [Structured Configs Tutorial](https://hydra.cc/docs/tutorials/structured_config/intro/)
- [Experiment Configuration Patterns](https://hydra.cc/docs/patterns/configuring_experiments/)
- [PyTorch Lightning + Hydra Template](https://github.com/ashleve/lightning-hydra-template)
- [Fairseq Hydra Integration](https://github.com/facebookresearch/fairseq/blob/main/docs/hydra_integration.md)
- [Configuring ML with Hydra](https://dramsch.net/articles/config-driven-machine-learning-development-with-hydra/)
- [MLflow + Hydra Sweeps](https://towardsdatascience.com/hyperparameters-tuning-with-mlflow-and-hydra-sweeps-7253d97d7897/)

---

## Unresolved Questions

1. **VLA-specific**: Should action decoder params be in separate config group or nested under main?
2. **Multirun reporting**: Best way to aggregate metrics across 100+ parallel experiments?
3. **Checkpoint loading**: Hydra 1.3+ lazy instantiation—does it affect checkpoint compatibility?
