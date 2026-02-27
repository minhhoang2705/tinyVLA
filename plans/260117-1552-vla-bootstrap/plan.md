---
title: "tinyVLA Bootstrap Implementation"
description: "Modular VLA research framework with composable blocks, backbone builders, and Hydra configs"
status: in-progress
priority: P1
effort: 40h
branch: main
tags: [vla, pytorch, hydra, vision-language-action, robotics]
created: 2026-01-17
---

# tinyVLA Bootstrap Plan

## Overview
Bootstrap a modular Vision-Language-Action (VLA) research framework with:
- Composable block primitives (attention, MLP, norms)
- Vision/language backbone builders via timm + transformers
- Perceiver Resampler fusion mechanism
- Discrete/Gaussian action heads
- Hydra-based hierarchical configuration

## Tech Stack
PyTorch 2.5+, Hydra 1.3+, Lightning 2.2+, timm 1.0+, transformers 4.40+, WandB

## Directory Structure
```
src/vla/
  registry/     # Component registries + factories
  nn/           # Primitives (attention, mlp, norm, pos_encoding)
  backbones/    # Vision (timm) + language (GPT-2)
  fusion/       # Perceiver Resampler, cross-attention
  policy/       # Action heads (discrete bins, Gaussian)
  models/       # VLA orchestration
  training/     # Lightning modules, callbacks
  data/         # OXE loaders, transforms
  utils/        # Logging, checkpoint, helpers
configs/        # Hydra YAML configs
scripts/        # train.py, eval.py, export.py
tests/          # Unit + integration tests
```

## Phases

| Phase | Description | Effort | Status |
|-------|-------------|--------|--------|
| [01](phase-01-project-setup.md) | Project setup (pyproject, structure) | 2h | ✅ Complete |
| [02](phase-02-core-registries.md) | Registries + factory patterns | 2h | ✅ Complete |
| [03](phase-03-nn-primitives.md) | NN primitives (attention, MLP, norm) | 4h | ✅ Complete |
| [04](phase-04-vision-backbone.md) | Vision backbone builder (timm) | 3h | ✅ Complete |
| [05](phase-05-language-backbone.md) | Language backbone (GPT-2) | 3h | ✅ Complete |
| [06](phase-06-fusion.md) | Fusion mechanisms (Perceiver) | 4h | ✅ Complete |
| [07](phase-07-action-heads.md) | Policy + action heads | 3h | ✅ Complete |
| [08](phase-08-vla-model.md) | VLA model orchestration | 4h | ✅ Complete |
| [09](phase-09-hydra-configs.md) | Hydra configuration system | 4h | ✅ Complete |
| [10](phase-10-data-pipeline.md) | Data loading (Dummy + LeRobot HF) | 5h | ✅ Complete |
| [11](phase-11-training.md) | Training infrastructure (Lightning) | 4h | 🔄 In Progress |
| [12](phase-12-testing.md) | Testing suite | 2h | Pending |

## Key Dependencies
- Phase 2 (registries) blocks all component phases
- Phases 3-7 can run in parallel after Phase 2
- Phase 8 requires Phases 3-7 complete
- Phase 9 can run parallel with Phases 3-8
- Phase 11 requires Phases 8-10

## Research Reports
- [VLA Architectures](../reports/researcher-260118-vla-architectures.md)
- [PyTorch VLA Patterns](../reports/researcher-260118-0228-pytorch-vla.md)
- [Hydra ML Config](../reports/researcher-260118-hydra-ml-config.md)
- [OXE Data Loading](../reports/researcher-260118-0228-oxe-data-loading.md)

## Success Criteria
1. All components instantiable via registry + config
2. End-to-end forward pass with dummy data
3. Hydra multirun sweeps functional
4. Lightning training loop executes
5. 80%+ test coverage on core modules
