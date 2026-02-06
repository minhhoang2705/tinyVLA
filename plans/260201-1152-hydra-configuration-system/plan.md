---
title: "Phase 09: Hydra Configuration System"
description: "Implement declarative YAML-based Hydra configuration for all VLA components"
status: pending
priority: P1
effort: 4h
branch: master
tags: [hydra, configuration, infrastructure, phase-09]
created: 2026-02-01
---

# Phase 09: Hydra Configuration System

## Context Links
- [Existing Phase 09 Blueprint](../260117-1552-vla-bootstrap/phase-09-hydra-configs.md)
- [Registry Factories](../../src/vla/registry/factories.py) -- already accepts DictConfig
- [VLA Configs Dataclasses](../../src/vla/models/vla_configs.py)

## Overview

Replace hardcoded Python configs with Hydra-based YAML composition. Enables reproducible experiments, CLI overrides, multirun sweeps. Unblocks Phase 10 (Data) and Phase 11 (Training).

**Key Insight:** `factories.py` already handles `DictConfig` via `hasattr(cfg, "_target_")` + registry `name` key pattern. Minimal Python code changes needed -- this is primarily a YAML authoring + utility phase.

## Registered Components (Source of Truth)

From codebase analysis, these are the exact registry names to match in YAML:

| Registry | Registered Names |
|----------|-----------------|
| VISION | `timm_vit`, `dinov2`, `siglip` |
| LANGUAGE | `gpt2`, `language_encoder` |
| FUSION | `perceiver_resampler`, `cross_attention_fusion`, `gated_fusion`, `concat_fusion`, `prepend_fusion`, `temporal_perceiver` |
| ACTION | `discrete_action`, `gaussian_action`, `hybrid_action` |
| MODEL | `vla_base`, `vla_temporal` |

## Phase Breakdown

| Phase | File | Status | Effort | Description |
|-------|------|--------|--------|-------------|
| 01 | [phase-01-directory-structure-and-main-config.md](phase-01-directory-structure-and-main-config.md) | Pending | 30m | Create configs/ tree + config.yaml entry point |
| 02 | [phase-02-component-yaml-config-files.md](phase-02-component-yaml-config-files.md) | Pending | 60m | Write 15+ YAML files for all config groups |
| 03 | [phase-03-factory-integration-with-hydra-dictconfig.md](phase-03-factory-integration-with-hydra-dictconfig.md) | Pending | 45m | Bridge Hydra DictConfig to existing factories + VLAConfig.from_hydra() |
| 04 | [phase-04-hydra-utility-functions-and-resolvers.md](phase-04-hydra-utility-functions-and-resolvers.md) | Complete | 45m | Utility functions: validation, resolvers, print/save helpers |
| 05 | [phase-05-testing-and-validation-scripts.md](phase-05-testing-and-validation-scripts.md) | Complete ✓ | 60m | Test script + pytest tests for config loading, overrides, sweeps |

## Dependencies
- Phase 1 (project structure) -- complete
- `hydra-core>=1.3.0` and `omegaconf>=2.3.0` -- already in pyproject.toml

## Risk Summary
- **Config path resolution**: Hydra uses relative paths from the decorated function's module. Must use `config_path` parameter correctly.
- **DictConfig vs dict**: `VLAConfig.from_dict()` expects plain dict. Need `OmegaConf.to_container()` bridge or new `from_hydra()` classmethod.
- **Cross-reference cycles**: Avoid circular `${}` interpolations between config groups.
