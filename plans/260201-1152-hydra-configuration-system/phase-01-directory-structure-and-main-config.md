# Phase 01: Directory Structure and Main Config Entry Point

## Context Links
- [Plan Overview](plan.md)
- [Existing Phase 09 Blueprint](../260117-1552-vla-bootstrap/phase-09-hydra-configs.md)
- [pyproject.toml](../../pyproject.toml) -- confirms hydra-core>=1.3.0 in deps

## Overview
| Field | Value |
|-------|-------|
| Priority | P1 |
| Status | Pending |
| Effort | 30m |
| Dependencies | None (Phase 1 complete) |

Create the `configs/` directory tree and the main `config.yaml` entry point that Hydra uses to compose all component configs.

## Key Insights
- Hydra's `defaults` list controls composition order; `_self_` at the end means local keys override defaults
- Each subdirectory is a "config group" -- Hydra selects one YAML per group
- `experiment/` uses `@package _global_` to override any top-level key
- No `configs/` directory exists yet; create from scratch

## Requirements

### Functional
- FR-01: `configs/` directory with 8 subdirectories (model, vision, language, fusion, action, train, data, experiment)
- FR-02: `config.yaml` with defaults list selecting one config per group
- FR-03: Project metadata (name, version, seed) in main config

### Non-Functional
- NFR-01: Consistent 2-space YAML indentation
- NFR-02: Comments in config.yaml explaining usage

## Architecture

```
configs/
├── config.yaml              # Main entry point with defaults list
├── model/
│   └── (Phase 02)
├── vision/
│   └── (Phase 02)
├── language/
│   └── (Phase 02)
├── fusion/
│   └── (Phase 02)
├── action/
│   └── (Phase 02)
├── train/
│   └── (Phase 02)
├── data/
│   └── (Phase 02)
└── experiment/
    └── (Phase 02)
```

## Related Code Files

### Files to Create
| Path | Purpose | Lines |
|------|---------|-------|
| `configs/config.yaml` | Main Hydra entry point | ~25 |

### Directories to Create
- `configs/model/`
- `configs/vision/`
- `configs/language/`
- `configs/fusion/`
- `configs/action/`
- `configs/train/`
- `configs/data/`
- `configs/experiment/`

## Implementation Steps

### Step 1: Create directory tree (2 min)
```bash
mkdir -p configs/{model,vision,language,fusion,action,train,data,experiment}
```

### Step 2: Create `configs/config.yaml` (15 min)

This is the main Hydra entry point. The `defaults` list composes configs from each group.

```yaml
# configs/config.yaml
# Main configuration entry point for tinyVLA.
#
# Usage:
#   python scripts/train.py                              # Load defaults
#   python scripts/train.py vision=dinov2                # Override vision
#   python scripts/train.py +experiment=baseline         # Load experiment preset
#   python scripts/train.py --multirun vision=vit_base,dinov2  # Sweep

defaults:
  - model: vla_base
  - vision: vit_base
  - language: gpt2
  - fusion: perceiver
  - action: discrete
  - train: default
  - data: dummy
  - _self_

# Project metadata
project:
  name: tinyVLA
  version: "0.1.0"

# Global seed for reproducibility
seed: 42

# Output directory (auto-set by Hydra runtime)
output_dir: ${hydra:runtime.output_dir}
```

**Critical details:**
- `_self_` must be last in defaults so local keys (project, seed) override group defaults
- Default selections match the most common/lightweight components for quick testing
- `data: dummy` is default so running without real data works out-of-the-box
- `output_dir` uses Hydra resolver to auto-populate at runtime

### Step 3: Verify directory structure (2 min)
```bash
find configs/ -type f -o -type d | sort
```

Expected output:
```
configs/
configs/action
configs/config.yaml
configs/data
configs/experiment
configs/fusion
configs/language
configs/model
configs/train
configs/vision
```

## Todo List
- [ ] Create `configs/` and all 8 subdirectories
- [ ] Write `configs/config.yaml` with defaults list
- [ ] Verify directory structure with `find` command

## Success Criteria
1. `configs/config.yaml` exists with valid YAML syntax
2. All 8 subdirectories exist under `configs/`
3. `python -c "import yaml; yaml.safe_load(open('configs/config.yaml'))"` passes (basic syntax check; Hydra resolvers won't resolve but YAML is valid)

## Risk Assessment
| Risk | Impact | Mitigation |
|------|--------|------------|
| Wrong defaults list syntax | High -- Hydra won't load | Copy exact syntax from Hydra docs; test immediately |
| Missing subdirectory | Medium -- group won't resolve | Use single `mkdir -p` command with brace expansion |

## Security Considerations
- No secrets in config files; sensitive values use `${oc.env:VAR}` resolver (Phase 04)

## Next Steps
- Phase 02: Populate each config group with YAML files
