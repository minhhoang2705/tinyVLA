# Planner Report: Hydra Configuration System (Phase 09)

**Date:** 2026-02-01
**Plan Dir:** `plans/260201-1152-hydra-configuration-system/`

## Summary

Created 5-phase implementation plan for Hydra configuration system. Total effort: ~4h.

## What Was Analyzed

- **Registry factories** (`src/vla/registry/factories.py`): Already accept DictConfig; minimal changes needed
- **All component constructors**: Verified exact parameter names for 10+ registered components across 5 registries
- **VLAConfig dataclass** (`src/vla/models/vla_configs.py`): Has `from_dict()`, needs `from_hydra()` bridge
- **VLAModel.__init__**: Uses dataclass config, not factory functions directly
- **pyproject.toml**: `hydra-core>=1.3.0` and `omegaconf>=2.3.0` already in deps

## Key Design Decisions

1. **YAML `name` fields map to exact registry keys** -- verified against all `@REGISTRY.register()` decorators
2. **Minimal Python changes**: Factories already handle DictConfig. Only additions: `VLAConfig.from_hydra()` + `build_vla_from_hydra()` + utility module
3. **No WandB in Phase 09** -- deferred to Phase 11 per user requirement
4. **Python files use underscores** (not kebab-case) since Python cannot import hyphenated modules
5. **Runtime-injected params** (`input_dim`, `vision_dim`, `language_dim`) omitted from YAML, set by model builder

## Phases

| # | Phase | Effort | Key Deliverables |
|---|-------|--------|-----------------|
| 01 | Directory + main config | 30m | `configs/` tree, `config.yaml` with defaults list |
| 02 | 17 YAML config files | 60m | All config groups populated with verified params |
| 03 | Factory integration | 45m | `VLAConfig.from_hydra()`, `build_vla_from_hydra()` |
| 04 | Utility functions | 45m | `hydra_config_helpers.py`: validation, resolvers, print/save |
| 05 | Testing + validation | 60m | `test-hydra-config.py` script + pytest suite |

## Files to Create (22 total)
- 17 YAML config files in `configs/`
- 1 Python utility module (`src/vla/utils/hydra_config_helpers.py`)
- 1 test script (`scripts/test-hydra-config.py`)
- 1 pytest file (`tests/unit/test_hydra_config_loading.py`)

## Files to Modify (3 total)
- `src/vla/models/vla_configs.py` -- add `from_hydra()` classmethod
- `src/vla/registry/factories.py` -- add `build_vla_from_hydra()`
- `src/vla/registry/__init__.py` -- export new function

## Risks
- Config path resolution with Hydra (mitigated: use `initialize_config_dir()` with absolute paths)
- DictConfig vs dict bridging (mitigated: `OmegaConf.to_container(resolve=True)`)
- Experiment overrides reference file names, not registry names (documented clearly)
