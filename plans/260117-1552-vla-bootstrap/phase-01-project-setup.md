# Phase 01: Project Setup

## Context Links
- [Tech Stack](../../docs/tech-stack.md)
- [PyTorch VLA Research](../reports/researcher-260118-0228-pytorch-vla.md)

## Overview
| Field | Value |
|-------|-------|
| Priority | P1 - Critical Path |
| Status | Pending |
| Effort | 2h |
| Dependencies | None |

Establish project foundation with proper dependency management, directory structure, and development tooling.

## Key Insights
- Use `uv` for fast dependency resolution (10x faster than pip)
- PyTorch 2.5+ required for FSDP2 and torch.compile
- Hydra 1.3+ for structured configs and multirun
- Keep files under 200 lines per YAGNI/KISS principles

## Requirements

### Functional
- FR-01: Python package installable via `pip install -e .`
- FR-02: All core dependencies pinned to compatible versions
- FR-03: Development tools (pytest, black, ruff, mypy) configured

### Non-Functional
- NFR-01: Package builds in <30s
- NFR-02: Import time <2s for core modules

## Architecture

```
tinyVLA/
├── pyproject.toml          # Package + dependencies
├── README.md               # Project overview
├── .gitignore              # Git exclusions
├── .python-version         # Python version (3.10+)
├── configs/                # Hydra configs (Phase 9)
├── src/
│   └── vla/
│       ├── __init__.py
│       ├── registry/       # Phase 2
│       ├── nn/             # Phase 3
│       ├── backbones/      # Phases 4-5
│       ├── fusion/         # Phase 6
│       ├── policy/         # Phase 7
│       ├── models/         # Phase 8
│       ├── training/       # Phase 11
│       ├── data/           # Phase 10
│       └── utils/
│           ├── __init__.py
│           └── logging.py
├── scripts/
│   ├── train.py            # Phase 11
│   ├── eval.py
│   └── export.py
└── tests/
    ├── __init__.py
    ├── conftest.py
    └── unit/
        └── __init__.py
```

## Related Code Files

### Files to Create
| Path | Purpose | Lines |
|------|---------|-------|
| `pyproject.toml` | Package config + deps | ~80 |
| `src/vla/__init__.py` | Package root | ~10 |
| `src/vla/utils/__init__.py` | Utils module | ~5 |
| `src/vla/utils/logging.py` | Logging setup | ~40 |
| `tests/conftest.py` | Pytest fixtures | ~30 |
| `.python-version` | Python version | 1 |

## Implementation Steps

### Step 1: Create pyproject.toml (30 min)
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "tinyvla"
version = "0.1.0"
description = "Modular VLA research framework"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.5.0",
    "torchvision>=0.20.0",
    "pytorch-lightning>=2.2.0",
    "hydra-core>=1.3.0",
    "omegaconf>=2.3.0",
    "timm>=1.0.0",
    "transformers>=4.40.0",
    "wandb>=0.16.0",
    "einops>=0.7.0",
    "numpy>=1.26.0",
    "pillow>=10.0.0",
    "h5py>=3.10.0",
    "webdataset>=0.2.86",
    "tensorboard>=2.15.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=4.0.0",
    "black>=24.0.0",
    "ruff>=0.3.0",
    "mypy>=1.8.0",
    "pre-commit>=3.6.0",
]

[tool.hatch.build.targets.wheel]
packages = ["src/vla"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --cov=vla --cov-report=term-missing"

[tool.black]
line-length = 100
target-version = ["py310"]

[tool.ruff]
line-length = 100
select = ["E", "F", "I", "W"]
ignore = ["E501"]

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
ignore_missing_imports = true
```

### Step 2: Create directory structure (15 min)
```bash
mkdir -p src/vla/{registry,nn,backbones,fusion,policy,models,training,data,utils}
mkdir -p configs/{experiment,model,vision,language,fusion,action,train,data}
mkdir -p scripts tests/unit tests/integration
```

### Step 3: Create __init__.py files (15 min)
- Root `src/vla/__init__.py` with version
- Empty `__init__.py` in each subpackage
- Defer imports to avoid circular dependencies

### Step 4: Create utils/logging.py (20 min)
```python
import logging
from pathlib import Path

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Configure module logger with console handler."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
```

### Step 5: Create tests/conftest.py (20 min)
```python
import pytest
import torch

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def batch_size():
    return 2

@pytest.fixture
def dummy_image(batch_size):
    return torch.randn(batch_size, 3, 224, 224)

@pytest.fixture
def dummy_text():
    return ["pick up the red block", "place on table"]
```

### Step 6: Verify installation (20 min)
```bash
uv pip install -e ".[dev]"
python -c "import vla; print(vla.__version__)"
pytest tests/ -v
```

## Todo List
- [ ] Create pyproject.toml with all dependencies
- [ ] Create directory structure
- [ ] Create __init__.py files for all packages
- [ ] Implement utils/logging.py
- [ ] Create tests/conftest.py with fixtures
- [ ] Install package in development mode
- [ ] Verify imports work correctly
- [ ] Run initial pytest to confirm setup

## Success Criteria
1. `pip install -e .` completes without errors
2. `python -c "import vla"` succeeds
3. `pytest tests/` runs (even if 0 tests)
4. `ruff check src/` passes
5. `black --check src/` passes

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Dependency conflicts | High | Pin exact versions, test in clean venv |
| CUDA version mismatch | Medium | Document CUDA requirements in README |
| Import circular deps | Medium | Use lazy imports, proper __init__.py |

## Security Considerations
- No secrets in pyproject.toml
- .gitignore includes .env, credentials
- Dependency versions from trusted sources (PyPI)

## Next Steps
- Phase 2: Implement registry patterns for component loading
