# Phase 03: scripts/eval.py Evaluation Entry Point

**Priority:** MEDIUM
**Status:** COMPLETE ✓
**Depends on:** Phase 01 (test_step must exist) — ✓ SATISFIED

## Context Links

- Training script pattern: `scripts/train.py`
- Lightning module: `src/vla/training/lightning_module.py`
- Hydra configs: `configs/config.yaml`
- VLAModel checkpoint API: `src/vla/models/vla_base.py`
- Brainstorm: `plans/reports/brainstorm-phase-12.md`

## Overview

`scripts/eval.py` is the CLI entry point for evaluating a trained checkpoint.
It mirrors `scripts/train.py`'s Hydra pattern but calls `trainer.test()` instead of `trainer.fit()`.
The checkpoint path is passed as a Hydra override: `python scripts/eval.py eval.checkpoint=path/to/ckpt.pt`

## Key Insights

- PL saves checkpoints as `.ckpt` files (via `ModelCheckpoint`); `VLAModel` saves `.pt` files.
  - If checkpoint is a `.ckpt` (PL format): use `VLALightningModule.load_from_checkpoint(path)`
  - If checkpoint is a `.pt` (raw VLAModel): load via `VLAModel.load_checkpoint(path)`, wrap in new `VLALightningModule`
  - Detect by file extension — keep it simple
- Reuse `_build_datamodule` logic from `train.py` to avoid duplication (DRY)
  - But `_build_datamodule` lives inside `train.py` — can't import it. Extract shared helper or just inline the minimal version in `eval.py` (YAGNI — eval only needs dummy/lerobot, same 10 lines)
- `trainer.test()` requires `test_dataloader` in the datamodule — same `VLADataModule` works
- `logger=False` by default for eval (no WandB run unless `eval.use_wandb=true`)

## Requirements

### Functional
- `@hydra.main` entry point, config from `configs/config.yaml` + optional `eval` sub-config
- Accepts `eval.checkpoint` path (required override, raises `ValueError` if missing)
- Loads checkpoint (`.ckpt` → PL, `.pt` → VLAModel)
- Runs `pl.Trainer.test(module, datamodule=dm)` and prints results
- Exits cleanly with test metrics printed to stdout

### Non-functional
- Under 100 LOC
- No new config files needed — `eval.checkpoint` passed as CLI override
- Follow same logger/docstring conventions as `train.py`

## Architecture

```
scripts/eval.py
└── @hydra.main(config_path="../configs", config_name="config")
    def main(cfg):
        checkpoint_path = cfg.get("eval", {}).get("checkpoint", None)
        if not checkpoint_path: raise ValueError(...)

        # Load module
        path = Path(checkpoint_path)
        if path.suffix == ".ckpt":
            module = VLALightningModule.load_from_checkpoint(path)
        else:
            model = VLAModel.load_checkpoint(str(path))
            module = VLALightningModule(model_cfg=model.config)
            module.model = model

        # Build datamodule (same as train.py)
        dm = _build_datamodule(cfg)

        # Evaluate
        trainer = pl.Trainer(accelerator="auto", logger=False, devices=1)
        results = trainer.test(module, datamodule=dm)
        logger.info(f"Test results: {results}")
```

**Why not add an `eval.yaml` config?**
YAGNI. The only required field is `eval.checkpoint`. Adding a full config file for one field adds
complexity without benefit. Hydra's `+eval.checkpoint=path` CLI syntax handles it cleanly.

**Alternative: `VLAModel.load_checkpoint` only (no PL checkpoint support)**
Simpler, but PL checkpoints include optimizer state, epoch number, and hparams — useful context.
Supporting both formats costs ~4 lines and future-proofs the script.

## Related Code Files

**Create:**
- `scripts/eval.py` — evaluation entry point

**Read (no changes):**
- `scripts/train.py` — reference for Hydra pattern, `_build_datamodule`, `_build_loggers`
- `src/vla/models/vla_base.py` — confirm `load_checkpoint` signature
- `src/vla/training/lightning_module.py` — confirm `load_from_checkpoint` is available (PL default)

## Implementation Steps

1. Create `scripts/eval.py`:

```python
"""Evaluation entry point for tinyVLA.

Loads a trained checkpoint and runs pl.Trainer.test() over the
configured dataset. Override checkpoint path from CLI:

    python scripts/eval.py +eval.checkpoint=outputs/checkpoints/last.ckpt
    python scripts/eval.py +eval.checkpoint=outputs/model.pt data=dummy

Metrics (test/loss, test/mse, test/mae) are printed to stdout.
"""

from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from vla.data import VLADataModule
from vla.models import VLAModel
from vla.training import VLALightningModule
from vla.utils import setup_logger

logger = setup_logger(__name__)


def _build_datamodule(cfg: DictConfig) -> VLADataModule:
    """Build VLADataModule from Hydra data config (mirrors train.py)."""
    data_cfg = cfg.data
    name_to_type = {"dummy": "dummy", "lerobot": "lerobot"}
    dataset_type = name_to_type.get(data_cfg.get("name", "dummy"), "dummy")

    kwargs = dict(
        dataset_type=dataset_type,
        batch_size=cfg.train.get("batch_size", 32),
        num_workers=cfg.train.get("num_workers", 4),
        total_samples=data_cfg.get("num_samples", 10000),
    )
    if dataset_type == "lerobot":
        if "repo_id" in data_cfg:
            kwargs["repo_id"] = data_cfg.repo_id
    return VLADataModule(**kwargs)


def _load_module(checkpoint_path: Path) -> VLALightningModule:
    """Load VLALightningModule from a .ckpt (PL) or .pt (VLAModel) file.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        VLALightningModule ready for evaluation

    Raises:
        FileNotFoundError: If checkpoint does not exist
        ValueError: If file extension is unrecognised
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if checkpoint_path.suffix == ".ckpt":
        # PL checkpoint — includes hparams; load_from_checkpoint handles config
        logger.info(f"Loading PL checkpoint: {checkpoint_path}")
        return VLALightningModule.load_from_checkpoint(str(checkpoint_path))

    if checkpoint_path.suffix == ".pt":
        # Raw VLAModel checkpoint — wrap in a fresh LightningModule
        logger.info(f"Loading VLAModel checkpoint: {checkpoint_path}")
        model = VLAModel.load_checkpoint(str(checkpoint_path))
        module = VLALightningModule(model_cfg=model.config)
        module.model = model
        return module

    raise ValueError(
        f"Unrecognised checkpoint extension '{checkpoint_path.suffix}'. "
        "Expected .ckpt (PyTorch Lightning) or .pt (VLAModel)."
    )


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run VLA evaluation on a trained checkpoint.

    Args:
        cfg: Composed Hydra config. Must include +eval.checkpoint=<path>.
    """
    logger.info("Eval config:\n" + OmegaConf.to_yaml(cfg, resolve=True))

    eval_cfg = cfg.get("eval", {})
    checkpoint_path = eval_cfg.get("checkpoint", None)
    if not checkpoint_path:
        raise ValueError(
            "Checkpoint path is required. Pass it with: +eval.checkpoint=path/to/ckpt"
        )

    module = _load_module(Path(checkpoint_path))
    dm = _build_datamodule(cfg)

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
    )

    logger.info(f"Running test on checkpoint: {checkpoint_path}")
    results = trainer.test(module, datamodule=dm)
    logger.info(f"Test results: {results}")


if __name__ == "__main__":
    main()
```

## Todo

- [x] Create `scripts/eval.py`
- [x] Manual smoke test: ✓ raises `ValueError` when `+eval.checkpoint` missing
- [x] Manual smoke test: ✓ raises `FileNotFoundError` when path doesn't exist
- [x] Run `black scripts/eval.py` + `ruff check scripts/eval.py` — ✓ passed
- [x] Run `mypy scripts/eval.py` — ✓ no type errors

## Success Criteria

- Script exists and is importable (`python scripts/eval.py --help` shows Hydra usage)
- Raises `ValueError` with helpful message when `+eval.checkpoint` is missing
- Raises `FileNotFoundError` when checkpoint path does not exist
- Loads a `.ckpt` and runs `trainer.test()` without error (manual test with a real checkpoint)
- Passes `black`, `ruff`, `mypy`

## Common Pitfalls

- **`load_from_checkpoint` requires `model_cfg` in saved hparams.** `VLALightningModule` saves hparams via `save_hyperparameters(ignore=["model_cfg"])` — so `model_cfg` is NOT in the checkpoint. PL will fail to reconstruct `__init__` without it. **Fix:** when loading `.ckpt`, check if hparams include enough info, or fall back to requiring the user to also pass `+model=vla` config override.
  - **Simpler mitigation:** document that `.ckpt` loading requires matching Hydra config to be active (which it is, since `@hydra.main` always loads `configs/config.yaml`). The model config comes from Hydra, not the checkpoint hparams.
  - Revised `.ckpt` loading: `VLALightningModule.load_from_checkpoint(path, model_cfg=VLAConfig.from_hydra(cfg.model))`
- **`_build_datamodule` duplication with `train.py`.** Acceptable for now (YAGNI — no shared module yet). If a third script needs it, extract to `src/vla/utils/script_utils.py`.
- **`devices=1` hardcoded.** Eval is typically single-GPU. If multi-GPU eval is needed, add `+eval.devices=auto` override later.

## Risk Assessment

| Risk | Probability | Mitigation |
|------|-------------|------------|
| `load_from_checkpoint` fails on missing `model_cfg` | HIGH | Pass `model_cfg=VLAConfig.from_hydra(cfg.model)` explicitly |
| `VLADataModule` missing `test_dataloader` | MEDIUM | Verify in Phase 02; dummy dataset reuses val split |
| Hydra output dir pollution during eval | LOW | Hydra writes to `outputs/` — acceptable, document in script docstring |
