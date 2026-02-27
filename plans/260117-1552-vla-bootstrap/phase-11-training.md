# Phase 11: Training Infrastructure

## Context Links
- [PyTorch VLA Research](../reports/researcher-260118-0228-pytorch-vla.md) - Training patterns
- [Tech Stack](../../docs/tech-stack.md) - Lightning section

## Overview
| Field | Value |
|-------|-------|
| Priority | P1 - Core Infrastructure |
| Status | 🔄 In Progress |
| Effort | 4h |
| Dependencies | Phases 8, 9, 10 |

Implement PyTorch Lightning training module with WandB logging, checkpointing, and distributed training support.

## Key Insights
- Lightning abstracts training loop boilerplate
- WandB integration for experiment tracking
- Gradient checkpointing for memory efficiency
- FSDP2 for multi-GPU training

## Requirements

### Functional
- FR-01: LightningModule wrapper for VLA model
- FR-02: Configurable optimizer and scheduler
- FR-03: WandB logging of metrics and samples
- FR-04: Checkpoint saving with best model tracking
- FR-05: Training/validation/test step implementations
- FR-06: train.py CLI entry point

### Non-Functional
- NFR-01: Resume from checkpoint seamlessly
- NFR-02: Multi-GPU scaling with DDP/FSDP

## Architecture

```
src/vla/training/
├── __init__.py
├── lightning_module.py   # LightningModule wrapper
├── callbacks.py          # Custom callbacks
└── trainer.py            # Trainer factory

scripts/
├── train.py              # Main training entry
├── eval.py               # Evaluation script
└── export.py             # Model export
```

## Related Code Files

### Files to Create
| Path | Purpose | Lines |
|------|---------|-------|
| `src/vla/training/__init__.py` | Exports | ~10 |
| `src/vla/training/lightning_module.py` | LightningModule | ~150 |
| `src/vla/training/callbacks.py` | Callbacks | ~80 |
| `src/vla/training/trainer.py` | Trainer factory | ~60 |
| `scripts/train.py` | Training CLI | ~80 |
| `tests/unit/test_training.py` | Training tests | ~80 |

## Implementation Steps

### Step 1: Implement lightning_module.py (75 min)
```python
"""PyTorch Lightning module for VLA training."""
import torch
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Any, Optional, List
from omegaconf import DictConfig
from hydra.utils import instantiate

from vla.models import VLAModel, VLAConfig
from vla.policy.action_utils import compute_action_loss


class VLALightningModule(pl.LightningModule):
    """Lightning wrapper for VLA model training.

    Args:
        config: Full Hydra config or VLAConfig
        model: Optional pre-built VLAModel
    """

    def __init__(
        self,
        config: DictConfig | VLAConfig,
        model: Optional[VLAModel] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        if isinstance(config, DictConfig):
            self.cfg = config
            # Build model from config
            if model is None:
                model_cfg = VLAConfig(
                    vision=config.vision,
                    language=config.language,
                    fusion=config.fusion,
                    action=config.action,
                    freeze_vision=config.model.get("freeze_vision", True),
                    freeze_language=config.model.get("freeze_language", True),
                )
                self.model = VLAModel(model_cfg)
            else:
                self.model = model
        else:
            self.cfg = config
            self.model = model or VLAModel(config)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        return self.model(
            images=batch["images"],
            texts=batch["texts"],
            target_actions=batch.get("actions"),
        )

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step."""
        output = self.forward(batch)
        loss = output["loss"]

        # Log metrics
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        # Log action statistics
        with torch.no_grad():
            actions = output["actions"]
            self.log("train/action_mean", actions.mean())
            self.log("train/action_std", actions.std())

        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step."""
        output = self.forward(batch)
        loss = output["loss"]

        # Compute metrics
        pred_actions = output["actions"]
        target_actions = batch["actions"]
        mae = (pred_actions - target_actions).abs().mean()

        # Log metrics
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/action_mae", mae, sync_dist=True)

        return {"loss": loss, "pred": pred_actions, "target": target_actions}

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step."""
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        # Get trainable parameters
        params = self.model.get_trainable_params()

        # Optimizer
        if hasattr(self.cfg, "train") and hasattr(self.cfg.train, "optimizer"):
            optimizer = instantiate(self.cfg.train.optimizer, params=params)
        else:
            optimizer = AdamW(
                params,
                lr=1e-4,
                weight_decay=0.01,
                betas=(0.9, 0.999),
            )

        # Scheduler
        if hasattr(self.cfg, "train") and hasattr(self.cfg.train, "scheduler"):
            scheduler = instantiate(
                self.cfg.train.scheduler,
                optimizer=optimizer,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        return optimizer

    def on_train_epoch_end(self):
        """Log learning rate at epoch end."""
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("train/lr", lr)


class VLADataModule(pl.LightningDataModule):
    """Lightning DataModule for VLA datasets."""

    def __init__(
        self,
        config: DictConfig,
        train_dataset=None,
        val_dataset=None,
        test_dataset=None,
    ):
        super().__init__()
        self.cfg = config
        self._train_dataset = train_dataset
        self._val_dataset = val_dataset
        self._test_dataset = test_dataset

    def setup(self, stage: Optional[str] = None):
        """Setup datasets."""
        from vla.data import DummyVLADataset, collate_vla_batch

        if self._train_dataset is None:
            if self.cfg.data.name == "dummy":
                self._train_dataset = DummyVLADataset(
                    num_samples=self.cfg.data.get("num_samples", 1000),
                    image_size=self.cfg.data.get("image_size", 224),
                    action_dim=self.cfg.action.action_dim,
                )
                self._val_dataset = DummyVLADataset(
                    num_samples=self.cfg.data.get("num_samples", 1000) // 10,
                    image_size=self.cfg.data.get("image_size", 224),
                    action_dim=self.cfg.action.action_dim,
                    seed=43,
                )

    def train_dataloader(self):
        from vla.data import collate_vla_batch
        return torch.utils.data.DataLoader(
            self._train_dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            num_workers=self.cfg.data.get("num_workers", 4),
            pin_memory=True,
            collate_fn=collate_vla_batch,
        )

    def val_dataloader(self):
        from vla.data import collate_vla_batch
        return torch.utils.data.DataLoader(
            self._val_dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.get("num_workers", 4),
            pin_memory=True,
            collate_fn=collate_vla_batch,
        )
```

### Step 2: Implement callbacks.py (45 min)
```python
"""Custom Lightning callbacks for VLA training."""
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import torch
import wandb
from typing import Any, Optional


class WandBSampleLogger(Callback):
    """Log sample predictions to WandB."""

    def __init__(self, log_every_n_epochs: int = 5, num_samples: int = 4):
        self.log_every_n_epochs = log_every_n_epochs
        self.num_samples = num_samples

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ):
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return

        if not hasattr(trainer, "val_dataloaders") or trainer.val_dataloaders is None:
            return

        # Get sample batch
        val_loader = trainer.val_dataloaders
        batch = next(iter(val_loader))
        batch = {k: v[:self.num_samples] if torch.is_tensor(v) else v[:self.num_samples]
                 for k, v in batch.items()}

        # Move to device
        batch = pl_module.transfer_batch_to_device(batch, pl_module.device, 0)

        # Get predictions
        pl_module.eval()
        with torch.no_grad():
            output = pl_module(batch)

        # Log to WandB
        if wandb.run is not None:
            # Create table
            columns = ["instruction", "target_action", "pred_action", "mae"]
            data = []
            for i in range(self.num_samples):
                target = batch["actions"][i].cpu().numpy()
                pred = output["actions"][i].cpu().numpy()
                mae = abs(target - pred).mean()
                data.append([
                    batch["texts"][i],
                    str(target.round(3).tolist()),
                    str(pred.round(3).tolist()),
                    round(mae, 4),
                ])
            table = wandb.Table(columns=columns, data=data)
            wandb.log({"val/samples": table})


class GradientMonitor(Callback):
    """Monitor gradient statistics during training."""

    def __init__(self, log_every_n_steps: int = 100):
        self.log_every_n_steps = log_every_n_steps

    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer: torch.optim.Optimizer,
    ):
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        grad_norms = {}
        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                grad_norms[f"grad_norm/{name}"] = param.grad.norm().item()

        if grad_norms:
            total_norm = sum(v ** 2 for v in grad_norms.values()) ** 0.5
            pl_module.log("grad_norm/total", total_norm)


class EarlyStoppingWithWarmup(pl.callbacks.EarlyStopping):
    """Early stopping that waits for warmup epochs."""

    def __init__(self, warmup_epochs: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.warmup_epochs = warmup_epochs

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if trainer.current_epoch < self.warmup_epochs:
            return
        super().on_validation_end(trainer, pl_module)
```

### Step 3: Implement trainer.py (30 min)
```python
"""Trainer factory for VLA training."""
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    RichProgressBar,
)
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from omegaconf import DictConfig
from typing import Optional, List

from .callbacks import WandBSampleLogger, GradientMonitor, EarlyStoppingWithWarmup


def create_trainer(
    config: DictConfig,
    callbacks: Optional[List[pl.callbacks.Callback]] = None,
) -> pl.Trainer:
    """Create Lightning Trainer from config.

    Args:
        config: Hydra config with train settings
        callbacks: Optional additional callbacks
    """
    train_cfg = config.train

    # Default callbacks
    default_callbacks = [
        ModelCheckpoint(
            dirpath=f"{config.output_dir}/checkpoints",
            filename="{epoch}-{val/loss:.4f}",
            monitor=train_cfg.checkpoint.get("monitor", "val/loss"),
            mode=train_cfg.checkpoint.get("mode", "min"),
            save_top_k=train_cfg.checkpoint.get("save_top_k", 3),
            save_last=train_cfg.checkpoint.get("save_last", True),
        ),
        LearningRateMonitor(logging_interval="step"),
        RichProgressBar(),
    ]

    # Optional callbacks
    if train_cfg.get("early_stopping"):
        default_callbacks.append(
            EarlyStoppingWithWarmup(
                monitor="val/loss",
                patience=train_cfg.early_stopping.patience,
                min_delta=train_cfg.early_stopping.min_delta,
                warmup_epochs=5,
            )
        )

    # WandB sample logging
    if config.get("logging", {}).get("wandb"):
        default_callbacks.append(WandBSampleLogger())

    if callbacks:
        default_callbacks.extend(callbacks)

    # Logger
    if config.get("logging", {}).get("wandb"):
        logger = WandbLogger(
            project=config.logging.wandb.project,
            name=config.logging.wandb.get("name"),
            tags=config.logging.wandb.get("tags", []),
            save_dir=config.output_dir,
        )
    else:
        logger = TensorBoardLogger(save_dir=config.output_dir)

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=train_cfg.max_epochs,
        accelerator="auto",
        devices="auto",
        strategy=train_cfg.get("strategy", "auto"),
        precision=train_cfg.get("precision", "32"),
        gradient_clip_val=train_cfg.get("gradient_clip_val", 1.0),
        accumulate_grad_batches=train_cfg.get("gradient_accumulation_steps", 1),
        val_check_interval=train_cfg.logging.get("val_check_interval", 1.0),
        log_every_n_steps=train_cfg.logging.get("log_every_n_steps", 50),
        callbacks=default_callbacks,
        logger=logger,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    return trainer
```

### Step 4: Create scripts/train.py (45 min)
```python
#!/usr/bin/env python
"""VLA training script with Hydra configuration."""
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import torch
import wandb

from vla.training import VLALightningModule, VLADataModule, create_trainer
from vla.utils.hydra_utils import print_config, validate_config


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> float:
    """Train VLA model.

    Args:
        cfg: Hydra configuration
    Returns:
        Best validation loss
    """
    # Print config
    print_config(cfg)
    validate_config(cfg)

    # Set seed
    pl.seed_everything(cfg.seed, workers=True)

    # Create data module
    data_module = VLADataModule(cfg)

    # Create model
    model = VLALightningModule(cfg)

    # Create trainer
    trainer = create_trainer(cfg)

    # Train
    trainer.fit(model, datamodule=data_module)

    # Test if test set available
    if data_module._test_dataset is not None:
        trainer.test(model, datamodule=data_module)

    # Return best val loss for hyperparameter optimization
    best_loss = trainer.callback_metrics.get("val/loss", float("inf"))
    if isinstance(best_loss, torch.Tensor):
        best_loss = best_loss.item()

    # Finish WandB run
    if wandb.run is not None:
        wandb.finish()

    return best_loss


if __name__ == "__main__":
    main()
```

### Step 5: Create __init__.py (10 min)
```python
"""Training infrastructure for VLA models."""
from .lightning_module import VLALightningModule, VLADataModule
from .callbacks import WandBSampleLogger, GradientMonitor, EarlyStoppingWithWarmup
from .trainer import create_trainer

__all__ = [
    "VLALightningModule",
    "VLADataModule",
    "create_trainer",
    "WandBSampleLogger",
    "GradientMonitor",
    "EarlyStoppingWithWarmup",
]
```

### Step 6: Write tests (45 min)
```python
"""Tests for training infrastructure."""
import pytest
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from vla.training import VLALightningModule, VLADataModule
from vla.data import DummyVLADataset


@pytest.fixture
def mock_config():
    return OmegaConf.create({
        "seed": 42,
        "output_dir": "/tmp/vla_test",
        "model": {
            "freeze_vision": True,
            "freeze_language": True,
        },
        "vision": {
            "name": "timm_vit",
            "model_name": "vit_tiny_patch16_224",
            "pretrained": False,
            "frozen": True,
            "output_mode": "spatial",
            "proj_dim": None,
        },
        "language": {
            "name": "gpt2",
            "model_name": "gpt2",
            "frozen": True,
            "output_mode": "mean",
            "max_length": 32,
            "proj_dim": None,
        },
        "fusion": {
            "name": "perceiver_resampler",
            "dim": 256,
            "num_latents": 16,
            "num_layers": 1,
            "num_heads": 4,
        },
        "action": {
            "name": "discrete_action",
            "action_dim": 7,
            "num_bins": 256,
            "hidden_dim": None,
        },
        "train": {
            "max_epochs": 2,
            "batch_size": 4,
            "gradient_clip_val": 1.0,
            "gradient_accumulation_steps": 1,
            "checkpoint": {"save_top_k": 1, "monitor": "val/loss", "mode": "min", "save_last": True},
            "logging": {"log_every_n_steps": 1, "val_check_interval": 1.0},
        },
        "data": {
            "name": "dummy",
            "num_samples": 100,
            "image_size": 224,
            "num_workers": 0,
        },
    })


class TestVLALightningModule:
    def test_forward(self, mock_config):
        module = VLALightningModule(mock_config)
        batch = {
            "images": torch.randn(2, 3, 224, 224),
            "texts": ["test instruction", "another test"],
            "actions": torch.rand(2, 7) * 2 - 1,
        }
        output = module(batch)
        assert "actions" in output
        assert "loss" in output

    def test_training_step(self, mock_config):
        module = VLALightningModule(mock_config)
        batch = {
            "images": torch.randn(2, 3, 224, 224),
            "texts": ["test", "test2"],
            "actions": torch.rand(2, 7) * 2 - 1,
        }
        loss = module.training_step(batch, 0)
        assert loss.ndim == 0

    def test_configure_optimizers(self, mock_config):
        module = VLALightningModule(mock_config)
        optimizer = module.configure_optimizers()
        assert isinstance(optimizer, torch.optim.Optimizer)


class TestVLADataModule:
    def test_setup(self, mock_config):
        dm = VLADataModule(mock_config)
        dm.setup()
        assert dm._train_dataset is not None
        assert dm._val_dataset is not None

    def test_train_dataloader(self, mock_config):
        dm = VLADataModule(mock_config)
        dm.setup()
        loader = dm.train_dataloader()
        batch = next(iter(loader))
        assert "images" in batch
        assert "texts" in batch
        assert "actions" in batch


class TestTrainingLoop:
    def test_fast_dev_run(self, mock_config, tmp_path):
        mock_config.output_dir = str(tmp_path)

        module = VLALightningModule(mock_config)
        dm = VLADataModule(mock_config)

        trainer = pl.Trainer(
            max_epochs=1,
            fast_dev_run=True,
            accelerator="cpu",
            enable_progress_bar=False,
            logger=False,
        )
        trainer.fit(module, datamodule=dm)
```

## Todo List
- [ ] Implement VLALightningModule
- [ ] Implement VLADataModule
- [ ] Implement custom callbacks
- [ ] Implement trainer factory
- [ ] Create train.py entry script
- [ ] Write training tests
- [ ] Test with real model (fast_dev_run)
- [ ] Test multi-GPU (if available)

## Success Criteria
1. Training loop runs without errors
2. Validation metrics logged correctly
3. Checkpoints saved according to config
4. WandB logging works (when enabled)
5. All tests pass

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| OOM during training | High | Gradient checkpointing, batch size |
| WandB rate limits | Low | Batch logging, reduce frequency |
| Checkpoint corruption | Medium | save_last=True as backup |

## Security Considerations
- WandB API key from environment variable
- No secrets in config or logs

## Next Steps
- Phase 12: Testing suite
