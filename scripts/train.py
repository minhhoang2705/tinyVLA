"""Main training entry point for tinyVLA.

Uses Hydra for config composition. Override any value from CLI:
    python scripts/train.py vision=dinov2 train.learning_rate=3e-4
    python scripts/train.py data.name=lerobot data.repo_id=lerobot/pusht
    python scripts/train.py --multirun train.learning_rate=1e-4,3e-4

Configs live in configs/ (see configs/config.yaml for defaults).
"""

from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from vla.utils import setup_logger

logger = setup_logger(__name__)


def _build_datamodule(cfg: DictConfig):
    """Instantiate VLADataModule from Hydra data config.

    Maps cfg.data.name to dataset_type and passes through common params.

    Args:
        cfg: Full Hydra config (uses cfg.data sub-config)

    Returns:
        Configured VLADataModule instance
    """
    from vla.data import VLADataModule

    data_cfg = cfg.data

    # Map config name to dataset_type string expected by VLADataModule
    name_to_type = {
        "dummy": "dummy",
        "lerobot": "lerobot",
    }
    dataset_type = name_to_type.get(data_cfg.get("name", "dummy"), "dummy")

    kwargs = dict(
        dataset_type=dataset_type,
        batch_size=cfg.train.get("batch_size", 32),
        num_workers=cfg.train.get("num_workers", 4),
        total_samples=data_cfg.get("num_samples", 1000),
    )

    # Pass LeRobot-specific params if present
    if dataset_type == "lerobot":
        if "repo_id" in data_cfg:
            kwargs["repo_id"] = data_cfg.repo_id
        if "image_key" in data_cfg:
            kwargs["image_key"] = data_cfg.image_key

    return VLADataModule(**kwargs)


def _build_callbacks(cfg: DictConfig, output_dir: Path) -> list:
    """Build PL callbacks from config.

    Always adds ModelCheckpoint. Adds EarlyStopping if configured.

    Args:
        cfg: Full Hydra config
        output_dir: Directory for saving checkpoints

    Returns:
        List of pytorch_lightning.Callback instances
    """
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

    train_cfg = cfg.train
    checkpoint_cfg = train_cfg.get("checkpoint", {})

    callbacks = [
        ModelCheckpoint(
            dirpath=str(output_dir / "checkpoints"),
            filename="vla-{epoch:03d}-{val/loss:.4f}",
            monitor=checkpoint_cfg.get("monitor", "val/loss"),
            mode=checkpoint_cfg.get("mode", "min"),
            save_top_k=checkpoint_cfg.get("save_top_k", 3),
            save_last=checkpoint_cfg.get("save_last", True),
            verbose=True,
        )
    ]

    early_stop_cfg = train_cfg.get("early_stopping", {})
    if early_stop_cfg:
        callbacks.append(
            EarlyStopping(
                monitor=checkpoint_cfg.get("monitor", "val/loss"),
                patience=early_stop_cfg.get("patience", 10),
                min_delta=early_stop_cfg.get("min_delta", 0.001),
                mode=checkpoint_cfg.get("mode", "min"),
                verbose=True,
            )
        )

    return callbacks


def _build_loggers(cfg: DictConfig) -> list:
    """Build PL loggers from config.

    Adds WandbLogger if train.use_wandb=true, else CSV logger only.

    Args:
        cfg: Full Hydra config

    Returns:
        List of pytorch_lightning logger instances
    """
    from pytorch_lightning.loggers import CSVLogger

    loggers = [CSVLogger(save_dir=".", name="csv_logs")]

    if cfg.train.get("use_wandb", False):
        from pytorch_lightning.loggers import WandbLogger

        loggers.append(
            WandbLogger(
                project=cfg.project.get("name", "tinyVLA"),
                name=cfg.train.get("name", "default"),
            )
        )
        logger.info("WandB logging enabled")

    return loggers


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run VLA training with the given Hydra config.

    Sets global seed, builds datamodule + lightning module + trainer,
    then calls trainer.fit().

    Args:
        cfg: Composed Hydra config (see configs/config.yaml for defaults)
    """
    # Log resolved config for reproducibility
    logger.info("Training config:\n" + OmegaConf.to_yaml(cfg, resolve=True))

    # Reproducibility — sets Python, numpy, torch seeds
    pl.seed_everything(cfg.get("seed", 42), workers=True)

    output_dir = Path(cfg.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Data ---
    datamodule = _build_datamodule(cfg)

    # --- Model ---
    from vla.training import VLALightningModule

    train_cfg = cfg.train
    lightning_module = VLALightningModule(
        model_cfg=cfg.model,
        learning_rate=train_cfg.get("learning_rate", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
        warmup_steps=train_cfg.get("warmup_steps", 1000),
    )

    # --- Trainer ---
    trainer = pl.Trainer(
        max_epochs=train_cfg.get("epochs", train_cfg.get("max_epochs", 100)),
        precision=train_cfg.get("precision", "bf16-mixed"),
        devices=train_cfg.get("devices", "auto"),
        gradient_clip_val=train_cfg.get("gradient_clip_val", 1.0),
        accumulate_grad_batches=train_cfg.get("gradient_accumulation_steps", 1),
        val_check_interval=train_cfg.get("val_check_interval", 0.25),
        log_every_n_steps=train_cfg.get("log_every_n_steps", 50),
        callbacks=_build_callbacks(cfg, output_dir),
        logger=_build_loggers(cfg),
        deterministic=False,  # set True for full reproducibility at cost of speed
    )

    logger.info("Starting training...")
    trainer.fit(lightning_module, datamodule=datamodule)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
