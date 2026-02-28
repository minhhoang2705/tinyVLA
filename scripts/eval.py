"""Evaluation entry point for tinyVLA.

Loads a trained checkpoint and runs pl.Trainer.test() over the configured
dataset, printing test/loss, test/mse, test/mae to stdout.

Usage:
    # Evaluate a PL checkpoint (.ckpt):
    python scripts/eval.py +eval.checkpoint=outputs/checkpoints/last.ckpt

    # Evaluate a raw VLAModel checkpoint (.pt):
    python scripts/eval.py +eval.checkpoint=outputs/model.pt data=dummy

Override data / training settings the same way as train.py:
    python scripts/eval.py +eval.checkpoint=last.ckpt data.name=dummy

Note:
    Hydra writes run outputs to outputs/ by default. This is expected behaviour.
"""

from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from vla.data import VLADataModule
from vla.models import VLAModel
from vla.models.vla_configs import VLAConfig
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
    if dataset_type == "lerobot" and "repo_id" in data_cfg:
        kwargs["repo_id"] = data_cfg.repo_id
    return VLADataModule(**kwargs)


def _load_module(checkpoint_path: Path, cfg: DictConfig) -> VLALightningModule:
    """Load VLALightningModule from a .ckpt (PL) or .pt (VLAModel) file.

    Args:
        checkpoint_path: Path to checkpoint file
        cfg: Full Hydra config (used to reconstruct model_cfg for .ckpt files)

    Returns:
        VLALightningModule ready for evaluation

    Raises:
        FileNotFoundError: If checkpoint does not exist
        ValueError: If file extension is unrecognised
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if checkpoint_path.suffix == ".ckpt":
        # PL checkpoint: model_cfg is excluded from saved hparams
        # (VLALightningModule uses save_hyperparameters(ignore=["model_cfg"])),
        # so we must supply it explicitly from the active Hydra config.
        logger.info(f"Loading PL checkpoint: {checkpoint_path}")
        model_cfg = VLAConfig.from_hydra(cfg.model)
        return VLALightningModule.load_from_checkpoint(str(checkpoint_path), model_cfg=model_cfg)

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
        raise ValueError("Checkpoint path is required. Pass it with: +eval.checkpoint=path/to/ckpt")

    module = _load_module(Path(checkpoint_path), cfg)
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
