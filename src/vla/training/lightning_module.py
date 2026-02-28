"""PyTorch Lightning training module for VLA models.

Wraps VLAModel in a LightningModule to enable PL training loop,
automatic logging, and optimizer/scheduler configuration.

Example:
    >>> from vla.training import VLALightningModule
    >>> from vla.models import VLAConfig
    >>> module = VLALightningModule(model_cfg=VLAConfig())
    >>> # Use with pytorch_lightning.Trainer
"""

from typing import Any, Dict, List, Union

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from vla.models import VLAModel
from vla.models.vla_configs import VLAConfig
from vla.utils import setup_logger

logger = setup_logger(__name__)


class VLALightningModule(pl.LightningModule):
    """PyTorch Lightning module for VLA model training.

    Wraps VLAModel with PL training/validation steps, optimizer setup,
    and cosine LR scheduling with linear warmup.

    Only fusion + action head parameters are passed to the optimizer
    (backbones are frozen per VLAModel config).

    Args:
        model_cfg: VLAConfig dataclass, plain dict, or Hydra DictConfig
        learning_rate: Peak learning rate for AdamW optimizer
        weight_decay: L2 regularization weight
        warmup_steps: Number of linear LR warmup steps

    Example:
        >>> cfg = VLAConfig()
        >>> module = VLALightningModule(cfg, learning_rate=3e-4)
        >>> # Training step example
        >>> batch = {"images": imgs, "texts": ["pick cube"], "actions": acts}
        >>> loss = module.training_step(batch, 0)
    """

    def __init__(
        self,
        model_cfg: Union[VLAConfig, dict, DictConfig],
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        warmup_steps: int = 1000,
    ):
        super().__init__()

        # Save hyperparameters (excludes model_cfg to avoid serialization issues)
        self.save_hyperparameters(ignore=["model_cfg"])

        # Build VLAModel from config — handles dict, DictConfig, and VLAConfig
        if isinstance(model_cfg, DictConfig):
            config = VLAConfig.from_hydra(model_cfg)
        elif isinstance(model_cfg, dict):
            config = VLAConfig.from_dict(model_cfg)
        else:
            config = model_cfg

        self.model = VLAModel(config)
        logger.info(
            f"VLALightningModule initialized: lr={learning_rate}, "
            f"weight_decay={weight_decay}, warmup_steps={warmup_steps}"
        )

    def forward(
        self,
        images: torch.Tensor,
        texts: List[str],
        target_actions: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Pass-through to VLAModel.forward().

        Args:
            images: Input images [B, 3, 224, 224]
            texts: List of instruction strings [B]
            target_actions: Ground truth actions [B, action_dim] (training only)

        Returns:
            Dict with "actions" and optionally "loss"
        """
        return self.model(images, texts=texts, target_actions=target_actions)

    def _shared_step(
        self, batch: Dict[str, Any], stage: str
    ) -> torch.Tensor:
        """Shared logic for train/val steps.

        Both steps have identical forward pass; only logging differs
        (on_step=True for train, sync_dist=True for val).

        Args:
            batch: Dict with "images", "texts", "actions" keys
            stage: "train" or "val" for metric naming

        Returns:
            Scalar loss tensor
        """
        images: torch.Tensor = batch["images"]
        texts: List[str] = batch["texts"]
        target_actions: torch.Tensor = batch["actions"]

        output = self.model(images, texts=texts, target_actions=target_actions)
        loss: torch.Tensor = output["loss"]

        # Log with different settings per stage
        is_train = stage == "train"
        self.log(
            f"{stage}/loss",
            loss,
            prog_bar=True,
            on_step=is_train,
            on_epoch=True,
            sync_dist=not is_train,  # sync_dist for distributed val
        )
        return loss

    def training_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> torch.Tensor:
        """Compute loss on a training batch.

        Args:
            batch: Dict with "images", "texts", "actions"
            batch_idx: Index of current batch (unused, required by PL)

        Returns:
            Scalar loss for backpropagation
        """
        return self._shared_step(batch, "train")

    def validation_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> torch.Tensor:
        """Compute loss on a validation batch.

        Args:
            batch: Dict with "images", "texts", "actions"
            batch_idx: Index of current batch (unused, required by PL)

        Returns:
            Scalar validation loss
        """
        return self._shared_step(batch, "val")

    def configure_optimizers(self) -> Dict[str, Any]:
        """Set up AdamW optimizer with cosine LR schedule + warmup.

        Only trainable params (fusion + action head) are passed to AdamW.
        Frozen backbone params have requires_grad=False so they are excluded
        via model.get_trainable_params().

        Returns:
            Dict with "optimizer" and "lr_scheduler" for PL
        """
        from transformers import get_cosine_schedule_with_warmup

        trainable_params = self.model.get_trainable_params()
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        # Total steps: use PL's estimate when available (requires trainer attached).
        # Fall back to a large number so warmup still fires correctly.
        # We check self.trainer explicitly rather than catching a broad Exception
        # so real errors (e.g. misconfigured datamodule) still propagate.
        if self.trainer is not None:
            total_steps = int(self.trainer.estimated_stepping_batches)
        else:
            total_steps = 100_000
            logger.warning(
                "Trainer not attached; could not estimate total training steps. "
                f"Defaulting to {total_steps} for LR schedule."
            )

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",   # update every step, not epoch
                "frequency": 1,
            },
        }
