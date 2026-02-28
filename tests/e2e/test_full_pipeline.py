"""End-to-end smoke test for PyTorch Lightning Trainer integration.

Exercises the full PL lifecycle (fit + test) with fast_dev_run=True,
which runs exactly 1 train + 1 val + 1 test batch. Catches hook/callback
integration bugs that raw PyTorch tests cannot detect.

Run with:
    pytest tests/e2e/ -v
"""

import pytorch_lightning as pl

from vla.data import VLADataModule
from vla.models.vla_configs import (
    ActionConfig,
    FusionConfig,
    LanguageConfig,
    VisionConfig,
    VLAConfig,
)
from vla.training import VLALightningModule


def _tiny_config() -> VLAConfig:
    """Minimal VLAConfig: no pretrained weights, tiny architecture for speed."""
    return VLAConfig(
        vision=VisionConfig(
            name="timm_vit",
            model_name="vit_tiny_patch16_224",
            pretrained=False,
            frozen=True,
        ),
        language=LanguageConfig(
            name="gpt2",
            model_name="gpt2",
            frozen=True,
        ),
        fusion=FusionConfig(num_latents=8, num_layers=1, num_heads=2),
        action=ActionConfig(action_dim=7, num_bins=16),
        freeze_vision=True,
        freeze_language=True,
    )


class TestLightningTrainerSmoke:
    """Smoke tests for the full PyTorch Lightning Trainer lifecycle.

    These tests are NOT about component correctness (that's in tests/unit/).
    They verify that PL hooks, callbacks, and lifecycle transitions work
    without raising exceptions when all components are wired together.
    """

    def test_trainer_fast_dev_run_completes(self):
        """Trainer.fit() + Trainer.test() must complete without raising.

        fast_dev_run=True limits to 1 batch per phase (train/val/test),
        making this fast enough for CI while exercising all PL hooks.
        Verifies test_step logs test/loss, test/mse, test/mae without error.
        """
        module = VLALightningModule(model_cfg=_tiny_config())
        dm = VLADataModule(
            dataset_type="dummy",
            batch_size=2,
            num_workers=0,  # No multiprocessing in tests (avoids fork issues)
        )

        trainer = pl.Trainer(
            fast_dev_run=True,
            accelerator="cpu",
            logger=False,  # Suppress CSV/WandB output in tests
            enable_checkpointing=False,  # Avoid filesystem side effects
        )

        # fit() exercises training_step + validation_step PL hooks
        trainer.fit(module, datamodule=dm)

        # test() exercises test_step (logs test/loss, test/mse, test/mae)
        trainer.test(module, datamodule=dm)
