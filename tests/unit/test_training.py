"""Unit tests for VLALightningModule.

Tests cover:
- Module initialisation and backbone freezing
- configure_optimizers returns only trainable params
- training_step computes and logs loss correctly
"""

from unittest.mock import MagicMock

import torch

from vla.models.vla_configs import (
    ActionConfig,
    FusionConfig,
    LanguageConfig,
    VisionConfig,
    VLAConfig,
)
from vla.training import VLALightningModule

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_config() -> VLAConfig:
    """Return a minimal VLAConfig that loads quickly (no pretrained weights)."""
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestVLALightningModuleInit:
    """Verify module builds correctly and applies backbone freezing."""

    def test_lightning_module_init(self):
        """Module initialises VLAModel without raising."""
        module = VLALightningModule(model_cfg=_tiny_config())
        assert module.model is not None

    def test_vision_backbone_is_frozen(self):
        """Vision backbone must have no trainable parameters after init."""
        module = VLALightningModule(model_cfg=_tiny_config())
        vision_params = list(module.model.vision.parameters())
        assert len(vision_params) > 0, "Vision backbone has no parameters"
        # All vision params should be frozen
        assert all(
            not p.requires_grad for p in vision_params
        ), "Vision backbone has trainable parameters — should be frozen"

    def test_language_backbone_is_frozen(self):
        """Language backbone must have no trainable parameters after init."""
        module = VLALightningModule(model_cfg=_tiny_config())
        lang_params = list(module.model.language.parameters())
        assert len(lang_params) > 0, "Language backbone has no parameters"
        assert all(
            not p.requires_grad for p in lang_params
        ), "Language backbone has trainable parameters — should be frozen"

    def test_fusion_and_action_head_are_trainable(self):
        """Fusion module and action head must remain trainable after init."""
        module = VLALightningModule(model_cfg=_tiny_config())
        fusion_params = list(module.model.fusion.parameters())
        action_params = list(module.model.action_head.parameters())
        assert any(p.requires_grad for p in fusion_params), "Fusion params should be trainable"
        assert any(p.requires_grad for p in action_params), "Action head params should be trainable"

    def test_hyperparameters_stored(self):
        """Hyperparameters should be accessible after init."""
        module = VLALightningModule(
            model_cfg=_tiny_config(),
            learning_rate=3e-4,
            weight_decay=1e-5,
            warmup_steps=500,
        )
        assert module.hparams.learning_rate == 3e-4
        assert module.hparams.weight_decay == 1e-5
        assert module.hparams.warmup_steps == 500


class TestConfigureOptimizers:
    """Verify optimizer only receives trainable parameters."""

    def test_configure_optimizers_returns_adamw(self):
        """Optimizer should be AdamW."""
        module = VLALightningModule(model_cfg=_tiny_config())

        # Attach a mock trainer so estimated_stepping_batches is available
        mock_trainer = MagicMock()
        mock_trainer.estimated_stepping_batches = 10_000
        module.trainer = mock_trainer

        result = module.configure_optimizers()
        optimizer = result["optimizer"]
        assert isinstance(optimizer, torch.optim.AdamW)

    def test_configure_optimizers_only_trainable_params(self):
        """Optimizer param groups must not contain frozen parameters."""
        module = VLALightningModule(model_cfg=_tiny_config())

        mock_trainer = MagicMock()
        mock_trainer.estimated_stepping_batches = 10_000
        module.trainer = mock_trainer

        result = module.configure_optimizers()
        optimizer = result["optimizer"]

        # Collect all param tensors in the optimizer
        opt_param_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}

        # Check no frozen param leaked into the optimizer
        for param in module.model.parameters():
            if not param.requires_grad:
                assert (
                    id(param) not in opt_param_ids
                ), "Frozen parameter found in optimizer — only trainable params should be optimized"

    def test_configure_optimizers_has_lr_scheduler(self):
        """Result dict should include a cosine LR scheduler."""
        module = VLALightningModule(model_cfg=_tiny_config())

        mock_trainer = MagicMock()
        mock_trainer.estimated_stepping_batches = 10_000
        module.trainer = mock_trainer

        result = module.configure_optimizers()
        assert "lr_scheduler" in result
        assert result["lr_scheduler"]["interval"] == "step"


class TestTrainingStep:
    """Verify training_step computes loss and logs it."""

    def _make_batch(self, batch_size: int = 2) -> dict:
        return {
            "images": torch.rand(batch_size, 3, 224, 224),
            "texts": ["pick up cube", "move to target"][:batch_size],
            "actions": torch.rand(batch_size, 7) * 2 - 1,  # range [-1, 1]
        }

    def test_training_step_returns_loss_tensor(self):
        """training_step must return a scalar loss tensor."""
        module = VLALightningModule(model_cfg=_tiny_config())
        module.eval()  # keep dropout/BN stable for deterministic test

        batch = self._make_batch()

        # Mock self.log to avoid needing a full Trainer
        module.log = MagicMock()

        loss = module.training_step(batch, batch_idx=0)
        assert isinstance(loss, torch.Tensor), "Loss must be a tensor"
        assert loss.ndim == 0, "Loss must be scalar (0-dimensional)"
        assert loss.item() > 0, "Loss should be positive"

    def test_training_step_calls_log(self):
        """training_step must log 'train/loss'."""
        module = VLALightningModule(model_cfg=_tiny_config())
        module.eval()

        batch = self._make_batch()
        module.log = MagicMock()

        module.training_step(batch, batch_idx=0)

        # Verify log was called with the expected metric name
        logged_keys = [call.args[0] for call in module.log.call_args_list]
        assert (
            "train/loss" in logged_keys
        ), f"Expected 'train/loss' to be logged, got: {logged_keys}"

    def test_validation_step_logs_val_loss(self):
        """validation_step must log 'val/loss' with sync_dist=True."""
        module = VLALightningModule(model_cfg=_tiny_config())
        module.eval()

        batch = self._make_batch()
        module.log = MagicMock()

        module.validation_step(batch, batch_idx=0)

        logged_keys = [call.args[0] for call in module.log.call_args_list]
        assert "val/loss" in logged_keys, f"Expected 'val/loss' to be logged, got: {logged_keys}"

        # sync_dist should be True for val
        val_call = next(c for c in module.log.call_args_list if c.args[0] == "val/loss")
        assert (
            val_call.kwargs.get("sync_dist") is True
        ), "val/loss should be logged with sync_dist=True for distributed training"

    def test_test_step_returns_loss_tensor(self):
        """test_step must return a scalar loss tensor."""
        module = VLALightningModule(model_cfg=_tiny_config())
        module.eval()

        batch = self._make_batch()
        module.log = MagicMock()

        loss = module.test_step(batch, batch_idx=0)
        assert isinstance(loss, torch.Tensor), "test_step loss must be a tensor"
        assert loss.ndim == 0, "test_step loss must be scalar"

    def test_test_step_logs_loss_mse_mae(self):
        """test_step must log test/loss, test/mse, test/mae."""
        module = VLALightningModule(model_cfg=_tiny_config())
        module.eval()

        batch = self._make_batch()
        module.log = MagicMock()

        module.test_step(batch, batch_idx=0)

        logged_keys = [call.args[0] for call in module.log.call_args_list]
        for expected_key in ("test/loss", "test/mse", "test/mae"):
            assert (
                expected_key in logged_keys
            ), f"Expected '{expected_key}' to be logged, got: {logged_keys}"

    def test_forward_passthrough_returns_dict(self):
        """forward() must return dict with 'actions' key."""
        module = VLALightningModule(model_cfg=_tiny_config())
        module.eval()

        images = torch.rand(2, 3, 224, 224)
        texts = ["pick up cube", "move to target"]

        with torch.no_grad():
            output = module(images, texts)

        assert "actions" in output, "forward() output must contain 'actions'"
        assert output["actions"].shape == (
            2,
            7,
        ), f"Expected actions shape (2, 7), got {output['actions'].shape}"
