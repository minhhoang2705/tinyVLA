"""End-to-end integration tests for VLA pipeline.

Tests complete workflow from data loading through model forward pass,
loss computation, and optimization steps. Verifies gradient flow and
checkpoint management.
"""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from vla.data import DummyVLADataset, VLADataModule, vla_collate_fn
from vla.models import VLAConfig, VLAModel


class TestEndToEndForwardPass:
    """Test complete forward pass pipeline."""

    @pytest.fixture
    def model(self):
        """Create VLA model for testing."""
        config = VLAConfig()
        return VLAModel(config)

    @pytest.fixture
    def dataloader(self):
        """Create data loader for testing."""
        dataset = DummyVLADataset(num_samples=20)
        return DataLoader(dataset, batch_size=4, collate_fn=vla_collate_fn)

    def test_forward_pass_with_loss(self, model, dataloader):
        """Test forward pass produces valid outputs and loss."""
        batch = next(iter(dataloader))

        output = model(
            images=batch["images"],
            texts=batch["texts"],
            target_actions=batch["actions"],
        )

        # Verify output structure
        assert "actions" in output
        assert "loss" in output

        # Verify shapes
        assert output["actions"].shape == batch["actions"].shape
        assert output["loss"].ndim == 0  # Scalar loss

        # Verify loss is positive and finite
        assert output["loss"].item() >= 0
        assert torch.isfinite(output["loss"])

    def test_inference_mode(self, model, dataloader):
        """Test inference without target actions."""
        batch = next(iter(dataloader))

        model.eval()
        with torch.no_grad():
            output = model(
                images=batch["images"],
                texts=batch["texts"],
            )

        # Should only have actions, no loss
        assert "actions" in output
        assert "loss" not in output
        assert output["actions"].shape == (4, 7)

    def test_predict_method(self, model, dataloader):
        """Test convenience predict method."""
        batch = next(iter(dataloader))

        actions = model.predict(batch["images"], batch["texts"])

        assert actions.shape == (4, 7)
        assert actions.min() >= -1.0
        assert actions.max() <= 1.0
        assert not actions.requires_grad


class TestTrainingLoop:
    """Test training loop with gradient flow."""

    @pytest.fixture
    def model(self):
        """Create VLA model for training."""
        config = VLAConfig()
        return VLAModel(config)

    @pytest.fixture
    def dataloader(self):
        """Create data loader."""
        dataset = DummyVLADataset(num_samples=32)
        return DataLoader(dataset, batch_size=8, collate_fn=vla_collate_fn)

    def test_single_training_step(self, model, dataloader):
        """Test single training step completes successfully."""
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        batch = next(iter(dataloader))

        # Forward pass
        model.train()
        output = model(
            images=batch["images"],
            texts=batch["texts"],
            target_actions=batch["actions"],
        )

        loss = output["loss"]

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Verify gradients were computed
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    def test_gradient_flow_to_trainable_only(self, model, dataloader):
        """Test gradients only flow to trainable parameters."""
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        batch = next(iter(dataloader))

        # Forward + backward
        model.train()
        output = model(
            images=batch["images"],
            texts=batch["texts"],
            target_actions=batch["actions"],
        )

        optimizer.zero_grad()
        output["loss"].backward()

        # Check frozen backbones have no gradients
        for name, param in model.vision.named_parameters():
            if not param.requires_grad:
                assert param.grad is None, f"Gradient computed for frozen vision param: {name}"

        for name, param in model.language.named_parameters():
            if not param.requires_grad:
                assert param.grad is None, f"Gradient computed for frozen language param: {name}"

        # Check trainable modules have gradients
        for name, param in model.fusion.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for fusion param: {name}"

        for name, param in model.action_head.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for action_head param: {name}"

    def test_multi_step_training(self, model, dataloader):
        """Test multiple training steps reduce loss."""
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        losses = []
        model.train()

        for i, batch in enumerate(dataloader):
            if i >= 5:  # Train for 5 steps
                break

            output = model(
                images=batch["images"],
                texts=batch["texts"],
                target_actions=batch["actions"],
            )

            loss = output["loss"]
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Loss should generally decrease (may not be monotonic)
        assert losses[-1] < losses[0], "Loss did not decrease after training"

    def test_no_gradient_in_eval_mode(self, model, dataloader):
        """Test no gradients computed in eval mode."""
        batch = next(iter(dataloader))

        model.eval()
        with torch.no_grad():
            output = model(
                images=batch["images"],
                texts=batch["texts"],
                target_actions=batch["actions"],
            )

        # Even with targets, loss should not require grad in eval mode
        if "loss" in output:
            assert not output["loss"].requires_grad
            assert not output["actions"].requires_grad


class TestCheckpointManagement:
    """Test model checkpoint save and load."""

    @pytest.fixture
    def model(self):
        """Create VLA model."""
        config = VLAConfig()
        return VLAModel(config)

    def test_checkpoint_save_load_roundtrip(self, model):
        """Test saving and loading preserves model state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.pt"

            # Save checkpoint
            model.save_checkpoint(str(checkpoint_path))
            assert checkpoint_path.exists()

            # Load checkpoint
            loaded_model = VLAModel.load_checkpoint(str(checkpoint_path))

            # Verify state dicts match
            original_state = model.state_dict()
            loaded_state = loaded_model.state_dict()

            assert set(original_state.keys()) == set(loaded_state.keys())

            for key in original_state.keys():
                assert torch.allclose(
                    original_state[key], loaded_state[key]
                ), f"State mismatch for {key}"

    def test_checkpoint_preserves_config(self, model):
        """Test config is saved and restored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.pt"

            # Save and load
            model.save_checkpoint(str(checkpoint_path))
            loaded_model = VLAModel.load_checkpoint(str(checkpoint_path))

            # Verify configs match
            assert model.config.vision.model_name == loaded_model.config.vision.model_name
            assert model.config.language.model_name == loaded_model.config.language.model_name
            assert model.config.fusion.num_latents == loaded_model.config.fusion.num_latents
            assert model.config.action.action_dim == loaded_model.config.action.action_dim

    def test_checkpoint_after_training(self, model):
        """Test checkpoint after training preserves learned weights."""
        # Create dummy data
        images = torch.rand(4, 3, 224, 224)
        texts = ["pick", "place", "move", "grasp"]
        actions = torch.rand(4, 7) * 2 - 1

        # Train for a few steps
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()

        for _ in range(3):
            output = model(images, texts=texts, target_actions=actions)
            optimizer.zero_grad()
            output["loss"].backward()
            optimizer.step()

        # Get predictions from trained model
        model.eval()
        with torch.no_grad():
            trained_actions = model.predict(images, texts)

        # Save and load checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "trained_model.pt"
            model.save_checkpoint(str(checkpoint_path))
            loaded_model = VLAModel.load_checkpoint(str(checkpoint_path))

        # Get predictions from loaded model
        loaded_model.eval()
        with torch.no_grad():
            loaded_actions = loaded_model.predict(images, texts)

        # Predictions should match exactly
        assert torch.allclose(trained_actions, loaded_actions)


class TestDataModuleIntegration:
    """Test VLADataModule integration with training."""

    def test_datamodule_with_model(self):
        """Test DataModule provides valid batches for model."""
        datamodule = VLADataModule(
            dataset_type="dummy",
            batch_size=8,
            num_workers=0,
        )
        datamodule.setup("fit")

        model = VLAModel(VLAConfig())
        model.train()

        # Get batch from datamodule
        train_loader = datamodule.train_dataloader()
        batch = next(iter(train_loader))

        # Forward pass should work
        output = model(
            images=batch["images"],
            texts=batch["texts"],
            target_actions=batch["actions"],
        )

        assert "actions" in output
        assert "loss" in output
        assert torch.isfinite(output["loss"])

    def test_full_epoch_training(self):
        """Test training for a full epoch."""
        datamodule = VLADataModule(
            dataset_type="dummy",
            batch_size=16,
            num_workers=0,
        )
        datamodule.setup("fit")

        model = VLAModel(VLAConfig())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        model.train()
        train_loader = datamodule.train_dataloader()

        epoch_losses = []
        for batch in train_loader:
            output = model(
                images=batch["images"],
                texts=batch["texts"],
                target_actions=batch["actions"],
            )

            loss = output["loss"]
            epoch_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Verify we processed batches
        assert len(epoch_losses) > 0
        # Verify all losses are finite
        assert all(torch.isfinite(torch.tensor(loss)) for loss in epoch_losses)


class TestComponentInteractions:
    """Test interactions between model components."""

    @pytest.fixture
    def model(self):
        """Create VLA model."""
        config = VLAConfig()
        return VLAModel(config)

    def test_vision_to_fusion_shape_compatibility(self, model):
        """Test vision features have correct shape for fusion."""
        images = torch.rand(2, 3, 224, 224)

        vision_features = model.vision(images)
        assert vision_features.ndim == 3  # [B, N, D]

        # Fusion should accept vision features
        texts = ["pick cube", "place cup"]
        language_features = model.language(texts=texts)

        fused = model.fusion(vision_features, language_features)
        assert fused.shape[0] == 2  # Batch size preserved
        assert fused.shape[1] == model.config.fusion.num_latents  # Fixed latents

    def test_fusion_to_action_shape_compatibility(self, model):
        """Test fusion output has correct shape for action head."""
        images = torch.rand(2, 3, 224, 224)
        texts = ["pick cube", "place cup"]

        vision_features = model.vision(images)
        language_features = model.language(texts=texts)
        fused = model.fusion(vision_features, language_features)

        # Action head should accept fused features
        actions, logits = model.action_head(fused, return_logits=True)

        assert actions.shape == (2, model.config.action.action_dim)
        assert logits is not None

    def test_loss_computation_matches_action_type(self, model):
        """Test loss computation uses correct method for action head type."""
        images = torch.rand(2, 3, 224, 224)
        texts = ["pick", "place"]
        target_actions = torch.rand(2, 7) * 2 - 1

        output = model(images, texts=texts, target_actions=target_actions)

        # Loss should be computed via ActionHead.compute_loss
        assert "loss" in output
        assert output["loss"].ndim == 0
        assert torch.isfinite(output["loss"])
