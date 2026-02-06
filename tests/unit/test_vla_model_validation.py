"""Unit tests for VLA model input validation integration.

Tests that VLAModel.forward() properly validates inputs and raises
clear error messages for invalid inputs.
"""

import pytest
import torch

from vla.models import VLAConfig, VLAModel, TemporalVLAModel


class TestVLAModelValidation:
    """Tests for VLAModel input validation."""

    @pytest.fixture
    def model(self):
        """Create a VLA model for testing."""
        config = VLAConfig()
        return VLAModel(config)

    @pytest.fixture
    def valid_inputs(self):
        """Create valid inputs for testing."""
        return {
            "images": torch.rand(2, 3, 224, 224),
            "texts": ["pick cube", "place cup"],
            "target_actions": torch.rand(2, 7) * 2 - 1,
        }

    def test_valid_forward_pass(self, model, valid_inputs):
        """Test forward pass with valid inputs."""
        output = model(**valid_inputs)
        assert "actions" in output
        assert "loss" in output
        assert output["actions"].shape == (2, 7)

    def test_invalid_image_shape(self, model, valid_inputs):
        """Test validation fails for invalid image shape."""
        valid_inputs["images"] = torch.rand(2, 4, 224, 224)  # Wrong channels
        with pytest.raises(ValueError, match="must have 3 channels"):
            model(**valid_inputs)

    def test_invalid_image_values(self, model, valid_inputs):
        """Test validation fails for image values outside [0, 1]."""
        valid_inputs["images"] = torch.randn(2, 3, 224, 224)  # Can be negative
        with pytest.raises(ValueError, match="values must be in range"):
            model(**valid_inputs)

    def test_batch_size_mismatch_texts(self, model, valid_inputs):
        """Test validation fails when batch sizes don't match."""
        valid_inputs["texts"] = ["pick cube"]  # Only 1 text but 2 images
        with pytest.raises(ValueError, match="Batch size mismatch"):
            model(**valid_inputs)

    def test_batch_size_mismatch_actions(self, model, valid_inputs):
        """Test validation fails when action batch size doesn't match."""
        valid_inputs["target_actions"] = torch.rand(1, 7)  # Only 1 action
        with pytest.raises(ValueError, match="Batch size mismatch"):
            model(**valid_inputs)

    def test_invalid_action_dimension(self, model, valid_inputs):
        """Test validation fails for wrong action dimension."""
        valid_inputs["target_actions"] = torch.rand(2, 10)  # Wrong action_dim
        with pytest.raises(ValueError, match="must have action_dim=7"):
            model(**valid_inputs)

    def test_invalid_action_values(self, model, valid_inputs):
        """Test validation fails for actions outside [-1, 1]."""
        valid_inputs["target_actions"] = torch.rand(2, 7) * 4 - 2  # Range [-2, 2]
        with pytest.raises(ValueError, match="values must be in range"):
            model(**valid_inputs)

    def test_both_texts_and_input_ids(self, model, valid_inputs):
        """Test validation fails when both texts and input_ids provided."""
        valid_inputs["input_ids"] = torch.randint(0, 1000, (2, 10))
        with pytest.raises(ValueError, match="Cannot provide both"):
            model(**valid_inputs)

    def test_neither_texts_nor_input_ids(self, model):
        """Test validation fails when neither texts nor input_ids provided."""
        images = torch.rand(2, 3, 224, 224)
        with pytest.raises(ValueError, match="Must provide either"):
            model(images)

    def test_inference_mode_validation(self, model):
        """Test validation works in inference mode (no target_actions)."""
        images = torch.rand(2, 3, 224, 224)
        texts = ["pick cube", "place cup"]

        model.eval()
        actions = model.predict(images, texts)
        assert actions.shape == (2, 7)

    def test_inference_mode_invalid_inputs(self, model):
        """Test validation catches errors in inference mode."""
        images = torch.randn(2, 3, 224, 224)  # Invalid values
        texts = ["pick cube", "place cup"]

        with pytest.raises(ValueError, match="values must be in range"):
            model.predict(images, texts)


class TestTemporalVLAModelValidation:
    """Tests for TemporalVLAModel input validation."""

    @pytest.fixture
    def model(self):
        """Create a temporal VLA model for testing."""
        config = VLAConfig()
        return TemporalVLAModel(config, num_frames=6)

    @pytest.fixture
    def valid_inputs(self):
        """Create valid temporal inputs for testing."""
        return {
            "image_sequence": [torch.rand(2, 3, 224, 224) for _ in range(6)],
            "texts": ["track ball", "catch it"],
            "target_actions": torch.rand(2, 7) * 2 - 1,
        }

    def test_valid_temporal_forward(self, model, valid_inputs):
        """Test forward pass with valid temporal inputs."""
        output = model(**valid_inputs)
        assert "actions" in output
        assert "loss" in output
        assert output["actions"].shape == (2, 7)

    def test_wrong_number_of_frames(self, model, valid_inputs):
        """Test validation fails for wrong number of frames."""
        valid_inputs["image_sequence"] = [torch.rand(2, 3, 224, 224) for _ in range(4)]
        with pytest.raises(ValueError, match="Expected 6 frames, got 4"):
            model(**valid_inputs)

    def test_frame_shape_invalid(self, model, valid_inputs):
        """Test validation fails when a frame has invalid shape."""
        valid_inputs["image_sequence"][3] = torch.rand(2, 4, 224, 224)  # Wrong channels
        with pytest.raises(ValueError, match="Frame 3 validation failed"):
            model(**valid_inputs)

    def test_inconsistent_batch_sizes_across_frames(self, model, valid_inputs):
        """Test validation fails when frames have different batch sizes."""
        valid_inputs["image_sequence"][3] = torch.rand(4, 3, 224, 224)  # Different batch
        with pytest.raises(ValueError, match="same batch size"):
            model(**valid_inputs)

    def test_temporal_batch_consistency(self, model, valid_inputs):
        """Test batch size consistency across temporal inputs."""
        valid_inputs["texts"] = ["track ball"]  # Only 1 text but 2 batch in images
        with pytest.raises(ValueError, match="Batch size mismatch"):
            model(**valid_inputs)

    def test_temporal_inference_mode(self, model):
        """Test temporal model inference with validation."""
        frames = [torch.rand(2, 3, 224, 224) for _ in range(6)]
        texts = ["track ball", "catch it"]

        model.eval()
        actions = model.predict(frames, texts)
        assert actions.shape == (2, 7)


class TestValidationErrorMessages:
    """Tests that validation errors have clear, helpful messages."""

    @pytest.fixture
    def model(self):
        """Create a VLA model for testing."""
        config = VLAConfig()
        return VLAModel(config)

    def test_image_shape_error_message_clarity(self, model):
        """Test that image shape errors provide clear guidance."""
        images = torch.rand(2, 4, 224, 224)  # Wrong channels
        texts = ["pick", "place"]

        with pytest.raises(ValueError) as exc_info:
            model(images, texts=texts)

        error_msg = str(exc_info.value)
        assert "3 channels" in error_msg
        assert "got 4" in error_msg

    def test_batch_mismatch_error_message_clarity(self, model):
        """Test that batch mismatch errors are clear."""
        images = torch.rand(4, 3, 224, 224)
        texts = ["pick", "place"]  # Only 2 texts

        with pytest.raises(ValueError) as exc_info:
            model(images, texts=texts)

        error_msg = str(exc_info.value)
        assert "Batch size mismatch" in error_msg
        assert "4 samples" in error_msg
        assert "2 strings" in error_msg

    def test_action_range_error_message_clarity(self, model):
        """Test that action range errors suggest normalization."""
        images = torch.rand(2, 3, 224, 224)
        texts = ["pick", "place"]
        actions = torch.rand(2, 7) * 4 - 2  # Range [-2, 2]

        with pytest.raises(ValueError) as exc_info:
            model(images, texts=texts, target_actions=actions)

        error_msg = str(exc_info.value)
        assert "must be in range [-1.0, 1.0]" in error_msg
        assert "normalized" in error_msg.lower()

    def test_image_value_range_error_suggests_normalization(self, model):
        """Test that image range errors mention normalization."""
        images = torch.randn(2, 3, 224, 224)  # Not normalized
        texts = ["pick", "place"]

        with pytest.raises(ValueError) as exc_info:
            model(images, texts=texts)

        error_msg = str(exc_info.value)
        assert "normalize" in error_msg.lower()
        assert "[0, 1]" in error_msg or "[0.0, 1.0]" in error_msg
