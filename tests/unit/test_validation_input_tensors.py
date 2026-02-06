"""Unit tests for input validation utilities.

Tests cover:
- validate_image_tensor: shape, dtype, value range validation
- validate_action_tensor: shape, value range validation
- validate_text_input: mutual exclusivity, type checking
- validate_batch_consistency: batch size matching across modalities
- validate_temporal_inputs: temporal sequence validation
"""

import pytest
import torch

from vla.utils import (
    validate_action_tensor,
    validate_batch_consistency,
    validate_image_tensor,
    validate_temporal_inputs,
    validate_text_input,
)


class TestValidateImageTensor:
    """Tests for validate_image_tensor function."""

    def test_valid_image_tensor(self):
        """Test validation passes for valid image tensor."""
        images = torch.rand(4, 3, 224, 224)  # Valid [B, C, H, W] in [0, 1]
        validate_image_tensor(images)  # Should not raise

    def test_wrong_dtype(self):
        """Test validation fails for wrong dtype."""
        images = torch.randint(0, 255, (4, 3, 224, 224), dtype=torch.uint8)
        with pytest.raises(TypeError, match="must have dtype"):
            validate_image_tensor(images)

    def test_wrong_ndim(self):
        """Test validation fails for wrong number of dimensions."""
        images = torch.rand(4, 224, 224)  # Missing channel dim
        with pytest.raises(ValueError, match="must be 4D tensor"):
            validate_image_tensor(images)

    def test_wrong_channels(self):
        """Test validation fails for wrong number of channels."""
        images = torch.rand(4, 4, 224, 224)  # 4 channels instead of 3
        with pytest.raises(ValueError, match="must have 3 channels"):
            validate_image_tensor(images)

    def test_wrong_spatial_dimensions(self):
        """Test validation fails for wrong spatial size."""
        images = torch.rand(4, 3, 256, 256)  # Wrong H, W
        with pytest.raises(ValueError, match="spatial dimensions must be"):
            validate_image_tensor(images)

    def test_values_out_of_range(self):
        """Test validation fails for values outside [0, 1]."""
        images = torch.randn(4, 3, 224, 224)  # Can have negative values
        with pytest.raises(ValueError, match="values must be in range"):
            validate_image_tensor(images)

    def test_values_below_min(self):
        """Test validation fails when values below minimum."""
        images = torch.rand(4, 3, 224, 224) - 0.5  # Range [-0.5, 0.5]
        with pytest.raises(ValueError, match="values must be in range"):
            validate_image_tensor(images)

    def test_values_above_max(self):
        """Test validation fails when values above maximum."""
        images = torch.rand(4, 3, 224, 224) * 2  # Range [0, 2]
        with pytest.raises(ValueError, match="values must be in range"):
            validate_image_tensor(images)

    def test_custom_expected_shape(self):
        """Test validation with custom expected shape."""
        images = torch.rand(2, 3, 128, 128)
        validate_image_tensor(images, expected_shape=(2, 3, 128, 128))

    def test_custom_value_range(self):
        """Test validation with custom value range."""
        images = torch.randn(4, 3, 224, 224)  # Range approx [-3, 3]
        validate_image_tensor(
            images,
            value_range=(-10.0, 10.0),  # Accept wider range
        )

    def test_not_tensor(self):
        """Test validation fails when input is not a tensor."""
        images = [[1, 2, 3]]  # List instead of tensor
        with pytest.raises(TypeError, match="must be torch.Tensor"):
            validate_image_tensor(images)


class TestValidateActionTensor:
    """Tests for validate_action_tensor function."""

    def test_valid_action_tensor(self):
        """Test validation passes for valid action tensor."""
        actions = torch.rand(4, 7) * 2 - 1  # Valid [B, action_dim] in [-1, 1]
        validate_action_tensor(actions)

    def test_wrong_ndim(self):
        """Test validation fails for wrong number of dimensions."""
        actions = torch.rand(4, 7, 1)  # 3D instead of 2D
        with pytest.raises(ValueError, match="must be 2D tensor"):
            validate_action_tensor(actions)

    def test_wrong_action_dim(self):
        """Test validation fails for wrong action dimension."""
        actions = torch.rand(4, 10)  # 10 dims instead of 7
        with pytest.raises(ValueError, match="must have action_dim=7"):
            validate_action_tensor(actions)

    def test_values_out_of_range(self):
        """Test validation fails for values outside [-1, 1]."""
        actions = torch.rand(4, 7) * 4 - 2  # Range [-2, 2]
        with pytest.raises(ValueError, match="values must be in range"):
            validate_action_tensor(actions)

    def test_custom_action_dim(self):
        """Test validation with custom action dimension."""
        actions = torch.rand(4, 14) * 2 - 1
        validate_action_tensor(actions, action_dim=14)

    def test_custom_value_range(self):
        """Test validation with custom value range."""
        actions = torch.rand(4, 7)  # Range [0, 1]
        validate_action_tensor(actions, value_range=(0.0, 1.0))

    def test_not_tensor(self):
        """Test validation fails when input is not a tensor."""
        actions = [[0.1, 0.2]]
        with pytest.raises(TypeError, match="must be torch.Tensor"):
            validate_action_tensor(actions)


class TestValidateTextInput:
    """Tests for validate_text_input function."""

    def test_valid_texts(self):
        """Test validation passes with texts only."""
        validate_text_input(texts=["pick cube", "place cup"])

    def test_valid_input_ids(self):
        """Test validation passes with input_ids only."""
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        validate_text_input(input_ids=input_ids)

    def test_both_texts_and_input_ids(self):
        """Test validation fails when both texts and input_ids provided."""
        texts = ["pick cube"]
        input_ids = torch.tensor([[1, 2, 3]])
        with pytest.raises(ValueError, match="Cannot provide both"):
            validate_text_input(texts=texts, input_ids=input_ids)

    def test_neither_texts_nor_input_ids(self):
        """Test validation fails when neither provided."""
        with pytest.raises(ValueError, match="Must provide either"):
            validate_text_input()

    def test_texts_not_list(self):
        """Test validation fails when texts is not a list."""
        with pytest.raises(TypeError, match="must be List"):
            validate_text_input(texts="pick cube")

    def test_texts_empty_list(self):
        """Test validation fails for empty texts list."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_text_input(texts=[])

    def test_texts_non_string_elements(self):
        """Test validation fails when texts contains non-strings."""
        with pytest.raises(TypeError, match="must be strings"):
            validate_text_input(texts=["pick cube", 123, "place cup"])

    def test_input_ids_not_tensor(self):
        """Test validation fails when input_ids is not a tensor."""
        with pytest.raises(TypeError, match="must be torch.Tensor"):
            validate_text_input(input_ids=[[1, 2, 3]])

    def test_input_ids_wrong_ndim(self):
        """Test validation fails when input_ids is not 2D."""
        input_ids = torch.tensor([1, 2, 3])  # 1D
        with pytest.raises(ValueError, match="must be 2D tensor"):
            validate_text_input(input_ids=input_ids)

    def test_attention_mask_shape_mismatch(self):
        """Test validation fails when attention_mask shape doesn't match input_ids."""
        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.tensor([[1, 1]])  # Wrong length
        with pytest.raises(ValueError, match="must match"):
            validate_text_input(input_ids=input_ids, attention_mask=attention_mask)

    def test_attention_mask_not_tensor(self):
        """Test validation fails when attention_mask is not a tensor."""
        input_ids = torch.tensor([[1, 2, 3]])
        with pytest.raises(TypeError, match="must be torch.Tensor"):
            validate_text_input(input_ids=input_ids, attention_mask=[[1, 1, 1]])


class TestValidateBatchConsistency:
    """Tests for validate_batch_consistency function."""

    def test_valid_consistency_with_texts(self):
        """Test validation passes when batch sizes match with texts."""
        images = torch.rand(4, 3, 224, 224)
        texts = ["pick", "place", "move", "grasp"]
        validate_batch_consistency(images, texts=texts)

    def test_valid_consistency_with_input_ids(self):
        """Test validation passes when batch sizes match with input_ids."""
        images = torch.rand(4, 3, 224, 224)
        input_ids = torch.randint(0, 1000, (4, 10))
        validate_batch_consistency(images, input_ids=input_ids)

    def test_valid_consistency_with_actions(self):
        """Test validation passes when batch sizes match with actions."""
        images = torch.rand(4, 3, 224, 224)
        texts = ["pick", "place", "move", "grasp"]
        actions = torch.rand(4, 7)
        validate_batch_consistency(images, texts=texts, target_actions=actions)

    def test_texts_batch_size_mismatch(self):
        """Test validation fails when texts batch size doesn't match."""
        images = torch.rand(4, 3, 224, 224)
        texts = ["pick", "place"]  # Only 2 texts
        with pytest.raises(ValueError, match="Batch size mismatch"):
            validate_batch_consistency(images, texts=texts)

    def test_input_ids_batch_size_mismatch(self):
        """Test validation fails when input_ids batch size doesn't match."""
        images = torch.rand(4, 3, 224, 224)
        input_ids = torch.randint(0, 1000, (2, 10))  # Only 2 samples
        with pytest.raises(ValueError, match="Batch size mismatch"):
            validate_batch_consistency(images, input_ids=input_ids)

    def test_actions_batch_size_mismatch(self):
        """Test validation fails when actions batch size doesn't match."""
        images = torch.rand(4, 3, 224, 224)
        texts = ["pick", "place", "move", "grasp"]
        actions = torch.rand(2, 7)  # Only 2 actions
        with pytest.raises(ValueError, match="Batch size mismatch"):
            validate_batch_consistency(images, texts=texts, target_actions=actions)

    def test_images_not_tensor(self):
        """Test validation fails when images is not a tensor."""
        with pytest.raises(TypeError, match="must be torch.Tensor"):
            validate_batch_consistency([[1, 2, 3]], texts=["pick"])

    def test_texts_not_list(self):
        """Test validation fails when texts is not a list."""
        images = torch.rand(4, 3, 224, 224)
        with pytest.raises(TypeError, match="must be List"):
            validate_batch_consistency(images, texts="pick cube")

    def test_input_ids_not_tensor(self):
        """Test validation fails when input_ids is not a tensor."""
        images = torch.rand(4, 3, 224, 224)
        with pytest.raises(TypeError, match="must be torch.Tensor"):
            validate_batch_consistency(images, input_ids=[[1, 2, 3]])

    def test_actions_not_tensor(self):
        """Test validation fails when actions is not a tensor."""
        images = torch.rand(4, 3, 224, 224)
        texts = ["pick", "place", "move", "grasp"]
        with pytest.raises(TypeError, match="must be torch.Tensor"):
            validate_batch_consistency(images, texts=texts, target_actions=[[1, 2]])


class TestValidateTemporalInputs:
    """Tests for validate_temporal_inputs function."""

    def test_valid_temporal_sequence(self):
        """Test validation passes for valid temporal sequence."""
        frames = [torch.rand(2, 3, 224, 224) for _ in range(6)]
        validate_temporal_inputs(frames, num_frames=6)

    def test_wrong_num_frames(self):
        """Test validation fails when number of frames doesn't match."""
        frames = [torch.rand(2, 3, 224, 224) for _ in range(4)]
        with pytest.raises(ValueError, match="Expected 6 frames, got 4"):
            validate_temporal_inputs(frames, num_frames=6)

    def test_not_list(self):
        """Test validation fails when image_sequence is not a list."""
        frames = torch.rand(6, 2, 3, 224, 224)  # Tensor instead of list
        with pytest.raises(TypeError, match="must be List"):
            validate_temporal_inputs(frames, num_frames=6)

    def test_frame_not_tensor(self):
        """Test validation fails when a frame is not a tensor."""
        frames = [torch.rand(2, 3, 224, 224) for _ in range(5)]
        frames.append([[1, 2, 3]])  # Non-tensor frame
        with pytest.raises(TypeError, match="must be torch.Tensor"):
            validate_temporal_inputs(frames, num_frames=6)

    def test_frame_invalid_shape(self):
        """Test validation fails when a frame has invalid shape."""
        frames = [torch.rand(2, 3, 224, 224) for _ in range(5)]
        frames.append(torch.rand(2, 4, 224, 224))  # Wrong channels
        with pytest.raises(ValueError, match="Frame 5 validation failed"):
            validate_temporal_inputs(frames, num_frames=6)

    def test_inconsistent_batch_sizes(self):
        """Test validation fails when frames have different batch sizes."""
        frames = [torch.rand(2, 3, 224, 224) for _ in range(3)]
        frames.extend([torch.rand(4, 3, 224, 224) for _ in range(3)])  # Different batch
        with pytest.raises(ValueError, match="same batch size"):
            validate_temporal_inputs(frames, num_frames=6)

    def test_custom_image_shape(self):
        """Test validation with custom expected image shape."""
        frames = [torch.rand(2, 3, 128, 128) for _ in range(4)]
        validate_temporal_inputs(
            frames,
            num_frames=4,
            expected_image_shape=(None, 3, 128, 128),
        )
