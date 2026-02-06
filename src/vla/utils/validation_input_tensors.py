"""Input validation utilities for VLA model tensors.

This module provides comprehensive validation functions for:
- Image tensors (shape, dtype, value range)
- Action tensors (shape, value range)
- Batch consistency across modalities
- Text input validation

All validation functions raise descriptive ValueError/TypeError exceptions
with clear error messages to help debug input issues quickly.

Example:
    >>> import torch
    >>> from vla.utils.validation import validate_image_tensor, validate_batch_consistency
    >>>
    >>> images = torch.randn(4, 3, 224, 224)
    >>> texts = ["pick cube", "place cup", "move left", "grasp"]
    >>>
    >>> validate_image_tensor(images)  # Passes
    >>> validate_batch_consistency(images, texts)  # Passes
"""

from typing import List, Optional, Tuple

import torch


def validate_image_tensor(
    images: torch.Tensor,
    expected_shape: Tuple[Optional[int], int, int, int] = (None, 3, 224, 224),
    dtype: torch.dtype = torch.float32,
    value_range: Tuple[float, float] = (0.0, 1.0),
    name: str = "images",
) -> None:
    """Validate image tensor shape, dtype, and value range.

    Args:
        images: Image tensor to validate
        expected_shape: Expected shape (B, C, H, W) where None means any batch size
        dtype: Expected data type
        value_range: Expected min/max values (inclusive)
        name: Tensor name for error messages

    Raises:
        TypeError: If images is not a torch.Tensor or has wrong dtype
        ValueError: If shape or value range is invalid

    Example:
        >>> images = torch.randn(2, 3, 224, 224).clamp(0, 1)
        >>> validate_image_tensor(images)  # Passes
        >>>
        >>> bad_images = torch.randn(2, 4, 224, 224)  # Wrong channels
        >>> validate_image_tensor(bad_images)  # Raises ValueError
    """
    # Check type
    if not isinstance(images, torch.Tensor):
        raise TypeError(
            f"{name} must be torch.Tensor, got {type(images).__name__}"
        )

    # Check dtype
    if images.dtype != dtype:
        raise TypeError(
            f"{name} must have dtype {dtype}, got {images.dtype}"
        )

    # Check ndim
    if images.ndim != 4:
        raise ValueError(
            f"{name} must be 4D tensor [B, C, H, W], got {images.ndim}D with shape {tuple(images.shape)}"
        )

    # Check shape components
    batch_size, channels, height, width = images.shape
    exp_batch, exp_channels, exp_height, exp_width = expected_shape

    if exp_batch is not None and batch_size != exp_batch:
        raise ValueError(
            f"{name} batch size must be {exp_batch}, got {batch_size}"
        )

    if channels != exp_channels:
        raise ValueError(
            f"{name} must have {exp_channels} channels, got {channels}. "
            f"Expected shape [B, {exp_channels}, {exp_height}, {exp_width}], "
            f"got {tuple(images.shape)}"
        )

    if height != exp_height or width != exp_width:
        raise ValueError(
            f"{name} spatial dimensions must be [{exp_height}, {exp_width}], "
            f"got [{height}, {width}]. "
            f"Expected shape [B, {exp_channels}, {exp_height}, {exp_width}], "
            f"got {tuple(images.shape)}"
        )

    # Check value range
    min_val = images.min().item()
    max_val = images.max().item()
    expected_min, expected_max = value_range

    if min_val < expected_min or max_val > expected_max:
        raise ValueError(
            f"{name} values must be in range [{expected_min}, {expected_max}], "
            f"got min={min_val:.4f}, max={max_val:.4f}. "
            f"Did you forget to normalize images to [0, 1]?"
        )


def validate_action_tensor(
    actions: torch.Tensor,
    action_dim: int = 7,
    value_range: Tuple[float, float] = (-1.0, 1.0),
    name: str = "actions",
) -> None:
    """Validate action tensor shape and value range.

    Args:
        actions: Action tensor to validate [B, action_dim]
        action_dim: Expected action dimension
        value_range: Expected min/max values (inclusive)
        name: Tensor name for error messages

    Raises:
        TypeError: If actions is not a torch.Tensor
        ValueError: If shape or value range is invalid

    Example:
        >>> actions = torch.randn(4, 7).clamp(-1, 1)
        >>> validate_action_tensor(actions)  # Passes
        >>>
        >>> bad_actions = torch.randn(4, 10)  # Wrong action_dim
        >>> validate_action_tensor(bad_actions)  # Raises ValueError
    """
    # Check type
    if not isinstance(actions, torch.Tensor):
        raise TypeError(
            f"{name} must be torch.Tensor, got {type(actions).__name__}"
        )

    # Check ndim
    if actions.ndim != 2:
        raise ValueError(
            f"{name} must be 2D tensor [B, action_dim], got {actions.ndim}D with shape {tuple(actions.shape)}"
        )

    # Check action dimension
    batch_size, actual_action_dim = actions.shape
    if actual_action_dim != action_dim:
        raise ValueError(
            f"{name} must have action_dim={action_dim}, got {actual_action_dim}. "
            f"Expected shape [B, {action_dim}], got {tuple(actions.shape)}"
        )

    # Check value range
    min_val = actions.min().item()
    max_val = actions.max().item()
    expected_min, expected_max = value_range

    if min_val < expected_min or max_val > expected_max:
        raise ValueError(
            f"{name} values must be in range [{expected_min}, {expected_max}], "
            f"got min={min_val:.4f}, max={max_val:.4f}. "
            f"Actions should be normalized to [-1, 1]."
        )


def validate_text_input(
    texts: Optional[List[str]] = None,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> None:
    """Validate text inputs (mutually exclusive texts vs input_ids).

    Args:
        texts: List of instruction strings
        input_ids: Pre-tokenized input IDs tensor
        attention_mask: Attention mask for input_ids

    Raises:
        ValueError: If both texts and input_ids provided, or neither provided

    Example:
        >>> validate_text_input(texts=["pick cube"])  # Passes
        >>> validate_text_input(input_ids=torch.tensor([[1, 2, 3]]))  # Passes
        >>> validate_text_input(texts=["pick"], input_ids=torch.tensor([[1]]))  # Raises
    """
    if texts is not None and input_ids is not None:
        raise ValueError(
            "Cannot provide both 'texts' and 'input_ids'. "
            "Use 'texts' for raw strings or 'input_ids' for pre-tokenized input."
        )

    if texts is None and input_ids is None:
        raise ValueError(
            "Must provide either 'texts' (List[str]) or 'input_ids' (torch.Tensor). "
            "Both cannot be None."
        )

    # Validate texts format
    if texts is not None:
        if not isinstance(texts, list):
            raise TypeError(
                f"'texts' must be List[str], got {type(texts).__name__}"
            )
        if len(texts) == 0:
            raise ValueError("'texts' cannot be empty list")
        if not all(isinstance(t, str) for t in texts):
            raise TypeError(
                f"All elements in 'texts' must be strings. "
                f"Got types: {[type(t).__name__ for t in texts]}"
            )

    # Validate input_ids format
    if input_ids is not None:
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError(
                f"'input_ids' must be torch.Tensor, got {type(input_ids).__name__}"
            )
        if input_ids.ndim != 2:
            raise ValueError(
                f"'input_ids' must be 2D tensor [B, L], got {input_ids.ndim}D"
            )

        # Validate attention_mask if provided
        if attention_mask is not None:
            if not isinstance(attention_mask, torch.Tensor):
                raise TypeError(
                    f"'attention_mask' must be torch.Tensor, got {type(attention_mask).__name__}"
                )
            if attention_mask.shape != input_ids.shape:
                raise ValueError(
                    f"'attention_mask' shape {tuple(attention_mask.shape)} "
                    f"must match 'input_ids' shape {tuple(input_ids.shape)}"
                )


def validate_batch_consistency(
    images: torch.Tensor,
    texts: Optional[List[str]] = None,
    input_ids: Optional[torch.Tensor] = None,
    target_actions: Optional[torch.Tensor] = None,
) -> None:
    """Validate batch size consistency across all inputs.

    Ensures that batch dimensions match across images, text inputs,
    and target actions (if provided).

    Args:
        images: Image tensor [B, C, H, W]
        texts: List of instruction strings [B]
        input_ids: Pre-tokenized input IDs [B, L]
        target_actions: Ground truth actions [B, action_dim]

    Raises:
        ValueError: If batch sizes don't match across inputs

    Example:
        >>> images = torch.randn(4, 3, 224, 224)
        >>> texts = ["pick", "place", "move", "grasp"]  # 4 texts
        >>> actions = torch.randn(4, 7)  # 4 actions
        >>> validate_batch_consistency(images, texts, target_actions=actions)  # Passes
        >>>
        >>> bad_texts = ["pick", "place"]  # Only 2 texts
        >>> validate_batch_consistency(images, bad_texts)  # Raises ValueError
    """
    if not isinstance(images, torch.Tensor):
        raise TypeError(f"images must be torch.Tensor, got {type(images).__name__}")

    batch_size = images.shape[0]

    # Check text batch size
    if texts is not None:
        if not isinstance(texts, list):
            raise TypeError(f"texts must be List[str], got {type(texts).__name__}")

        text_batch_size = len(texts)
        if text_batch_size != batch_size:
            raise ValueError(
                f"Batch size mismatch: images have {batch_size} samples, "
                f"but texts has {text_batch_size} strings. "
                f"Expected {batch_size} text instructions."
            )

    # Check input_ids batch size
    if input_ids is not None:
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError(
                f"input_ids must be torch.Tensor, got {type(input_ids).__name__}"
            )

        input_ids_batch_size = input_ids.shape[0]
        if input_ids_batch_size != batch_size:
            raise ValueError(
                f"Batch size mismatch: images have {batch_size} samples, "
                f"but input_ids has {input_ids_batch_size} samples. "
                f"Expected shape [{batch_size}, L], got {tuple(input_ids.shape)}."
            )

    # Check target_actions batch size
    if target_actions is not None:
        if not isinstance(target_actions, torch.Tensor):
            raise TypeError(
                f"target_actions must be torch.Tensor, got {type(target_actions).__name__}"
            )

        actions_batch_size = target_actions.shape[0]
        if actions_batch_size != batch_size:
            raise ValueError(
                f"Batch size mismatch: images have {batch_size} samples, "
                f"but target_actions has {actions_batch_size} samples. "
                f"Expected shape [{batch_size}, action_dim], got {tuple(target_actions.shape)}."
            )


def validate_temporal_inputs(
    image_sequence: List[torch.Tensor],
    num_frames: int,
    expected_image_shape: Tuple[Optional[int], int, int, int] = (None, 3, 224, 224),
) -> None:
    """Validate temporal image sequence for TemporalVLAModel.

    Args:
        image_sequence: List of image tensors
        num_frames: Expected number of frames
        expected_image_shape: Expected shape for each frame

    Raises:
        TypeError: If image_sequence is not a list or contains non-tensors
        ValueError: If number of frames or shapes are invalid

    Example:
        >>> frames = [torch.randn(2, 3, 224, 224) for _ in range(6)]
        >>> validate_temporal_inputs(frames, num_frames=6)  # Passes
    """
    if not isinstance(image_sequence, list):
        raise TypeError(
            f"image_sequence must be List[torch.Tensor], got {type(image_sequence).__name__}"
        )

    if len(image_sequence) != num_frames:
        raise ValueError(
            f"Expected {num_frames} frames, got {len(image_sequence)}. "
            f"Ensure image_sequence length matches model's num_frames parameter."
        )

    # Validate each frame
    for i, frame in enumerate(image_sequence):
        if not isinstance(frame, torch.Tensor):
            raise TypeError(
                f"Frame {i} must be torch.Tensor, got {type(frame).__name__}"
            )

        try:
            validate_image_tensor(
                frame,
                expected_shape=expected_image_shape,
                name=f"image_sequence[{i}]",
            )
        except (TypeError, ValueError) as e:
            raise ValueError(f"Frame {i} validation failed: {e}") from e

    # Check all frames have same batch size
    batch_sizes = [frame.shape[0] for frame in image_sequence]
    if len(set(batch_sizes)) > 1:
        raise ValueError(
            f"All frames must have same batch size, got {batch_sizes}. "
            f"Ensure all frames in the sequence have consistent batch dimension."
        )
