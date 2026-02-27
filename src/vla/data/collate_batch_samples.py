"""Batch collation utilities for VLA datasets.

Provides collate functions to combine individual samples into batches
for DataLoader usage. Handles stacking images/actions and collecting texts.

Example:
    >>> from torch.utils.data import DataLoader
    >>> from vla.data import DummyVLADataset, vla_collate_fn
    >>>
    >>> dataset = DummyVLADataset(num_samples=100)
    >>> loader = DataLoader(dataset, batch_size=16, collate_fn=vla_collate_fn)
    >>> batch = next(iter(loader))
    >>> batch['images'].shape  # torch.Size([16, 3, 224, 224])
"""

from typing import Any, Dict, List

import torch


def vla_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor | List[str]]:
    """Collate VLA samples into a batch.

    Stacks image and action tensors along batch dimension, and collects
    text instructions into a list.

    Args:
        batch: List of sample dictionaries from DummyVLADataset
               Each sample contains: {'image': Tensor, 'text': str, 'action': Tensor}

    Returns:
        Dictionary containing:
            - images: Stacked image tensor [B, C, H, W]
            - texts: List of text strings [B]
            - actions: Stacked action tensor [B, action_dim]

    Example:
        >>> from vla.data import DummyVLADataset
        >>> dataset = DummyVLADataset(num_samples=10)
        >>> samples = [dataset[i] for i in range(4)]
        >>> batch = vla_collate_fn(samples)
        >>> batch['images'].shape
        torch.Size([4, 3, 224, 224])
        >>> len(batch['texts'])
        4
    """
    # Stack images: List of [C, H, W] -> [B, C, H, W]
    images = torch.stack([sample["image"] for sample in batch])

    # Collect text instructions: List of strings
    texts = [sample["text"] for sample in batch]

    # Stack actions: List of [action_dim] -> [B, action_dim]
    actions = torch.stack([sample["action"] for sample in batch])

    return {
        "images": images,
        "texts": texts,
        "actions": actions,
    }


def temporal_vla_collate_fn(
    batch: List[Dict[str, Any]],
) -> Dict[str, List[torch.Tensor] | List[str] | torch.Tensor]:
    """Collate temporal VLA samples into a batch.

    For temporal models with frame sequences, this collates sequences
    while preserving the temporal dimension.

    Args:
        batch: List of temporal sample dictionaries
               Each sample contains: {'image_sequence': List[Tensor], 'text': str, 'action': Tensor}

    Returns:
        Dictionary containing:
            - image_sequences: List of [B, C, H, W] tensors (one per frame)
            - texts: List of text strings [B]
            - actions: Stacked action tensor [B, action_dim]

    Example:
        >>> from vla.data import DummyTemporalVLADataset
        >>> dataset = DummyTemporalVLADataset(num_samples=10, num_frames=6)
        >>> samples = [dataset[i] for i in range(2)]
        >>> batch = temporal_vla_collate_fn(samples)
        >>> len(batch['image_sequences'])  # 6 frames
        6
        >>> batch['image_sequences'][0].shape  # First frame
        torch.Size([2, 3, 224, 224])
    """
    # Get number of frames from first sample
    num_frames = len(batch[0]["image_sequence"])

    # Stack each frame across batch: List[List[Tensor]] -> List[Tensor]
    # For each frame index, stack all samples' frames
    image_sequences = [
        torch.stack([sample["image_sequence"][frame_idx] for sample in batch])
        for frame_idx in range(num_frames)
    ]

    # Collect text instructions
    texts = [sample["text"] for sample in batch]

    # Stack actions
    actions = torch.stack([sample["action"] for sample in batch])

    return {
        "image_sequences": image_sequences,
        "texts": texts,
        "actions": actions,
    }
