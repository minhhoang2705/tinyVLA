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

from typing import TYPE_CHECKING, Any, Callable, Dict, List

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


def vla_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate VLA samples into a batch.

    Stacks image and action tensors along batch dimension, and collects
    text instructions into a list. Optionally stacks state tensors when
    every sample in the batch contains a "state" key.

    Args:
        batch: List of sample dictionaries from DummyVLADataset
               Each sample contains: {'image': Tensor, 'text': str, 'action': Tensor}

    Returns:
        Dictionary containing:
            - images: Stacked image tensor [B, C, H, W]
            - texts: List of text strings [B]
            - actions: Stacked action tensor [B, action_dim]
            - states: Stacked state tensor [B, state_dim] (only when all samples have "state")

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

    result: Dict[str, Any] = {
        "images": images,
        "texts": texts,
        "actions": actions,
    }
    # Stack state only when every sample has it — partial batches drop silently to avoid shape errors
    if all("state" in sample for sample in batch):
        result["states"] = torch.stack([sample["state"] for sample in batch])
    return result


def make_tokenized_collate_fn(
    tokenizer: "PreTrainedTokenizerBase",
    max_length: int = 77,
) -> Callable:
    """Return a collate_fn that tokenizes text in DataLoader workers.

    Moves CPU tokenization out of the GPU forward pass, eliminating
    CPU-GPU sync stalls and enabling parallel tokenization across workers.
    The returned collate_fn produces 'input_ids' and 'attention_mask' keys
    instead of 'texts', which VLAModel.forward() already supports.

    Args:
        tokenizer: HuggingFace tokenizer (must be picklable for multiprocessing).
        max_length: Maximum token sequence length. Default 77 (CLIP-style).

    Returns:
        collate_fn compatible with DataLoader's collate_fn parameter.

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> tokenizer.pad_token = tokenizer.eos_token
        >>> collate_fn = make_tokenized_collate_fn(tokenizer, max_length=77)
        >>> batch = collate_fn([dataset[i] for i in range(4)])
        >>> batch['input_ids'].shape  # torch.Size([4, 77])
    """

    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        images = torch.stack([sample["image"] for sample in batch])
        texts = [sample["text"] for sample in batch]
        actions = torch.stack([sample["action"] for sample in batch])

        # Tokenize in worker process — runs in parallel with GPU compute
        encoded = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        result = {
            "images": images,
            "input_ids": encoded["input_ids"],           # [B, max_length]
            "attention_mask": encoded["attention_mask"],  # [B, max_length]
            "actions": actions,
            "texts": texts,  # Keep for debugging/logging
        }
        if all("state" in sample for sample in batch):
            result["states"] = torch.stack([sample["state"] for sample in batch])
        return result

    return collate_fn


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
