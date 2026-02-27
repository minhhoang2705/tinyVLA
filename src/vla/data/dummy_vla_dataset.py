"""Dummy VLA dataset for testing and prototyping.

Generates random synthetic samples with images, text instructions, and actions.
Useful for:
- Testing model architecture without real data
- Prototyping training loops
- Benchmarking inference speed
- Unit testing data pipeline

Example:
    >>> from vla.data import DummyVLADataset
    >>> dataset = DummyVLADataset(num_samples=1000, action_dim=7)
    >>> sample = dataset[0]
    >>> print(sample['image'].shape)  # torch.Size([3, 224, 224])
    >>> print(sample['text'])  # "pick up cube"
    >>> print(sample['action'].shape)  # torch.Size([7])
"""

from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset


class DummyVLADataset(Dataset):
    """Synthetic VLA dataset generating random samples.

    Creates random images, text instructions, and action vectors for testing
    VLA models without requiring real robotics data.

    Args:
        num_samples: Number of samples in the dataset
        image_size: Image spatial dimensions (H, W)
        action_dim: Dimensionality of action vectors
        seed: Random seed for reproducibility

    Attributes:
        instructions: List of template instruction strings

    Example:
        >>> dataset = DummyVLADataset(num_samples=100, action_dim=7)
        >>> len(dataset)
        100
        >>> sample = dataset[42]
        >>> sample['image'].shape
        torch.Size([3, 224, 224])
        >>> sample['action'].shape
        torch.Size([7])
    """

    # Template instructions for common robot tasks
    INSTRUCTIONS = [
        "pick up the cube",
        "place the cup on the table",
        "move to the target location",
        "grasp the object",
        "push the block forward",
        "pull the drawer open",
        "rotate the object clockwise",
        "lift the plate",
        "press the button",
        "turn the knob",
        "slide the door",
        "stack the blocks",
        "pour water into the cup",
        "open the lid",
        "close the container",
        "wipe the surface",
        "flip the switch",
        "insert the key",
        "remove the cap",
        "align the pieces",
    ]

    def __init__(
        self,
        num_samples: int = 1000,
        image_size: Tuple[int, int] = (224, 224),
        action_dim: int = 7,
        seed: int = 42,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.image_size = image_size
        self.action_dim = action_dim
        self.seed = seed

        # Set seed for reproducibility
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)

        # Cache instruction list for faster access
        self.instructions = self.INSTRUCTIONS

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        """Get a single sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
                - image: Random RGB image [3, H, W] in range [0, 1]
                - text: Random instruction string
                - action: Random action vector [action_dim] in range [-1, 1]

        Example:
            >>> dataset = DummyVLADataset(num_samples=10)
            >>> sample = dataset[5]
            >>> print(sample.keys())
            dict_keys(['image', 'text', 'action'])
        """
        # Generate deterministic random values based on index
        # This ensures same sample for same index across calls
        sample_seed = self.seed + idx
        generator = torch.Generator()
        generator.manual_seed(sample_seed)

        # Generate random RGB image in [0, 1]
        image = torch.rand(3, *self.image_size, generator=generator)

        # Select instruction cyclically (deterministic for same idx)
        instruction_idx = idx % len(self.instructions)
        text = self.instructions[instruction_idx]

        # Generate random action in [-1, 1]
        action = torch.rand(self.action_dim, generator=generator) * 2 - 1

        return {
            "image": image,
            "text": text,
            "action": action,
        }

    def get_batch(self, batch_size: int) -> Dict[str, torch.Tensor | List[str]]:
        """Get a batch of samples (convenience method for testing).

        Args:
            batch_size: Number of samples to retrieve

        Returns:
            Dictionary containing batched tensors and list of texts

        Example:
            >>> dataset = DummyVLADataset(num_samples=100)
            >>> batch = dataset.get_batch(batch_size=4)
            >>> batch['images'].shape
            torch.Size([4, 3, 224, 224])
            >>> len(batch['texts'])
            4
        """
        samples = [self[i] for i in range(batch_size)]

        return {
            "images": torch.stack([s["image"] for s in samples]),
            "texts": [s["text"] for s in samples],
            "actions": torch.stack([s["action"] for s in samples]),
        }


class DummyTemporalVLADataset(DummyVLADataset):
    """Synthetic temporal VLA dataset with frame sequences.

    Extends DummyVLADataset to generate sequences of frames instead of
    single images. Useful for testing temporal VLA models.

    Args:
        num_samples: Number of samples in the dataset
        image_size: Image spatial dimensions (H, W)
        action_dim: Dimensionality of action vectors
        num_frames: Number of frames per sequence
        seed: Random seed for reproducibility

    Example:
        >>> dataset = DummyTemporalVLADataset(num_samples=100, num_frames=6)
        >>> sample = dataset[0]
        >>> len(sample['image_sequence'])
        6
        >>> sample['image_sequence'][0].shape
        torch.Size([3, 224, 224])
    """

    def __init__(
        self,
        num_samples: int = 1000,
        image_size: Tuple[int, int] = (224, 224),
        action_dim: int = 7,
        num_frames: int = 6,
        seed: int = 42,
    ):
        super().__init__(
            num_samples=num_samples,
            image_size=image_size,
            action_dim=action_dim,
            seed=seed,
        )
        self.num_frames = num_frames

    def __getitem__(self, idx: int) -> Dict[str, List[torch.Tensor] | str | torch.Tensor]:
        """Get a single temporal sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
                - image_sequence: List of [3, H, W] frames in range [0, 1]
                - text: Instruction string
                - action: Action vector [action_dim] in range [-1, 1]

        Example:
            >>> dataset = DummyTemporalVLADataset(num_samples=10, num_frames=4)
            >>> sample = dataset[5]
            >>> len(sample['image_sequence'])
            4
        """
        # Generate deterministic random values based on index
        sample_seed = self.seed + idx
        generator = torch.Generator()
        generator.manual_seed(sample_seed)

        # Generate sequence of random frames
        image_sequence = [
            torch.rand(3, *self.image_size, generator=generator) for _ in range(self.num_frames)
        ]

        # Select instruction cyclically
        instruction_idx = idx % len(self.instructions)
        text = self.instructions[instruction_idx]

        # Generate random action
        action = torch.rand(self.action_dim, generator=generator) * 2 - 1

        return {
            "image_sequence": image_sequence,
            "text": text,
            "action": action,
        }
