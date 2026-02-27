# Phase 10: Data Loading Pipeline

## Context Links
- [OXE Data Research](../reports/researcher-260118-0228-oxe-data-loading.md)
- [Tech Stack](../../docs/tech-stack.md) - Data loading section

## Overview
| Field | Value |
|-------|-------|
| Priority | P1 - Core Infrastructure |
| Status | ✅ Complete |
| Effort | 5h |
| Dependencies | Phases 1, 9 |

Implement data loading for Open X-Embodiment (OXE) and custom datasets. Support WebDataset streaming, HDF5 local cache, and multi-dataset mixing.

## Key Insights
- OXE uses RLDS format (TFRecord) - need conversion or streaming
- WebDataset for streaming large datasets
- HDF5 for fast local random access
- Multi-dataset mixing with configurable weights

## Requirements

### Functional
- FR-01: Dummy dataset for testing
- FR-02: WebDataset loader for OXE streaming
- FR-03: HDF5 dataset for local cached data
- FR-04: Multi-dataset mixing with sampling weights
- FR-05: Data augmentation pipeline (optional)
- FR-06: Action normalization per dataset

### Non-Functional
- NFR-01: >1000 samples/sec on NVMe SSD
- NFR-02: Async prefetching for I/O hiding

## Architecture

```
src/vla/data/
├── __init__.py
├── dummy.py              # Dummy dataset for testing
├── transforms.py         # Image/action transforms
├── webdataset_loader.py  # WebDataset streaming
├── hdf5_dataset.py       # HDF5 local dataset
├── mixture.py            # Multi-dataset mixing
└── oxe/
    ├── __init__.py
    └── rlds_adapter.py   # RLDS to PyTorch adapter
```

## Related Code Files

### Files to Create
| Path | Purpose | Lines |
|------|---------|-------|
| `src/vla/data/__init__.py` | Exports | ~20 |
| `src/vla/data/dummy.py` | Dummy dataset | ~60 |
| `src/vla/data/transforms.py` | Data transforms | ~80 |
| `src/vla/data/webdataset_loader.py` | WebDataset | ~100 |
| `src/vla/data/hdf5_dataset.py` | HDF5 dataset | ~100 |
| `src/vla/data/mixture.py` | Dataset mixing | ~80 |
| `tests/unit/test_data.py` | Data tests | ~100 |

## Implementation Steps

### Step 1: Implement dummy.py (30 min)
```python
"""Dummy dataset for testing and development."""
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, Any
import random


class DummyVLADataset(Dataset):
    """Synthetic VLA dataset for testing.

    Generates random images, text instructions, and actions.

    Args:
        num_samples: Number of samples in dataset
        image_size: Image resolution
        action_dim: Action vector dimension
        num_frames: Number of frames per sample (for temporal models)
        seed: Random seed for reproducibility
    """

    INSTRUCTIONS = [
        "pick up the red block",
        "place the cube on the table",
        "move arm to the left",
        "grasp the yellow object",
        "push the box forward",
        "rotate gripper 90 degrees",
        "lift the container",
        "release the object",
    ]

    def __init__(
        self,
        num_samples: int = 1000,
        image_size: int = 224,
        action_dim: int = 7,
        num_frames: int = 1,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.action_dim = action_dim
        self.num_frames = num_frames

        # Generate deterministic random data
        self.rng = random.Random(seed)
        torch.manual_seed(seed)

        # Pre-generate all data for consistency
        self.images = torch.randn(num_samples, num_frames, 3, image_size, image_size)
        self.actions = torch.rand(num_samples, action_dim) * 2 - 1  # [-1, 1]
        self.instructions = [
            self.rng.choice(self.INSTRUCTIONS) for _ in range(num_samples)
        ]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        images = self.images[idx]
        if self.num_frames == 1:
            images = images.squeeze(0)  # [C, H, W]

        return {
            "image": images,
            "text": self.instructions[idx],
            "action": self.actions[idx],
            "idx": idx,
        }


def create_dummy_dataloader(
    num_samples: int = 1000,
    batch_size: int = 32,
    num_workers: int = 4,
    **kwargs,
) -> DataLoader:
    """Create DataLoader with dummy dataset."""
    dataset = DummyVLADataset(num_samples=num_samples, **kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
```

### Step 2: Implement transforms.py (45 min)
```python
"""Data transforms for VLA training."""
import torch
import torch.nn as nn
from torchvision import transforms as T
from typing import Optional, Dict, Any, Callable


class VLATransform:
    """Composable transforms for VLA data.

    Handles image preprocessing and optional augmentation.
    """

    def __init__(
        self,
        image_size: int = 224,
        normalize: bool = True,
        augment: bool = False,
        mean: tuple = (0.485, 0.456, 0.406),
        std: tuple = (0.229, 0.224, 0.225),
    ):
        # Base transforms
        base = [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ]

        # Optional augmentation
        if augment:
            aug = [
                T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            ]
            base = aug + base[1:]  # Replace resize with random crop

        # Normalization
        if normalize:
            base.append(T.Normalize(mean=mean, std=std))

        self.image_transform = T.Compose(base)

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply transforms to sample."""
        result = dict(sample)

        if "image" in sample:
            result["image"] = self.image_transform(sample["image"])

        return result


class ActionNormalize:
    """Normalize actions based on dataset statistics."""

    def __init__(
        self,
        action_min: torch.Tensor,
        action_max: torch.Tensor,
    ):
        self.action_min = action_min
        self.action_max = action_max
        self.action_range = action_max - action_min

    def normalize(self, actions: torch.Tensor) -> torch.Tensor:
        """Normalize to [-1, 1]."""
        return 2 * (actions - self.action_min) / self.action_range - 1

    def denormalize(self, actions: torch.Tensor) -> torch.Tensor:
        """Denormalize from [-1, 1]."""
        return (actions + 1) / 2 * self.action_range + self.action_min

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(sample)
        if "action" in sample:
            result["action"] = self.normalize(sample["action"])
        return result


def collate_vla_batch(batch: list) -> Dict[str, torch.Tensor]:
    """Custom collate function for VLA batches.

    Handles variable-length text and stacks tensors.
    """
    images = torch.stack([s["image"] for s in batch])
    texts = [s["text"] for s in batch]
    actions = torch.stack([s["action"] for s in batch])

    result = {
        "images": images,
        "texts": texts,
        "actions": actions,
    }

    # Optional fields
    if "idx" in batch[0]:
        result["idx"] = torch.tensor([s["idx"] for s in batch])

    return result
```

### Step 3: Implement hdf5_dataset.py (60 min)
```python
"""HDF5 dataset for fast local data loading."""
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List


class HDF5VLADataset(Dataset):
    """VLA dataset stored in HDF5 format.

    Expected HDF5 structure:
    /images: [N, C, H, W] or [N, T, C, H, W] for temporal
    /actions: [N, action_dim]
    /instructions: [N,] string array
    /metadata: optional metadata group

    Args:
        hdf5_path: Path to HDF5 file
        transform: Optional transform to apply
        max_samples: Limit number of samples (for debugging)
    """

    def __init__(
        self,
        hdf5_path: str,
        transform: Optional[callable] = None,
        max_samples: Optional[int] = None,
    ):
        self.hdf5_path = Path(hdf5_path)
        self.transform = transform

        # Open file to get metadata (close immediately)
        with h5py.File(hdf5_path, "r") as f:
            self.num_samples = len(f["images"])
            self.image_shape = f["images"].shape[1:]
            self.action_dim = f["actions"].shape[1]

            # Check if instructions are strings or bytes
            self._instruction_dtype = f["instructions"].dtype

        if max_samples:
            self.num_samples = min(self.num_samples, max_samples)

        # Lazy file handle
        self._file = None

    def _open_file(self):
        """Lazy file opening for multiprocessing compatibility."""
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, "r")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        self._open_file()

        image = torch.from_numpy(self._file["images"][idx]).float()
        action = torch.from_numpy(self._file["actions"][idx]).float()

        # Handle string encoding
        instruction = self._file["instructions"][idx]
        if isinstance(instruction, bytes):
            instruction = instruction.decode("utf-8")

        sample = {
            "image": image,
            "text": instruction,
            "action": action,
            "idx": idx,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def close(self):
        """Close HDF5 file handle."""
        if self._file is not None:
            self._file.close()
            self._file = None


def create_hdf5_dataset(
    output_path: str,
    images: np.ndarray,
    actions: np.ndarray,
    instructions: List[str],
    chunk_size: int = 100,
):
    """Create HDF5 dataset from numpy arrays.

    Args:
        output_path: Output HDF5 file path
        images: [N, C, H, W] image array
        actions: [N, action_dim] action array
        instructions: List of instruction strings
        chunk_size: Chunk size for compression
    """
    N = len(images)
    assert len(actions) == N
    assert len(instructions) == N

    with h5py.File(output_path, "w") as f:
        # Create datasets with compression
        f.create_dataset(
            "images",
            data=images,
            chunks=(min(chunk_size, N),) + images.shape[1:],
            compression="gzip",
            compression_opts=4,
        )
        f.create_dataset(
            "actions",
            data=actions,
            chunks=(min(chunk_size, N), actions.shape[1]),
        )

        # String array for instructions
        dt = h5py.special_dtype(vlen=str)
        f.create_dataset("instructions", data=instructions, dtype=dt)

        # Metadata
        f.attrs["num_samples"] = N
        f.attrs["image_shape"] = images.shape[1:]
        f.attrs["action_dim"] = actions.shape[1]
```

### Step 4: Implement webdataset_loader.py (60 min)
```python
"""WebDataset loader for streaming large datasets."""
import torch
from torch.utils.data import DataLoader, IterableDataset
import webdataset as wds
from typing import Optional, Iterator, Dict, Any
import io
from PIL import Image


def create_webdataset(
    shards: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    shuffle_buffer: int = 1000,
    resampled: bool = True,
) -> DataLoader:
    """Create WebDataset DataLoader for streaming.

    Args:
        shards: Shard URL pattern (e.g., "/data/oxe-{000..099}.tar")
        batch_size: Batch size
        num_workers: Number of worker processes
        image_size: Target image size
        shuffle_buffer: Shuffle buffer size
        resampled: Whether to resample for infinite iteration
    """
    # Image preprocessing
    def preprocess_image(image_bytes: bytes) -> torch.Tensor:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((image_size, image_size))
        tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        return tensor

    # Sample processing
    def process_sample(sample: Dict) -> Dict:
        return {
            "image": preprocess_image(sample["jpg"] if "jpg" in sample else sample["png"]),
            "text": sample["txt"].decode() if isinstance(sample["txt"], bytes) else sample["txt"],
            "action": torch.from_numpy(np.frombuffer(sample["action.npy"], dtype=np.float32)),
        }

    # Build pipeline
    if resampled:
        dataset = wds.WebDataset(shards, resampled=True)
    else:
        dataset = wds.WebDataset(shards)

    dataset = (
        dataset
        .shuffle(shuffle_buffer)
        .map(process_sample)
        .batched(batch_size, collation_fn=collate_wds_batch)
    )

    return DataLoader(
        dataset,
        batch_size=None,  # Batching done in WebDataset
        num_workers=num_workers,
        pin_memory=True,
    )


def collate_wds_batch(samples: list) -> Dict[str, torch.Tensor]:
    """Collate WebDataset batch."""
    return {
        "images": torch.stack([s["image"] for s in samples]),
        "texts": [s["text"] for s in samples],
        "actions": torch.stack([s["action"] for s in samples]),
    }


class StreamingVLADataset(IterableDataset):
    """Custom streaming dataset for VLA data.

    Wraps WebDataset with VLA-specific processing.
    """

    def __init__(
        self,
        shards: str,
        transform: Optional[callable] = None,
        shuffle_buffer: int = 1000,
    ):
        self.shards = shards
        self.transform = transform
        self.shuffle_buffer = shuffle_buffer

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        dataset = wds.WebDataset(self.shards, resampled=True)
        dataset = dataset.shuffle(self.shuffle_buffer)

        for sample in dataset:
            processed = self._process_sample(sample)
            if self.transform:
                processed = self.transform(processed)
            yield processed

    def _process_sample(self, sample: Dict) -> Dict[str, Any]:
        # Extract image
        if "jpg" in sample:
            image_bytes = sample["jpg"]
        elif "png" in sample:
            image_bytes = sample["png"]
        else:
            raise ValueError("No image found in sample")

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        # Extract text
        text = sample.get("txt", b"").decode() if isinstance(sample.get("txt", ""), bytes) else sample.get("txt", "")

        # Extract action
        action = torch.from_numpy(np.frombuffer(sample["action.npy"], dtype=np.float32))

        return {
            "image": image_tensor,
            "text": text,
            "action": action,
        }
```

### Step 5: Implement mixture.py (45 min)
```python
"""Multi-dataset mixing for training."""
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
import numpy as np
from typing import List, Dict, Any, Optional, Iterator
import random


class DatasetMixture(Dataset):
    """Mix multiple datasets with weighted sampling.

    Args:
        datasets: List of datasets to mix
        weights: Sampling weights (will be normalized)
        seed: Random seed
    """

    def __init__(
        self,
        datasets: List[Dataset],
        weights: Optional[List[float]] = None,
        seed: int = 42,
    ):
        self.datasets = datasets
        self.dataset_sizes = [len(d) for d in datasets]
        self.total_size = sum(self.dataset_sizes)

        # Normalize weights
        if weights is None:
            weights = self.dataset_sizes
        weights = np.array(weights, dtype=np.float32)
        self.weights = weights / weights.sum()

        # Precompute sampling probabilities
        self.rng = np.random.default_rng(seed)
        self._precompute_indices()

    def _precompute_indices(self):
        """Precompute dataset and sample indices."""
        # Sample which dataset each index comes from
        self.dataset_indices = self.rng.choice(
            len(self.datasets),
            size=self.total_size,
            p=self.weights,
        )

        # Sample index within each dataset
        self.sample_indices = np.zeros(self.total_size, dtype=np.int64)
        for i, ds_idx in enumerate(self.dataset_indices):
            self.sample_indices[i] = self.rng.integers(0, self.dataset_sizes[ds_idx])

    def __len__(self) -> int:
        return self.total_size

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ds_idx = self.dataset_indices[idx]
        sample_idx = self.sample_indices[idx]

        sample = self.datasets[ds_idx][sample_idx]
        sample["dataset_idx"] = ds_idx

        return sample


class IterableDatasetMixture(IterableDataset):
    """Mix multiple iterable datasets with weighted sampling."""

    def __init__(
        self,
        datasets: List[IterableDataset],
        weights: Optional[List[float]] = None,
    ):
        self.datasets = datasets

        if weights is None:
            weights = [1.0] * len(datasets)
        weights = np.array(weights, dtype=np.float32)
        self.weights = weights / weights.sum()

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        # Create iterators for each dataset
        iterators = [iter(ds) for ds in self.datasets]

        while True:
            # Sample dataset
            ds_idx = np.random.choice(len(self.datasets), p=self.weights)

            try:
                sample = next(iterators[ds_idx])
                sample["dataset_idx"] = ds_idx
                yield sample
            except StopIteration:
                # Restart exhausted iterator
                iterators[ds_idx] = iter(self.datasets[ds_idx])


def create_mixture_dataloader(
    datasets: List[Dataset],
    weights: List[float],
    batch_size: int = 32,
    num_workers: int = 4,
    **loader_kwargs,
) -> DataLoader:
    """Create DataLoader for dataset mixture."""
    from .transforms import collate_vla_batch

    mixture = DatasetMixture(datasets, weights)
    return DataLoader(
        mixture,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_vla_batch,
        **loader_kwargs,
    )
```

### Step 6: Create __init__.py (15 min)
```python
"""Data loading modules for VLA training."""
from .dummy import DummyVLADataset, create_dummy_dataloader
from .transforms import VLATransform, ActionNormalize, collate_vla_batch
from .hdf5_dataset import HDF5VLADataset, create_hdf5_dataset
from .webdataset_loader import create_webdataset, StreamingVLADataset
from .mixture import DatasetMixture, IterableDatasetMixture, create_mixture_dataloader

__all__ = [
    # Datasets
    "DummyVLADataset",
    "HDF5VLADataset",
    "StreamingVLADataset",
    "DatasetMixture",
    "IterableDatasetMixture",
    # Transforms
    "VLATransform",
    "ActionNormalize",
    "collate_vla_batch",
    # Factory functions
    "create_dummy_dataloader",
    "create_hdf5_dataset",
    "create_webdataset",
    "create_mixture_dataloader",
]
```

### Step 7: Write tests (45 min)
```python
"""Tests for data loading."""
import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path
from vla.data import (
    DummyVLADataset,
    HDF5VLADataset,
    create_hdf5_dataset,
    DatasetMixture,
    VLATransform,
    collate_vla_batch,
)


class TestDummyDataset:
    def test_length(self):
        dataset = DummyVLADataset(num_samples=100)
        assert len(dataset) == 100

    def test_sample_structure(self):
        dataset = DummyVLADataset()
        sample = dataset[0]
        assert "image" in sample
        assert "text" in sample
        assert "action" in sample
        assert sample["image"].shape == (3, 224, 224)
        assert sample["action"].shape == (7,)

    def test_reproducibility(self):
        ds1 = DummyVLADataset(seed=42)
        ds2 = DummyVLADataset(seed=42)
        assert torch.allclose(ds1[0]["image"], ds2[0]["image"])


class TestHDF5Dataset:
    @pytest.fixture
    def hdf5_path(self, tmp_path):
        path = tmp_path / "test.h5"
        images = np.random.randn(50, 3, 224, 224).astype(np.float32)
        actions = np.random.randn(50, 7).astype(np.float32)
        instructions = ["test instruction"] * 50
        create_hdf5_dataset(str(path), images, actions, instructions)
        return path

    def test_length(self, hdf5_path):
        dataset = HDF5VLADataset(str(hdf5_path))
        assert len(dataset) == 50

    def test_sample_loading(self, hdf5_path):
        dataset = HDF5VLADataset(str(hdf5_path))
        sample = dataset[0]
        assert sample["image"].shape == (3, 224, 224)
        assert sample["action"].shape == (7,)
        assert sample["text"] == "test instruction"


class TestDatasetMixture:
    def test_mixture_sampling(self):
        ds1 = DummyVLADataset(num_samples=100, seed=1)
        ds2 = DummyVLADataset(num_samples=100, seed=2)
        mixture = DatasetMixture([ds1, ds2], weights=[0.7, 0.3])

        assert len(mixture) == 200

        # Check we get samples from both datasets
        dataset_indices = [mixture[i]["dataset_idx"] for i in range(200)]
        assert 0 in dataset_indices
        assert 1 in dataset_indices


class TestCollate:
    def test_collate_batch(self):
        dataset = DummyVLADataset(num_samples=10)
        batch = [dataset[i] for i in range(4)]
        collated = collate_vla_batch(batch)

        assert collated["images"].shape == (4, 3, 224, 224)
        assert len(collated["texts"]) == 4
        assert collated["actions"].shape == (4, 7)
```

## Todo List
- [ ] Implement DummyVLADataset
- [ ] Implement VLATransform and ActionNormalize
- [ ] Implement HDF5VLADataset
- [ ] Implement create_hdf5_dataset utility
- [ ] Implement WebDataset loader
- [ ] Implement DatasetMixture
- [ ] Implement collate_vla_batch
- [ ] Write comprehensive tests
- [ ] Test with real OXE data (if available)

## Success Criteria
1. Dummy dataset generates valid samples
2. HDF5 dataset loads from file correctly
3. WebDataset streams without loading all data
4. Mixture samples from datasets according to weights
5. All tests pass with >85% coverage

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| OXE format changes | Medium | Abstract RLDS adapter |
| Memory leak in streaming | High | Test long-running loading |
| Multiprocessing issues | Medium | Test with num_workers > 0 |

## Security Considerations
- Validate file paths before loading
- No arbitrary code execution from data files

## Next Steps
- Phase 11: Training infrastructure
