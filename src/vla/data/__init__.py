"""Data loading and preprocessing utilities for VLA training.

This module provides:
- DummyVLADataset: Synthetic dataset for testing and prototyping
- VLADataModule: PyTorch Lightning DataModule for training
- vla_collate_fn: Batch collation utility

Example:
    >>> from vla.data import DummyVLADataset, VLADataModule
    >>>
    >>> # Create dummy dataset
    >>> dataset = DummyVLADataset(num_samples=1000)
    >>> sample = dataset[0]
    >>> print(sample.keys())  # dict_keys(['image', 'text', 'action'])
    >>>
    >>> # Use with DataModule
    >>> datamodule = VLADataModule(dataset_type='dummy', batch_size=32)
    >>> datamodule.setup()
    >>> train_loader = datamodule.train_dataloader()
"""

from vla.data.collate_batch_samples import vla_collate_fn
from vla.data.datamodule_lightning import VLADataModule
from vla.data.dummy_vla_dataset import DummyVLADataset
from vla.data.lerobot_dataset import LeRobotVLADataset

__all__ = [
    "DummyVLADataset",
    "LeRobotVLADataset",
    "VLADataModule",
    "vla_collate_fn",
]
