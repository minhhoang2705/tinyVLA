"""PyTorch Lightning DataModule for VLA training.

Provides a standardized interface for data loading with train/val/test splits,
configurable batch size, and automatic worker management.

Example:
    >>> from vla.data import VLADataModule
    >>> from pytorch_lightning import Trainer
    >>>
    >>> # Create data module
    >>> datamodule = VLADataModule(
    ...     dataset_type='dummy',
    ...     batch_size=32,
    ...     num_workers=4,
    ... )
    >>>
    >>> # Use with PyTorch Lightning Trainer
    >>> trainer = Trainer(max_epochs=10)
    >>> trainer.fit(model, datamodule=datamodule)
"""

from typing import Optional, Dict, Any

import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split

from vla.data.collate_batch_samples import vla_collate_fn
from vla.data.dummy_vla_dataset import DummyVLADataset
from vla.data.transform_wrapper import TransformWrapper
from vla.utils import setup_logger

# LeRobotVLADataset is imported lazily inside setup() to avoid requiring
# the `lerobot` package when only using the dummy dataset.

logger = setup_logger(__name__)


class VLADataModule(LightningDataModule):
    """PyTorch Lightning DataModule for VLA datasets.

    Handles data loading, splitting, and batch preparation for VLA training.
    Supports dummy datasets for testing and can be extended for real datasets.

    Args:
        dataset_type: Type of dataset ('dummy' or 'lerobot')
        batch_size: Batch size for training and validation
        num_workers: Number of DataLoader worker processes
        train_split: Fraction of data for training (rest goes to validation)
        image_size: Image spatial dimensions (H, W)
        action_dim: Dimensionality of action vectors. If None, uses all dims from dataset.
        seed: Random seed for reproducibility
        repo_id: HuggingFace repo ID (only used when dataset_type='lerobot')
        image_key: Camera key override (only used when dataset_type='lerobot')
        tokenizer_name: HuggingFace tokenizer name for CPU tokenization in workers
        max_token_length: Maximum token sequence length (default 77, CLIP-style)
        use_tokenized_collate: If True, tokenize text in DataLoader workers for
            15-30% throughput improvement by eliminating CPU-GPU sync stalls

    Attributes:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset

    Example:
        >>> datamodule = VLADataModule(dataset_type='dummy', batch_size=16)
        >>> datamodule.setup('fit')
        >>> train_loader = datamodule.train_dataloader()
        >>> val_loader = datamodule.val_dataloader()
    """

    def __init__(
        self,
        dataset_type: str = "dummy",
        batch_size: int = 32,
        num_workers: int = 4,
        train_split: float = 0.8,
        image_size: tuple[int, int] = (224, 224),
        action_dim: Optional[int] = None,
        seed: int = 42,
        total_samples: int = 1000,
        # LeRobot-specific params (only used when dataset_type="lerobot")
        repo_id: str = "lerobot/pusht",
        image_key: Optional[str] = None,
        # Tokenization params for CPU tokenization in DataLoader workers
        tokenizer_name: str = "gpt2",
        max_token_length: int = 77,
        use_tokenized_collate: bool = True,
    ):
        super().__init__()
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.image_size = image_size
        self.action_dim = action_dim
        self.seed = seed
        self.total_samples = total_samples
        self.repo_id = repo_id
        self.image_key = image_key
        self.tokenizer_name = tokenizer_name
        self.max_token_length = max_token_length
        self.use_tokenized_collate = use_tokenized_collate

        # Dataset placeholders (set in setup())
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

        # Collate function (set in setup() after tokenizer is built)
        self._collate_fn = None

        logger.info(
            f"Initialized VLADataModule: type={dataset_type}, "
            f"batch_size={batch_size}, num_workers={num_workers}, "
            f"use_tokenized_collate={use_tokenized_collate}"
        )

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training, validation, or testing.

        Args:
            stage: 'fit' for train/val, 'test' for test, None for all

        Raises:
            ValueError: If dataset_type is not supported
        """
        # Build collate_fn once — tokenizer is picklable so safe for workers
        if self._collate_fn is None:
            if self.use_tokenized_collate:
                from transformers import AutoTokenizer
                from vla.data.collate_batch_samples import make_tokenized_collate_fn

                tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                self._collate_fn = make_tokenized_collate_fn(tokenizer, self.max_token_length)
                logger.info(
                    f"Using tokenized collate_fn: tokenizer={self.tokenizer_name}, "
                    f"max_length={self.max_token_length}"
                )
            else:
                self._collate_fn = vla_collate_fn

        if stage == "fit" or stage is None:
            if self.dataset_type == "dummy":
                # Calculate split sizes
                total_samples = self.total_samples
                train_size = int(total_samples * self.train_split)
                val_size = total_samples - train_size

                logger.info(f"Creating dummy datasets: train={train_size}, val={val_size}")

                # Create train dataset
                self.train_dataset = DummyVLADataset(
                    num_samples=train_size,
                    image_size=self.image_size,
                    action_dim=self.action_dim or 7,
                    seed=self.seed,
                )

                # Create validation dataset with different seed
                self.val_dataset = DummyVLADataset(
                    num_samples=val_size,
                    image_size=self.image_size,
                    action_dim=self.action_dim or 7,
                    seed=self.seed + 1000,  # Different seed for val
                )

            elif self.dataset_type == "lerobot":
                from vla.data.lerobot_dataset import LeRobotVLADataset

                full_dataset = LeRobotVLADataset(
                    repo_id=self.repo_id,
                    image_key=self.image_key,
                    action_dim=self.action_dim,
                    image_size=self.image_size,
                    split="train",
                )
                # Split train dataset into train/val because many LeRobot
                # datasets only have a "train" split (no "test" split).
                total = len(full_dataset)
                train_size = int(total * self.train_split)
                val_size = total - train_size
                train_subset, val_subset = random_split(
                    full_dataset,
                    [train_size, val_size],
                    generator=torch.Generator().manual_seed(self.seed),
                )

                # Apply training augmentation ONLY to the train split
                train_transform = transforms.Compose([
                    transforms.RandomResizedCrop(
                        self.image_size,
                        scale=(0.75, 1.0),
                        interpolation=transforms.InterpolationMode.BICUBIC,
                        antialias=True
                    ),
                    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)
                ])

                def apply_augmentation(sample: Dict[str, Any]) -> Dict[str, Any]:
                    # Augment image(s)
                    if "images" in sample:
                        sample["images"] = train_transform(sample["images"])
                    # Augment temporal sequence if present
                    if "image_sequence" in sample:
                        sample["image_sequence"] = [train_transform(img) for img in sample["image_sequence"]]
                    return sample

                self.train_dataset = TransformWrapper(train_subset, apply_augmentation)
                self.val_dataset = val_subset

                logger.info(
                    f"LeRobot datasets loaded: repo={self.repo_id}, "
                    f"train={train_size} (with augmentation), val={val_size}, "
                    f"image_key={full_dataset.image_key}, "
                    f"action_dim={full_dataset.action_dim}"
                )

            else:
                raise ValueError(
                    f"Unsupported dataset_type: {self.dataset_type}. "
                    f"Supported types: 'dummy', 'lerobot'."
                )

        if stage == "test" or stage is None:
            if self.dataset_type == "dummy":
                # Mirror the val split: use the non-training fraction of total_samples
                # so test_size scales with the user's total_samples setting.
                test_size = self.total_samples - int(self.total_samples * self.train_split)
                logger.info(f"Creating dummy test dataset: test={test_size}")

                self.test_dataset = DummyVLADataset(
                    num_samples=test_size,
                    image_size=self.image_size,
                    action_dim=self.action_dim or 7,
                    seed=self.seed + 2000,  # Different seed for test
                )
            elif self.dataset_type == "lerobot":
                from vla.data.lerobot_dataset import LeRobotVLADataset

                self.test_dataset = LeRobotVLADataset(
                    repo_id=self.repo_id,
                    image_key=self.image_key,
                    action_dim=self.action_dim,
                    image_size=self.image_size,
                    split="train",
                )
            else:
                raise ValueError(f"Unsupported dataset_type: {self.dataset_type}")

    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader.

        Returns:
            DataLoader for training with shuffling enabled

        Example:
            >>> datamodule = VLADataModule()
            >>> datamodule.setup('fit')
            >>> loader = datamodule.train_dataloader()
            >>> batch = next(iter(loader))
        """
        if self.train_dataset is None:
            raise RuntimeError("Call setup('fit') before accessing train_dataloader")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn or vla_collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation DataLoader.

        Returns:
            DataLoader for validation (no shuffling)

        Example:
            >>> datamodule = VLADataModule()
            >>> datamodule.setup('fit')
            >>> loader = datamodule.val_dataloader()
        """
        if self.val_dataset is None:
            raise RuntimeError("Call setup('fit') before accessing val_dataloader")

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn or vla_collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        """Create test DataLoader.

        Returns:
            DataLoader for testing (no shuffling)

        Example:
            >>> datamodule = VLADataModule()
            >>> datamodule.setup('test')
            >>> loader = datamodule.test_dataloader()
        """
        if self.test_dataset is None:
            raise RuntimeError("Call setup('test') before accessing test_dataloader")

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn or vla_collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def teardown(self, stage: Optional[str] = None):
        """Cleanup after training/testing.

        Args:
            stage: Stage being torn down ('fit', 'test', or None)
        """
        # Release dataset references
        if stage == "fit" or stage is None:
            self.train_dataset = None
            self.val_dataset = None
        if stage == "test" or stage is None:
            self.test_dataset = None

        logger.info(f"Teardown completed for stage: {stage}")
