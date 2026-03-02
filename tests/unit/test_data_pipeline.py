"""Unit tests for VLA data pipeline.

Tests cover:
- DummyVLADataset: sample generation, reproducibility
- DummyTemporalVLADataset: temporal sequences
- vla_collate_fn: batch collation
- temporal_vla_collate_fn: temporal batch collation
- VLADataModule: Lightning DataModule interface
- LeRobotVLADataset: LeRobot format adapter (mocked)
"""

import types
from typing import Dict
from unittest.mock import MagicMock, patch

import pytest
import torch
import sys
from unittest.mock import MagicMock

# Mock lerobot package for testing
lerobot_mock = MagicMock()
sys.modules["lerobot"] = lerobot_mock
sys.modules["lerobot.datasets"] = MagicMock()
sys.modules["lerobot.datasets.lerobot_dataset"] = MagicMock()
sys.modules["lerobot.datasets.lerobot_dataset"].LeRobotDataset = MagicMock


from torch.utils.data import DataLoader

from vla.data import DummyVLADataset, VLADataModule, vla_collate_fn
from vla.data.collate_batch_samples import temporal_vla_collate_fn
from vla.data.dummy_vla_dataset import DummyTemporalVLADataset


class TestDummyVLADataset:
    """Tests for DummyVLADataset."""

    def test_dataset_length(self):
        """Test dataset reports correct length."""
        dataset = DummyVLADataset(num_samples=100)
        assert len(dataset) == 100

    def test_sample_structure(self):
        """Test sample contains required keys."""
        dataset = DummyVLADataset(num_samples=10)
        sample = dataset[0]

        assert "image" in sample
        assert "text" in sample
        assert "action" in sample

    def test_image_shape(self):
        """Test image has correct shape."""
        dataset = DummyVLADataset(num_samples=10, image_size=(224, 224))
        sample = dataset[0]

        assert sample["image"].shape == (3, 224, 224)
        assert sample["image"].dtype == torch.float32

    def test_image_value_range(self):
        """Test image values are in [0, 1]."""
        dataset = DummyVLADataset(num_samples=10)
        sample = dataset[0]

        assert sample["image"].min() >= 0.0
        assert sample["image"].max() <= 1.0

    def test_text_is_string(self):
        """Test text is a string."""
        dataset = DummyVLADataset(num_samples=10)
        sample = dataset[0]

        assert isinstance(sample["text"], str)
        assert len(sample["text"]) > 0

    def test_action_shape(self):
        """Test action has correct shape."""
        dataset = DummyVLADataset(num_samples=10, action_dim=7)
        sample = dataset[0]

        assert sample["action"].shape == (7,)
        assert sample["action"].dtype == torch.float32

    def test_action_value_range(self):
        """Test action values are in [-1, 1]."""
        dataset = DummyVLADataset(num_samples=10)
        sample = dataset[0]

        assert sample["action"].min() >= -1.0
        assert sample["action"].max() <= 1.0

    def test_reproducibility_same_index(self):
        """Test same index returns same sample."""
        dataset = DummyVLADataset(num_samples=10, seed=42)

        sample1 = dataset[5]
        sample2 = dataset[5]

        assert torch.allclose(sample1["image"], sample2["image"])
        assert sample1["text"] == sample2["text"]
        assert torch.allclose(sample1["action"], sample2["action"])

    def test_reproducibility_same_seed(self):
        """Test same seed produces same dataset."""
        dataset1 = DummyVLADataset(num_samples=10, seed=42)
        dataset2 = DummyVLADataset(num_samples=10, seed=42)

        sample1 = dataset1[3]
        sample2 = dataset2[3]

        assert torch.allclose(sample1["image"], sample2["image"])
        assert sample1["text"] == sample2["text"]
        assert torch.allclose(sample1["action"], sample2["action"])

    def test_different_seed_different_data(self):
        """Test different seeds produce different data."""
        dataset1 = DummyVLADataset(num_samples=10, seed=42)
        dataset2 = DummyVLADataset(num_samples=10, seed=99)

        sample1 = dataset1[0]
        sample2 = dataset2[0]

        # Same text (cyclic) but different images/actions
        assert not torch.allclose(sample1["image"], sample2["image"])
        assert not torch.allclose(sample1["action"], sample2["action"])

    def test_custom_image_size(self):
        """Test custom image size."""
        dataset = DummyVLADataset(num_samples=10, image_size=(128, 128))
        sample = dataset[0]

        assert sample["image"].shape == (3, 128, 128)

    def test_custom_action_dim(self):
        """Test custom action dimension."""
        dataset = DummyVLADataset(num_samples=10, action_dim=14)
        sample = dataset[0]

        assert sample["action"].shape == (14,)

    def test_get_batch(self):
        """Test get_batch convenience method."""
        dataset = DummyVLADataset(num_samples=10)
        batch = dataset.get_batch(batch_size=4)

        assert batch["images"].shape == (4, 3, 224, 224)
        assert len(batch["texts"]) == 4
        assert batch["actions"].shape == (4, 7)


class TestDummyTemporalVLADataset:
    """Tests for DummyTemporalVLADataset."""

    def test_temporal_sample_structure(self):
        """Test temporal sample contains required keys."""
        dataset = DummyTemporalVLADataset(num_samples=10, num_frames=6)
        sample = dataset[0]

        assert "image_sequence" in sample
        assert "text" in sample
        assert "action" in sample

    def test_temporal_sequence_length(self):
        """Test image sequence has correct number of frames."""
        dataset = DummyTemporalVLADataset(num_samples=10, num_frames=6)
        sample = dataset[0]

        assert len(sample["image_sequence"]) == 6

    def test_temporal_frame_shape(self):
        """Test each frame has correct shape."""
        dataset = DummyTemporalVLADataset(num_samples=10, num_frames=6)
        sample = dataset[0]

        for frame in sample["image_sequence"]:
            assert frame.shape == (3, 224, 224)
            assert frame.dtype == torch.float32

    def test_temporal_frame_value_range(self):
        """Test each frame values are in [0, 1]."""
        dataset = DummyTemporalVLADataset(num_samples=10, num_frames=6)
        sample = dataset[0]

        for frame in sample["image_sequence"]:
            assert frame.min() >= 0.0
            assert frame.max() <= 1.0

    def test_temporal_custom_num_frames(self):
        """Test custom number of frames."""
        dataset = DummyTemporalVLADataset(num_samples=10, num_frames=12)
        sample = dataset[0]

        assert len(sample["image_sequence"]) == 12


class TestVLACollateFn:
    """Tests for vla_collate_fn."""

    def test_collate_stacks_images(self):
        """Test images are stacked correctly."""
        dataset = DummyVLADataset(num_samples=10)
        samples = [dataset[i] for i in range(4)]
        batch = vla_collate_fn(samples)

        assert batch["images"].shape == (4, 3, 224, 224)

    def test_collate_collects_texts(self):
        """Test texts are collected into list."""
        dataset = DummyVLADataset(num_samples=10)
        samples = [dataset[i] for i in range(4)]
        batch = vla_collate_fn(samples)

        assert len(batch["texts"]) == 4
        assert all(isinstance(text, str) for text in batch["texts"])

    def test_collate_stacks_actions(self):
        """Test actions are stacked correctly."""
        dataset = DummyVLADataset(num_samples=10)
        samples = [dataset[i] for i in range(4)]
        batch = vla_collate_fn(samples)

        assert batch["actions"].shape == (4, 7)

    def test_collate_single_sample(self):
        """Test collate works with single sample."""
        dataset = DummyVLADataset(num_samples=10)
        samples = [dataset[0]]
        batch = vla_collate_fn(samples)

        assert batch["images"].shape == (1, 3, 224, 224)
        assert len(batch["texts"]) == 1
        assert batch["actions"].shape == (1, 7)

    def test_collate_with_dataloader(self):
        """Test collate function works with DataLoader."""
        dataset = DummyVLADataset(num_samples=20)
        loader = DataLoader(dataset, batch_size=8, collate_fn=vla_collate_fn)

        batch = next(iter(loader))
        assert batch["images"].shape == (8, 3, 224, 224)
        assert len(batch["texts"]) == 8
        assert batch["actions"].shape == (8, 7)


class TestTemporalVLACollateFn:
    """Tests for temporal_vla_collate_fn."""

    def test_temporal_collate_structure(self):
        """Test temporal batch has correct structure."""
        dataset = DummyTemporalVLADataset(num_samples=10, num_frames=6)
        samples = [dataset[i] for i in range(2)]
        batch = temporal_vla_collate_fn(samples)

        assert "image_sequences" in batch
        assert "texts" in batch
        assert "actions" in batch

    def test_temporal_collate_frame_count(self):
        """Test temporal batch has correct number of frames."""
        dataset = DummyTemporalVLADataset(num_samples=10, num_frames=6)
        samples = [dataset[i] for i in range(2)]
        batch = temporal_vla_collate_fn(samples)

        assert len(batch["image_sequences"]) == 6

    def test_temporal_collate_frame_shape(self):
        """Test each frame in batch has correct shape."""
        dataset = DummyTemporalVLADataset(num_samples=10, num_frames=6)
        samples = [dataset[i] for i in range(2)]
        batch = temporal_vla_collate_fn(samples)

        for frame_batch in batch["image_sequences"]:
            assert frame_batch.shape == (2, 3, 224, 224)

    def test_temporal_collate_actions(self):
        """Test actions are stacked correctly."""
        dataset = DummyTemporalVLADataset(num_samples=10, num_frames=6)
        samples = [dataset[i] for i in range(3)]
        batch = temporal_vla_collate_fn(samples)

        assert batch["actions"].shape == (3, 7)


class TestVLADataModule:
    """Tests for VLADataModule."""

    def test_datamodule_initialization(self):
        """Test DataModule initializes correctly."""
        datamodule = VLADataModule(
            dataset_type="dummy",
            batch_size=16,
            num_workers=0,  # Use 0 for testing
        )
        assert datamodule.batch_size == 16
        assert datamodule.dataset_type == "dummy"

    def test_datamodule_setup_fit(self):
        """Test setup creates train and val datasets."""
        datamodule = VLADataModule(dataset_type="dummy", num_workers=0)
        datamodule.setup("fit")

        assert datamodule.train_dataset is not None
        assert datamodule.val_dataset is not None
        assert len(datamodule.train_dataset) > 0
        assert len(datamodule.val_dataset) > 0

    def test_datamodule_setup_test(self):
        """Test setup creates test dataset."""
        datamodule = VLADataModule(dataset_type="dummy", num_workers=0)
        datamodule.setup("test")

        assert datamodule.test_dataset is not None
        assert len(datamodule.test_dataset) > 0

    def test_train_dataloader(self):
        """Test train_dataloader returns valid DataLoader."""
        datamodule = VLADataModule(dataset_type="dummy", batch_size=8, num_workers=0)
        datamodule.setup("fit")

        train_loader = datamodule.train_dataloader()
        batch = next(iter(train_loader))

        assert batch["images"].shape[0] == 8  # Batch size
        assert len(batch["texts"]) == 8
        assert batch["actions"].shape[0] == 8

    def test_val_dataloader(self):
        """Test val_dataloader returns valid DataLoader."""
        datamodule = VLADataModule(dataset_type="dummy", batch_size=8, num_workers=0)
        datamodule.setup("fit")

        val_loader = datamodule.val_dataloader()
        batch = next(iter(val_loader))

        assert batch["images"].shape[0] == 8
        assert len(batch["texts"]) == 8
        assert batch["actions"].shape[0] == 8

    def test_test_dataloader(self):
        """Test test_dataloader returns valid DataLoader."""
        datamodule = VLADataModule(dataset_type="dummy", batch_size=8, num_workers=0)
        datamodule.setup("test")

        test_loader = datamodule.test_dataloader()
        batch = next(iter(test_loader))

        assert batch["images"].shape[0] == 8

    def test_datamodule_without_setup_raises(self):
        """Test accessing dataloaders without setup raises error."""
        datamodule = VLADataModule(dataset_type="dummy", num_workers=0)

        with pytest.raises(RuntimeError, match="Call setup"):
            datamodule.train_dataloader()

    def test_unsupported_dataset_type(self):
        """Test unsupported dataset type raises error."""
        datamodule = VLADataModule(dataset_type="unsupported", num_workers=0)

        with pytest.raises(ValueError, match="Unsupported dataset_type"):
            datamodule.setup("fit")

    def test_train_val_split_ratios(self):
        """Test train/val split ratios are respected."""
        datamodule = VLADataModule(
            dataset_type="dummy",
            train_split=0.8,
            num_workers=0,
        )
        datamodule.setup("fit")

        train_size = len(datamodule.train_dataset)
        val_size = len(datamodule.val_dataset)
        total_size = train_size + val_size

        # Check split is approximately correct (within 1% tolerance)
        train_ratio = train_size / total_size
        assert abs(train_ratio - 0.8) < 0.01

    def test_datamodule_teardown(self):
        """Test teardown cleans up datasets."""
        datamodule = VLADataModule(dataset_type="dummy", num_workers=0)
        datamodule.setup("fit")

        assert datamodule.train_dataset is not None
        datamodule.teardown("fit")
        assert datamodule.train_dataset is None


# ---------------------------------------------------------------------------
# Helpers for LeRobot tests — build a fake LeRobotDataset without real data
# ---------------------------------------------------------------------------


def _make_fake_lerobot_module() -> types.ModuleType:
    """Create a fake `lerobot` package so imports don't fail."""
    lerobot_mod = types.ModuleType("lerobot")
    common_mod = types.ModuleType("lerobot.common")
    datasets_mod = types.ModuleType("lerobot.common.datasets")
    ds_mod = types.ModuleType("lerobot.common.datasets.lerobot_dataset")
    lerobot_mod.common = common_mod
    common_mod.datasets = datasets_mod
    datasets_mod.lerobot_dataset = ds_mod
    return lerobot_mod, ds_mod


def _make_mock_lerobot_ds(
    num_samples: int = 10,
    action_dim: int = 2,
    task_descriptions: Dict[int, str] | None = None,
) -> MagicMock:
    """Build a MagicMock that mimics LeRobotDataset's interface."""
    import torch

    if task_descriptions is None:
        task_descriptions = {0: "push the T block to the target zone"}

    # Build fake samples: each has image tensor, action tensor, task_index
    fake_image = torch.rand(3, 96, 96)  # raw camera resolution
    fake_action = torch.tensor([0.5, -0.3] + [0.0] * (action_dim - 2), dtype=torch.float32)

    def getitem(idx: int) -> Dict:
        return {
            "observation.images.top": fake_image.clone(),
            "action": fake_action.clone(),
            "task_index": torch.tensor(0),
        }

    mock_ds = MagicMock()
    mock_ds.__len__ = MagicMock(return_value=num_samples)
    mock_ds.__getitem__ = MagicMock(side_effect=getitem)
    mock_ds.features = {
        "observation.images.top": {"shape": [3, 96, 96]},
        "action": {"shape": [action_dim]},
    }
    mock_ds.tasks = task_descriptions
    mock_ds.stats = {
        "action": {
            "mean": [0.0] * action_dim,
            "std": [1.0] * action_dim,
        }
    }
    return mock_ds


class TestLeRobotVLADataset:
    """Tests for LeRobotVLADataset (uses mocked LeRobotDataset)."""

    def _make_adapter(self, **kwargs):
        """Instantiate the adapter with a fully mocked lerobot dependency."""
        import sys

        lerobot_mod, ds_mod = _make_fake_lerobot_module()
        mock_inner = _make_mock_lerobot_ds(**kwargs)
        ds_mod.LeRobotDataset = MagicMock(return_value=mock_inner)

        # Inject mock modules so `import lerobot` succeeds inside the adapter
        with patch.dict(
            sys.modules,
            {
                "lerobot": lerobot_mod,
                "lerobot.common": lerobot_mod.common,
                "lerobot.datasets": lerobot_mod.datasets,
                "lerobot.datasets.lerobot_dataset": ds_mod,
            },
        ):
            from vla.data.lerobot_dataset import LeRobotVLADataset

            adapter = LeRobotVLADataset(repo_id="lerobot/pusht")
        return adapter, mock_inner

    def test_lerobot_dataset_init_with_mock(self):
        """Adapter initializes and maps samples correctly."""
        adapter, _ = self._make_adapter()

        assert len(adapter) == 10
        sample = adapter[0]
        assert set(sample.keys()) == {"image", "text", "action"}

    def test_lerobot_image_resize(self):
        """Output image is always resized to [3, 224, 224]."""
        adapter, _ = self._make_adapter()

        sample = adapter[0]
        # Raw image was 96×96; adapter must resize to (224, 224)
        assert sample["image"].shape == (3, 224, 224)
        assert sample["image"].dtype == import_torch().float32

    def test_lerobot_image_value_range(self):
        """Image is float32, finite, and within ImageNet-normalized range.

        After ImageNet normalization (required for DINOv2/SigLIP), values are
        outside [0, 1]. Typical range: ~[-2.5, 2.7].
        """
        adapter, _ = self._make_adapter()

        img = adapter[0]["image"]
        assert img.dtype == import_torch().float32
        assert import_torch().isfinite(img).all()
        # ImageNet-normalized range (approx): [-2.5, 2.7]
        assert img.min() >= -3.0
        assert img.max() <= 3.0

    def test_lerobot_text_lookup(self):
        """task_index is mapped to the correct task description string."""
        tasks = {0: "push the T block to the target zone", 1: "place object on shelf"}
        adapter, _ = self._make_adapter(task_descriptions=tasks)

        text = adapter[0]["text"]
        assert text == "push the T block to the target zone"
        assert isinstance(text, str)

    def test_lerobot_action_normalization(self):
        """Normalized actions are clamped to [-1, 1]."""
        adapter, mock_inner = self._make_adapter(action_dim=2)

        # Override stats to force extreme raw values that need clamping
        mock_inner.stats = {"action": {"mean": [100.0, 100.0], "std": [1.0, 1.0]}}
        # Reload normalization stats on the already-built adapter
        adapter._action_mean, adapter._action_std = adapter._load_action_stats()

        action = adapter[0]["action"]
        assert action.shape == (2,)
        assert action.min() >= -1.0
        assert action.max() <= 1.0

    def test_lerobot_action_dim_slice(self):
        """action_dim parameter correctly slices the action tensor."""
        import sys

        lerobot_mod, ds_mod = _make_fake_lerobot_module()
        mock_inner = _make_mock_lerobot_ds(action_dim=6)
        ds_mod.LeRobotDataset = MagicMock(return_value=mock_inner)

        with patch.dict(
            sys.modules,
            {
                "lerobot": lerobot_mod,
                "lerobot.common": lerobot_mod.common,
                "lerobot.datasets": lerobot_mod.datasets,
                "lerobot.datasets.lerobot_dataset": ds_mod,
            },
        ):
            from vla.data.lerobot_dataset import LeRobotVLADataset

            adapter = LeRobotVLADataset(repo_id="lerobot/pusht", action_dim=3)

        action = adapter[0]["action"]
        assert action.shape == (3,)

    def test_lerobot_missing_image_key_raises(self):
        """Accessing a non-existent image key raises KeyError."""
        import sys

        lerobot_mod, ds_mod = _make_fake_lerobot_module()
        mock_inner = _make_mock_lerobot_ds()
        ds_mod.LeRobotDataset = MagicMock(return_value=mock_inner)

        with patch.dict(
            sys.modules,
            {
                "lerobot": lerobot_mod,
                "lerobot.common": lerobot_mod.common,
                "lerobot.datasets": lerobot_mod.datasets,
                "lerobot.datasets.lerobot_dataset": ds_mod,
            },
        ):
            from vla.data.lerobot_dataset import LeRobotVLADataset

            # Force wrong image_key after init to trigger per-sample error
            adapter = LeRobotVLADataset(repo_id="lerobot/pusht")
            adapter.image_key = "observation.images.nonexistent"

        with pytest.raises(KeyError, match="nonexistent"):
            _ = adapter[0]


class TestLeRobotStateExtraction:
    """Tests for optional observation.state extraction in LeRobotVLADataset."""

    def test_process_state_normalizes_to_minus_one_one(self):
        """_process_state clamps output to [-1, 1] using fixed-scale fallback."""
        from vla.data.lerobot_dataset import LeRobotVLADataset
        # Bypass __init__ — set required attributes manually
        ds = LeRobotVLADataset.__new__(LeRobotVLADataset)
        ds._state_mean = None  # forces fixed-scale normalization path
        ds._state_std = None
        result = ds._process_state({"observation.state": torch.tensor([0.0, 512.0, 3.14])})
        assert result is not None
        assert result.min() >= -1.0
        assert result.max() <= 1.0

    def test_process_state_returns_none_when_missing(self):
        """_process_state returns None when observation.state key absent."""
        from vla.data.lerobot_dataset import LeRobotVLADataset
        ds = LeRobotVLADataset.__new__(LeRobotVLADataset)
        ds._state_mean = None
        ds._state_std = None
        result = ds._process_state({})  # no observation.state key
        assert result is None

    def test_collate_fn_stacks_states(self):
        """vla_collate_fn produces 'states' key when all samples have 'state'."""
        from vla.data.collate_batch_samples import vla_collate_fn
        samples = [
            {"image": torch.rand(3, 224, 224), "text": "t", "action": torch.rand(2),
             "state": torch.tensor([0.1, 0.2, 0.3])},
            {"image": torch.rand(3, 224, 224), "text": "t", "action": torch.rand(2),
             "state": torch.tensor([0.4, 0.5, 0.6])},
        ]
        batch = vla_collate_fn(samples)
        assert "states" in batch
        assert batch["states"].shape == (2, 3)

    def test_collate_fn_omits_states_when_absent(self):
        """vla_collate_fn omits 'states' key when samples lack 'state'."""
        from vla.data.collate_batch_samples import vla_collate_fn
        samples = [
            {"image": torch.rand(3, 224, 224), "text": "t", "action": torch.rand(2)},
            {"image": torch.rand(3, 224, 224), "text": "t", "action": torch.rand(2)},
        ]
        batch = vla_collate_fn(samples)
        assert "states" not in batch


def import_torch():
    """Small helper to import torch (avoids top-level import duplication)."""
    import torch

    return torch
