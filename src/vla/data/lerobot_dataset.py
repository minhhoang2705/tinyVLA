"""LeRobot dataset adapter for tinyVLA.

Bridges HuggingFace LeRobot format to tinyVLA's expected sample format.

LeRobot format uses:
- Parquet files for states/actions
- MP4 video files for camera frames
- meta/info.json: schema, shapes, FPS
- meta/stats.json: normalization stats (mean/std/min/max)
- meta/tasks.jsonl: task descriptions → int IDs

tinyVLA expects samples with keys:
- "image": torch.Tensor [3, H, W], float32 in [0, 1]
- "text": str — the task instruction
- "action": torch.Tensor [action_dim], float32 in [-1, 1]

Installation:
    pip install lerobot>=0.4.0
    # Or from source: pip install git+https://github.com/huggingface/lerobot.git

Example:
    >>> from vla.data.lerobot_dataset import LeRobotVLADataset
    >>> dataset = LeRobotVLADataset(repo_id="lerobot/pusht")
    >>> sample = dataset[0]
    >>> print(sample["image"].shape)   # torch.Size([3, 224, 224])
    >>> print(sample["text"])          # "push the T block..."
    >>> print(sample["action"].shape)  # torch.Size([2])
"""

from typing import Any, Dict, List, Optional, Tuple, Union, cast

import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from vla.utils import setup_logger

logger = setup_logger(__name__)


class LeRobotVLADataset(Dataset):
    """Adapter that wraps LeRobotDataset to match tinyVLA's sample format.

    Loads a robotics dataset from HuggingFace using the LeRobot format and
    converts each sample to the dict format expected by tinyVLA models:
    {"image": Tensor[3,H,W], "text": str, "action": Tensor[action_dim]}.

    Args:
        repo_id: HuggingFace dataset repo ID (e.g. "lerobot/pusht").
        image_key: Camera key to use (e.g. "observation.images.top").
            If None, auto-detects the first available camera.
        action_dim: Number of action dimensions to use. If None, uses
            all dimensions from the dataset.
        image_size: Target image size (H, W). Default: (224, 224).
        split: Dataset split to load. Default: "train".
        normalize_actions: If True, normalizes actions to [-1, 1] using
            stats.json mean/std. Default: True.
        root: Local path if dataset is already downloaded. If None,
            downloads from HuggingFace Hub.

    Raises:
        ImportError: If the `lerobot` package is not installed.
        KeyError: If the specified image_key is not found in the dataset.
        ValueError: If action_dim exceeds the dataset's native action dim.

    Example:
        >>> dataset = LeRobotVLADataset(
        ...     repo_id="lerobot/pusht",
        ...     image_size=(224, 224),
        ...     normalize_actions=True,
        ... )
        >>> sample = dataset[0]
        >>> sample["image"].shape
        torch.Size([3, 224, 224])
        >>> sample["action"].shape
        torch.Size([2])
    """

    def __init__(
        self,
        repo_id: str,
        image_key: Optional[str] = None,
        action_dim: Optional[int] = None,
        image_size: Tuple[int, int] = (224, 224),
        split: str = "train",
        normalize_actions: bool = True,
        include_state: bool = True,
        root: Optional[str] = None,
    ):
        super().__init__()
        self._import_lerobot()

        self.repo_id = repo_id
        self.image_size = image_size
        self.split = split
        self.normalize_actions = normalize_actions

        # Load the underlying LeRobot dataset
        self._lerobot_ds: Any = self._load_dataset(repo_id, split, root)

        # Resolve which image key to use
        self.image_key = self._resolve_image_key(image_key)

        # Build task_index → description lookup from the dataset
        self._task_lookup: Dict[int, str] = self._build_task_lookup()

        # Resolve action dimension (cap if requested)
        native_action_dim = self._get_native_action_dim()
        if action_dim is not None and action_dim > native_action_dim:
            raise ValueError(
                f"Requested action_dim={action_dim} exceeds dataset's "
                f"native action_dim={native_action_dim} for {repo_id}"
            )
        self.action_dim = action_dim if action_dim is not None else native_action_dim

        # Pre-load normalization stats (mean/std for actions)
        self._action_mean, self._action_std = self._load_action_stats()

        # State observation support (optional — absent in most datasets)
        self.include_state = include_state
        self._has_state: bool = include_state and self._detect_state_key()
        self._state_mean: Optional[torch.Tensor] = None
        self._state_std: Optional[torch.Tensor] = None
        if self._has_state:
            self._state_mean, self._state_std = self._load_state_stats()

        logger.info(
            f"LeRobotVLADataset loaded: repo={repo_id}, split={split}, "
            f"len={len(self)}, image_key={self.image_key}, "
            f"action_dim={self.action_dim}, tasks={len(self._task_lookup)}"
        )

    def _import_lerobot(self) -> None:
        """Verify lerobot package is installed."""
        try:
            import lerobot  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "The `lerobot` package is required. Install with:\n"
                "  pip install lerobot>=0.4.0\n"
                "Or from source:\n"
                "  pip install git+https://github.com/huggingface/lerobot.git"
            ) from e

    def _load_dataset(self, repo_id: str, split: str, root: Optional[str]) -> Any:
        """Load the LeRobotDataset from HuggingFace or local path."""
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        kwargs: Dict[str, Any] = {}
        if root is not None:
            kwargs["root"] = root

        logger.info(f"Loading LeRobotDataset: {repo_id}")
        return LeRobotDataset(repo_id, **kwargs)

    def _resolve_image_key(self, image_key: Optional[str]) -> str:
        """Find the camera key to use; auto-detect if not specified."""
        # LeRobot exposes camera keys in meta_data or features
        available = self._get_available_image_keys()
        if not available:
            raise KeyError(
                f"No image keys found in dataset {self.repo_id}. "
                f"Expected keys like 'observation.images.<cam>'."
            )

        if image_key is None:
            # Auto-detect: use the first available camera
            resolved = available[0]
            logger.info(f"Auto-detected image key: '{resolved}' from {available}")
            return resolved

        if image_key not in available:
            raise KeyError(f"Image key '{image_key}' not found. " f"Available keys: {available}")
        return image_key

    def _get_available_image_keys(self) -> List[str]:
        """Extract available image/camera keys from dataset features."""
        # LeRobotDataset exposes features dict with data types
        features = getattr(self._lerobot_ds, "features", {})
        return [k for k in features if k.startswith("observation.images")]

    def _build_task_lookup(self) -> Dict[int, str]:
        """Build int → task description map from dataset metadata."""
        # LeRobotDataset stores tasks in meta_data.tasks or tasks attribute
        lookup: Dict[int, str] = {}

        # Try tasks attribute (dict or list depending on lerobot version)
        tasks = getattr(self._lerobot_ds, "tasks", None)
        if tasks is not None:
            if isinstance(tasks, dict):
                # Format: {task_index: "description"}
                lookup = {int(k): str(v) for k, v in tasks.items()}
            elif isinstance(tasks, list):
                # Format: [{"task_index": 0, "task": "description"}, ...]
                for entry in tasks:
                    idx = int(entry.get("task_index", entry.get("index", 0)))
                    desc = str(entry.get("task", entry.get("description", "")))
                    lookup[idx] = desc

        if not lookup:
            logger.warning(
                "Could not build task lookup from dataset metadata. "
                "Text will default to 'perform the task'."
            )
        return lookup

    def _get_native_action_dim(self) -> int:
        """Get the dataset's native action dimension from features/shapes."""
        features = getattr(self._lerobot_ds, "features", {})
        if "action" in features:
            shape = features["action"].get("shape", None)
            if shape:
                return int(shape[0])

        # Fallback: check the first sample
        if len(self._lerobot_ds) > 0:
            sample = self._lerobot_ds[0]
            action = sample.get("action", None)
            if action is not None:
                return action.shape[-1] if hasattr(action, "shape") else len(action)

        raise ValueError(f"Cannot determine action dimension for {self.repo_id}")

    def _load_action_stats(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Load action mean/std from dataset stats for normalization."""
        if not self.normalize_actions:
            return None, None

        # LeRobotDataset stores stats in meta.stats (newer API) or .stats (older API).
        # Validate it's a real dict before using, to avoid MagicMock / None issues.
        stats = None
        if hasattr(self._lerobot_ds, "meta") and hasattr(self._lerobot_ds.meta, "stats"):
            candidate = self._lerobot_ds.meta.stats
            if isinstance(candidate, dict):
                stats = candidate
        if stats is None:
            stats = getattr(self._lerobot_ds, "stats", None)

        if stats is None:
            logger.warning("No stats found; actions will not be normalized.")
            return None, None

        action_stats = stats.get("action", {})
        mean = action_stats.get("mean", None)
        std = action_stats.get("std", None)

        if mean is None or std is None:
            logger.warning("action mean/std missing in stats; skipping normalization.")
            return None, None

        # Convert to tensors, keep only the requested action dims
        mean_t = torch.tensor(mean, dtype=torch.float32)[: self.action_dim]
        std_t = torch.tensor(std, dtype=torch.float32)[: self.action_dim]

        # Avoid division by zero for constant dimensions
        std_t = torch.clamp(std_t, min=1e-6)
        logger.info("Loaded action normalization stats (mean/std).")
        return mean_t, std_t

    def __len__(self) -> int:
        """Return number of samples in the dataset."""
        return len(self._lerobot_ds)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        """Get a single sample converted to tinyVLA format.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with keys:
                - "image": Float32 tensor [3, H, W] in range [0, 1].
                - "text": Task instruction string.
                - "action": Float32 tensor [action_dim] in range [-1, 1].
                - "state": Float32 tensor [state_dim] in range [-1, 1] (optional).

        Raises:
            KeyError: If image_key is missing from the sample.
        """
        raw = self._lerobot_ds[idx]

        image = self._process_image(raw)
        text = self._process_text(raw)
        action = self._process_action(raw)

        sample: Dict[str, Union[torch.Tensor, str]] = {"image": image, "text": text, "action": action}
        if self._has_state:
            state = self._process_state(raw)
            if state is not None:
                sample["state"] = state
        return sample


    def _detect_state_key(self) -> bool:
        """Check if observation.state exists in this dataset's features."""
        features = getattr(self._lerobot_ds, "features", {})
        return "observation.state" in features

    def _load_state_stats(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Load observation.state mean/std from dataset stats for normalization."""
        stats = None
        if hasattr(self._lerobot_ds, "meta") and hasattr(self._lerobot_ds.meta, "stats"):
            candidate = self._lerobot_ds.meta.stats
            if isinstance(candidate, dict):
                stats = candidate
        if stats is None:
            stats = getattr(self._lerobot_ds, "stats", None)

        if stats is None:
            logger.warning("No stats found; state will use fixed-scale normalization.")
            return None, None

        state_stats = stats.get("observation.state", {})
        mean = state_stats.get("mean", None)
        std = state_stats.get("std", None)

        if mean is None or std is None:
            logger.warning("observation.state mean/std missing; using fixed-scale normalization.")
            return None, None

        mean_t = torch.tensor(mean, dtype=torch.float32)
        std_t = torch.clamp(torch.tensor(std, dtype=torch.float32), min=1e-6)
        logger.info("Loaded observation.state normalization stats.")
        return mean_t, std_t

    def _process_state(self, raw: Dict) -> Optional[torch.Tensor]:
        """Extract and normalize observation.state to [-1, 1]."""
        state = raw.get("observation.state", None)
        if state is None:
            return None

        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        state = state.float()

        # Normalize using stats if available, else fixed scale
        if self._state_mean is not None and self._state_std is not None:
            state = (state - self._state_mean) / self._state_std
        else:
            # Fixed scale: maps pixel range [0, 512] → [-1, 1]
            state = state / 256.0 - 1.0

        return torch.clamp(state, -1.0, 1.0)

    def _process_image(self, raw: Dict) -> torch.Tensor:
        """Convert raw LeRobot image to [3, H, W] float32 in [0, 1]."""
        if self.image_key not in raw:
            raise KeyError(
                f"Image key '{self.image_key}' missing in sample. "
                f"Available keys: {list(raw.keys())}"
            )
        img = raw[self.image_key]

        # LeRobot may return PIL Image or a uint8/float tensor
        if not isinstance(img, torch.Tensor):
            # Convert PIL Image → tensor [C, H, W] in [0, 1]
            img = TF.to_tensor(img)
        elif img.dtype == torch.uint8:
            img = img.float() / 255.0

        # Ensure [C, H, W] layout (some datasets return [H, W, C])
        if img.ndim == 3 and img.shape[2] in (1, 3, 4):
            img = img.permute(2, 0, 1)

        # Resize to target image_size if needed
        h, w = img.shape[-2], img.shape[-1]
        if (h, w) != self.image_size:
            img = TF.resize(img, list(self.image_size))

        # Keep only RGB channels (drop alpha if present)
        img = cast(torch.Tensor, img[:3].float())

        # ImageNet Normalization (required for DINOv2 / SigLIP)
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return img

    def _process_text(self, raw: Dict) -> str:
        """Look up task description from task_index."""
        task_index = raw.get("task_index", 0)
        if isinstance(task_index, torch.Tensor):
            task_index = task_index.item()
        return self._task_lookup.get(int(task_index), "perform the task")

    def _process_action(self, raw: Dict) -> torch.Tensor:
        """Extract, slice, normalize, and clamp action to [-1, 1]."""
        action = raw.get("action")
        if action is None:
            raise KeyError(f"'action' key missing in LeRobot sample for {self.repo_id}")

        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32)
        action = action.float()[: self.action_dim]

        # Normalize using mean/std from stats.json
        if self._action_mean is not None and self._action_std is not None:
            action = (action - self._action_mean) / self._action_std

        # Clamp to [-1, 1] to satisfy tinyVLA interface
        return torch.clamp(action, -1.0, 1.0)
