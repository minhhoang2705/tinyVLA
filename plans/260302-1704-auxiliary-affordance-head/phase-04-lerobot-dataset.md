# Phase 04: Update LeRobot Dataset Adapter

**Status:** Pending
**Depends on:** Phase 02

## Overview

Extract `observation.state` from LeRobot samples and normalize it to `[-1, 1]` using dataset stats. The `"state"` key is **Optional** — only included when `observation.state` exists in the dataset.

## Related Code Files

- **Modify:** `src/vla/data/lerobot_dataset.py`

## State Normalization Strategy

LeRobot PushT `observation.state` = `[block_x, block_y, block_angle]`:
- `block_x`, `block_y` ∈ `[0, 512]` (pixel space)
- `block_angle` ∈ `[0, 2π]` (radians)

**Approach:** Use dataset `meta.stats["observation.state"]` mean/std (same pattern as action normalization), then clamp to `[-1, 1]`. This is dataset-agnostic and consistent with action normalization.

**Fallback:** If stats missing, apply fixed scale: `state / 256.0 - 1.0` for all dims (safe for pixel range 0-512; angle in radians ≪ 512 so will clamp).

## Implementation Steps

### 1. Add `__init__` parameters

```python
def __init__(
    self,
    repo_id: str,
    image_key: Optional[str] = None,
    action_dim: Optional[int] = None,
    image_size: Tuple[int, int] = (224, 224),
    split: str = "train",
    normalize_actions: bool = True,
    include_state: bool = True,      # ← NEW: extract observation.state if available
    root: Optional[str] = None,
):
    ...
    self.include_state = include_state
    # Detect whether observation.state exists in this dataset
    self._has_state: bool = self._detect_state_key()
    # Load normalization stats for state
    self._state_mean, self._state_std = self._load_state_stats() if self._has_state else (None, None)
```

### 2. Add `_detect_state_key()` method

```python
def _detect_state_key(self) -> bool:
    """Check if observation.state exists in this dataset's features."""
    features = getattr(self._lerobot_ds, "features", {})
    return "observation.state" in features
```

### 3. Add `_load_state_stats()` method (mirrors `_load_action_stats`)

```python
def _load_state_stats(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Load observation.state mean/std from dataset stats for normalization."""
    stats = None
    if hasattr(self._lerobot_ds, "meta") and hasattr(self._lerobot_ds.meta, "stats"):
        stats = self._lerobot_ds.meta.stats
    else:
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
```

### 4. Add `_process_state()` method

```python
def _process_state(self, raw: Dict) -> Optional[torch.Tensor]:
    """Extract and normalize observation.state to [-1, 1].

    Args:
        raw: Raw LeRobot sample dict

    Returns:
        Normalized state tensor [state_dim], or None if not available
    """
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
```

### 5. Update `__getitem__` to include state

```python
def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
    raw = self._lerobot_ds[idx]

    image = self._process_image(raw)
    text = self._process_text(raw)
    action = self._process_action(raw)

    sample: Dict[str, Union[torch.Tensor, str]] = {
        "image": image,
        "text": text,
        "action": action,
    }

    # Include state only if dataset has it and caller requested it
    if self.include_state and self._has_state:
        state = self._process_state(raw)
        if state is not None:
            sample["state"] = state

    return sample
```

## Common Pitfalls

- **Don't crash on missing state** — many datasets have no `observation.state`
- **Don't hardcode 3 dims** — state shape from dataset may vary; just use whatever is present
- **Fixed-scale fallback is imprecise for angle** — angle in radians (max ~6.28) normalizes to ~-0.975 to -0.975 range, not full [-1,1]; acceptable as fallback

## Success Criteria

- [ ] PushT dataset returns `sample["state"]` with shape `[3]` and values in `[-1, 1]`
- [ ] Dataset without `observation.state` returns sample without `"state"` key (no KeyError)
- [ ] `include_state=False` suppresses state extraction entirely
