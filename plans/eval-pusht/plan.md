# Implementation Plan: `scripts/eval_pusht.py`

**Date:** 2026-03-03
**Status:** Draft
**Source:** [Brainstorm Report](../reports/brainstorm-eval-pusht.md)
**Branch:** master
**Estimated LOC:** ~170 (single file + pyproject.toml edit)

---

## Overview

Closed-loop evaluation script for tinyVLA on the LeRobot PushT gym environment. Loads a trained checkpoint, runs N episodes in simulation, and reports success rate, average reward, and average episode length.

---

## Phase 1: Add Eval Dependencies to `pyproject.toml`

**Priority:** P0 — blocks everything else
**Files Modified:** `pyproject.toml`

### What & Why

`gymnasium` and `gym-pusht` are NOT installed in this project. Without them, the script cannot create the PushT environment. We add them under an `[eval]` optional-dependency group so they don't bloat the base install.

### Implementation Steps

1. Open `pyproject.toml`
2. Add a new `eval` group under `[project.optional-dependencies]`:

```toml
[project.optional-dependencies]
dev = [...]  # existing
eval = [
    "gymnasium>=0.29.0",
    "gym-pusht>=0.1.0",
    "imageio>=2.34.0",
    "imageio-ffmpeg>=0.4.9",
]
```

3. Install locally: `uv pip install -e ".[eval]"`

### Why These Versions

- `gymnasium>=0.29.0` — stable Gymnasium API (v0.29+ has consistent `reset(seed=)` and `step()` signatures)
- `gym-pusht>=0.1.0` — registers `gym_pusht/PushT-v0` environment
- `imageio` + `imageio-ffmpeg` — optional video saving (future extension, cheap to include now)

### Success Criteria

- `python -c "import gymnasium; import gym_pusht"` runs without error
- `pip install -e ".[eval]"` installs cleanly

---

## Phase 2: Implement `scripts/eval_pusht.py`

**Priority:** P0
**Files Created:** `scripts/eval_pusht.py`

### Architecture

```
CLI (argparse)
    │
    ├─ load_model()       → VLAModel on device
    ├─ load_action_stats() → action_mean, action_std tensors
    ├─ preprocess_image()  → mirrors LeRobotVLADataset._process_image()
    ├─ run_episodes()      → main eval loop
    └─ print_summary()     → formatted results table
```

### 2.1 CLI Interface (argparse)

**Why argparse over Hydra:** This is a standalone eval script with ~6 params. Hydra adds config-group complexity that provides no benefit here. argparse is simpler, has no side-effects (no output dirs), and makes the script self-contained.

```
Arguments:
  --checkpoint PATH      Required. Path to .ckpt or .pt file
  --num-episodes INT     Default: 50
  --seed INT             Default: 42
  --device STR           Default: "auto" (cuda if available, else cpu)
  --max-steps INT        Default: 300 (PushT default)
  --task-text STR        Default: "Push the T-shaped block onto the T-shaped target."
  --save-video           Flag. If set, save episode videos (future extension, skip in v1)
  --config-dir PATH      Default: "configs" (Hydra configs dir, needed for .ckpt loading)
  --config-name STR      Default: "config" (Hydra config entry point)
  --overrides LIST       Hydra overrides for model config (e.g., "+experiment=pusht_baseline")
```

### 2.2 Checkpoint Loading

**Why two paths:** The project has two checkpoint formats:
- `.pt` files are self-contained (config + state_dict inside the file)
- `.ckpt` files (PyTorch Lightning) excluded `model_cfg` from `save_hyperparameters`, so we must reconstruct VLAConfig from Hydra configs

```python
def load_model(args) -> VLAModel:
    path = Path(args.checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if path.suffix == ".pt":
        # Self-contained: config stored inside checkpoint
        model = VLAModel.load_checkpoint(str(path), map_location="cpu")

    elif path.suffix == ".ckpt":
        # PL checkpoint: need external config to reconstruct VLAConfig
        # Use Hydra compose API (not @hydra.main) to load config without CLI side-effects
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(config_dir=str(Path(args.config_dir).resolve()), version_base="1.3"):
            cfg = compose(config_name=args.config_name, overrides=args.overrides or [])

        model_cfg = VLAConfig.from_hydra(cfg.model)
        module = VLALightningModule.load_from_checkpoint(str(path), model_cfg=model_cfg)
        model = module.model
    else:
        raise ValueError(f"Unknown checkpoint format: {path.suffix}. Expected .ckpt or .pt")

    return model
```

**Alternative considered:** Hardcoding PushT model config values. Rejected because it couples the script to a single experiment config and breaks when architecture changes.

### 2.3 Action Denormalization (CRITICAL)

**Why z-score, NOT min/max:** During training, `LeRobotVLADataset._process_action()` (line 394-396 in `lerobot_dataset.py`) normalizes actions with:

```python
action = (action - mean) / std  # z-score normalization
```

At eval time, we must invert this EXACTLY:

```python
action_raw = action_normalized * std + mean
```

The `ActionNormalizer` class elsewhere uses min/max normalization — that is a DIFFERENT normalizer and MUST NOT be used here. Using the wrong normalizer will systematically shift all actions and produce 0% success.

**Stats source:** Load from `lerobot/pusht` dataset via `LeRobotDataset.meta.stats["action"]`. This is the same source used during training, guaranteeing consistency.

```python
def load_action_stats(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load action mean/std from lerobot/pusht dataset stats.

    Returns tensors on the specified device for efficient denormalization.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("lerobot/pusht")
    stats = ds.meta.stats
    action_mean = torch.tensor(stats["action"]["mean"], dtype=torch.float32).to(device)
    action_std = torch.tensor(stats["action"]["std"], dtype=torch.float32).to(device)
    # Guard against division by zero (same as LeRobotVLADataset)
    action_std = torch.clamp(action_std, min=1e-6)
    return action_mean, action_std
```

**Post-denorm clamp:** PushT actions are absolute positions in `[0, 512]`. We clamp after denormalization to prevent env crashes from out-of-range values.

### 2.4 Image Preprocessing

**Requirement:** Must EXACTLY match `LeRobotVLADataset._process_image()` (lines 345-375 in `lerobot_dataset.py`).

**Pipeline:**
1. `uint8 HWC [96, 96, 3]` → `float CHW [3, 96, 96]` via `permute + /255.0`
2. Resize → `[3, 224, 224]` via `TF.resize()`
3. ImageNet normalize via `TF.normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`
4. Add batch dim → `[1, 3, 224, 224]`

```python
def preprocess_obs_image(pixels: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert gym observation pixels to model-ready tensor.

    Mirrors LeRobotVLADataset._process_image() exactly.
    """
    # pixels: [96, 96, 3] uint8 from gym env
    img = torch.from_numpy(pixels).permute(2, 0, 1).float() / 255.0  # [3, 96, 96]
    img = TF.resize(img, [224, 224], antialias=True)
    img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return img.unsqueeze(0).to(device)  # [1, 3, 224, 224]
```

**Why `antialias=True`:** `TF.resize()` defaults to `antialias=True` in newer torchvision (matches training). Explicit for clarity.

**Why ImageNet normalization:** Vision backbones (DINOv2, ViT) were pretrained on ImageNet-normalized data. Skipping this produces garbage features.

### 2.5 Episode Loop

```python
def run_episodes(model, env, action_mean, action_std, args) -> List[Dict]:
    results = []

    for ep in range(args.num_episodes):
        obs, info = env.reset(seed=args.seed + ep)
        episode_reward = 0.0

        for step in range(args.max_steps):
            # Preprocess observation image
            image = preprocess_obs_image(obs["pixels"], device)

            # Model predicts normalized action in [-1, 1]
            with torch.no_grad():
                action_norm = model.predict(image, [args.task_text])  # [1, 2]

            # Denormalize: z-score inverse → raw pixel coords
            action_raw = action_norm.squeeze(0).cpu() * action_std.cpu() + action_mean.cpu()
            action_raw = action_raw.clamp(0.0, 512.0).numpy()

            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action_raw)
            episode_reward += reward

            if terminated or truncated:
                break

        # Record episode result
        success = info.get("is_success", False)
        results.append({
            "episode": ep + 1,
            "success": success,
            "reward": episode_reward / (step + 1),  # avg reward per step
            "steps": step + 1,
        })

        # Log per-episode progress
        status = "SUCCESS" if success else "FAIL"
        logger.info(
            f"Episode {ep+1:3d}/{args.num_episodes}: "
            f"{status:7s}  reward={results[-1]['reward']:.3f}  steps={step+1}"
        )

    return results
```

**Key decisions:**
- `env.reset(seed=args.seed + ep)` — deterministic seeding for reproducibility
- `model.predict()` — calls `model.eval()` + `torch.no_grad()` internally, safe to use
- `action_raw.clamp(0.0, 512.0)` — safety clamp to PushT coordinate range
- `info.get("is_success", False)` — gym-pusht reports success in info dict

### 2.6 Environment Setup

```python
import gymnasium as gym
import gym_pusht  # noqa: F401 — side-effect import registers the env

env = gym.make(
    "gym_pusht/PushT-v0",
    obs_type="pixels_agent_pos",  # returns dict: {"pixels": [96,96,3], "agent_pos": [2]}
    render_mode="rgb_array",       # headless-compatible (no pygame window)
)
```

**Why `pixels_agent_pos`:** Gives both pixels (for model input) and agent position (for debugging). Model only uses pixels.

**Why `render_mode="rgb_array"`:** Required for headless servers (SSH, CI). On servers, also use `xvfb-run python scripts/eval_pusht.py`.

### 2.7 Summary Report

```
=== PushT Evaluation Results ===
Checkpoint: outputs/checkpoints/epoch=49.ckpt
Episodes:   50
Device:     cuda:0

Episode  1/50: SUCCESS  reward=0.970  steps=142
Episode  2/50: FAIL     reward=0.450  steps=300
...

=== Summary ===
Success Rate: 34/50 (68.0%)
Avg Reward:   0.72 ± 0.23
Avg Steps:    218.4 ± 67.2
```

### 2.8 Sanity Checks (built into the script)

Before entering the episode loop, print diagnostic info:

1. Model output shape verification: run one dummy forward pass, assert output is `[1, 2]`
2. First denormalized action range: print and verify in `[0, 512]`
3. Observation image shape: verify `[96, 96, 3]` from env
4. Action stats values: print mean/std for visual confirmation

These are `logger.info()` calls, not assertions — they inform but don't block.

---

## Phase 3: Verification

**Priority:** P0
**Files Modified:** None (manual testing)

### 3.1 Smoke Test (no trained model)

```bash
# Install eval deps
uv pip install -e ".[eval]"

# Quick syntax/import check
python -c "import scripts.eval_pusht"  # should not error

# Run with untrained model (expect ~0% success, validates loop works)
# (Requires a dummy checkpoint — can create via model.save_checkpoint())
```

### 3.2 Integration Test (with trained checkpoint)

```bash
# Run 3 episodes to verify loop completes
python scripts/eval_pusht.py \
    --checkpoint outputs/checkpoints/last.ckpt \
    --num-episodes 3 \
    --overrides "+experiment=pusht_baseline"

# Verify output format matches expected summary
```

### 3.3 Headless Server Test

```bash
xvfb-run python scripts/eval_pusht.py \
    --checkpoint model.pt \
    --num-episodes 5
```

---

## Common Pitfalls

| Pitfall | Impact | Mitigation |
|---------|--------|------------|
| Using `ActionNormalizer` (min/max) instead of z-score | 0% success — systematic action shift | Use dataset stats mean/std only |
| Forgetting ImageNet normalization | Garbage vision features | Copy exact pipeline from `_process_image()` |
| Missing `model.eval()` | BatchNorm/Dropout active → wrong behavior | `model.predict()` calls `eval()` internally |
| Device mismatch (model on CUDA, stats on CPU) | Silent broadcasting errors | Move all tensors to same device at startup |
| Forgetting `import gym_pusht` | `gym.make()` fails (env not registered) | Import at top of file (side-effect import) |
| Not clamping denormalized actions | Env may crash on out-of-range values | `clamp(0.0, 512.0)` after denorm |
| .ckpt without model config | `load_from_checkpoint` fails (missing model_cfg) | Use Hydra compose API to load config |

---

## File-Level Todo Checklist

- [ ] **pyproject.toml** — Add `[eval]` optional dependency group
- [ ] **scripts/eval_pusht.py** — Create complete eval script
  - [ ] argparse CLI with all arguments
  - [ ] `load_model()` — support .ckpt and .pt
  - [ ] `load_action_stats()` — z-score stats from lerobot/pusht
  - [ ] `preprocess_obs_image()` — exact match to `_process_image()`
  - [ ] `run_episodes()` — main eval loop
  - [ ] `print_summary()` — formatted results output
  - [ ] Sanity check logging
- [ ] **Manual verification** — smoke test + integration test

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| `lerobot/pusht` dataset download is slow (~2GB first time) | Medium | Blocks first run by 5-10min | Document in script help text; stats cached after first download |
| `gym-pusht` version incompatibility with gymnasium | Low | Script crashes at env creation | Pin compatible version ranges in pyproject.toml |
| PushT pygame dependency issues on headless server | Medium | Crash on `gym.make()` | Document `xvfb-run` usage; set `render_mode="rgb_array"` |
| Model architecture mismatch with checkpoint | Low | Load failure | Clear error messages; document required `--overrides` for .ckpt |

---

## Out of Scope (Future Extensions)

- Video recording of episodes (imageio infrastructure added but not wired)
- Batch/vectorized environments for parallel eval
- Support for TemporalVLAModel (frame buffering)
- W&B / TensorBoard metric logging
- Stats caching inside checkpoints

---

## Unresolved Questions

1. **Exact lerobot/pusht stats values** — Always loaded dynamically, never hardcoded. Verify on installed version.
2. **Stats caching in checkpoint** — Should `VLALightningModule` save normalization stats? Deferred — not needed for v1.
3. **`observation.image` vs `observation.images.top`** — PushT uses `observation.image` per `pusht_baseline.yaml`. Auto-detection in `LeRobotVLADataset` should handle either key.
