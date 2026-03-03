# Brainstorm: `scripts/eval_pusht.py` — Closed-Loop PushT Evaluation

**Date:** 2026-03-03
**Status:** Design brainstorm
**Goal:** Design a script that loads a trained tinyVLA checkpoint, runs closed-loop inference on the LeRobot PushT gym environment, and reports success rate.

---

## 1. Problem Statement

The existing `scripts/eval.py` evaluates offline metrics (loss, MSE, MAE) on a held-out dataset. It cannot measure **real task performance** because it never closes the loop — it doesn't feed predicted actions back into an environment.

`eval_pusht.py` fills this gap: run the model in the PushT simulator, let it control the agent step-by-step, and measure whether the T-block reaches the target zone.

---

## 2. Dependencies

| Package | Purpose | Install |
|---------|---------|---------|
| `gymnasium` | Env interface (NOT legacy `gym`) | `pip install gymnasium` |
| `gym-pusht` | Registers `gym_pusht/PushT-v0` | `pip install gym-pusht` |
| `lerobot>=0.4.0` | Dataset stats (action mean/std) | `pip install lerobot` |
| `pytorch-lightning` | Checkpoint loading (`.ckpt`) | already in project |
| `hydra-core` | Config composition | already in project |
| `torchvision` | Image transforms | already in project |
| `numpy` | Array conversion for env | already in project |

**Optional:**
- `imageio` / `imageio-ffmpeg` — save episode videos
- `tqdm` — progress bars
- `xvfb` — headless rendering on servers (`xvfb-run python scripts/eval_pusht.py`)

---

## 3. Architecture Overview

```
┌───────────────────────────────────────────────────────────┐
│                     eval_pusht.py                         │
├───────────────────────────────────────────────────────────┤
│                                                           │
│  1. LOAD CHECKPOINT                                       │
│     ├── .ckpt → VLALightningModule.load_from_checkpoint() │
│     └── .pt  → VLAModel.load_checkpoint()                 │
│                                                           │
│  2. LOAD NORMALIZATION STATS                              │
│     └── LeRobotDataset("lerobot/pusht").meta.stats        │
│         → action_mean [2], action_std [2]                 │
│                                                           │
│  3. CREATE ENVIRONMENT                                    │
│     └── gym.make("gym_pusht/PushT-v0",                    │
│              obs_type="pixels_agent_pos",                 │
│              render_mode="rgb_array")                     │
│                                                           │
│  4. EPISODE LOOP (N episodes)                             │
│     ├── obs, info = env.reset(seed=episode_seed)          │
│     ├── STEP LOOP (max 300 steps)                         │
│     │   ├── obs["pixels"] → preprocess_image()            │
│     │   │   ├── uint8 HWC [96,96,3] → float CHW [3,96,96]│
│     │   │   ├── resize → [3, 224, 224]                    │
│     │   │   ├── ImageNet normalize                        │
│     │   │   └── unsqueeze → [1, 3, 224, 224]              │
│     │   ├── model.predict(image, [text]) → [1, 2]         │
│     │   │   └── actions in [-1, 1] (normalized)           │
│     │   ├── DENORMALIZE:                                  │
│     │   │   action_raw = action_norm * std + mean          │
│     │   │   clip to [0, 512]                              │
│     │   ├── env.step(action_raw.numpy())                  │
│     │   └── check terminated / truncated                  │
│     └── record success, reward, episode_length            │
│                                                           │
│  5. REPORT                                                │
│     ├── success_rate = sum(successes) / N                  │
│     ├── avg_reward per step                                │
│     ├── avg_episode_length                                 │
│     └── per-episode details table                          │
└───────────────────────────────────────────────────────────┘
```

---

## 4. Key Design Decisions

### 4.1 Checkpoint Loading Strategy

Two checkpoint formats exist in tinyVLA:

| Format | Source | Loader |
|--------|--------|--------|
| `.ckpt` | PyTorch Lightning `ModelCheckpoint` | `VLALightningModule.load_from_checkpoint(path, model_cfg=cfg)` |
| `.pt` | `VLAModel.save_checkpoint()` | `VLAModel.load_checkpoint(path)` |

**Decision:** Support both, auto-detect by file extension. Reuse the pattern from `scripts/eval.py:_load_module()`. Extract the raw `VLAModel` from the Lightning wrapper for inference (avoid PL overhead).

```python
if path.suffix == ".ckpt":
    module = VLALightningModule.load_from_checkpoint(path, model_cfg=cfg)
    model = module.model
elif path.suffix == ".pt":
    model = VLAModel.load_checkpoint(path)
```

### 4.2 Action Denormalization (CRITICAL)

This is the most error-prone part. The model outputs actions in `[-1, 1]`, but PushT env expects raw pixel coordinates in `[0, 512]`.

**Training normalization formula** (from `lerobot_dataset.py:394`):
```
action_normalized = (action_raw - mean) / std
```

**Denormalization (inverse):**
```
action_raw = action_normalized * std + mean
```

**Where do stats come from?**
```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("lerobot/pusht")
action_mean = ds.meta.stats["action"]["mean"]  # ~[256, 256]
action_std  = ds.meta.stats["action"]["std"]   # ~[83, 80]
```

**Why NOT min/max normalization?** The `LeRobotVLADataset` uses **z-score (mean/std)** normalization, not min/max. The `ActionNormalizer` class in `action_utils.py` uses min/max — this is a **different normalizer** and must NOT be used for denormalization here. We must match the exact normalization used during training.

**Alternative approach — also considered:** Load stats directly from the dataset the model was trained on, rather than hardcoding. This makes the script robust to different dataset versions.

### 4.3 Image Preprocessing Pipeline

Must exactly match `LeRobotVLADataset._process_image()`:

```python
def preprocess_obs_image(pixels: np.ndarray) -> torch.Tensor:
    """Convert gym obs pixels to model-ready tensor.

    Pipeline: uint8 HWC → float CHW → resize 224 → ImageNet norm → batch dim
    """
    # pixels shape: [96, 96, 3], dtype: uint8
    img = torch.from_numpy(pixels).permute(2, 0, 1).float() / 255.0  # [3, 96, 96]
    img = TF.resize(img, [224, 224], antialias=True)                  # [3, 224, 224]
    img = TF.normalize(img, mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])                # ImageNet norm
    return img.unsqueeze(0)  # [1, 3, 224, 224]
```

**Why ImageNet normalization?** The vision backbones (DINOv2, SigLIP, ViT) were pretrained on ImageNet-normalized images. Skipping this will produce garbage features.

### 4.4 Environment Configuration

```python
import gymnasium as gym
import gym_pusht  # side-effect: registers the env

env = gym.make(
    "gym_pusht/PushT-v0",
    obs_type="pixels_agent_pos",  # gives both pixels AND agent position
    render_mode="rgb_array",       # needed for headless servers
)
```

- `obs_type="pixels_agent_pos"` → obs dict with `"pixels"` (96x96x3 uint8) and `"agent_pos"` (2D float)
- `obs_type="pixels"` → only pixels, no agent pos (simpler but less info for debugging)
- `max_episode_steps=300` by default (set by the gym registration)

### 4.5 Text Instruction

PushT has a single task. From `lerobot/pusht` dataset metadata, the task is something like `"Push the T-shaped block onto the T-shaped target."` We should load this from the dataset's task lookup, or default to a reasonable string:

```python
task_text = "Push the T-shaped block onto the T-shaped target."
```

### 4.6 Episode Seeding

For reproducibility, seed each episode deterministically:

```python
for ep in range(num_episodes):
    obs, info = env.reset(seed=base_seed + ep)
```

This ensures results are reproducible across runs.

---

## 5. Script Interface (CLI)

Use Hydra for consistency with the rest of the codebase:

```bash
# Basic usage
python scripts/eval_pusht.py \
    +eval.checkpoint=outputs/checkpoints/last.ckpt \
    +eval.num_episodes=50

# With custom settings
python scripts/eval_pusht.py \
    +eval.checkpoint=model.pt \
    +eval.num_episodes=100 \
    +eval.seed=42 \
    +eval.save_video=true \
    +eval.device=cuda
```

**Alternative:** Use `argparse` for simplicity since this is a standalone eval script, not a training entry point. Hydra adds complexity for a script that doesn't need config composition.

**Recommendation:** Use `argparse`. The script has ~5 parameters, doesn't need Hydra's config group system, and is conceptually separate from train/eval offline.

---

## 6. Output Format

```
=== PushT Evaluation Results ===
Checkpoint: outputs/checkpoints/epoch=49.ckpt
Episodes:   50
Device:     cuda:0

Episode  1/50: success=True   reward=0.97  steps=142
Episode  2/50: success=False  reward=0.45  steps=300
Episode  3/50: success=True   reward=0.96  steps=189
...

=== Summary ===
Success Rate: 34/50 (68.0%)
Avg Reward:   0.72 ± 0.23
Avg Steps:    218.4 ± 67.2
```

---

## 7. Potential Pitfalls

### 7.1 Normalization Mismatch (HIGHEST RISK)

**Problem:** If denormalization formula doesn't exactly invert the training normalization, the model's actions will be systematically wrong.

**Scenarios:**
- Using `ActionNormalizer` (min/max) instead of z-score (mean/std) → wrong mapping
- Loading stats from a different dataset version → shifted distribution
- Forgetting to clamp denormalized actions to `[0, 512]` → env may crash or clip silently
- Stats tensor device mismatch (CPU stats vs CUDA model output) → silent broadcasting error

**Mitigation:** Load stats from the SAME `lerobot/pusht` dataset version used for training. Always clamp after denorm. Add a sanity check that prints a few denormalized actions to verify they're in the expected range.

### 7.2 Image Preprocessing Drift

**Problem:** Even small differences in image preprocessing between training and eval cause feature drift.

**Scenarios:**
- Forgetting ImageNet normalization → model sees out-of-distribution inputs
- Using `PIL.Image.resize` (bilinear) vs `TF.resize` (different interpolation) → subtle pixel differences
- Color channel order: gym returns RGB, but some OpenCV paths return BGR
- Input range: gym gives uint8 `[0, 255]`, model expects float `[0, 1]` pre-normalization

**Mitigation:** Extract image preprocessing into a shared utility used by both `LeRobotVLADataset` and `eval_pusht.py`. Or at minimum, write the eval preprocessing to exactly mirror `_process_image()`.

### 7.3 Model in Wrong Mode

**Problem:** Calling `model.forward()` without `model.eval()` or `torch.no_grad()` → BatchNorm/Dropout active, gradients computed (slow + wrong behavior).

**Mitigation:** Always call `model.eval()` before the loop. Use `torch.no_grad()` context manager. The `model.predict()` method already handles both — prefer using it.

### 7.4 Discrete vs Continuous Action Head

**Problem:** DiscreteActionHead outputs actions via `argmax` over 256 bins, then `bins_to_continuous` → values in `[-1, 1]`. GaussianActionHead outputs `clamp(mean, -1, 1)`. Both produce `[-1, 1]` outputs, so denormalization works the same way.

**However:** If the model was trained with `GaussianActionHead(sample=True)`, the training distribution includes noise. At eval time, we should use `sample=False` (deterministic mean prediction) for best performance. The `model.predict()` method uses `return_logits=False` which doesn't trigger sampling — this is correct.

### 7.5 PushT Action Semantics

**Problem:** PushT actions are **absolute target positions** in `[0, 512]`, not velocities or deltas. A PD controller inside the physics sim moves the pusher toward the target. This means:
- Repeated identical actions → pusher stays at that position (converges)
- Very different consecutive actions → large jumps (potentially unstable)
- The model must learn to output smooth trajectories

**No mitigation needed in the script** — just awareness for debugging weird behavior.

### 7.6 Device Mismatch

**Problem:** Model on CUDA, stats tensors on CPU, image tensor on CPU → silent broadcasting or crash.

**Mitigation:** Move everything to the same device at startup:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
action_mean = action_mean.to(device)
action_std = action_std.to(device)
# In loop:
image_tensor = preprocess(obs["pixels"]).to(device)
```

### 7.7 Headless Rendering

**Problem:** `gym_pusht` uses `pygame` for rendering. On headless servers (SSH, CI), no display is available → crash.

**Mitigation:**
- Use `render_mode="rgb_array"` (doesn't open a window)
- Run with `xvfb-run python scripts/eval_pusht.py` on servers
- Or set `SDL_VIDEODRIVER=dummy` env variable

### 7.8 Stats Loading Without Full Dataset Download

**Problem:** `LeRobotDataset("lerobot/pusht")` downloads the entire dataset (~2GB) just to get stats. For eval-only, this is wasteful.

**Alternatives:**
1. Cache stats in the checkpoint (save `action_mean`, `action_std` alongside model weights)
2. Provide stats as CLI args: `+eval.action_mean=[256,256] +eval.action_std=[83,80]`
3. Download only `meta/stats.json` via HuggingFace Hub API
4. Accept the download — it's cached after first run

**Recommendation:** Option 1 for production, option 4 for prototyping. The script should first check if stats are in the checkpoint, then fall back to loading from dataset.

---

## 8. Pseudocode

```python
def main():
    args = parse_args()
    device = select_device(args.device)

    # 1. Load model
    model = load_checkpoint(args.checkpoint, device)
    model.eval()

    # 2. Load denormalization stats
    action_mean, action_std = load_action_stats("lerobot/pusht")
    action_mean = action_mean.to(device)
    action_std = action_std.to(device)

    # 3. Create env
    env = gym.make("gym_pusht/PushT-v0",
                   obs_type="pixels_agent_pos",
                   render_mode="rgb_array")

    # 4. Run episodes
    results = []
    for ep in range(args.num_episodes):
        obs, info = env.reset(seed=args.seed + ep)
        episode_reward = 0.0

        for step in range(300):  # max_episode_steps
            # Preprocess observation
            image = preprocess_obs_image(obs["pixels"]).to(device)

            # Predict action (normalized [-1, 1])
            with torch.no_grad():
                action_norm = model.predict(image, [args.task_text])

            # Denormalize to env scale [0, 512]
            action_raw = action_norm.squeeze(0).cpu() * action_std + action_mean
            action_raw = action_raw.clamp(0.0, 512.0).numpy()

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action_raw)
            episode_reward += reward

            if terminated or truncated:
                break

        success = info.get("is_success", terminated)
        results.append({
            "episode": ep,
            "success": success,
            "reward": episode_reward / (step + 1),
            "steps": step + 1,
        })

    # 5. Report
    print_summary(results, args)
    env.close()
```

---

## 9. File Structure

```
scripts/
├── eval.py           # Existing offline eval (dataset loss/MSE/MAE)
├── eval_pusht.py     # NEW: closed-loop PushT evaluation
└── train.py          # Training entry point
```

Single file, ~150-180 lines. No need for additional modules — the script is self-contained.

---

## 10. Testing the Script

### Smoke Test (no trained model needed)
```python
# Test with random model (expect ~0% success rate)
python scripts/eval_pusht.py --checkpoint=none --num_episodes=3
```

### Sanity Checks to Add
1. Print first denormalized action and verify it's in `[0, 512]` range
2. Print first observation image shape and verify it's `[96, 96, 3]`
3. Verify model output shape is `[1, 2]` (batch=1, action_dim=2)
4. Run 1 episode and verify episode terminates (doesn't hang)

---

## 11. Future Extensions

- **Video recording:** Save episode rollouts as MP4 for qualitative analysis
- **Multi-task:** Generalize to other LeRobot envs (ALOHA, DORA, etc.)
- **Temporal model support:** Handle `TemporalVLAModel` by buffering last N frames
- **Batch evaluation:** Run multiple envs in parallel (vectorized envs)
- **Metric logging:** Log to W&B or TensorBoard

---

## Unresolved Questions

1. **Exact stats values:** Need to verify `lerobot/pusht` stats on the installed version. Always load dynamically, never hardcode.
2. **`observation.state` dimensionality:** Dataset metadata says 5D `[agent_x, agent_y, block_x, block_y, block_angle]`, but project's `AffordanceConfig` defaults to `state_dim=3`. Clarify if model was trained with 3D or 5D states.
3. **Training data image key:** Config says `observation.image` but LeRobot >= 0.4.0 uses `observation.images.top`. Auto-detection in `LeRobotVLADataset` should handle this, but verify env obs matches.
4. **Should eval_pusht.py use Hydra or argparse?** Hydra is consistent with codebase but overkill for ~5 params. Recommend argparse for simplicity.
5. **Stats caching in checkpoint:** Should we modify `VLALightningModule` to save normalization stats in the checkpoint? This would decouple eval from dataset availability.
