"""Closed-loop PushT evaluation for tinyVLA models.

Loads a trained checkpoint, runs N episodes in the gym-pusht simulation,
and reports success rate, average reward, and average episode length.

Usage:
    python scripts/eval_pusht.py --checkpoint path/to/model.pt --num-episodes 50

Requirements:
    pip install -e ".[eval]"
    # On headless servers: xvfb-run python scripts/eval_pusht.py ...

Note:
    First run downloads ~2GB of lerobot/pusht stats. Subsequent runs use cache.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import numpy as np
import torch
import torchvision.transforms.functional as TF

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for PushT evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a tinyVLA model on the PushT gym environment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .ckpt or .pt file")
    parser.add_argument("--num-episodes", type=int, default=50, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for reproducibility")
    parser.add_argument("--device", type=str, default="auto", help="Device: 'auto', 'cpu', or 'cuda:0'")
    parser.add_argument("--max-steps", type=int, default=300, help="Max steps per episode")
    parser.add_argument(
        "--task-text",
        type=str,
        default="Push the T-shaped block onto the T-shaped target.",
        help="Task instruction text fed to the language encoder",
    )
    parser.add_argument("--config-dir", type=str, default="configs", help="Hydra configs dir (.ckpt only)")
    parser.add_argument("--config-name", type=str, default="config", help="Hydra config name (.ckpt only)")
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help='Hydra overrides for .ckpt loading (e.g. "+experiment=pusht_baseline")',
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def resolve_device(device_str: str) -> torch.device:
    """Resolve 'auto' to cuda/cpu; pass through explicit device strings.

    Args:
        device_str: 'auto', 'cpu', or 'cuda:N'

    Returns:
        torch.device ready for tensor placement
    """
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def load_model(args: argparse.Namespace, device: torch.device) -> Any:
    """Load a VLAModel from a .pt or .ckpt checkpoint.

    Two formats exist because:
    - .pt files are self-contained (config + weights stored together)
    - .ckpt files (PyTorch Lightning) excluded model_cfg from save_hyperparameters,
      so we reconstruct VLAConfig from Hydra configs at load time.

    Args:
        args: CLI arguments (checkpoint path, config-dir, config-name, overrides)
        device: Target device for the model

    Returns:
        VLAModel instance in eval mode on the target device
    """
    from vla.models import VLAModel
    from vla.models.vla_configs import VLAConfig

    path = Path(args.checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if path.suffix == ".pt":
        # Self-contained checkpoint: config is stored inside the file
        model = VLAModel.load_checkpoint(str(path), map_location="cpu")

    elif path.suffix == ".ckpt":
        # PL checkpoint: need external Hydra config to reconstruct VLAConfig
        from hydra import compose, initialize_config_dir

        from vla.training.lightning_module import VLALightningModule

        config_dir = str(Path(args.config_dir).resolve())
        with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
            cfg = compose(config_name=args.config_name, overrides=args.overrides or [])

        model_cfg = VLAConfig.from_hydra(cfg.model)

        # Strip torch.compile() artifact: checkpoint saved while model was compiled
        # adds "_orig_mod." prefix to all keys. Remove it before loading.
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        fixed_state = {
            k.replace("._orig_mod.", "."): v
            for k, v in ckpt["state_dict"].items()
        }
        ckpt["state_dict"] = fixed_state

        module = VLALightningModule(model_cfg=model_cfg)
        module.load_state_dict(ckpt["state_dict"], strict=True)
        model = module.model
    else:
        raise ValueError(f"Unknown checkpoint format: {path.suffix}. Expected .ckpt or .pt")

    model = model.to(device)
    model.eval()
    logger.info(f"Model loaded from {path} → {device}")
    return model


# ---------------------------------------------------------------------------
# Action statistics (z-score denormalization)
# ---------------------------------------------------------------------------


def load_action_stats(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load action mean/std from lerobot/pusht dataset stats.

    During training, LeRobotVLADataset normalizes actions with z-score:
        action_norm = (action - mean) / std
    At eval time we invert this:
        action_raw = action_norm * std + mean

    CRITICAL: This is z-score inverse, NOT min/max (ActionNormalizer).
    Using the wrong normalizer produces 0% success.

    Returns:
        Tuple of (action_mean, action_std) tensors on the specified device.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    logger.info("Loading lerobot/pusht dataset stats (may download ~2GB on first run)...")
    ds = LeRobotDataset("lerobot/pusht")
    stats = ds.meta.stats
    action_mean = torch.tensor(stats["action"]["mean"], dtype=torch.float32).to(device)
    action_std = torch.tensor(stats["action"]["std"], dtype=torch.float32).to(device)
    # Guard against division by zero (matches LeRobotVLADataset._load_action_stats)
    action_std = torch.clamp(action_std, min=1e-6)
    logger.info(f"Action stats loaded — mean: {action_mean.tolist()}, std: {action_std.tolist()}")
    return action_mean, action_std


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------


def preprocess_obs_image(pixels: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert gym observation pixels to model-ready tensor.

    MUST exactly match LeRobotVLADataset._process_image() (lerobot_dataset.py:345-375):
    1. uint8 HWC → float CHW via permute + /255.0
    2. Resize to 224x224
    3. ImageNet normalize (required for DINOv2/ViT pretrained backbones)
    4. Add batch dimension

    Args:
        pixels: Raw observation from gym env, shape [96, 96, 3], dtype uint8
        device: Target device for the output tensor

    Returns:
        Tensor of shape [1, 3, 224, 224], ImageNet-normalized
    """
    # [H, W, C] uint8 → [C, H, W] float in [0, 1]
    img = torch.from_numpy(pixels).permute(2, 0, 1).float() / 255.0
    # Resize to match training resolution (antialias=True matches torchvision default)
    img = TF.resize(img, [224, 224], antialias=True)
    # ImageNet normalization — vision backbones (DINOv2, ViT) expect this
    img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return cast(torch.Tensor, img).unsqueeze(0).to(device)


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------


def run_sanity_checks(
    model: Any,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    device: torch.device,
    task_text: str,
    first_obs_pixels: np.ndarray,
) -> None:
    """Run diagnostic checks before the episode loop.

    These are informational logger.info() calls, not assertions.
    They help catch obvious misconfigurations early.

    Args:
        model: The loaded VLAModel
        action_mean: Action mean tensor
        action_std: Action std tensor
        device: Compute device
        task_text: Task instruction text
        first_obs_pixels: First observation pixels from env.reset() [96,96,3] uint8
    """
    # Use real observation (not torch.randn which exceeds ImageNet norm range)
    dummy_image = preprocess_obs_image(first_obs_pixels, device)
    with torch.no_grad():
        dummy_action = model.predict(dummy_image, [task_text])
    logger.info(f"Sanity check — model output shape: {dummy_action.shape} (expected [1, 2])")

    # Verify denormalized action range
    dummy_raw = dummy_action.squeeze(0).cpu() * action_std.cpu() + action_mean.cpu()
    logger.info(f"Sanity check — first denormalized action: {dummy_raw.tolist()}")

    # Log action stats for visual confirmation
    logger.info(f"Sanity check — action_mean: {action_mean.tolist()}")
    logger.info(f"Sanity check — action_std: {action_std.tolist()}")


# ---------------------------------------------------------------------------
# Episode loop
# ---------------------------------------------------------------------------


def run_episodes(
    model: Any,
    env: Any,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    """Run closed-loop evaluation episodes.

    Each episode:
    1. Reset env with deterministic seed (seed + episode_idx)
    2. Preprocess observation image
    3. Model predicts normalized action
    4. Denormalize with z-score inverse: action_raw = action_norm * std + mean
    5. Clamp to [0, 512] (PushT coordinate range)
    6. Step the environment

    Args:
        model: VLAModel in eval mode
        env: Gymnasium PushT environment
        action_mean: Mean tensor for z-score denormalization
        action_std: Std tensor for z-score denormalization
        device: Compute device
        args: CLI arguments

    Returns:
        List of per-episode result dicts with keys: episode, success, reward, steps
    """
    results: List[Dict[str, Any]] = []
    # Pre-move stats to CPU once (avoids repeated .cpu() calls per step)
    mean_cpu = action_mean.cpu()
    std_cpu = action_std.cpu()

    for ep in range(args.num_episodes):
        obs, _info = env.reset(seed=args.seed + ep)
        episode_reward = 0.0
        step = 0

        for step in range(args.max_steps):  # noqa: B007 — used after loop as num_steps
            # Preprocess observation — must match training pipeline exactly
            image = preprocess_obs_image(obs["pixels"], device)

            # Model predicts normalized action in roughly [-1, 1]
            with torch.no_grad():
                action_norm = model.predict(image, [args.task_text])  # [1, action_dim]

            # Denormalize: z-score inverse → raw pixel coordinates
            action_raw = action_norm.squeeze(0).cpu() * std_cpu + mean_cpu
            # Safety clamp to PushT coordinate range [0, 512]
            action_raw = action_raw.clamp(0.0, 512.0).numpy()

            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action_raw)
            episode_reward += reward

            if terminated or truncated:
                break

        # gym-pusht reports success in the info dict
        success = info.get("is_success", False)
        num_steps = step + 1
        results.append(
            {
                "episode": ep + 1,
                "success": bool(success),
                "reward": episode_reward / num_steps,
                "steps": num_steps,
            }
        )

        status = "SUCCESS" if success else "FAIL"
        logger.info(
            f"Episode {ep + 1:3d}/{args.num_episodes}: "
            f"{status:7s}  reward={results[-1]['reward']:.3f}  steps={num_steps}"
        )

    return results


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------


def print_summary(results: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    """Print formatted evaluation summary.

    Args:
        results: List of per-episode result dicts
        args: CLI arguments (for checkpoint path display)
    """
    successes = sum(1 for r in results if r["success"])
    total = len(results)
    rewards = [r["reward"] for r in results]
    steps = [r["steps"] for r in results]

    reward_mean = np.mean(rewards)
    reward_std = np.std(rewards)
    steps_mean = np.mean(steps)
    steps_std = np.std(steps)

    print("\n=== PushT Evaluation Results ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Episodes:   {total}")
    print(f"Device:     {args.device}")
    print("\n=== Summary ===")
    print(f"Success Rate: {successes}/{total} ({100.0 * successes / total:.1f}%)")
    print(f"Avg Reward:   {reward_mean:.2f} +/- {reward_std:.2f}")
    print(f"Avg Steps:    {steps_mean:.1f} +/- {steps_std:.1f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point: parse args, load model, run evaluation, print summary."""
    args = parse_args()
    device = resolve_device(args.device)
    logger.info(f"Using device: {device}")

    # -- Load model --
    model = load_model(args, device)

    # -- Load action normalization stats --
    action_mean, action_std = load_action_stats(device)

    # -- Create PushT environment --
    import gymnasium as gym

    try:
        import gym_pusht  # noqa: F401 — side-effect import registers the env
    except ImportError:
        logger.error(
            "gym-pusht is not installed. Install with: pip install -e '.[eval]'"
        )
        sys.exit(1)

    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
    )
    logger.info("PushT environment created (obs_type=pixels_agent_pos, render_mode=rgb_array)")

    # Verify observation shape
    obs, _ = env.reset(seed=args.seed)
    logger.info(f"Sanity check — observation pixels shape: {obs['pixels'].shape} (expected (96, 96, 3))")

    # -- Sanity checks --
    run_sanity_checks(model, action_mean, action_std, device, args.task_text, obs["pixels"])

    # -- Run episodes --
    results = run_episodes(model, env, action_mean, action_std, device, args)

    # -- Print summary --
    print_summary(results, args)

    env.close()


if __name__ == "__main__":
    main()
