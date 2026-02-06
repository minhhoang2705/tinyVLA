#!/usr/bin/env python
"""Manual verification script for Hydra configuration system.

Run from project root:
    python scripts/test-hydra-config.py
    python scripts/test-hydra-config.py vision=dinov2
    python scripts/test-hydra-config.py +experiment=baseline
    python scripts/test-hydra-config.py --multirun vision=vit-base,dinov2

Tests:
    1. Default config loads successfully
    2. CLI overrides work
    3. Experiment presets load
    4. Config validation passes
    5. Config auto-saves to .hydra/
"""

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from vla.utils.hydra_config_helpers import (  # noqa: E402
    flatten_config,
    register_resolvers,
    validate_config,
)

# Register custom resolvers before Hydra loads configs
register_resolvers()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for config testing.

    Args:
        cfg: Composed Hydra configuration
    """
    print("=" * 60)
    print("tinyVLA Hydra Configuration Test")
    print("=" * 60)

    # Test 1: Config loaded
    print("\n[1/5] Config loaded successfully")
    print(f"  Vision: {cfg.vision.name}")
    print(f"  Language: {cfg.language.name}")
    print(f"  Fusion: {cfg.fusion.name}")
    print(f"  Action: {cfg.action.name}")

    # Test 2: Print full config
    print("\n[2/5] Full resolved config:")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # Test 3: Validate config
    print("[3/5] Validating config...")
    try:
        validate_config(cfg)
        print("  ✓ Validation PASSED")
    except ValueError as e:
        print(f"  ✗ Validation FAILED: {e}")
        sys.exit(1)

    # Test 4: Check output directory
    output_dir = Path(cfg.output_dir)
    print(f"\n[4/5] Output directory: {output_dir}")
    hydra_dir = output_dir / ".hydra"
    if hydra_dir.exists():
        print("  ✓ .hydra/ directory exists with auto-saved configs")
        for f in sorted(hydra_dir.iterdir()):
            print(f"    - {f.name}")
    else:
        print("  ℹ .hydra/ directory will be created by Hydra")

    # Test 5: Flatten config (useful for WandB logging)
    flat = flatten_config(cfg)
    print(f"\n[5/5] Flattened config has {len(flat)} keys")
    print("  Sample keys:")
    for key in sorted(flat.keys())[:5]:
        print(f"    {key}: {flat[key]}")

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
