#!/usr/bin/env python3
"""Verify tinyVLA installation and environment setup.

This script checks:
1. Python version compatibility (3.10+)
2. PyTorch installation and version
3. CUDA availability and version
4. All required dependencies installed
5. Basic import tests for vla modules
6. GPU memory availability (if CUDA available)

Run this after setup_env.sh to ensure environment is correctly configured.

Usage:
    python scripts/verify-installation-and-environment.py
"""

import sys
from pathlib import Path


def print_section(title: str):
    """Print formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def print_result(check: str, passed: bool, details: str = ""):
    """Print check result with color coding."""
    status = "✓ PASS" if passed else "✗ FAIL"
    color = "\033[92m" if passed else "\033[91m"
    reset = "\033[0m"
    print(f"{color}{status}{reset} - {check}")
    if details:
        print(f"       {details}")


def check_python_version() -> bool:
    """Check if Python version is 3.10+."""
    version = sys.version_info
    is_valid = version.major == 3 and version.minor >= 10
    details = f"Python {version.major}.{version.minor}.{version.micro}"
    print_result("Python Version", is_valid, details)
    return is_valid


def check_pytorch() -> tuple[bool, bool]:
    """Check PyTorch installation and CUDA availability.

    Returns:
        (pytorch_installed, cuda_available)
    """
    try:
        import torch

        pytorch_installed = True
        version = torch.__version__
        print_result("PyTorch Installation", True, f"Version {version}")

        # Check CUDA
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_version = torch.version.cuda
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            details = f"CUDA {cuda_version}, {device_count} device(s), Primary: {device_name}"
            print_result("CUDA Support", True, details)

            # Check GPU memory
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            details = f"Total GPU memory: {total_memory:.1f} GB"
            sufficient_memory = total_memory >= 8.0
            print_result("GPU Memory", sufficient_memory, details)
        else:
            print_result("CUDA Support", False, "CUDA not available (CPU only)")

        return pytorch_installed, cuda_available

    except ImportError as e:
        print_result("PyTorch Installation", False, f"Import error: {e}")
        return False, False


def check_dependencies() -> bool:
    """Check if all required dependencies are installed."""
    dependencies = [
        ("torchvision", "torchvision"),
        ("pytorch_lightning", "pytorch-lightning"),
        ("hydra", "hydra-core"),
        ("omegaconf", "omegaconf"),
        ("timm", "timm"),
        ("transformers", "transformers"),
        ("wandb", "wandb"),
        ("einops", "einops"),
        ("numpy", "numpy"),
        ("PIL", "pillow"),
        ("h5py", "h5py"),
        ("webdataset", "webdataset"),
    ]

    all_passed = True
    for import_name, package_name in dependencies:
        try:
            __import__(import_name)
            print_result(f"Dependency: {package_name}", True)
        except ImportError:
            print_result(f"Dependency: {package_name}", False, "Not installed")
            all_passed = False

    return all_passed


def check_dev_tools() -> bool:
    """Check if development tools are installed."""
    dev_tools = [
        ("pytest", "pytest"),
        ("black", "black"),
        ("ruff", "ruff"),
        ("mypy", "mypy"),
    ]

    all_passed = True
    for import_name, package_name in dev_tools:
        try:
            __import__(import_name)
            print_result(f"Dev Tool: {package_name}", True)
        except ImportError:
            print_result(f"Dev Tool: {package_name}", False, "Not installed")
            all_passed = False

    return all_passed


def check_vla_imports() -> bool:
    """Check if vla package can be imported."""
    try:
        import vla
        from vla.registry import (
            VISION_REGISTRY,
            LANGUAGE_REGISTRY,
            FUSION_REGISTRY,
            ACTION_REGISTRY,
            MODEL_REGISTRY,
        )
        from vla.models import VLAModel, VLAConfig
        from vla.utils import setup_logger

        print_result("VLA Package Import", True, "All core modules importable")

        # Check registry contents
        vision_count = len(VISION_REGISTRY.list_available())
        language_count = len(LANGUAGE_REGISTRY.list_available())
        fusion_count = len(FUSION_REGISTRY.list_available())
        action_count = len(ACTION_REGISTRY.list_available())
        model_count = len(MODEL_REGISTRY.list_available())

        details = (
            f"Registered components: "
            f"Vision={vision_count}, Language={language_count}, "
            f"Fusion={fusion_count}, Action={action_count}, Models={model_count}"
        )
        print_result("Registry Population", True, details)

        return True

    except ImportError as e:
        print_result("VLA Package Import", False, f"Import error: {e}")
        return False


def run_basic_functionality_test() -> bool:
    """Run a basic forward pass test."""
    try:
        import torch
        from vla.models import VLAModel, VLAConfig

        # Create minimal config
        config = VLAConfig()
        model = VLAModel(config)

        # Create dummy inputs (images must be in [0, 1] range)
        images = torch.rand(1, 3, 224, 224)  # Use rand (uniform [0,1]), not randn (normal)
        texts = ["pick up block"]

        # Forward pass
        model.eval()
        with torch.no_grad():
            actions = model.predict(images, texts)

        is_valid = actions.shape == (1, 7)
        details = f"Output shape: {tuple(actions.shape)}"
        print_result("Basic Forward Pass", is_valid, details)

        return is_valid

    except Exception as e:
        print_result("Basic Forward Pass", False, f"Error: {e}")
        return False


def main():
    """Run all verification checks."""
    print_section("tinyVLA Installation Verification")

    # Track overall status
    all_checks_passed = True

    # Python version
    print_section("Python Environment")
    all_checks_passed &= check_python_version()

    # PyTorch and CUDA
    print_section("PyTorch & CUDA")
    pytorch_ok, cuda_ok = check_pytorch()
    all_checks_passed &= pytorch_ok

    # Dependencies
    print_section("Required Dependencies")
    all_checks_passed &= check_dependencies()

    # Dev tools
    print_section("Development Tools")
    dev_tools_ok = check_dev_tools()
    # Don't fail overall if dev tools missing (they're optional)

    # VLA package
    print_section("VLA Package")
    vla_ok = check_vla_imports()
    all_checks_passed &= vla_ok

    # Basic functionality test
    if pytorch_ok and vla_ok:
        print_section("Functionality Test")
        all_checks_passed &= run_basic_functionality_test()

    # Summary
    print_section("Summary")
    if all_checks_passed:
        print("\033[92m✓ All critical checks passed!\033[0m")
        print("\nYou can now:")
        print("  - Run tests: pytest tests/ -v")
        print("  - Train models: python scripts/train.py")
        print("  - Benchmark: python scripts/benchmark-model.py")
        if not cuda_ok:
            print("\n\033[93mNote: CUDA not available. Training will be slower on CPU.\033[0m")
        sys.exit(0)
    else:
        print("\033[91m✗ Some checks failed. Please review errors above.\033[0m")
        sys.exit(1)


if __name__ == "__main__":
    main()
