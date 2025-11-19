#!/usr/bin/env python3
"""
Requirements Checker for GSFlow Module

This script checks whether all required dependencies for the GSFlow module
and optical flow supervision are properly installed.
"""

import sys
import importlib
from typing import List, Tuple


def check_package(package_name: str, import_name: str = None) -> Tuple[bool, str]:
    """
    Check if a package is installed.
    
    Args:
        package_name: Display name of the package
        import_name: Name to use for import (defaults to package_name)
        
    Returns:
        Tuple of (is_installed, version_or_error)
    """
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown version')
        return True, version
    except ImportError as e:
        return False, str(e)


def check_cuda_extensions() -> Tuple[bool, str]:
    """
    Check if custom CUDA extensions are built and available.
    
    Returns:
        Tuple of (is_available, status_message)
    """
    try:
        import diff_gaussian_rasterization
        return True, "diff_gaussian_rasterization available"
    except ImportError:
        return False, "diff_gaussian_rasterization not found - need to build submodules"


def main():
    """Main function to check all requirements."""
    print("=" * 60)
    print("Checking Requirements for GSFlow Module")
    print("=" * 60)
    print()
    
    # Core dependencies
    core_packages = [
        ("torch", "torch"),
        ("numpy", "numpy"),
        ("opencv-python", "cv2"),
        ("scipy", "scipy"),
        ("imageio", "imageio"),
        ("tqdm", "tqdm"),
        ("plyfile", "plyfile"),
    ]
    
    # Optional but recommended packages
    optional_packages = [
        ("lpips", "lpips"),
        ("tensorboard", "torch.utils.tensorboard"),
        ("wandb", "wandb"),
    ]
    
    all_ok = True
    
    # Check core packages
    print("Core Dependencies:")
    print("-" * 60)
    for package_name, import_name in core_packages:
        is_installed, version_or_error = check_package(package_name, import_name)
        status = "✓ OK" if is_installed else "✗ MISSING"
        print(f"{status:8} {package_name:20} {version_or_error}")
        if not is_installed:
            all_ok = False
    
    print()
    
    # Check optional packages
    print("Optional Dependencies:")
    print("-" * 60)
    for package_name, import_name in optional_packages:
        is_installed, version_or_error = check_package(package_name, import_name)
        status = "✓ OK" if is_installed else "- OPTIONAL"
        print(f"{status:8} {package_name:20} {version_or_error}")
    
    print()
    
    # Check CUDA extensions
    print("CUDA Extensions:")
    print("-" * 60)
    is_available, message = check_cuda_extensions()
    status = "✓ OK" if is_available else "✗ MISSING"
    print(f"{status:8} {message}")
    if not is_available:
        all_ok = False
    
    print()
    
    # Check PyTorch CUDA availability
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        status = "✓ OK" if cuda_available else "✗ WARNING"
        device_count = torch.cuda.device_count() if cuda_available else 0
        print(f"{status:8} CUDA available: {cuda_available} (devices: {device_count})")
    except:
        print("✗ MISSING Unable to check CUDA availability")
        all_ok = False
    
    print()
    print("=" * 60)
    
    if all_ok:
        print("✓ All required dependencies are installed!")
        print()
        print("GSFlow module is ready to use.")
        return 0
    else:
        print("✗ Some required dependencies are missing.")
        print()
        print("To install missing dependencies, run:")
        print("  pip install -r requirements.txt")
        print()
        print("To build CUDA extensions, run:")
        print("  cd submodules/diff-gaussian-rasterization-w-pose && pip install -e .")
        print("  cd submodules/flow-diff-gaussian-rasterization && pip install -e .")
        return 1


if __name__ == "__main__":
    sys.exit(main())
