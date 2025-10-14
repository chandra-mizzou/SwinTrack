#!/usr/bin/env python3
"""
Test script to verify SwinTrack inference installation.

This script checks if all required dependencies are installed
and the SwinTrack modules can be imported correctly.
"""

import sys
import os

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch: {e}")
        return False
    
    try:
        import torchvision
        print(f"✓ Torchvision {torchvision.__version__}")
    except ImportError as e:
        print(f"✗ Torchvision: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy: {e}")
        return False
    
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV: {e}")
        return False
    
    try:
        from PIL import Image
        print("✓ Pillow")
    except ImportError as e:
        print(f"✗ Pillow: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"✗ Matplotlib: {e}")
        return False
    
    try:
        import yaml
        print("✓ PyYAML")
    except ImportError as e:
        print(f"✗ PyYAML: {e}")
        return False
    
    try:
        import timm
        print(f"✓ TIMM {timm.__version__}")
    except ImportError as e:
        print(f"✗ TIMM: {e}")
        return False
    
    return True

def test_swintrack_modules():
    """Test if SwinTrack modules can be imported."""
    print("\nTesting SwinTrack modules...")
    
    try:
        from miscellanies.yaml_ops import load_yaml
        print("✓ YAML operations")
    except ImportError as e:
        print(f"✗ YAML operations: {e}")
        return False
    
    try:
        from models.methods.builder import build_model
        print("✓ Model builder")
    except ImportError as e:
        print(f"✗ Model builder: {e}")
        return False
    
    try:
        from core.run.event_dispatcher.register import EventRegister
        print("✓ Event dispatcher")
    except ImportError as e:
        print(f"✗ Event dispatcher: {e}")
        return False
    
    return True

def test_config_files():
    """Test if config files exist."""
    print("\nTesting config files...")
    
    config_path = "config/SwinTrack/Base/config.yaml"
    if os.path.exists(config_path):
        print(f"✓ Config file found: {config_path}")
        return True
    else:
        print(f"✗ Config file not found: {config_path}")
        return False

def test_cuda():
    """Test CUDA availability."""
    print("\nTesting CUDA...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠ CUDA not available (CPU mode only)")
            return True
    except Exception as e:
        print(f"✗ CUDA test failed: {e}")
        return False

def main():
    """Main test function."""
    print("SwinTrack Inference Installation Test")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Test basic imports
    if not test_imports():
        all_tests_passed = False
    
    # Test SwinTrack modules
    if not test_swintrack_modules():
        all_tests_passed = False
    
    # Test config files
    if not test_config_files():
        all_tests_passed = False
    
    # Test CUDA
    test_cuda()  # This doesn't fail the test
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("✓ All tests passed! SwinTrack inference is ready to use.")
        print("\nNext steps:")
        print("1. Download a pretrained SwinTrack model")
        print("2. Prepare your image sequence")
        print("3. Run: python swintrack_inference.py --help")
    else:
        print("✗ Some tests failed. Please install missing dependencies.")
        print("\nInstall dependencies with:")
        print("pip install -r requirements_inference.txt")
    
    return all_tests_passed

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)