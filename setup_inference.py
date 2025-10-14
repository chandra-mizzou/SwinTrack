#!/usr/bin/env python3
"""
Setup script for SwinTrack inference.

This script helps set up the environment for SwinTrack inference.
"""

import os
import sys
import subprocess
import argparse

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"  Error: {e.stderr}")
        return False

def install_dependencies():
    """Install required dependencies."""
    print("Installing dependencies...")
    
    # Install basic requirements
    if not run_command("pip install -r requirements_inference.txt", "Installing basic requirements"):
        return False
    
    # Install additional dependencies if needed
    additional_deps = [
        "psutil",  # For process management
        "wandb",   # For logging (optional)
    ]
    
    for dep in additional_deps:
        run_command(f"pip install {dep}", f"Installing {dep}")
    
    return True

def create_directories():
    """Create necessary directories."""
    print("Creating directories...")
    
    directories = [
        "sample_images",
        "example_output",
        "output",
        "models"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    return True

def check_python_version():
    """Check Python version compatibility."""
    print("Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print(f"✗ Python {version.major}.{version.minor} is not supported.")
        print("Please use Python 3.7 or higher.")
        return False
    
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def download_sample_model():
    """Download a sample model (placeholder)."""
    print("Model setup...")
    
    model_path = "models/swintrack_sample.pth"
    if os.path.exists(model_path):
        print(f"✓ Model already exists: {model_path}")
        return True
    
    print("⚠ No model found. You need to download a pretrained SwinTrack model.")
    print("Please download a model from the official SwinTrack repository and place it in the models/ directory.")
    print("For testing, you can create a dummy model file:")
    print(f"  touch {model_path}")
    
    return True

def run_tests():
    """Run installation tests."""
    print("Running installation tests...")
    
    if not run_command("python test_installation.py", "Running installation tests"):
        return False
    
    return True

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description='Setup SwinTrack inference environment')
    parser.add_argument('--skip-deps', action='store_true', help='Skip dependency installation')
    parser.add_argument('--skip-tests', action='store_true', help='Skip installation tests')
    parser.add_argument('--force', action='store_true', help='Force reinstallation of dependencies')
    
    args = parser.parse_args()
    
    print("SwinTrack Inference Setup")
    print("=" * 30)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create directories
    if not create_directories():
        return False
    
    # Install dependencies
    if not args.skip_deps:
        if not install_dependencies():
            print("Dependency installation failed. Please install manually.")
            return False
    else:
        print("Skipping dependency installation")
    
    # Setup model
    download_sample_model()
    
    # Run tests
    if not args.skip_tests:
        if not run_tests():
            print("Installation tests failed. Please check the errors above.")
            return False
    else:
        print("Skipping installation tests")
    
    print("\n" + "=" * 30)
    print("✓ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Download a pretrained SwinTrack model and place it in models/ directory")
    print("2. Prepare your image sequence")
    print("3. Run: python swintrack_inference.py --help")
    print("4. Or try the example: python example_usage.py")
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)