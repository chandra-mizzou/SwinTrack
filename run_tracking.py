#!/usr/bin/env python3
"""
Helper script to run SwinTrack tracking from any directory.

This script automatically handles the path setup and runs the tracking.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def find_swintrack_root():
    """Find the SwinTrack repository root directory."""
    current_dir = Path.cwd()
    
    # Look for SwinTrack repository indicators
    indicators = [
        'main.py',
        'config/SwinTrack',
        'models/methods/SwinTrack',
        'miscellanies'
    ]
    
    # Check current directory
    if all((current_dir / indicator).exists() for indicator in indicators):
        return current_dir
    
    # Check parent directories
    for parent in current_dir.parents:
        if all((parent / indicator).exists() for indicator in indicators):
            return parent
    
    return None

def main():
    parser = argparse.ArgumentParser(description='Run SwinTrack tracking')
    parser.add_argument('--input_folder', type=str, required=True,
                       help='Path to folder containing input images')
    parser.add_argument('--output_folder', type=str, required=True,
                       help='Path to folder for saving outputs')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to pretrained SwinTrack model weights')
    parser.add_argument('--config_path', type=str, 
                       default='config/SwinTrack/Base/config.yaml',
                       help='Path to SwinTrack config file')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run inference on')
    parser.add_argument('--initial_bbox', type=float, nargs=4, metavar=('X', 'Y', 'W', 'H'),
                       help='Initial bounding box in normalized coordinates (x, y, w, h)')
    parser.add_argument('--use_standalone', action='store_true',
                       help='Use standalone version instead of full SwinTrack')
    
    args = parser.parse_args()
    
    # Find SwinTrack root
    swintrack_root = find_swintrack_root()
    
    if swintrack_root is None:
        print("Error: Could not find SwinTrack repository root directory.")
        print("Please run this script from within the SwinTrack repository or")
        print("use the --use_standalone flag to use the simplified version.")
        return
    
    print(f"Found SwinTrack repository at: {swintrack_root}")
    
    # Change to SwinTrack root directory
    os.chdir(swintrack_root)
    print(f"Changed working directory to: {os.getcwd()}")
    
    # Build command
    if args.use_standalone:
        script_name = "swintrack_standalone.py"
    else:
        script_name = "swintrack_inference.py"
    
    cmd = [
        sys.executable, script_name,
        "--input_folder", args.input_folder,
        "--output_folder", args.output_folder,
        "--model_path", args.model_path,
        "--config_path", args.config_path,
        "--device", args.device
    ]
    
    if args.initial_bbox:
        cmd.extend(["--initial_bbox"] + [str(x) for x in args.initial_bbox])
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Run the command
    try:
        result = subprocess.run(cmd, check=True)
        print("Tracking completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error running tracking: {e}")
        return
    except FileNotFoundError:
        print(f"Error: {script_name} not found in SwinTrack directory.")
        print("Make sure the script files are in the SwinTrack repository root.")

if __name__ == '__main__':
    main()