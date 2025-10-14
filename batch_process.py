#!/usr/bin/env python3
"""
Batch processing script for SwinTrack inference.

This script processes multiple image sequences in batch mode.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from swintrack_inference import SwinTrackInference


def find_image_sequences(input_dir):
    """Find all image sequences in the input directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    sequences = {}
    
    for root, dirs, files in os.walk(input_dir):
        # Get all image files in this directory
        image_files = [f for f in files if any(f.lower().endswith(ext) for ext in image_extensions)]
        
        if image_files:
            # Sort files naturally
            image_files.sort()
            sequence_paths = [os.path.join(root, f) for f in image_files]
            sequence_name = os.path.relpath(root, input_dir)
            if sequence_name == '.':
                sequence_name = 'root'
            sequences[sequence_name] = sequence_paths
    
    return sequences


def process_sequence(tracker, sequence_name, image_paths, output_base_dir, initial_bbox=None):
    """Process a single image sequence."""
    print(f"\nProcessing sequence: {sequence_name}")
    print(f"  Images: {len(image_paths)}")
    
    # Create output directory for this sequence
    sequence_output_dir = os.path.join(output_base_dir, sequence_name)
    
    try:
        tracker.process_image_sequence(image_paths, sequence_output_dir, initial_bbox)
        print(f"  ✓ Completed: {sequence_output_dir}")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Batch process image sequences with SwinTrack')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Root directory containing image sequences')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for all results')
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
    parser.add_argument('--sequences', type=str, nargs='*',
                       help='Specific sequences to process (by name). If not provided, process all.')
    parser.add_argument('--max_sequences', type=int, default=None,
                       help='Maximum number of sequences to process')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        return
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file {args.model_path} does not exist")
        return
    
    if not os.path.exists(args.config_path):
        print(f"Error: Config file {args.config_path} does not exist")
        return
    
    # Find image sequences
    print("Scanning for image sequences...")
    sequences = find_image_sequences(args.input_dir)
    
    if not sequences:
        print(f"No image sequences found in {args.input_dir}")
        return
    
    print(f"Found {len(sequences)} image sequences:")
    for name, paths in sequences.items():
        print(f"  {name}: {len(paths)} images")
    
    # Filter sequences if specified
    if args.sequences:
        sequences = {name: paths for name, paths in sequences.items() if name in args.sequences}
        print(f"Filtered to {len(sequences)} specified sequences")
    
    # Limit number of sequences if specified
    if args.max_sequences:
        sequences = dict(list(sequences.items())[:args.max_sequences])
        print(f"Limited to {len(sequences)} sequences")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize SwinTrack
    print(f"\nInitializing SwinTrack on {args.device}...")
    try:
        tracker = SwinTrackInference(args.config_path, args.model_path, args.device)
        print("SwinTrack initialized successfully")
    except Exception as e:
        print(f"Error initializing SwinTrack: {e}")
        return
    
    # Process sequences
    print(f"\nProcessing {len(sequences)} sequences...")
    results = {}
    successful = 0
    failed = 0
    
    for i, (sequence_name, image_paths) in enumerate(sequences.items(), 1):
        print(f"\n[{i}/{len(sequences)}] Processing: {sequence_name}")
        
        success = process_sequence(tracker, sequence_name, image_paths, args.output_dir, args.initial_bbox)
        
        results[sequence_name] = {
            'success': success,
            'num_images': len(image_paths),
            'output_dir': os.path.join(args.output_dir, sequence_name)
        }
        
        if success:
            successful += 1
        else:
            failed += 1
    
    # Save batch results
    batch_results = {
        'total_sequences': len(sequences),
        'successful': successful,
        'failed': failed,
        'sequences': results,
        'config': {
            'input_dir': args.input_dir,
            'output_dir': args.output_dir,
            'model_path': args.model_path,
            'device': args.device,
            'initial_bbox': args.initial_bbox
        }
    }
    
    results_path = os.path.join(args.output_dir, 'batch_results.json')
    with open(results_path, 'w') as f:
        json.dump(batch_results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 50)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 50)
    print(f"Total sequences: {len(sequences)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Results saved to: {args.output_dir}")
    print(f"Batch results: {results_path}")
    
    if failed > 0:
        print("\nFailed sequences:")
        for name, result in results.items():
            if not result['success']:
                print(f"  - {name}")
    
    print("\nBatch processing completed!")


if __name__ == '__main__':
    main()