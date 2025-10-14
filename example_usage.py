#!/usr/bin/env python3
"""
Example usage of SwinTrack inference script.

This script demonstrates how to use the SwinTrackInference class
for object tracking on image sequences.
"""

import os
import cv2
import numpy as np
from swintrack_inference import SwinTrackInference


def create_sample_images(output_dir, num_images=10):
    """Create sample images for testing."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a simple moving object animation
    for i in range(num_images):
        # Create a white background
        img = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        # Add a moving red rectangle
        x = int(50 + i * 20)  # Move horizontally
        y = int(200 + 30 * np.sin(i * 0.5))  # Add some vertical movement
        cv2.rectangle(img, (x, y), (x + 50, y + 50), (0, 0, 255), -1)
        
        # Add some noise
        noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
        
        # Save image
        filename = os.path.join(output_dir, f'frame_{i:04d}.jpg')
        cv2.imwrite(filename, img)
        print(f"Created sample image: {filename}")
    
    return [os.path.join(output_dir, f'frame_{i:04d}.jpg') for i in range(num_images)]


def main():
    """Main example function."""
    print("SwinTrack Inference Example")
    print("=" * 40)
    
    # Create sample images
    print("Creating sample images...")
    sample_dir = "sample_images"
    image_paths = create_sample_images(sample_dir, num_images=15)
    
    # Configuration
    config_path = "config/SwinTrack/Base/config.yaml"
    model_path = "swintrack_model.pth"  # You need to provide this
    output_dir = "example_output"
    device = "cuda" if os.system("nvidia-smi") == 0 else "cpu"
    
    print(f"Using device: {device}")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Warning: Model file {model_path} not found.")
        print("Please download a pretrained SwinTrack model and place it in the current directory.")
        print("You can use a dummy model for testing by creating an empty file.")
        return
    
    try:
        # Initialize SwinTrack
        print("Initializing SwinTrack...")
        tracker = SwinTrackInference(config_path, model_path, device)
        print("SwinTrack initialized successfully!")
        
        # Process the image sequence
        print("Processing image sequence...")
        initial_bbox = (0.1, 0.4, 0.15, 0.15)  # x, y, w, h in normalized coordinates
        tracker.process_image_sequence(image_paths, output_dir, initial_bbox)
        
        print(f"\nExample completed! Check the output in '{output_dir}' folder.")
        print("Output includes:")
        print("- visualizations/: Images with tracking results and track vectors")
        print("- response_maps/: Raw response maps from SwinTrack")
        print("- tracking_results.json: Detailed tracking data")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        print("Make sure you have:")
        print("1. Installed all required dependencies")
        print("2. A valid SwinTrack model weights file")
        print("3. Proper CUDA setup if using GPU")


if __name__ == '__main__':
    main()