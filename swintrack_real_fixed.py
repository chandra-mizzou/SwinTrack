#!/usr/bin/env python3
"""
Real SwinTrack Object Tracking Script - FIXED VERSION

This version correctly uses SwinTrack for object tracking with proper input formats.

Usage:
    python swintrack_real_fixed.py --input_folder /path/to/images --output_folder /path/to/output --model_path /path/to/model.pth
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import warnings
warnings.filterwarnings("ignore")

# Add SwinTrack repository to path
current_dir = os.path.dirname(os.path.abspath(__file__))
swintrack_root = current_dir  # Assuming we're in the SwinTrack repo
sys.path.insert(0, swintrack_root)

print(f"Looking for SwinTrack modules in: {swintrack_root}")
print(f"Current working directory: {os.getcwd()}")

# Try to import SwinTrack modules
try:
    from miscellanies.yaml_ops import load_yaml
    from models.methods.builder import build_model
    from core.run.event_dispatcher.register import EventRegister
    print("✓ Successfully imported SwinTrack modules")
    SWINTRACK_AVAILABLE = True
except ImportError as e:
    print(f"✗ Failed to import SwinTrack modules: {e}")
    print("Available files in current directory:")
    for f in os.listdir('.'):
        print(f"  {f}")
    SWINTRACK_AVAILABLE = False


class SwinTrackTracker:
    """Real SwinTrack-based object tracker with correct input formats."""
    
    def __init__(self, config_path, model_path, device='cuda'):
        if not SWINTRACK_AVAILABLE:
            raise ImportError("SwinTrack modules not available. Cannot use SwinTrack tracker.")
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load config
        self.config = load_yaml(config_path)
        print(f"Loaded config from: {config_path}")
        
        # Build model
        self.model = self._build_model()
        self._load_weights(model_path)
        self.model.eval()
        
        # Image preprocessing parameters from config
        self.template_size = self.config['data']['template_size']  # [H, W]
        self.search_size = self.config['data']['search_size']      # [H, W]
        self.imagenet_normalization = self.config['data']['imagenet_normalization']
        
        print(f"Template size: {self.template_size}")
        print(f"Search size: {self.search_size}")
        print(f"ImageNet normalization: {self.imagenet_normalization}")
        
        # Tracking state
        self.template_features = None
        self.track_id = 0
        self.tracking_history = []
        self.initial_bbox = None
    
    def _build_model(self):
        """Build SwinTrack model from config."""
        print("Building SwinTrack model...")
        
        class RuntimeVars:
            def __init__(self, device):
                self.resume = None
                self.weight_path = None
                self.device = str(device)
                self.distributed = False
                self.local_rank = 0
                self.enable_autograd_detect_anomaly = False
        
        runtime_vars = RuntimeVars(self.device)
        
        try:
            model, _ = build_model(
                self.config, 
                runtime_vars, 
                max_batch_size=1, 
                num_epochs=1, 
                iterations_per_epoch=1, 
                event_register=EventRegister('inference/'),
                has_training_run=False
            )
            print("✓ SwinTrack model built successfully")
            return model
        except Exception as e:
            print(f"✗ Failed to build SwinTrack model: {e}")
            raise
    
    def _load_weights(self, model_path):
        """Load pretrained weights into the model."""
        print(f"Loading weights from: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"✗ Model weights not found at {model_path}")
            raise FileNotFoundError(f"Model weights not found: {model_path}")
        
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            print(f"Checkpoint keys: {list(checkpoint.keys())}")
            
            if 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'])
                print("✓ Loaded model weights from checkpoint['model']")
            else:
                self.model.load_state_dict(checkpoint)
                print("✓ Loaded model weights directly from checkpoint")
                
        except Exception as e:
            print(f"✗ Failed to load weights: {e}")
            raise
    
    def _crop_bbox_region(self, image, bbox):
        """Crop bounding box region from image."""
        h, w = image.shape[:2]
        x, y, bbox_w, bbox_h = bbox
        
        # Convert normalized coordinates to pixel coordinates
        x1 = int(x * w)
        y1 = int(y * h)
        x2 = int((x + bbox_w) * w)
        y2 = int((y + bbox_h) * h)
        
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(x1+1, min(x2, w))
        y2 = max(y1+1, min(y2, h))
        
        # Crop the region
        cropped = image[y1:y2, x1:x2]
        return cropped
    
    def _preprocess_image_for_swintrack(self, image, target_size, bbox=None, is_template=False):
        """
        Preprocess image for SwinTrack inference with correct format.
        
        The SwinTrack model expects (B, H*W, 3) format, but the backbone expects (B, C, H, W).
        We need to provide (B, C, H, W) format to the backbone.
        """
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # For template, crop the bbox region first
        if is_template and bbox is not None:
            image = self._crop_bbox_region(image, bbox)
            print(f"Cropped template region from bbox {bbox}, new size: {image.shape}")
        
        # Resize image to target size
        pil_image = Image.fromarray(image)
        pil_image = pil_image.resize((target_size[1], target_size[0]), Image.BILINEAR)
        
        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(np.array(pil_image)).float() / 255.0
        
        if self.imagenet_normalization:
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])
            image_tensor = (image_tensor - mean) / std
        
        # Convert from (H, W, C) to (B, C, H, W) format expected by backbone
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # Shape: (1, 3, H, W)
        
        return image_tensor.to(self.device)
    
    def _extract_bbox_from_response(self, response_map, search_size):
        """
        Extract bounding box from SwinTrack response map.
        
        SwinTrack outputs response maps of size (H, W) where H and W correspond
        to the search image spatial dimensions.
        """
        if len(response_map.shape) == 3:
            response_map = response_map.squeeze(0)  # Remove batch dimension if present
        
        h, w = response_map.shape
        max_idx = torch.argmax(response_map.flatten())
        y, x = max_idx // w, max_idx % w
        
        # Convert to normalized coordinates
        x_norm = (x + 0.5) / w  # Center of the response
        y_norm = (y + 0.5) / h
        
        # Use a fixed bbox size relative to search image
        # This could be made configurable or learned
        bbox_w = 0.2  # 20% of search image width
        bbox_h = 0.2  # 20% of search image height
        
        # Convert to (x, y, w, h) format in normalized coordinates
        bbox_x = max(0, x_norm - bbox_w / 2)
        bbox_y = max(0, y_norm - bbox_h / 2)
        bbox_w = min(bbox_w, 1.0 - bbox_x)
        bbox_h = min(bbox_h, 1.0 - bbox_y)
        
        return (float(bbox_x), float(bbox_y), float(bbox_w), float(bbox_h))
    
    def initialize_tracking(self, template_image, bbox=None):
        """Initialize tracking with a template image."""
        print("Initializing SwinTrack tracking...")
        
        # Use provided bbox or default center bbox
        if bbox is None:
            h, w = template_image.shape[:2]
            # Default to center 20% of image
            bbox = (0.4, 0.4, 0.2, 0.2)
        
        self.initial_bbox = bbox
        print(f"Using bbox for template: {bbox}")
        
        # Preprocess template image with correct format - CROP from bbox first
        template_tensor = self._preprocess_image_for_swintrack(
            template_image, self.template_size, bbox, is_template=True
        )
        print(f"Template tensor shape: {template_tensor.shape}")
        
        # Initialize template features using SwinTrack's initialize method
        with torch.no_grad():
            self.template_features = self.model.initialize(template_tensor)
            print(f"Template features shape: {self.template_features.shape}")
        
        self.track_id += 1
        self.tracking_history = [bbox]
        print(f"✓ SwinTrack tracking initialized. Track ID: {self.track_id}")
        print(f"Initial bbox: {bbox}")
    
    def track(self, search_image):
        """Track object in search image using SwinTrack."""
        if self.template_features is None:
            raise ValueError("Tracking not initialized. Call initialize_tracking() first.")
        
        # Preprocess search image with correct format (no bbox cropping for search)
        search_tensor = self._preprocess_image_for_swintrack(
            search_image, self.search_size, is_template=False
        )
        print(f"Search tensor shape: {search_tensor.shape}")
        
        # Track using SwinTrack's track method
        with torch.no_grad():
            outputs = self.model.track(self.template_features, search_tensor)
            print(f"SwinTrack outputs type: {type(outputs)}")
            if isinstance(outputs, dict):
                print(f"SwinTrack output keys: {list(outputs.keys())}")
                for k, v in outputs.items():
                    print(f"  {k}: {v.shape if hasattr(v, 'shape') else type(v)}")
        
        # Extract response map from outputs
        if isinstance(outputs, dict):
            if 'class_score' in outputs:
                response_map = outputs['class_score'].squeeze().cpu()
                print("Using 'class_score' output")
            elif 'bbox' in outputs:
                response_map = outputs['bbox'].squeeze().cpu()
                print("Using 'bbox' output")
            else:
                # Use the first available output
                key = list(outputs.keys())[0]
                response_map = outputs[key].squeeze().cpu()
                print(f"Using '{key}' output")
        else:
            response_map = outputs.squeeze().cpu()
            print("Using direct output")
        
        print(f"Response map shape: {response_map.shape}")
        
        # Extract bounding box from response map
        bbox = self._extract_bbox_from_response(response_map, self.search_size)
        self.tracking_history.append(bbox)
        
        print(f"SwinTrack bbox: {bbox}")
        return bbox, response_map
    
    def visualize_tracking(self, image, bbox, response_map, save_path=None, show_template=False):
        """Visualize tracking results with bounding box and response map."""
        if show_template and self.initial_bbox is not None:
            fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        else:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original image with bounding box
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        h, w = image.shape[:2]
        x, y, bbox_w, bbox_h = bbox
        rect = patches.Rectangle((x * w, y * h), bbox_w * w, bbox_h * h, 
                               linewidth=2, edgecolor='red', facecolor='none')
        axes[0].add_patch(rect)
        axes[0].set_title('SwinTrack Tracking Result')
        axes[0].axis('off')
        
        # Show template extraction if requested
        if show_template and self.initial_bbox is not None:
            template_cropped = self._crop_bbox_region(image, self.initial_bbox)
            axes[1].imshow(template_cropped)
            axes[1].set_title('Extracted Template')
            axes[1].axis('off')
            response_ax = axes[2]
        else:
            response_ax = axes[1]
        
        # Response map
        response_np = response_map.numpy()
        im = response_ax.imshow(response_np, cmap='hot', interpolation='nearest')
        response_ax.set_title('SwinTrack Response Map')
        response_ax.axis('off')
        plt.colorbar(im, ax=response_ax)
        
        # Add track vector
        if len(self.tracking_history) > 1:
            prev_bbox = self.tracking_history[-2]
            curr_bbox = self.tracking_history[-1]
            
            prev_center = (prev_bbox[0] + prev_bbox[2]/2, prev_bbox[1] + prev_bbox[3]/2)
            curr_center = (curr_bbox[0] + curr_bbox[2]/2, curr_bbox[1] + curr_bbox[3]/2)
            
            # Only draw arrow if there's significant movement
            movement = np.sqrt((curr_center[0] - prev_center[0])**2 + (curr_center[1] - prev_center[1])**2)
            if movement > 0.01:
                axes[0].arrow(prev_center[0] * w, prev_center[1] * h, 
                             (curr_center[0] - prev_center[0]) * w, 
                             (curr_center[1] - prev_center[1]) * h,
                             head_width=10, head_length=10, fc='blue', ec='blue', alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        # Convert figure to numpy array
        fig.canvas.draw()
        
        import io
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        pil_img = Image.open(buf)
        vis_array = np.array(pil_img.convert('RGB'))
        
        plt.close(fig)
        
        return vis_array
    
    def process_image_sequence(self, image_paths, output_dir, initial_bbox=None):
        """Process a sequence of images for object tracking using SwinTrack."""
        print(f"Processing {len(image_paths)} images with SwinTrack...")
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'response_maps'), exist_ok=True)
        
        tracking_results = []
        
        for i, image_path in enumerate(image_paths):
            print(f"\n--- Processing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)} ---")
            
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not load image {image_path}")
                continue
            
            if i == 0:
                # Initialize tracking with first image
                self.initialize_tracking(image, initial_bbox)
                bbox = self.initial_bbox
                response_map = torch.zeros(self.search_size)
            else:
                # Track in subsequent images
                bbox, response_map = self.track(image)
            
            # Save response map
            response_map_np = response_map.numpy()
            response_path = os.path.join(output_dir, 'response_maps', f'response_{i:04d}.npy')
            np.save(response_path, response_map_np)
            
            # Create visualization (show template extraction for first frame)
            show_template = (i == 0)
            vis_image = self.visualize_tracking(image, bbox, response_map, show_template=show_template)
            vis_path = os.path.join(output_dir, 'visualizations', f'vis_{i:04d}.jpg')
            cv2.imwrite(vis_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            
            result = {
                'frame': i,
                'image_path': image_path,
                'bbox': bbox,
                'response_map_path': response_path
            }
            tracking_results.append(result)
            
            print(f"  Final bbox: {bbox}")
        
        # Save results
        results_path = os.path.join(output_dir, 'tracking_results.json')
        with open(results_path, 'w') as f:
            json.dump(tracking_results, f, indent=2)
        
        print(f"\n✓ SwinTrack tracking completed! Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Real SwinTrack Object Tracking - Fixed Version')
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
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.input_folder):
        print(f"Error: Input folder {args.input_folder} does not exist")
        return
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file {args.model_path} does not exist")
        return
    
    if not os.path.exists(args.config_path):
        print(f"Error: Config file {args.config_path} does not exist")
        return
    
    # Get list of image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_paths = []
    for file in os.listdir(args.input_folder):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_paths.append(os.path.join(args.input_folder, file))
    
    if not image_paths:
        print(f"Error: No image files found in {args.input_folder}")
        return
    
    image_paths.sort()
    print(f"Found {len(image_paths)} images to process")
    
    # Initialize SwinTrack tracker
    if not SWINTRACK_AVAILABLE:
        print("Error: SwinTrack modules not available. Cannot proceed.")
        print("Make sure you're running this script from the SwinTrack repository root.")
        return
    
    try:
        print("Initializing SwinTrack tracker...")
        tracker = SwinTrackTracker(args.config_path, args.model_path, args.device)
        print("✓ SwinTrack tracker initialized successfully")
    except Exception as e:
        print(f"Error initializing SwinTrack tracker: {e}")
        return
    
    # Process images
    try:
        tracker.process_image_sequence(image_paths, args.output_folder, args.initial_bbox)
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == '__main__':
    main()