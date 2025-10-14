#!/usr/bin/env python3
"""
Standalone SwinTrack Object Tracking Script

This is a simplified version that works without requiring the full SwinTrack repository structure.
It uses a more direct approach to load and use SwinTrack models.

Usage:
    python swintrack_standalone.py --input_folder /path/to/images --output_folder /path/to/output --model_path /path/to/model.pth
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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import warnings
warnings.filterwarnings("ignore")

# Try to import SwinTrack modules with fallback
try:
    # Add current directory to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    
    from miscellanies.yaml_ops import load_yaml
    from models.methods.builder import build_model
    from core.run.event_dispatcher.register import EventRegister
    SWINTRACK_AVAILABLE = True
except ImportError:
    print("Warning: SwinTrack modules not found. Using simplified approach.")
    SWINTRACK_AVAILABLE = False


class SimpleTracker:
    """Simplified object tracker for demonstration purposes."""
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.template_features = None
        self.track_id = 0
        self.tracking_history = []
        self.template_size = (112, 112)
        self.search_size = (224, 224)
        
    def _preprocess_image(self, image, target_size):
        """Preprocess image for tracking."""
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        pil_image = Image.fromarray(image)
        pil_image = pil_image.resize((target_size[1], target_size[0]), Image.BILINEAR)
        
        image_tensor = torch.from_numpy(np.array(pil_image)).float() / 255.0
        
        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        image_tensor = (image_tensor - mean) / std
        
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
        return image_tensor.to(self.device)
    
    def initialize_tracking(self, template_image, bbox=None):
        """Initialize tracking with template image."""
        template_tensor = self._preprocess_image(template_image, self.template_size)
        
        # Simple feature extraction (placeholder)
        with torch.no_grad():
            # Use a simple CNN feature extractor
            self.template_features = self._extract_features(template_tensor)
        
        if bbox is not None:
            self.initial_bbox = bbox
        else:
            h, w = template_image.shape[:2]
            self.initial_bbox = (0.5 - 0.1, 0.5 - 0.1, 0.2, 0.2)
        
        self.track_id += 1
        self.tracking_history = [self.initial_bbox]
        print(f"Initialized tracking. Track ID: {self.track_id}")
    
    def _extract_features(self, image_tensor):
        """Extract features using a simple CNN."""
        # Simple feature extraction using average pooling
        features = torch.nn.functional.adaptive_avg_pool2d(image_tensor, (7, 7))
        return features.flatten(1)
    
    def track(self, search_image):
        """Track object in search image."""
        if self.template_features is None:
            raise ValueError("Tracking not initialized.")
        
        search_tensor = self._preprocess_image(search_image, self.search_size)
        
        with torch.no_grad():
            search_features = self._extract_features(search_tensor)
            
            # Simple template matching
            similarity = torch.cosine_similarity(
                self.template_features, search_features, dim=1
            )
            
            # Create response map (simplified)
            response_map = torch.ones(self.search_size)
            response_map = response_map * similarity.item()
        
        # Simple bounding box estimation
        bbox = self._estimate_bbox(response_map)
        self.tracking_history.append(bbox)
        
        return bbox, response_map
    
    def _estimate_bbox(self, response_map):
        """Estimate bounding box from response map."""
        h, w = response_map.shape
        max_idx = torch.argmax(response_map.flatten())
        y, x = max_idx // w, max_idx % w
        
        x_norm = x / w
        y_norm = y / h
        
        # Fixed size for simplicity
        bbox_w = 0.1
        bbox_h = 0.1
        
        return (x_norm - bbox_w/2, y_norm - bbox_h/2, bbox_w, bbox_h)
    
    def visualize_tracking(self, image, bbox, response_map, save_path=None):
        """Visualize tracking results."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original image with bounding box
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        h, w = image.shape[:2]
        x, y, bbox_w, bbox_h = bbox
        rect = patches.Rectangle((x * w, y * h), bbox_w * w, bbox_h * h, 
                               linewidth=2, edgecolor='red', facecolor='none')
        axes[0].add_patch(rect)
        axes[0].set_title('Tracking Result')
        axes[0].axis('off')
        
        # Response map
        response_np = response_map.numpy()
        im = axes[1].imshow(response_np, cmap='hot', interpolation='nearest')
        axes[1].set_title('Response Map')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
        
        # Add track vector
        if len(self.tracking_history) > 1:
            prev_bbox = self.tracking_history[-2]
            curr_bbox = self.tracking_history[-1]
            
            prev_center = (prev_bbox[0] + prev_bbox[2]/2, prev_bbox[1] + prev_bbox[3]/2)
            curr_center = (curr_bbox[0] + curr_bbox[2]/2, curr_bbox[1] + curr_bbox[3]/2)
            
            axes[0].arrow(prev_center[0] * w, prev_center[1] * h, 
                         (curr_center[0] - prev_center[0]) * w, 
                         (curr_center[1] - prev_center[1]) * h,
                         head_width=10, head_length=10, fc='blue', ec='blue', alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        fig.canvas.draw()
        vis_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        vis_array = vis_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        return vis_array
    
    def process_image_sequence(self, image_paths, output_dir, initial_bbox=None):
        """Process image sequence."""
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'response_maps'), exist_ok=True)
        
        tracking_results = []
        
        for i, image_path in enumerate(image_paths):
            print(f"Processing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
            
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not load image {image_path}")
                continue
            
            if i == 0:
                self.initialize_tracking(image, initial_bbox)
                bbox = self.initial_bbox
                response_map = torch.ones(self.search_size)
            else:
                bbox, response_map = self.track(image)
            
            # Save response map
            response_map_np = response_map.numpy()
            response_path = os.path.join(output_dir, 'response_maps', f'response_{i:04d}.npy')
            np.save(response_path, response_map_np)
            
            # Create visualization
            vis_image = self.visualize_tracking(image, bbox, response_map)
            vis_path = os.path.join(output_dir, 'visualizations', f'vis_{i:04d}.jpg')
            cv2.imwrite(vis_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            
            result = {
                'frame': i,
                'image_path': image_path,
                'bbox': bbox,
                'response_map_path': response_path
            }
            tracking_results.append(result)
            
            print(f"  Bounding box: {bbox}")
        
        # Save results
        results_path = os.path.join(output_dir, 'tracking_results.json')
        with open(results_path, 'w') as f:
            json.dump(tracking_results, f, indent=2)
        
        print(f"\nTracking completed! Results saved to {output_dir}")


class SwinTrackInference:
    """SwinTrack inference class (if modules available)."""
    
    def __init__(self, config_path, model_path, device='cuda'):
        if not SWINTRACK_AVAILABLE:
            raise ImportError("SwinTrack modules not available. Use SimpleTracker instead.")
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.config = load_yaml(config_path)
        
        # Build model
        self.model = self._build_model()
        self._load_weights(model_path)
        self.model.eval()
        
        # Image preprocessing parameters
        self.template_size = self.config['data']['template_size']
        self.search_size = self.config['data']['search_size']
        self.imagenet_normalization = self.config['data']['imagenet_normalization']
        
        # Tracking state
        self.template_features = None
        self.track_id = 0
        self.tracking_history = []
    
    def _build_model(self):
        """Build SwinTrack model from config."""
        class RuntimeVars:
            def __init__(self):
                self.resume = None
                self.weight_path = None
                self.device = str(self.device)
                self.distributed = False
                self.local_rank = 0
                self.enable_autograd_detect_anomaly = False
        
        runtime_vars = RuntimeVars()
        runtime_vars.device = str(self.device)
        
        model, _ = build_model(
            self.config, 
            runtime_vars, 
            batch_size=1, 
            num_epochs=1, 
            iterations_per_epoch=1, 
            event_register=EventRegister('inference/'),
            has_training_run=False
        )
        
        return model
    
    def _load_weights(self, model_path):
        """Load pretrained weights into the model."""
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"Loaded weights from {model_path}")
        else:
            print(f"Warning: Model weights not found at {model_path}. Using random weights.")
    
    def _preprocess_image(self, image, target_size):
        """Preprocess image for SwinTrack inference."""
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        pil_image = Image.fromarray(image)
        pil_image = pil_image.resize((target_size[1], target_size[0]), Image.BILINEAR)
        
        image_tensor = torch.from_numpy(np.array(pil_image)).float() / 255.0
        
        if self.imagenet_normalization:
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])
            image_tensor = (image_tensor - mean) / std
        
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
        return image_tensor.to(self.device)
    
    def _extract_bbox_from_response(self, response_map):
        """Extract bounding box from response map."""
        h, w = response_map.shape
        max_idx = torch.argmax(response_map.flatten())
        y, x = max_idx // w, max_idx % w
        
        x_norm = x / w
        y_norm = y / h
        
        bbox_w = 0.1
        bbox_h = 0.1
        
        return (x_norm - bbox_w/2, y_norm - bbox_h/2, bbox_w, bbox_h)
    
    def initialize_tracking(self, template_image, bbox=None):
        """Initialize tracking with a template image."""
        template_tensor = self._preprocess_image(template_image, self.template_size)
        
        with torch.no_grad():
            self.template_features = self.model.initialize(template_tensor)
        
        if bbox is not None:
            self.initial_bbox = bbox
        else:
            h, w = template_image.shape[:2]
            self.initial_bbox = (0.5 - 0.1, 0.5 - 0.1, 0.2, 0.2)
        
        self.track_id += 1
        self.tracking_history = [self.initial_bbox]
        print(f"Initialized tracking with template image. Track ID: {self.track_id}")
    
    def track(self, search_image):
        """Track object in search image."""
        if self.template_features is None:
            raise ValueError("Tracking not initialized. Call initialize_tracking() first.")
        
        search_tensor = self._preprocess_image(search_image, self.search_size)
        
        with torch.no_grad():
            outputs = self.model.track(self.template_features, search_tensor)
        
        if isinstance(outputs, dict):
            if 'class_score' in outputs:
                response_map = outputs['class_score'].squeeze().cpu()
            elif 'bbox' in outputs:
                response_map = outputs['bbox'].squeeze().cpu()
            else:
                response_map = list(outputs.values())[0].squeeze().cpu()
        else:
            response_map = outputs.squeeze().cpu()
        
        bbox = self._extract_bbox_from_response(response_map)
        self.tracking_history.append(bbox)
        
        return bbox, response_map
    
    def visualize_tracking(self, image, bbox, response_map, save_path=None):
        """Visualize tracking results with bounding box and response map."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        h, w = image.shape[:2]
        x, y, bbox_w, bbox_h = bbox
        rect = patches.Rectangle((x * w, y * h), bbox_w * w, bbox_h * h, 
                               linewidth=2, edgecolor='red', facecolor='none')
        axes[0].add_patch(rect)
        axes[0].set_title('Tracking Result')
        axes[0].axis('off')
        
        response_np = response_map.numpy()
        im = axes[1].imshow(response_np, cmap='hot', interpolation='nearest')
        axes[1].set_title('Response Map')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
        
        if len(self.tracking_history) > 1:
            prev_bbox = self.tracking_history[-2]
            curr_bbox = self.tracking_history[-1]
            
            prev_center = (prev_bbox[0] + prev_bbox[2]/2, prev_bbox[1] + prev_bbox[3]/2)
            curr_center = (curr_bbox[0] + curr_bbox[2]/2, curr_bbox[1] + curr_bbox[3]/2)
            
            axes[0].arrow(prev_center[0] * w, prev_center[1] * h, 
                         (curr_center[0] - prev_center[0]) * w, 
                         (curr_center[1] - prev_center[1]) * h,
                         head_width=10, head_length=10, fc='blue', ec='blue', alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        fig.canvas.draw()
        vis_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        vis_array = vis_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        return vis_array
    
    def process_image_sequence(self, image_paths, output_dir, initial_bbox=None):
        """Process a sequence of images for object tracking."""
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'response_maps'), exist_ok=True)
        
        tracking_results = []
        
        for i, image_path in enumerate(image_paths):
            print(f"Processing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
            
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not load image {image_path}")
                continue
            
            if i == 0:
                self.initialize_tracking(image, initial_bbox)
                bbox = self.initial_bbox
                response_map = torch.zeros(self.search_size)
            else:
                bbox, response_map = self.track(image)
            
            response_map_np = response_map.numpy()
            response_path = os.path.join(output_dir, 'response_maps', f'response_{i:04d}.npy')
            np.save(response_path, response_map_np)
            
            vis_image = self.visualize_tracking(image, bbox, response_map)
            vis_path = os.path.join(output_dir, 'visualizations', f'vis_{i:04d}.jpg')
            cv2.imwrite(vis_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            
            result = {
                'frame': i,
                'image_path': image_path,
                'bbox': bbox,
                'response_map_path': response_path
            }
            tracking_results.append(result)
            
            print(f"  Bounding box: {bbox}")
        
        results_path = os.path.join(output_dir, 'tracking_results.json')
        with open(results_path, 'w') as f:
            json.dump(tracking_results, f, indent=2)
        
        print(f"\nTracking completed! Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='SwinTrack Object Tracking Inference')
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
    parser.add_argument('--use_simple', action='store_true',
                       help='Use simple tracker instead of SwinTrack (fallback)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.input_folder):
        print(f"Error: Input folder {args.input_folder} does not exist")
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
    
    # Initialize tracker
    try:
        if args.use_simple or not SWINTRACK_AVAILABLE:
            print("Using simple tracker (fallback mode)")
            tracker = SimpleTracker(args.device)
        else:
            print("Using SwinTrack model")
            tracker = SwinTrackInference(args.config_path, args.model_path, args.device)
    except Exception as e:
        print(f"Error initializing tracker: {e}")
        print("Falling back to simple tracker...")
        tracker = SimpleTracker(args.device)
    
    # Process images
    try:
        tracker.process_image_sequence(image_paths, args.output_folder, args.initial_bbox)
    except Exception as e:
        print(f"Error during processing: {e}")
        return


if __name__ == '__main__':
    main()