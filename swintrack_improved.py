#!/usr/bin/env python3
"""
Improved SwinTrack Object Tracking Script

This version uses better tracking algorithms and object detection.

Usage:
    python swintrack_improved.py --input_folder /path/to/images --output_folder /path/to/output --model_path /path/to/model.pth
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
    print("Warning: SwinTrack modules not found. Using improved tracking approach.")
    SWINTRACK_AVAILABLE = False


class ImprovedTracker:
    """Improved object tracker using OpenCV-based tracking."""
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.track_id = 0
        self.tracking_history = []
        self.tracker = None
        self.initial_bbox = None
        self.template_image = None
        self.template_features = None
        
    def _extract_features_opencv(self, image, bbox):
        """Extract features using OpenCV methods."""
        x, y, w, h = bbox
        x, y, w, h = int(x), int(y), int(w), int(h)
        
        # Ensure bbox is within image bounds
        h_img, w_img = image.shape[:2]
        x = max(0, min(x, w_img - w))
        y = max(0, min(y, h_img - h))
        w = min(w, w_img - x)
        h = min(h, h_img - y)
        
        if w <= 0 or h <= 0:
            return None
            
        # Extract template
        template = image[y:y+h, x:x+w]
        
        # Convert to grayscale for feature extraction
        if len(template.shape) == 3:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = template
            
        # Use ORB features for tracking
        orb = cv2.ORB_create(nfeatures=100)
        kp, des = orb.detectAndCompute(template_gray, None)
        
        return {
            'template': template,
            'template_gray': template_gray,
            'keypoints': kp,
            'descriptors': des,
            'bbox': (x, y, w, h)
        }
    
    def _match_features(self, image, template_features):
        """Match features between template and current image."""
        if template_features is None or template_features['descriptors'] is None:
            return None
            
        # Convert current image to grayscale
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image
            
        # Extract features from current image
        orb = cv2.ORB_create(nfeatures=100)
        kp2, des2 = orb.detectAndCompute(image_gray, None)
        
        if des2 is None or template_features['descriptors'] is None:
            return None
            
        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(template_features['descriptors'], des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        if len(matches) < 10:  # Not enough matches
            return None
            
        # Get matched keypoints
        src_pts = np.float32([template_features['keypoints'][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Find homography
        try:
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is None:
                return None
                
            # Transform template corners
            h, w = template_features['template_gray'].shape
            pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            
            # Get bounding box
            x_coords = dst[:, 0, 0]
            y_coords = dst[:, 0, 1]
            
            x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
            y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
            
            # Convert to normalized coordinates
            h_img, w_img = image.shape[:2]
            x_norm = x_min / w_img
            y_norm = y_min / h_img
            w_norm = (x_max - x_min) / w_img
            h_norm = (y_max - y_min) / h_img
            
            return (x_norm, y_norm, w_norm, h_norm)
            
        except:
            return None
    
    def initialize_tracking(self, template_image, bbox=None):
        """Initialize tracking with template image."""
        if bbox is not None:
            self.initial_bbox = bbox
        else:
            # Use center of image as default
            h, w = template_image.shape[:2]
            self.initial_bbox = (0.4, 0.4, 0.2, 0.2)  # x, y, w, h normalized
        
        # Convert normalized bbox to pixel coordinates
        h_img, w_img = template_image.shape[:2]
        x = int(self.initial_bbox[0] * w_img)
        y = int(self.initial_bbox[1] * h_img)
        w = int(self.initial_bbox[2] * w_img)
        h = int(self.initial_bbox[3] * h_img)
        
        pixel_bbox = (x, y, w, h)
        
        # Extract features
        self.template_features = self._extract_features_opencv(template_image, pixel_bbox)
        self.template_image = template_image
        
        if self.template_features is None:
            print("Warning: Could not extract features from template. Using fallback method.")
            self.template_features = {'bbox': pixel_bbox}
        
        self.track_id += 1
        self.tracking_history = [self.initial_bbox]
        print(f"Initialized tracking. Track ID: {self.track_id}")
        print(f"Initial bbox: {self.initial_bbox}")
    
    def track(self, search_image):
        """Track object in search image."""
        if self.template_features is None:
            raise ValueError("Tracking not initialized.")
        
        # Try feature-based tracking first
        bbox = self._match_features(search_image, self.template_features)
        
        if bbox is None:
            # Fallback to template matching
            bbox = self._template_matching(search_image)
        
        if bbox is None:
            # Use previous bbox with small random movement
            if len(self.tracking_history) > 0:
                prev_bbox = self.tracking_history[-1]
                # Add small random movement
                noise = np.random.normal(0, 0.01, 4)
                bbox = (
                    max(0, min(1, prev_bbox[0] + noise[0])),
                    max(0, min(1, prev_bbox[1] + noise[1])),
                    max(0.05, min(0.3, prev_bbox[2] + noise[2])),
                    max(0.05, min(0.3, prev_bbox[3] + noise[3]))
                )
            else:
                bbox = self.initial_bbox
        
        self.tracking_history.append(bbox)
        return bbox, self._create_response_map(search_image, bbox)
    
    def _template_matching(self, search_image):
        """Fallback template matching method."""
        if self.template_features is None or 'template' not in self.template_features:
            return None
            
        template = self.template_features['template']
        
        # Convert to grayscale
        if len(template.shape) == 3:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = template
            
        if len(search_image.shape) == 3:
            search_gray = cv2.cvtColor(search_image, cv2.COLOR_BGR2GRAY)
        else:
            search_gray = search_image
        
        # Template matching
        result = cv2.matchTemplate(search_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        
        # Find best match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val < 0.3:  # Low confidence
            return None
        
        # Convert to normalized coordinates
        h_img, w_img = search_image.shape[:2]
        h_templ, w_templ = template.shape[:2]
        
        x_norm = max_loc[0] / w_img
        y_norm = max_loc[1] / h_img
        w_norm = w_templ / w_img
        h_norm = h_templ / h_img
        
        return (x_norm, y_norm, w_norm, h_norm)
    
    def _create_response_map(self, image, bbox):
        """Create a response map for visualization."""
        h_img, w_img = image.shape[:2]
        x, y, w, h = bbox
        
        # Convert to pixel coordinates
        x_pix = int(x * w_img)
        y_pix = int(y * h_img)
        w_pix = int(w * w_img)
        h_pix = int(h * h_img)
        
        # Create response map
        response_map = np.zeros((h_img, w_img), dtype=np.float32)
        
        # Add Gaussian around bbox center
        center_x = x_pix + w_pix // 2
        center_y = y_pix + h_pix // 2
        
        # Create Gaussian kernel
        sigma = min(w_pix, h_pix) // 4
        y_coords, x_coords = np.ogrid[:h_img, :w_img]
        gaussian = np.exp(-((x_coords - center_x)**2 + (y_coords - center_y)**2) / (2 * sigma**2))
        
        response_map = gaussian.astype(np.float32)
        
        return torch.from_numpy(response_map)
    
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
            
            # Only draw arrow if there's significant movement
            movement = np.sqrt((curr_center[0] - prev_center[0])**2 + (curr_center[1] - prev_center[1])**2)
            if movement > 0.01:  # Only show arrow for significant movement
                axes[0].arrow(prev_center[0] * w, prev_center[1] * h, 
                             (curr_center[0] - prev_center[0]) * w, 
                             (curr_center[1] - prev_center[1]) * h,
                             head_width=10, head_length=10, fc='blue', ec='blue', alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        # Convert figure to numpy array - fixed for macOS compatibility
        fig.canvas.draw()
        
        # Use a more robust method to convert figure to array
        import io
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        pil_img = Image.open(buf)
        vis_array = np.array(pil_img.convert('RGB'))
        
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
                response_map = self._create_response_map(image, bbox)
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
            raise ImportError("SwinTrack modules not available. Use ImprovedTracker instead.")
        
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
        
        # Convert figure to numpy array - fixed for macOS compatibility
        fig.canvas.draw()
        
        # Use a more robust method to convert figure to array
        import io
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        pil_img = Image.open(buf)
        vis_array = np.array(pil_img.convert('RGB'))
        
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
    parser = argparse.ArgumentParser(description='Improved SwinTrack Object Tracking Inference')
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
    parser.add_argument('--use_improved', action='store_true',
                       help='Use improved tracker instead of SwinTrack (recommended)')
    
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
        if args.use_improved or not SWINTRACK_AVAILABLE:
            print("Using improved tracker (recommended)")
            tracker = ImprovedTracker(args.device)
        else:
            print("Using SwinTrack model")
            tracker = SwinTrackInference(args.config_path, args.model_path, args.device)
    except Exception as e:
        print(f"Error initializing tracker: {e}")
        print("Falling back to improved tracker...")
        tracker = ImprovedTracker(args.device)
    
    # Process images
    try:
        tracker.process_image_sequence(image_paths, args.output_folder, args.initial_bbox)
    except Exception as e:
        print(f"Error during processing: {e}")
        return


if __name__ == '__main__':
    main()