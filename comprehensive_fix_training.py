import os
import random
import shutil
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import vgg19
import csv
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import sys
import os
import numpy as np
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# --- COMPREHENSIVE FIX: Simplified Model + Better Loss + Better Training ---

# First, let's create a SIMPLIFIED but CORRECT model
class SimpleFusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Simple encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Bottleneck with attention
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Simple attention mechanism
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, 1),
            nn.Sigmoid()
        )
        
        # Decoder
        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Final output
        self.final = nn.Conv2d(32, 3, 1)
        
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
    def forward(self, rgb, thermal):
        # Encode RGB
        r1 = self.enc1(rgb)
        r2 = self.enc2(self.pool(r1))
        r3 = self.enc3(self.pool(r2))
        
        # Encode Thermal
        t1 = self.enc1(thermal)
        t2 = self.enc2(self.pool(t1))
        t3 = self.enc3(self.pool(t2))
        
        # Fuse features
        fused = r3 + t3  # Simple addition
        fused = self.bottleneck(fused)
        
        # Apply attention
        att = self.attention(fused)
        fused = fused * att
        
        # Decode
        d3 = self.dec3(self.up(fused))
        d2 = self.dec2(self.up(d3))
        d1 = self.dec1(self.up(d2))
        
        # Final output
        out = self.final(d1)
        
        # Ensure output matches input size
        if out.shape[-2:] != rgb.shape[-2:]:
            out = F.interpolate(out, size=rgb.shape[-2:], mode='bilinear', align_corners=False)
        
        return torch.sigmoid(out)

# --- IMPROVED LOSS FUNCTION ---
class ImprovedFusionLoss(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        
        # VGG for perceptual loss
        vgg = vgg19(pretrained=True)
        self.vgg_features = nn.Sequential(*list(vgg.features)[:16]).eval().to(device)
        for param in self.vgg_features.parameters():
            param.requires_grad = False
    
    def forward(self, fused, rgb, thermal, gt):
        # Ensure all tensors have the same spatial size
        target_size = gt.shape[-2:]
        
        if fused.shape[-2:] != target_size:
            fused = F.interpolate(fused, size=target_size, mode='bilinear', align_corners=False)
        if rgb.shape[-2:] != target_size:
            rgb = F.interpolate(rgb, size=target_size, mode='bilinear', align_corners=False)
        if thermal.shape[-2:] != target_size:
            thermal = F.interpolate(thermal, size=target_size, mode='bilinear', align_corners=False)
        
        # 1. PRIMARY LOSS: L1 to GT (most important)
        l1_loss = F.l1_loss(fused, gt)
        
        # 2. SSIM LOSS: Structural similarity
        ssim_loss = 1 - self.ssim(fused, gt)
        
        # 3. PERCEPTUAL LOSS: High-level features
        perceptual_loss = self.perceptual_loss(fused, gt)
        
        # 4. CONTENT PRESERVATION: Ensure both modalities are represented
        rgb_preservation = F.l1_loss(fused, rgb)
        thermal_preservation = F.l1_loss(fused, thermal)
        
        # 5. GRADIENT LOSS: Edge preservation
        gradient_loss = self.gradient_loss(fused, gt)
        
        # SIMPLIFIED LOSS: Focus on the most important components
        total_loss = (2.0 * l1_loss +           # Primary objective
                     1.0 * ssim_loss +          # Structural similarity
                     0.5 * perceptual_loss +    # High-level features
                     0.3 * rgb_preservation +   # RGB content
                     0.3 * thermal_preservation + # Thermal content
                     0.1 * gradient_loss)       # Edge preservation
        
        return total_loss, {
            'l1_loss': l1_loss.item(),
            'ssim_loss': ssim_loss.item(),
            'perceptual_loss': perceptual_loss.item(),
            'rgb_preservation': rgb_preservation.item(),
            'thermal_preservation': thermal_preservation.item(),
            'gradient_loss': gradient_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def ssim(self, pred, target, window_size=11):
        """Simplified SSIM implementation"""
        pred = pred.clamp(0, 1)
        target = target.clamp(0, 1)
        
        mu_pred = F.avg_pool2d(pred, window_size, stride=1, padding=window_size // 2)
        mu_target = F.avg_pool2d(target, window_size, stride=1, padding=window_size // 2)
        
        sigma_pred = F.avg_pool2d(pred**2, window_size, stride=1, padding=window_size // 2) - mu_pred**2
        sigma_target = F.avg_pool2d(target**2, window_size, stride=1, padding=window_size // 2) - mu_target**2
        sigma_pred_target = F.avg_pool2d(pred * target, window_size, stride=1, padding=window_size // 2) - mu_pred * mu_target
        
        c1, c2 = 0.01**2, 0.03**2
        ssim_map = ((2 * mu_pred * mu_target + c1) * (2 * sigma_pred_target + c2)) / \
                ((mu_pred**2 + mu_target**2 + c1) * (sigma_pred + sigma_target + c2))
        
        return ssim_map.mean()
    
    def perceptual_loss(self, fused, gt):
        """Perceptual loss using VGG features"""
        if fused.size(1) == 1:
            fused_3ch = fused.repeat(1, 3, 1, 1)
            gt_3ch = gt.repeat(1, 3, 1, 1)
        else:
            fused_3ch = fused
            gt_3ch = gt
        
        # Denormalize from [-1, 1] to [0, 1]
        fused_3ch = (fused_3ch + 1) / 2
        gt_3ch = (gt_3ch + 1) / 2
        
        # Downsample for memory efficiency
        if fused_3ch.shape[-1] > 256:
            scale_factor = 256 / fused_3ch.shape[-1]
            fused_3ch = F.interpolate(fused_3ch, scale_factor=scale_factor, mode='bilinear', align_corners=False)
            gt_3ch = F.interpolate(gt_3ch, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        
        return F.mse_loss(self.vgg_features(fused_3ch), self.vgg_features(gt_3ch))
    
    def gradient_loss(self, pred, target):
        """Gradient loss for edge preservation"""
        # Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(self.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(self.device)
        
        # Convert to grayscale
        if pred.size(1) == 3:
            pred_gray = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
            target_gray = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]
        else:
            pred_gray = pred
            target_gray = target
        
        # Compute gradients
        pred_grad_x = F.conv2d(pred_gray, sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred_gray, sobel_y, padding=1)
        target_grad_x = F.conv2d(target_gray, sobel_x, padding=1)
        target_grad_y = F.conv2d(target_gray, sobel_y, padding=1)
        
        return F.l1_loss(pred_grad_x, target_grad_x) + F.l1_loss(pred_grad_y, target_grad_y)

# --- TRAINING SETUP ---
num_epochs = 150
MAX_SIZE = 256  # Even smaller for faster training and better convergence

# Define paths
folder1 = '/usr/mvl2/cvrfr/KeyPonFuse_V1/Codes/models_22aug25/training_datasets_full/training_dataset_20k/vi'
folder2 = '/usr/mvl2/cvrfr/KeyPonFuse_V1/Codes/models_22aug25/training_datasets_full/training_dataset_20k/ir'
test_output_folder = '/usr/mvl2/cvrfr/KeyPonFuse_V1/Datasets/Training_Dataset_full/training_dataset_20k/test_ops'  
folder3 = '/usr/mvl2/cvrfr/KeyPonFuse_V1/Codes/models_22aug25/model_outputs/SGT_DIrect_Fusion'

os.makedirs(test_output_folder, exist_ok=True)

# Get all filenames
filenames = sorted(os.listdir(folder1))
random.shuffle(filenames)

# Split
n = len(filenames)
train_idx = int(0.7 * n)
val_idx = int(0.9 * n)

train_files = filenames[:train_idx]
val_files = filenames[train_idx:val_idx]
test_files = filenames[val_idx:] 

# SIMPLIFIED Dataset
class SimpleDataset(Dataset):
    def __init__(self, folder1, folder2, folder3, filenames, is_training=True, max_size=256):
        self.folder1 = folder1
        self.folder2 = folder2
        self.folder3 = folder3
        self.filenames = filenames
        self.is_training = is_training
        self.max_size = max_size

        if is_training:
            self.transform = transforms.Compose([
                transforms.Resize((max_size, max_size)),  # Fixed size for stability
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((max_size, max_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img1_path = os.path.join(self.folder1, self.filenames[idx])
        img2_path = os.path.join(self.folder2, self.filenames[idx])
        gt_path = os.path.join(self.folder3, self.filenames[idx])

        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        gt = Image.open(gt_path).convert('RGB')

        img1 = self.transform(img1)
        img2 = self.transform(img2)
        gt = self.transform(gt)

        return img1, img2, gt, self.filenames[idx]

# Create datasets
train_dataset = SimpleDataset(folder1, folder2, folder3, train_files, is_training=True, max_size=MAX_SIZE)
val_dataset = SimpleDataset(folder1, folder2, folder3, val_files, is_training=False, max_size=MAX_SIZE)
test_dataset = SimpleDataset(folder1, folder2, folder3, test_files, is_training=False, max_size=MAX_SIZE)

# TRAINING FUNCTION
def train_model(model, train_loader, val_loader, epochs, device='cpu'):
    model = model.to(device)
    criterion = ImprovedFusionLoss(device=device)
    
    # SIMPLE OPTIMIZER
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # SIMPLE SCHEDULER
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    scaler = GradScaler()
    best_loss = float('inf')
    early_stop_patience = 20
    early_stop_counter = 0

    for epoch in range(epochs):
        print(f"\n🟢 Epoch {epoch + 1}/{epochs} - Training...")
        model.train()
        total_loss = 0.0

        for step, (img1_batch, img2_batch, gt_batch, _) in enumerate(train_loader):
            img1_batch = img1_batch.to(device)
            img2_batch = img2_batch.to(device)
            gt_batch = gt_batch.to(device)

            optimizer.zero_grad()

            with autocast():
                output = model(img1_batch, img2_batch)
                loss, loss_dict = criterion(output, img1_batch, img2_batch, gt_batch)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️ Skipping step {step} due to NaN/Inf loss.")
                continue

            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            del loss, output

        total_loss /= len(train_loader)
        print(f"✅ Epoch {epoch + 1} Training Loss: {total_loss:.4f}")

        # Validation
        print(f"🔍 Validating Epoch {epoch + 1}...")
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img1_batch, img2_batch, gt_batch, _ in val_loader:
                img1_batch = img1_batch.to(device)
                img2_batch = img2_batch.to(device)
                gt_batch = gt_batch.to(device)

                with autocast():
                    output = model(img1_batch, img2_batch)
                    loss, loss_dict = criterion(output, img1_batch, img2_batch, gt_batch)

                if torch.isnan(loss) or torch.isinf(loss):
                    print("⚠️ Skipping validation batch due to NaN/Inf.")
                    continue

                val_loss += loss.item()
                del loss, output

        val_loss /= len(val_loader)
        print(f"📉 Epoch {epoch + 1} Validation Loss: {val_loss:.4f}")
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"📊 Learning Rate: {current_lr:.2e}")

        # Early stopping
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            early_stop_counter = 0
            print("💾 Model improved. Saving...")

            torch.save(model.state_dict(),
                '/usr/mvl2/cvrfr/KeyPonFuse_V1/Codes/models_22aug25/model_training/trained_models/UNet_simple_fusion_model.pth')
        else:
            early_stop_counter += 1
            print(f"⏳ No improvement. Patience: {early_stop_counter}/{early_stop_patience}")
            if early_stop_counter >= early_stop_patience:
                print("🛑 Early stopping triggered.")
                break

        torch.cuda.empty_cache()

# --- Main Execution ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Use the SIMPLIFIED model
model = SimpleFusionModel()

# SIMPLE training
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

print("🚀 Starting COMPREHENSIVE FIX Training...")
print(f"📊 Image size: {MAX_SIZE}x{MAX_SIZE} (fixed for stability)")
print(f"📊 Batch size: 8 (larger for better gradients)")
print(f"📊 Model: Simplified architecture without attention bugs")
print(f"📊 Loss: Simplified and rebalanced")
print(f"📊 Optimizer: Simple Adam with StepLR")
print(f"📊 Learning Rate: 1e-3 with step decay")

train_model(model, train_loader, val_loader, epochs=num_epochs, device=device)

print("Training Done.")