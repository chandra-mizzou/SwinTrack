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
from memory_efficient_transformer import MemoryEfficientRGBThermalFusion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

# --- MEMORY EFFICIENT LOSS FUNCTION ---
class MemoryEfficientLoss(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        
        # VGG for perceptual loss (smaller version)
        vgg = vgg19(pretrained=True)
        self.vgg_features = nn.Sequential(*list(vgg.features)[:8]).eval().to(device)  # Reduced layers
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
        
        # 1. PRIMARY LOSS: L1 to GT
        l1_loss = F.l1_loss(fused, gt)
        
        # 2. SSIM LOSS: Structural similarity
        ssim_loss = 1 - self.ssim(fused, gt)
        
        # 3. PERCEPTUAL LOSS: High-level features (downsampled for memory)
        perceptual_loss = self.perceptual_loss(fused, gt)
        
        # 4. CONTENT PRESERVATION: Ensure both modalities are represented
        rgb_preservation = F.l1_loss(fused, rgb)
        thermal_preservation = F.l1_loss(fused, thermal)
        
        # 5. GRADIENT LOSS: Edge preservation
        gradient_loss = self.gradient_loss(fused, gt)
        
        # SIMPLIFIED LOSS
        total_loss = (3.0 * l1_loss +           # Primary objective
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
        """SSIM implementation"""
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
        """Perceptual loss using VGG features (memory efficient)"""
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
        if fused_3ch.shape[-1] > 128:  # Reduced from 256
            scale_factor = 128 / fused_3ch.shape[-1]
            fused_3ch = F.interpolate(fused_3ch, scale_factor=scale_factor, mode='bilinear', align_corners=False)
            gt_3ch = F.interpolate(gt_3ch, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        
        return F.mse_loss(self.vgg_features(fused_3ch), self.vgg_features(gt_3ch))
    
    def gradient_loss(self, pred, target):
        """Gradient loss for edge preservation"""
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(self.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(self.device)
        
        if pred.size(1) == 3:
            pred_gray = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
            target_gray = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]
        else:
            pred_gray = pred
            target_gray = target
        
        pred_grad_x = F.conv2d(pred_gray, sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred_gray, sobel_y, padding=1)
        target_grad_x = F.conv2d(target_gray, sobel_x, padding=1)
        target_grad_y = F.conv2d(target_gray, sobel_y, padding=1)
        
        return F.l1_loss(pred_grad_x, target_grad_x) + F.l1_loss(pred_grad_y, target_grad_y)

# --- DATASET CLASS ---
class MemoryEfficientDataset(Dataset):
    def __init__(self, folder1, folder2, folder3, filenames, is_training=True, max_size=256):
        self.folder1 = folder1
        self.folder2 = folder2
        self.folder3 = folder3
        self.filenames = filenames
        self.is_training = is_training
        self.max_size = max_size

        if is_training:
            self.transform = transforms.Compose([
                transforms.Resize((max_size, max_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(degrees=5),  # Reduced rotation
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),  # Reduced jitter
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

# --- TRAINING FUNCTION ---
def train_memory_efficient_model(model, train_loader, val_loader, epochs, device='cpu'):
    model = model.to(device)
    criterion = MemoryEfficientLoss(device=device)
    
    # Simple optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # Simple scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    
    scaler = GradScaler()
    best_loss = float('inf')
    early_stop_patience = 20
    early_stop_counter = 0

    for epoch in range(epochs):
        print(f"\n🟢 Epoch {epoch + 1}/{epochs} - Training...")
        model.train()
        total_loss = 0.0
        loss_components = {}

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Reduced clipping
            
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            
            # Accumulate loss components
            for key, value in loss_dict.items():
                if key not in loss_components:
                    loss_components[key] = 0
                loss_components[key] += value
            
            del loss, output
            
            # Clear cache every few steps
            if step % 10 == 0:
                torch.cuda.empty_cache()

        total_loss /= len(train_loader)
        for key in loss_components:
            loss_components[key] /= len(train_loader)
        
        print(f"✅ Epoch {epoch + 1} Training Loss: {total_loss:.4f}")
        print(f"   L1: {loss_components['l1_loss']:.4f}, SSIM: {loss_components['ssim_loss']:.4f}")

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
                '/usr/mvl2/cvrfr/KeyPonFuse_V1/Codes/models_22aug25/model_training/trained_models/memory_efficient_transformer.pth')
        else:
            early_stop_counter += 1
            print(f"⏳ No improvement. Patience: {early_stop_counter}/{early_stop_patience}")
            if early_stop_counter >= early_stop_patience:
                print("🛑 Early stopping triggered.")
                break

        torch.cuda.empty_cache()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Define paths
    folder1 = '/usr/mvl2/cvrfr/KeyPonFuse_V1/Codes/models_22aug25/training_datasets_full/training_dataset_20k/vi'
    folder2 = '/usr/mvl2/cvrfr/KeyPonFuse_V1/Codes/models_22aug25/training_datasets_full/training_dataset_20k/ir'
    test_output_folder = '/usr/mvl2/cvrfr/KeyPonFuse_V1/Datasets/Training_Dataset_full/training_dataset_20k/test_ops'  
    folder3 = '/usr/mvl2/cvrfr/KeyPonFuse_V1/Codes/models_22aug25/model_outputs/SGT_DIrect_Fusion'

    os.makedirs(test_output_folder, exist_ok)

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

    # Create datasets
    train_dataset = MemoryEfficientDataset(folder1, folder2, folder3, train_files, is_training=True, max_size=256)
    val_dataset = MemoryEfficientDataset(folder1, folder2, folder3, val_files, is_training=False, max_size=256)
    test_dataset = MemoryEfficientDataset(folder1, folder2, folder3, test_files, is_training=False, max_size=256)

    # Create model
    model = MemoryEfficientRGBThermalFusion(
        input_channels=3,
        d_model=128,  # Reduced from 256
        num_heads=4,  # Reduced from 8
        num_cross_attention=1,  # Reduced from 2
        dropout=0.1
    )

    # Create data loaders with smaller batch size
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2)  # Reduced batch size
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2)

    print("🚀 Starting Memory Efficient Transformer Training...")
    print(f"📊 Image size: 256x256")
    print(f"📊 Batch size: 2 (reduced for memory)")
    print(f"📊 Model: MemoryEfficientRGBThermalFusion")
    print(f"📊 d_model: 128 (reduced from 256)")
    print(f"📊 num_heads: 4 (reduced from 8)")
    print(f"📊 Cross attention blocks: 1 (reduced from 2)")
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_memory_efficient_model(model, train_loader, val_loader, epochs=100, device=device)

    print("Training Done.")