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
from best_model_for_training import UNet_FusionTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# --- Training Configuration ---
num_epochs = 100
MAX_SIZE = 1024  # Set to 1024 for full resolution training
BATCH_SIZE = 1   # Small batch for 1024x1024

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

# OPTIMIZED Dataset for 1024x1024 training
class Optimized1024Dataset(Dataset):
    def __init__(self, folder1, folder2, folder3, filenames, is_training=True):
        self.folder1 = folder1
        self.folder2 = folder2
        self.folder3 = folder3
        self.filenames = filenames
        self.is_training = is_training

        if is_training:
            self.img_transform = transforms.Compose([
                transforms.Lambda(self.resize_if_needed),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),  # Reduced rotation for large images
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # Reduced jitter
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            self.gt_transform = transforms.Compose([
                transforms.Lambda(self.resize_if_needed),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.img_transform = transforms.Compose([
                transforms.Lambda(self.resize_if_needed),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            self.gt_transform = transforms.Compose([
                transforms.Lambda(self.resize_if_needed),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def resize_if_needed(self, img):
        """Resize only if image is larger than MAX_SIZE"""
        w, h = img.size
        if max(w, h) > MAX_SIZE:
            if w > h:
                new_w = MAX_SIZE
                new_h = int(h * MAX_SIZE / w)
            else:
                new_h = MAX_SIZE
                new_w = int(w * MAX_SIZE / h)
            img = img.resize((new_w, new_h), Image.BILINEAR)
        return img

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # Load images
        img1_path = os.path.join(self.folder1, self.filenames[idx])
        img2_path = os.path.join(self.folder2, self.filenames[idx])
        gt_path = os.path.join(self.folder3, self.filenames[idx])

        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        gt = Image.open(gt_path).convert('RGB')

        # Ensure all are RGB
        if img1.mode != 'RGB':
            img1 = img1.convert('RGB')
        if img2.mode != 'RGB':
            img2 = img2.convert('RGB')
        if gt.mode != 'RGB':
            gt = gt.convert('RGB')

        # Apply transforms
        img1 = self.img_transform(img1)
        img2 = self.img_transform(img2)
        gt = self.gt_transform(gt)

        return img1, img2, gt, self.filenames[idx]

# Create datasets
train_dataset = Optimized1024Dataset(folder1, folder2, folder3, train_files, is_training=True)
val_dataset = Optimized1024Dataset(folder1, folder2, folder3, val_files, is_training=False)
test_dataset = Optimized1024Dataset(folder1, folder2, folder3, test_files, is_training=False)

def custom_collate(batch):
    """Custom collate function for variable sizes"""
    img1_list, img2_list, gt_list, names = zip(*batch)
    
    # Find the maximum size in the batch
    max_h = max(img.shape[1] for img in img1_list)
    max_w = max(img.shape[2] for img in img1_list)
    
    # Pad all images to the same size
    padded_img1 = []
    padded_img2 = []
    padded_gt = []
    
    for img1, img2, gt in zip(img1_list, img2_list, gt_list):
        # Pad to max size
        pad_h = max_h - img1.shape[1]
        pad_w = max_w - img1.shape[2]
        
        img1_padded = F.pad(img1, (0, pad_w, 0, pad_h), mode='reflect')
        img2_padded = F.pad(img2, (0, pad_w, 0, pad_h), mode='reflect')
        gt_padded = F.pad(gt, (0, pad_w, 0, pad_h), mode='reflect')
        
        padded_img1.append(img1_padded)
        padded_img2.append(img2_padded)
        padded_gt.append(gt_padded)
    
    return torch.stack(padded_img1), torch.stack(padded_img2), torch.stack(padded_gt), names

# MEMORY-OPTIMIZED Loss Function
class MemoryOptimizedLoss(nn.Module):
    def __init__(self, device='cpu', 
                 lambda_gt=2.0,
                 lambda_perceptual=0.2,  # Reduced for memory
                 lambda_structural=0.8,  # Reduced for memory
                 lambda_modality=0.8,
                 lambda_gradient=0.3,
                 lambda_rgb=0.5, 
                 lambda_thermal=0.5):
        super(MemoryOptimizedLoss, self).__init__()
        self.lambda_gt = lambda_gt
        self.lambda_perceptual = lambda_perceptual
        self.lambda_structural = lambda_structural
        self.lambda_modality = lambda_modality
        self.lambda_gradient = lambda_gradient
        self.lambda_rgb = lambda_rgb
        self.lambda_thermal = lambda_thermal
        
        # VGG for perceptual loss (fewer layers for memory)
        vgg = vgg19(pretrained=True)
        self.vgg_features = nn.Sequential(*list(vgg.features)[:12]).eval().to(device)  # Even fewer layers
        for param in self.vgg_features.parameters():
            param.requires_grad = False
        self.device = device

    def forward(self, fused, rgb, thermal, gt):
        # Ensure all tensors are the same spatial size
        target_size = gt.shape[-2:]
        
        if fused.shape[-2:] != target_size:
            fused = F.interpolate(fused, size=target_size, mode='bilinear', align_corners=False)
        if rgb.shape[-2:] != target_size:
            rgb = F.interpolate(rgb, size=target_size, mode='bilinear', align_corners=False)
        if thermal.shape[-2:] != target_size:
            thermal = F.interpolate(thermal, size=target_size, mode='bilinear', align_corners=False)

        # Ground truth loss
        l1_gt = F.l1_loss(fused, gt)
        l2_gt = F.mse_loss(fused, gt)
        ssim_gt = self.ssim(fused, gt)
        ssim_loss_gt = 1 - ssim_gt
        
        gt_loss = 0.4 * l1_gt + 0.3 * l2_gt + 0.3 * ssim_loss_gt
        
        # Perceptual loss (aggressive downsampling for memory)
        if fused.size(1) == 1:
            fused_3ch = fused.repeat(1, 3, 1, 1)
            gt_3ch = gt.repeat(1, 3, 1, 1)
        else:
            fused_3ch = fused
            gt_3ch = gt
            
        # Downsample to 256x256 for VGG to save memory
        if fused_3ch.shape[-1] > 256:
            scale_factor = 256 / fused_3ch.shape[-1]
            fused_3ch = F.interpolate(fused_3ch, scale_factor=scale_factor, mode='bilinear', align_corners=False)
            gt_3ch = F.interpolate(gt_3ch, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        
        # Denormalize for VGG
        fused_3ch = self.denormalize(fused_3ch)
        gt_3ch = self.denormalize(gt_3ch)
        
        perceptual_loss = F.mse_loss(self.vgg_features(fused_3ch), self.vgg_features(gt_3ch))
        
        # Structural loss (downsample for memory)
        if fused.shape[-1] > 256:
            scale_factor = 256 / fused.shape[-1]
            fused_down = F.interpolate(fused, scale_factor=scale_factor, mode='bilinear', align_corners=False)
            gt_down = F.interpolate(gt, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        else:
            fused_down = fused
            gt_down = gt
            
        structural_loss = 1 - self.ms_ssim(fused_down, gt_down, levels=2)  # Reduced levels
        
        # Modality-specific loss
        modality_mse_loss = (self.lambda_rgb * F.mse_loss(fused, rgb) +
                           self.lambda_thermal * F.mse_loss(fused, thermal))
        
        # Gradient loss
        grad_loss = (self.gradient_loss(fused, gt) + 
                    0.3 * self.gradient_loss(fused, rgb) +
                    0.3 * self.gradient_loss(fused, thermal))
        
        # Total loss
        total_loss = (self.lambda_gt * gt_loss + 
                     self.lambda_perceptual * perceptual_loss +
                     self.lambda_structural * structural_loss +
                     self.lambda_modality * modality_mse_loss +
                     self.lambda_gradient * grad_loss)
        
        return total_loss, {
            'gt_loss': gt_loss.item(),
            'perceptual_loss': perceptual_loss.item(),
            'structural_loss': structural_loss.item(),
            'modality_loss': modality_mse_loss.item(),
            'gradient_loss': grad_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def denormalize(self, tensor):
        """Denormalize tensor from [-1,1] normalization"""
        return (tensor + 1) / 2

    def gradient_loss(self, pred, target):
        if pred.shape[-2:] != target.shape[-2:]:
            target = F.interpolate(target, size=pred.shape[-2:], mode='bilinear', align_corners=False)

        H, W = pred.shape[2], pred.shape[3]
        H = H - 1 if H % 2 != 0 else H
        W = W - 1 if W % 2 != 0 else W
        pred = pred[:, :, :H, :W]
        target = target[:, :, :H, :W]

        grad_pred_x = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        grad_pred_y = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
        grad_target_x = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
        grad_target_y = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])

        loss_x = F.l1_loss(grad_pred_x, grad_target_x)
        loss_y = F.l1_loss(grad_pred_y, grad_target_y)

        return loss_x + loss_y

    def ms_ssim(self, pred, target, levels=2):  # Reduced levels
        """Multi-scale SSIM implementation"""
        pred = pred.clamp(-1, 1)
        target = target.clamp(-1, 1)

        ssim_values = []
        for i in range(levels):
            if i == 0:
                ssim_values.append(self.ssim(pred, target))
            else:
                pred_down = F.avg_pool2d(pred, 2**i)
                target_down = F.avg_pool2d(target, 2**i)
                ssim_values.append(self.ssim(pred_down, target_down))
        
        return torch.prod(torch.stack(ssim_values))

    def ssim(self, pred, target, window_size=11):
        """Custom SSIM implementation for [-1,1] range"""
        pred = pred.clamp(-1, 1)
        target = target.clamp(-1, 1)

        mu_pred = F.avg_pool2d(pred, window_size, stride=1, padding=window_size // 2)
        mu_target = F.avg_pool2d(target, window_size, stride=1, padding=window_size // 2)
        
        sigma_pred = F.avg_pool2d(pred**2, window_size, stride=1, padding=window_size // 2) - mu_pred**2
        sigma_target = F.avg_pool2d(target**2, window_size, stride=1, padding=window_size // 2) - mu_target**2
        sigma_pred_target = F.avg_pool2d(pred * target, window_size, stride=1, padding=window_size // 2) - mu_pred * mu_target
        
        c1, c2 = 0.01**2, 0.03**2
        ssim_map = ((2 * mu_pred * mu_target + c1) * (2 * sigma_pred_target + c2)) / \
                ((mu_pred**2 + mu_target**2 + c1) * (sigma_pred + sigma_target + c2))
        
        return ssim_map.mean()

# OPTIMIZED Training Function
def optimized_1024_train(model, train_loader, val_loader, test_dataset, epochs, device='cpu'):
    model = model.to(device)
    criterion = MemoryOptimizedLoss(device=device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2, eta_min=1e-6
    )
    
    scaler = GradScaler()
    best_loss = float('inf')
    early_stop_patience = 25
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

                if output.shape[-2:] != gt_batch.shape[-2:]:
                    output = F.interpolate(output, size=gt_batch.shape[-2:], mode='bilinear', align_corners=False)

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

                    if output.shape[-2:] != gt_batch.shape[-2:]:
                        output = F.interpolate(output, size=gt_batch.shape[-2:], mode='bilinear', align_corners=False)

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
                '/usr/mvl2/cvrfr/KeyPonFuse_V1/Codes/models_22aug25/model_training/trained_models/UNet_1024_optimized.pth')
        else:
            early_stop_counter += 1
            print(f"⏳ No improvement. Patience: {early_stop_counter}/{early_stop_patience}")
            if early_stop_counter >= early_stop_patience:
                print("🛑 Early stopping triggered.")
                break

        torch.cuda.empty_cache()

# --- Main Execution ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Use the corrected model
model = UNet_FusionTransformer()

# OPTIMIZED: Very small batch size for 1024x1024
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate, num_workers=1)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate, num_workers=1)

print("🚀 Starting Optimized 1024x1024 Training...")
print(f"📊 Max image size: {MAX_SIZE}px")
print(f"📊 Batch size: {BATCH_SIZE}")
print(f"📊 Memory optimizations: Enabled")

optimized_1024_train(model, train_loader, val_loader, test_dataset, epochs=num_epochs, device=device)

print("Training Done.")