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

# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# --- (a) Data Loading and Splitting ---
num_epochs = 100  # Increased epochs

# Define paths - IMPORTANT: Use proper ground truth, not averaged inputs
folder1 = '/usr/mvl2/cvrfr/KeyPonFuse_V1/Codes/models_22aug25/training_datasets_full/training_dataset_20k/vi'
folder2 = '/usr/mvl2/cvrfr/KeyPonFuse_V1/Codes/models_22aug25/training_datasets_full/training_dataset_20k/ir'
test_output_folder = '/usr/mvl2/cvrfr/KeyPonFuse_V1/Datasets/Training_Dataset_full/training_dataset_20k/test_ops'  

# CRITICAL FIX: Use proper ground truth, not averaged inputs
# You need to replace this with actual fusion ground truth
folder3 = '/usr/mvl2/cvrfr/KeyPonFuse_V1/Codes/models_22aug25/model_outputs/SGT_DIrect_Fusion'

os.makedirs(test_output_folder, exist_ok=True)

# Get all filenames (assuming same names in all folders)
filenames = sorted(os.listdir(folder1))
random.shuffle(filenames)

# Split
n = len(filenames)
train_idx = int(0.7 * n)
val_idx = int(0.9 * n)  # 70% + 20% = 90%

train_files = filenames[:train_idx]             # 0% - 70%
val_files = filenames[train_idx:val_idx]        # 70% - 90%
test_files = filenames[val_idx:] 

# IMPROVED Dataset class with better augmentation
class ImprovedTripleFolderDataset(Dataset):
    def __init__(self, folder1, folder2, folder3, filenames, is_training=True):
        self.folder1 = folder1
        self.folder2 = folder2
        self.folder3 = folder3
        self.filenames = filenames
        self.is_training = is_training

        # IMPROVED: Better augmentation and normalization
        if is_training:
            self.img_transform = transforms.Compose([
                transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),  # Increased rotation
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),  # Increased jitter
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Added affine
                transforms.ToTensor(),
                # IMPROVED: Use [0,1] normalization instead of ImageNet
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1] range
            ])
            self.gt_transform = transforms.Compose([
                transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            self.gt_transform = transforms.Compose([
                transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

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

        # Ensure all are RGB (3 channels)
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
train_dataset = ImprovedTripleFolderDataset(folder1, folder2, folder3, train_files, is_training=True)
val_dataset = ImprovedTripleFolderDataset(folder1, folder2, folder3, val_files, is_training=False)
test_dataset = ImprovedTripleFolderDataset(folder1, folder2, folder3, test_files, is_training=False)

def custom_collate(batch):
    img1_list, img2_list, gt_list, names = zip(*batch)
    img1_batch = torch.stack(img1_list)  # shape: (B, C, H, W)
    img2_batch = torch.stack(img2_list)
    gt_batch = torch.stack(gt_list)
    return img1_batch, img2_batch, gt_batch, names

# IMPROVED Loss Function with better calibration
class ImprovedRGBTFusionLossV2(nn.Module):
    def __init__(self, device='cpu', 
                 lambda_gt=2.0,  # Increased weight
                 lambda_perceptual=0.5,  # Increased weight
                 lambda_structural=1.0,  # Increased weight
                 lambda_modality=0.8,  # Increased weight
                 lambda_gradient=0.3,  # Increased weight
                 lambda_ssim=1.5,  # Increased weight
                 lambda_rgb=0.5, 
                 lambda_thermal=0.5):
        super(ImprovedRGBTFusionLossV2, self).__init__()
        self.lambda_gt = lambda_gt
        self.lambda_perceptual = lambda_perceptual
        self.lambda_structural = lambda_structural
        self.lambda_modality = lambda_modality
        self.lambda_gradient = lambda_gradient
        self.lambda_ssim = lambda_ssim
        self.lambda_rgb = lambda_rgb
        self.lambda_thermal = lambda_thermal
        
        # VGG for perceptual loss
        vgg = vgg19(pretrained=True)
        self.vgg_features = nn.Sequential(*list(vgg.features)[:23]).eval().to(device)
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

        # IMPROVED: Better ground truth loss with proper weighting
        l1_gt = F.l1_loss(fused, gt)
        l2_gt = F.mse_loss(fused, gt)
        
        # Use custom SSIM implementation
        ssim_gt = self.ssim(fused, gt)
        ssim_loss_gt = 1 - ssim_gt
        
        # Weighted ground truth loss
        gt_loss = 0.4 * l1_gt + 0.3 * l2_gt + 0.3 * ssim_loss_gt
        
        # IMPROVED: Perceptual loss with proper channel handling
        if fused.size(1) == 1:
            fused_3ch = fused.repeat(1, 3, 1, 1)
            gt_3ch = gt.repeat(1, 3, 1, 1)
        else:
            fused_3ch = fused
            gt_3ch = gt
            
        # Denormalize for VGG (now using [-1,1] normalization)
        fused_3ch = self.denormalize(fused_3ch)
        gt_3ch = self.denormalize(gt_3ch)
        
        perceptual_loss = F.mse_loss(self.vgg_features(fused_3ch), self.vgg_features(gt_3ch))
        
        # IMPROVED: Structural loss using MS-SSIM
        structural_loss = 1 - self.ms_ssim(fused, gt)
        
        # IMPROVED: Modality-specific loss with better balance
        modality_mse_loss = (self.lambda_rgb * F.mse_loss(fused, rgb) +
                           self.lambda_thermal * F.mse_loss(fused, thermal))
        
        # Gradient loss for edge preservation
        grad_loss = (self.gradient_loss(fused, gt) + 
                    0.3 * self.gradient_loss(fused, rgb) +
                    0.3 * self.gradient_loss(fused, thermal))
        
        # IMPROVED: Total loss with better weighting
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
        return (tensor + 1) / 2  # Convert from [-1,1] to [0,1]

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

    def ms_ssim(self, pred, target, levels=5):
        """Multi-scale SSIM implementation"""
        pred = pred.clamp(-1, 1)  # Clamp to [-1,1] range
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
        # Clamp inputs to [-1, 1] range
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

# IMPROVED Training Function
def improved_train_model_v2(model, train_loader, val_loader, test_dataset, epochs, device='cpu'):
    model = model.to(device)
    criterion = ImprovedRGBTFusionLossV2(device=device)
    
    # IMPROVED: Better optimizer with higher learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)  # Increased LR
    
    # IMPROVED: Better learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    scaler = GradScaler()
    best_loss = float('inf')
    early_stop_patience = 15  # Increased patience
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
            
            # IMPROVED: Gradient clipping for stability
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            del loss, output

        total_loss /= len(train_loader)
        print(f"✅ Epoch {epoch + 1} Training Loss: {total_loss:.4f}")

        # ---- Validation ----
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
        
        # IMPROVED: Learning rate scheduling
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"📊 Learning Rate: {current_lr:.2e}")

        # ---- Early Stopping ----
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            early_stop_counter = 0
            print("💾 Model improved. Saving...")

            torch.save(model.state_dict(),
                '/usr/mvl2/cvrfr/KeyPonFuse_V1/Codes/models_22aug25/model_training/trained_models/UNet_cross_self_attnTransformerSGTbase_fullres_improved_v2.pth')
        else:
            early_stop_counter += 1
            print(f"⏳ No improvement. Patience: {early_stop_counter}/{early_stop_patience}")
            if early_stop_counter >= early_stop_patience:
                print("🛑 Early stopping triggered.")
                break

        torch.cuda.empty_cache()

# --- Main Execution ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ✅ Use the corrected model
model = UNet_FusionTransformer()

# IMPROVED: Larger batch size for better training stability
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=custom_collate, num_workers=4)  # Increased batch size
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=custom_collate, num_workers=4)

# Replace num_epochs with a defined integer, e.g., 50
improved_train_model_v2(model, train_loader, val_loader, test_dataset, epochs=num_epochs, device=device)

print("Training Done.")