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
from fixed_model import FIXED_UNet_FusionTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# --- FIXED SIZE TRAINING ---
num_epochs = 75

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

# Dataset with fixed size
class FixedSizeDataset(Dataset):
    def __init__(self, folder1, folder2, folder3, filenames, is_training=True, target_size=512):
        self.folder1 = folder1
        self.folder2 = folder2
        self.folder3 = folder3
        self.filenames = filenames
        self.is_training = is_training
        self.target_size = target_size

        if is_training:
            self.img_transform = transforms.Compose([
                transforms.Resize((target_size, target_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            self.gt_transform = transforms.Compose([
                transforms.Resize((target_size, target_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize((target_size, target_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            self.gt_transform = transforms.Compose([
                transforms.Resize((target_size, target_size)),
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

        # Apply transforms
        img1 = self.img_transform(img1)
        img2 = self.img_transform(img2)
        gt = self.gt_transform(gt)

        return img1, img2, gt, self.filenames[idx]

# Create datasets
train_dataset = FixedSizeDataset(folder1, folder2, folder3, train_files, is_training=True, target_size=512)
val_dataset = FixedSizeDataset(folder1, folder2, folder3, val_files, is_training=False, target_size=512)
test_dataset = FixedSizeDataset(folder1, folder2, folder3, test_files, is_training=False, target_size=512)

def simple_collate(batch):
    """Simple collate function - all images are same size now"""
    img1_list, img2_list, gt_list, names = zip(*batch)
    return torch.stack(img1_list), torch.stack(img2_list), torch.stack(gt_list), names

# Content Preservation Loss
class ContentPreservationLoss(nn.Module):
    def __init__(self, device='cpu'):
        super(ContentPreservationLoss, self).__init__()
        self.device = device
        
        # VGG for perceptual loss
        vgg = vgg19(pretrained=True)
        self.vgg_features = nn.Sequential(*list(vgg.features)[:16]).eval().to(device)
        for param in self.vgg_features.parameters():
            param.requires_grad = False

    def forward(self, fused, rgb, thermal, gt):
        # 1. REFERENCE GT LOSS - Learn from the reference fusion
        l1_gt = F.l1_loss(fused, gt)
        ssim_gt = self.ssim(fused, gt)
        ssim_loss_gt = 1 - ssim_gt
        gt_loss = 0.7 * l1_gt + 0.3 * ssim_loss_gt
        
        # 2. RGB CONTENT PRESERVATION
        rgb_l1 = F.l1_loss(fused, rgb)
        rgb_ssim = self.ssim(fused, rgb)
        rgb_content_loss = 0.7 * rgb_l1 + 0.3 * (1 - rgb_ssim)
        
        # 3. THERMAL CONTENT PRESERVATION
        thermal_l1 = F.l1_loss(fused, thermal)
        thermal_ssim = self.ssim(fused, thermal)
        thermal_content_loss = 0.7 * thermal_l1 + 0.3 * (1 - thermal_ssim)
        
        # 4. PERCEPTUAL LOSS - High-level features from GT
        perceptual_loss = self.perceptual_loss(fused, gt)
        
        # TOTAL LOSS - Focus on content preservation + perceptual quality
        total_loss = (1.0 * gt_loss + 
                     0.8 * rgb_content_loss +
                     0.8 * thermal_content_loss +
                     0.3 * perceptual_loss)
        
        return total_loss, {
            'gt_loss': gt_loss.item(),
            'rgb_content_loss': rgb_content_loss.item(),
            'thermal_content_loss': thermal_content_loss.item(),
            'perceptual_loss': perceptual_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def perceptual_loss(self, fused, gt):
        """Calculate perceptual loss using VGG features"""
        if fused.size(1) == 1:
            fused_3ch = fused.repeat(1, 3, 1, 1)
            gt_3ch = gt.repeat(1, 3, 1, 1)
        else:
            fused_3ch = fused
            gt_3ch = gt
            
        # Denormalize for VGG
        fused_3ch = self.denormalize(fused_3ch)
        gt_3ch = self.denormalize(gt_3ch)
        
        return F.mse_loss(self.vgg_features(fused_3ch), self.vgg_features(gt_3ch))
    
    def denormalize(self, tensor):
        """Denormalize tensor from [-1,1] normalization"""
        return (tensor + 1) / 2
    
    def ssim(self, pred, target, window_size=11):
        """SSIM implementation"""
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

# Training Function
def train_model(model, train_loader, val_loader, test_dataset, epochs, device='cpu'):
    model = model.to(device)
    criterion = ContentPreservationLoss(device=device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
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
                '/usr/mvl2/cvrfr/KeyPonFuse_V1/Codes/models_22aug25/model_training/trained_models/UNet_fixed_size_content_preservation.pth')
        else:
            early_stop_counter += 1
            print(f"⏳ No improvement. Patience: {early_stop_counter}/{early_stop_patience}")
            if early_stop_counter >= early_stop_patience:
                print("🛑 Early stopping triggered.")
                break

        torch.cuda.empty_cache()

# --- Main Execution ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Use the FIXED model
model = FIXED_UNet_FusionTransformer()

# Training with fixed size
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=simple_collate, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=simple_collate, num_workers=4)

print("🚀 Starting Fixed Size Content Preservation Training...")
print(f"📊 Image size: 512x512 (fixed)")
print(f"📊 Batch size: 4")
print(f"📊 Focus: Retain content from both RGB and Thermal")
print(f"📊 Reference: Learn from GT fusion output")
print(f"📊 Loss: L1 + SSIM + Perceptual + Content Preservation")
print(f"📊 FIXED model ensures output size matches input size!")

train_model(model, train_loader, val_loader, test_dataset, epochs=num_epochs, device=device)

print("Training Done.")