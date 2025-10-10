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
from rgb_thermal_fusion_transformer import RGBThermalFusionTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# --- COMPREHENSIVE LOSS FUNCTION ---
class TransformerFusionLoss(nn.Module):
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
        
        # 1. PRIMARY LOSS: L1 to GT
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
        
        # 6. FEATURE SIMILARITY: Encourage similar high-level features
        feature_similarity = self.feature_similarity_loss(fused, gt)
        
        # BALANCED LOSS
        total_loss = (2.0 * l1_loss +           # Primary objective
                     1.5 * ssim_loss +          # Structural similarity
                     0.8 * perceptual_loss +    # High-level features
                     0.4 * rgb_preservation +   # RGB content
                     0.4 * thermal_preservation + # Thermal content
                     0.2 * gradient_loss +      # Edge preservation
                     0.3 * feature_similarity)  # Feature similarity
        
        return total_loss, {
            'l1_loss': l1_loss.item(),
            'ssim_loss': ssim_loss.item(),
            'perceptual_loss': perceptual_loss.item(),
            'rgb_preservation': rgb_preservation.item(),
            'thermal_preservation': thermal_preservation.item(),
            'gradient_loss': gradient_loss.item(),
            'feature_similarity': feature_similarity.item(),
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
    
    def feature_similarity_loss(self, fused, gt):
        """Feature similarity loss using VGG features"""
        if fused.size(1) == 1:
            fused_3ch = fused.repeat(1, 3, 1, 1)
            gt_3ch = gt.repeat(1, 3, 1, 1)
        else:
            fused_3ch = fused
            gt_3ch = gt
        
        fused_3ch = (fused_3ch + 1) / 2
        gt_3ch = (gt_3ch + 1) / 2
        
        if fused_3ch.shape[-1] > 256:
            scale_factor = 256 / fused_3ch.shape[-1]
            fused_3ch = F.interpolate(fused_3ch, scale_factor=scale_factor, mode='bilinear', align_corners=False)
            gt_3ch = F.interpolate(gt_3ch, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        
        fused_features = self.vgg_features(fused_3ch)
        gt_features = self.vgg_features(gt_3ch)
        
        return F.mse_loss(fused_features, gt_features)

# --- DATASET CLASS ---
class TransformerDataset(Dataset):
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
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
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
def train_transformer_model(model, train_loader, val_loader, epochs, device='cpu'):
    model = model.to(device)
    criterion = TransformerFusionLoss(device=device)
    
    # Optimizer with different learning rates for different parts
    encoder_params = list(model.rgb_encoder.parameters()) + list(model.thermal_encoder.parameters())
    attention_params = list(model.cross_attention_blocks.parameters()) + list(model.self_attention.parameters())
    decoder_params = list(model.feature_fusion.parameters()) + list(model.decoder.parameters())
    
    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': 1e-4},
        {'params': attention_params, 'lr': 2e-4},
        {'params': decoder_params, 'lr': 1e-4}
    ], weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)
    
    scaler = GradScaler()
    best_loss = float('inf')
    early_stop_patience = 25
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            
            # Accumulate loss components
            for key, value in loss_dict.items():
                if key not in loss_components:
                    loss_components[key] = 0
                loss_components[key] += value
            
            del loss, output

        total_loss /= len(train_loader)
        for key in loss_components:
            loss_components[key] /= len(train_loader)
        
        print(f"✅ Epoch {epoch + 1} Training Loss: {total_loss:.4f}")
        print(f"   L1: {loss_components['l1_loss']:.4f}, SSIM: {loss_components['ssim_loss']:.4f}, Perceptual: {loss_components['perceptual_loss']:.4f}")

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
                '/usr/mvl2/cvrfr/KeyPonFuse_V1/Codes/models_22aug25/model_training/trained_models/transformer_fusion_model.pth')
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
    train_dataset = TransformerDataset(folder1, folder2, folder3, train_files, is_training=True, max_size=256)
    val_dataset = TransformerDataset(folder1, folder2, folder3, val_files, is_training=False, max_size=256)
    test_dataset = TransformerDataset(folder1, folder2, folder3, test_files, is_training=False, max_size=256)

    # Create model
    model = RGBThermalFusionTransformer(
        input_channels=3,
        d_model=256,
        num_heads=8,
        num_cross_attention=2,
        dropout=0.1
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

    print("🚀 Starting Transformer Fusion Training...")
    print(f"📊 Image size: 256x256")
    print(f"📊 Batch size: 4")
    print(f"📊 Model: RGBThermalFusionTransformer")
    print(f"📊 Cross attention blocks: 2")
    print(f"📊 Self attention: 1 (before fusion)")
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_transformer_model(model, train_loader, val_loader, epochs=100, device=device)

    print("Training Done.")