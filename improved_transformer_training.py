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
import warnings
from typing import Optional, Tuple

# Suppress interpolation warnings
warnings.filterwarnings("ignore", message=".*interpolate.*recompute_scale_factor.*")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

# Import the improved model (assuming it's saved as improved_transformer.py)
# For now, I'll include the model code here
class PositionalEncoding(nn.Module):
    """Positional encoding that accepts input of shape (B, L, D) and returns same shape."""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("PositionalEncoding expects input shape (B, L, D)")
        L = x.size(1)
        return x + self.pe[:, :L, :].to(x.dtype)

class MemoryEfficientAttention(nn.Module):
    """Chunked, memory-efficient multi-head attention."""
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, chunk_size: int = 64):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.chunk_size = chunk_size

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _prepare_mask(self, mask: Optional[torch.Tensor], device: torch.device,
                      dtype: torch.dtype, q_len: int, kv_len: int) -> Optional[torch.Tensor]:
        if mask is None:
            return None
        mask = mask.to(device=device)
        if mask.dtype != torch.bool:
            mask_bool = mask != 0
        else:
            mask_bool = mask

        if mask_bool.dim() == 2:
            mask_out = mask_bool.unsqueeze(1).unsqueeze(1)
        elif mask_bool.dim() == 3:
            B, a, b = mask_bool.shape
            if a == 1 or a == q_len:
                if a == q_len:
                    mask_out = mask_bool.unsqueeze(1)
                else:
                    mask_out = mask_bool.unsqueeze(1)
            else:
                mask_out = mask_bool.unsqueeze(1)
        elif mask_bool.dim() == 4:
            mask_out = mask_bool
        else:
            raise ValueError(f"Unsupported mask dimension: {mask_bool.dim()}")
        return mask_out

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, Lq, _ = query.shape
        _, Lk, _ = key.shape
        device = query.device
        dtype = query.dtype

        Q = self.w_q(query).view(B, Lq, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(B, Lk, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(B, Lk, self.num_heads, self.d_k).transpose(1, 2)

        norm_mask = self._prepare_mask(mask, device, dtype, Lq, Lk)

        out_chunks = []
        for start in range(0, Lq, self.chunk_size):
            end = min(start + self.chunk_size, Lq)
            Q_chunk = Q[:, :, start:end, :]

            scores = torch.matmul(Q_chunk, K.transpose(-2, -1)) / self.scale

            if norm_mask is not None:
                if norm_mask.size(-2) == Lq:
                    mask_chunk = norm_mask[:, :, start:end, :]
                else:
                    mask_chunk = norm_mask
                scores = scores.masked_fill(~mask_chunk, float('-1e9'))

            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)

            context = torch.matmul(attn, V)
            out_chunks.append(context)

        context = torch.cat(out_chunks, dim=2)
        context = context.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        out = self.w_o(context)
        return out, None

class CrossAttentionBlock(nn.Module):
    """Cross attention between two streams."""
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, chunk_size: int = 64):
        super().__init__()
        self.cross_attention = MemoryEfficientAttention(d_model, num_heads, dropout, chunk_size)
        self.norm_rgb_1 = nn.LayerNorm(d_model)
        self.norm_thermal_1 = nn.LayerNorm(d_model)
        self.norm_rgb_2 = nn.LayerNorm(d_model)
        self.norm_thermal_2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        self.dropout = nn.Dropout(dropout)

        for p in self.feed_forward.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, rgb_features: torch.Tensor, thermal_features: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        cross_out_rgb, _ = self.cross_attention(rgb_features, thermal_features, thermal_features, mask=mask)
        rgb_features = self.norm_rgb_1(rgb_features + self.dropout(cross_out_rgb))

        cross_out_thermal, _ = self.cross_attention(thermal_features, rgb_features, rgb_features, mask=mask)
        thermal_features = self.norm_thermal_1(thermal_features + self.dropout(cross_out_thermal))

        rgb_ff = self.feed_forward(rgb_features)
        rgb_features = self.norm_rgb_2(rgb_features + self.dropout(rgb_ff))

        thermal_ff = self.feed_forward(thermal_features)
        thermal_features = self.norm_thermal_2(thermal_features + self.dropout(thermal_ff))

        return rgb_features, thermal_features

class SelfAttentionBlock(nn.Module):
    """Self attention refinement block"""
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, chunk_size: int = 64):
        super().__init__()
        self.self_attention = MemoryEfficientAttention(d_model, num_heads, dropout, chunk_size)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        self.dropout = nn.Dropout(dropout)

        for p in self.feed_forward.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, features: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        self_out, _ = self.self_attention(features, features, features, mask=mask)
        features = self.norm1(features + self.dropout(self_out))
        ff_out = self.feed_forward(features)
        features = self.norm2(features + self.dropout(ff_out))
        return features

class ImprovedRGBThermalFusion(nn.Module):
    """Improved RGB-Thermal Fusion Model with transformer-style attention."""
    def __init__(self,
                 input_channels: int = 3,
                 d_model: int = 128,
                 num_heads: int = 8,
                 num_cross_attention: int = 2,
                 dropout: float = 0.1,
                 chunk_size: int = 64):
        super().__init__()
        self.d_model = d_model

        def encoder_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        self.rgb_encoder = encoder_block(input_channels, d_model)
        self.thermal_encoder = encoder_block(input_channels, d_model)

        self.pos_encoding = PositionalEncoding(d_model)

        self.cross_attention_blocks = nn.ModuleList([
            CrossAttentionBlock(d_model, num_heads, dropout, chunk_size)
            for _ in range(num_cross_attention)
        ])

        self.self_attention_rgb = SelfAttentionBlock(d_model, num_heads, dropout, chunk_size)
        self.self_attention_thermal = SelfAttentionBlock(d_model, num_heads, dropout, chunk_size)

        self.feature_fusion = nn.Sequential(
            nn.Conv2d(d_model * 2, d_model, 1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model, d_model, 3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(d_model, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, input_channels, 1)
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, rgb: torch.Tensor, thermal: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, C, H, W = rgb.shape

        rgb_features = self.rgb_encoder(rgb)
        thermal_features = self.thermal_encoder(thermal)

        max_attn_res = 64
        if H > max_attn_res or W > max_attn_res:
            scale_factor = min(max_attn_res / float(H), max_attn_res / float(W))
            new_h = max(1, int(H * scale_factor))
            new_w = max(1, int(W * scale_factor))
            rgb_features = F.interpolate(rgb_features, size=(new_h, new_w), mode='bilinear', align_corners=False, recompute_scale_factor=True)
            thermal_features = F.interpolate(thermal_features, size=(new_h, new_w), mode='bilinear', align_corners=False, recompute_scale_factor=True)
            H_attn, W_attn = new_h, new_w
        else:
            H_attn, W_attn = H, W

        L = H_attn * W_attn
        rgb_seq = rgb_features.view(B, self.d_model, L).transpose(1, 2)
        thermal_seq = thermal_features.view(B, self.d_model, L).transpose(1, 2)

        rgb_seq = self.pos_encoding(rgb_seq)
        thermal_seq = self.pos_encoding(thermal_seq)

        for cross_attn in self.cross_attention_blocks:
            rgb_seq, thermal_seq = cross_attn(rgb_seq, thermal_seq, mask=mask)

        rgb_seq = self.self_attention_rgb(rgb_seq, mask=mask)
        thermal_seq = self.self_attention_thermal(thermal_seq, mask=mask)

        rgb_feats_spatial = rgb_seq.transpose(1, 2).contiguous().view(B, self.d_model, H_attn, W_attn)
        thermal_feats_spatial = thermal_seq.transpose(1, 2).contiguous().view(B, self.d_model, H_attn, W_attn)

        if (H_attn, W_attn) != (H, W):
            rgb_feats_spatial = F.interpolate(rgb_feats_spatial, size=(H, W), mode='bilinear', align_corners=False, recompute_scale_factor=True)
            thermal_feats_spatial = F.interpolate(thermal_feats_spatial, size=(H, W), mode='bilinear', align_corners=False, recompute_scale_factor=True)

        fused = torch.cat([rgb_feats_spatial, thermal_feats_spatial], dim=1)
        fused = self.feature_fusion(fused)
        out = self.decoder(fused)

        return torch.sigmoid(out)

# --- IMPROVED LOSS FUNCTION ---
class ImprovedFusionLoss(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        
        vgg = vgg19(pretrained=True)
        self.vgg_features = nn.Sequential(*list(vgg.features)[:8]).eval().to(device)
        for param in self.vgg_features.parameters():
            param.requires_grad = False
    
    def forward(self, fused, rgb, thermal, gt):
        target_size = gt.shape[-2:]
        
        if fused.shape[-2:] != target_size:
            fused = F.interpolate(fused, size=target_size, mode='bilinear', align_corners=False, recompute_scale_factor=True)
        if rgb.shape[-2:] != target_size:
            rgb = F.interpolate(rgb, size=target_size, mode='bilinear', align_corners=False, recompute_scale_factor=True)
        if thermal.shape[-2:] != target_size:
            thermal = F.interpolate(thermal, size=target_size, mode='bilinear', align_corners=False, recompute_scale_factor=True)
        
        l1_loss = F.l1_loss(fused, gt)
        ssim_loss = 1 - self.ssim(fused, gt)
        perceptual_loss = self.perceptual_loss(fused, gt)
        rgb_preservation = F.l1_loss(fused, rgb)
        thermal_preservation = F.l1_loss(fused, thermal)
        gradient_loss = self.gradient_loss(fused, gt)
        
        total_loss = (3.0 * l1_loss +           
                     1.5 * ssim_loss +          
                     0.8 * perceptual_loss +    
                     0.4 * rgb_preservation +   
                     0.4 * thermal_preservation + 
                     0.2 * gradient_loss)       
        
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
        if fused.size(1) == 1:
            fused_3ch = fused.repeat(1, 3, 1, 1)
            gt_3ch = gt.repeat(1, 3, 1, 1)
        else:
            fused_3ch = fused
            gt_3ch = gt
        
        fused_3ch = (fused_3ch + 1) / 2
        gt_3ch = (gt_3ch + 1) / 2
        
        if fused_3ch.shape[-1] > 128:
            scale_factor = 128 / fused_3ch.shape[-1]
            fused_3ch = F.interpolate(fused_3ch, scale_factor=scale_factor, mode='bilinear', align_corners=False, recompute_scale_factor=True)
            gt_3ch = F.interpolate(gt_3ch, scale_factor=scale_factor, mode='bilinear', align_corners=False, recompute_scale_factor=True)
        
        return F.mse_loss(self.vgg_features(fused_3ch), self.vgg_features(gt_3ch))
    
    def gradient_loss(self, pred, target):
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
class ImprovedDataset(Dataset):
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
                transforms.RandomRotation(degrees=5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
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
def train_improved_model(model, train_loader, val_loader, epochs, device='cpu'):
    model = model.to(device)
    criterion = ImprovedFusionLoss(device=device)
    
    # Different learning rates for different parts
    encoder_params = list(model.rgb_encoder.parameters()) + list(model.thermal_encoder.parameters())
    attention_params = list(model.cross_attention_blocks.parameters()) + list(model.self_attention_rgb.parameters()) + list(model.self_attention_thermal.parameters())
    decoder_params = list(model.feature_fusion.parameters()) + list(model.decoder.parameters())
    
    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': 8e-5},
        {'params': attention_params, 'lr': 1.5e-4},
        {'params': decoder_params, 'lr': 8e-5}
    ], weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=1e-6)
    
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
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            
            for key, value in loss_dict.items():
                if key not in loss_components:
                    loss_components[key] = 0
                loss_components[key] += value
            
            del loss, output
            
            if step % 10 == 0:
                torch.cuda.empty_cache()

        total_loss /= len(train_loader)
        for key in loss_components:
            loss_components[key] /= len(train_loader)
        
        print(f"✅ Epoch {epoch + 1} Training Loss: {total_loss:.4f}")
        print(f"   L1: {loss_components['l1_loss']:.4f}, SSIM: {loss_components['ssim_loss']:.4f}, Perceptual: {loss_components['perceptual_loss']:.4f}")

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

        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            early_stop_counter = 0
            print("💾 Model improved. Saving...")

            torch.save(model.state_dict(),
                '/usr/mvl2/cvrfr/KeyPonFuse_V1/Codes/models_22aug25/model_training/trained_models/improved_transformer_fusion.pth')
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
    train_dataset = ImprovedDataset(folder1, folder2, folder3, train_files, is_training=True, max_size=256)
    val_dataset = ImprovedDataset(folder1, folder2, folder3, val_files, is_training=False, max_size=256)
    test_dataset = ImprovedDataset(folder1, folder2, folder3, test_files, is_training=False, max_size=256)

    # Create model
    model = ImprovedRGBThermalFusion(
        input_channels=3,
        d_model=128,
        num_heads=8,
        num_cross_attention=2,
        dropout=0.1,
        chunk_size=64
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=3, shuffle=False, num_workers=2)

    print("🚀 Starting IMPROVED Transformer Fusion Training...")
    print(f"📊 Image size: 256x256")
    print(f"📊 Batch size: 3")
    print(f"📊 Model: ImprovedRGBThermalFusion")
    print(f"📊 d_model: 128")
    print(f"📊 num_heads: 8")
    print(f"📊 Cross attention blocks: 2")
    print(f"📊 Separate self-attention for RGB/Thermal")
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_improved_model(model, train_loader, val_loader, epochs=100, device=device)

    print("Training Done.")