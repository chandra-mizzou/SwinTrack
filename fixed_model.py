import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 2D Sin-Cos Positional Encoding ---
def add_2d_sincos_pe(x, H, W):
    # x: (B, N, C), N = H*W
    pe = get_2d_sincos_pos_embed(x.shape[-1], H, W, device=x.device)
    return x + pe

def get_2d_sincos_pos_embed(embed_dim, grid_h, grid_w, device=None):
    """
    Borrowed/modified from MAE/GPT-NeoX
    """
    import numpy as np
    def get_1d_pos_embed(embed_dim, pos):
        omega = np.arange(embed_dim // 2) / (embed_dim / 2)
        omega = 1. / 10000**omega
        out = np.einsum('m,d->md', pos, omega)
        emb = np.concatenate([np.sin(out), np.cos(out)], axis=1)
        return emb

    grid = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing='ij')
    grid = np.stack(grid, axis=-1).reshape(-1, 2)
    emb_h = get_1d_pos_embed(embed_dim // 2, grid[:, 0])
    emb_w = get_1d_pos_embed(embed_dim // 2, grid[:, 1])
    pe = np.concatenate([emb_h, emb_w], axis=1)
    pe = torch.from_numpy(pe).float()
    return pe.unsqueeze(0).to(device)  # (1, N, C)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=False),
        )
    def forward(self, x):
        return self.double_conv(x)

def center_crop(enc_feature, target_feature):
    _, _, h, w = target_feature.size()
    _, _, H, W = enc_feature.size()
    dh = (H - h) // 2
    dw = (W - w) // 2
    return enc_feature[:, :, dh:dh+h, dw:dw+w]

# --- 1x1 FuseBlock for multi-scale fusion ---
class FuseBlock(nn.Module):
    def __init__(self, c1, c2, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(c1 + c2, out_ch, 1)
        self.norm = nn.InstanceNorm2d(out_ch, affine=True)
    def forward(self, feat1, feat2):
        return self.norm(self.conv1(torch.cat([feat1, feat2], 1)))

# --- Squeeze-and-Excitation (SE) block for optional attention gating ---
class SEBlock(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, ch // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch // r, ch, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(self.avg(x).view(x.size(0), -1)).view(x.size(0), x.size(1), 1, 1)
        return x * w

# --- CORRECTED Cross-Attention with proper scaling and normalization ---
class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.d_k = dim // heads  # CORRECTED: Calculate d_k properly
        self.scale = self.d_k ** -0.5  # CORRECTED: Use d_k for scaling
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim)
    def forward(self, query, key_value, pe=None):
        # Optionally add positional encoding
        if pe is not None:
            query = query + pe
            key_value = key_value + pe
        q = self.norm_q(query)
        kv_input = self.norm_kv(key_value)
        q = self.q_proj(q)
        kv = self.kv_proj(kv_input)
        B, N, C = q.shape
        H = self.heads
        q = q.view(B, N, H, self.d_k).transpose(1, 2)  # CORRECTED: Use self.d_k
        kv = kv.view(B, -1, H, 2*self.d_k).transpose(1, 2)  # CORRECTED: Use self.d_k
        k, v = kv.chunk(2, dim=-1)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        return self.norm_out(self.out_proj(out))

# --- CORRECTED Self-Attention with proper scaling and normalization ---
class LightweightSelfAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.d_k = dim // heads  # CORRECTED: Calculate d_k properly
        self.scale = self.d_k ** -0.5  # CORRECTED: Use d_k for scaling
        self.to_qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.norm = nn.InstanceNorm2d(dim, affine=True)
    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(B, self.heads, self.d_k, H * W)  # CORRECTED: Use self.d_k
        k = k.view(B, self.heads, self.d_k, H * W)  # CORRECTED: Use self.d_k
        v = v.view(B, self.heads, self.d_k, H * W)  # CORRECTED: Use self.d_k
        attn = torch.einsum('bhcn,bhcm->bhnm', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bhnm,bhcm->bhcn', attn, v)
        out = out.contiguous().view(B, C, H, W)
        return self.norm(self.out_proj(out))

class FIXED_UNet_FusionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoders
        self.rgb_enc1 = DoubleConv(3, 64)
        self.rgb_enc2 = DoubleConv(64, 128)
        self.rgb_enc3 = DoubleConv(128, 256)
        self.th_enc1 = DoubleConv(3, 64)
        self.th_enc2 = DoubleConv(64, 128)
        self.th_enc3 = DoubleConv(128, 256)
        self.pool = nn.MaxPool2d(2)
        # Bottleneck
        self.bottleneck_conv = DoubleConv(256, 256)
        # Skip fusion blocks
        self.fuse1 = FuseBlock(64, 64, 64)
        self.fuse2 = FuseBlock(128, 128, 128)
        self.fuse3 = FuseBlock(256, 256, 256)
        # Cross and Self Attention
        self.cross_attn = CrossAttentionBlock(dim=256, heads=4)
        self.self_attn = LightweightSelfAttention(dim=256, heads=4)
        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(128 + 256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(64 + 128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(32 + 64, 32)
        self.final_conv = nn.Conv2d(32, 3, kernel_size=1)

    def forward(self, x_rgb, x_th):
        # Store original input size
        original_size = x_rgb.shape[-2:]
        
        # Encoding RGB
        r1 = self.rgb_enc1(x_rgb)
        r2 = self.rgb_enc2(self.pool(r1))
        r3 = self.rgb_enc3(self.pool(r2))
        # Encoding Thermal
        t1 = self.th_enc1(x_th)
        t2 = self.th_enc2(self.pool(t1))
        t3 = self.th_enc3(self.pool(t2))
        # Bottleneck pooling
        r3_pooled = self.pool(r3)
        t3_pooled = self.pool(t3)
        r_bottleneck = self.bottleneck_conv(r3_pooled)
        t_bottleneck = self.bottleneck_conv(t3_pooled)
        # Bottleneck symmetric cross-attention fusion
        B, C, H, W = r_bottleneck.shape
        r_flat = r_bottleneck.view(B, C, -1).permute(0, 2, 1)
        t_flat = t_bottleneck.view(B, C, -1).permute(0, 2, 1)
        pe = get_2d_sincos_pos_embed(C, H, W, device=r_flat.device)
        fused_rt = self.cross_attn(r_flat, t_flat, pe=pe)
        fused_tr = self.cross_attn(t_flat, r_flat, pe=pe)
        fused_flat = 0.5 * (fused_rt + fused_tr)   # Symmetric bi-directional fusion
        fused_map = fused_flat.permute(0, 2, 1).view(B, C, H, W)
        # Self-attn refine (downsampled for memory)
        fused_map_down = F.avg_pool2d(fused_map, kernel_size=2)
        refined_down = self.self_attn(fused_map_down)
        # FIXED: Interpolate to match the size after 3 pooling operations
        # After 3 pooling operations, size is original_size // 8
        expected_size = (original_size[0] // 8, original_size[1] // 8)
        refined = F.interpolate(refined_down, size=expected_size, mode='bilinear', align_corners=False)
        # Decoder with skip fusion (RGB + thermal feature at each scale)
        d3 = self.up3(refined)
        r3_c = center_crop(r3, d3)
        t3_c = center_crop(t3, d3)
        f3_c = self.fuse3(r3_c, t3_c)
        d3 = self.dec3(torch.cat([d3, f3_c], dim=1))
        d2 = self.up2(d3)
        r2_c = center_crop(r2, d2)
        t2_c = center_crop(t2, d2)
        f2_c = self.fuse2(r2_c, t2_c)
        d2 = self.dec2(torch.cat([d2, f2_c], dim=1))
        d1 = self.up1(d2)
        r1_c = center_crop(r1, d1)
        t1_c = center_crop(t1, d1)
        f1_c = self.fuse1(r1_c, t1_c)
        d1 = self.dec1(torch.cat([d1, f1_c], dim=1))
        
        # FIXED: Ensure output matches original input size
        out = self.final_conv(d1)
        if out.shape[-2:] != original_size:
            out = F.interpolate(out, size=original_size, mode='bilinear', align_corners=False)
        
        out = torch.sigmoid(out)
        return out