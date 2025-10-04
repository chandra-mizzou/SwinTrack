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

# --- Enhanced FuseBlock with attention ---
class EnhancedFuseBlock(nn.Module):
    def __init__(self, c1, c2, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(c1 + c2, out_ch, 1)
        self.norm = nn.InstanceNorm2d(out_ch, affine=True)
        # Add channel attention
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, out_ch // 16, 1),
            nn.ReLU(),
            nn.Conv2d(out_ch // 16, out_ch, 1),
            nn.Sigmoid()
        )
        
    def forward(self, feat1, feat2):
        fused = self.norm(self.conv1(torch.cat([feat1, feat2], 1)))
        attn = self.channel_attn(fused)
        return fused * attn

# --- Squeeze-and-Excitation (SE) block ---
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

# --- Enhanced Cross-Attention with fusion gates ---
class EnhancedCrossAttentionBlock(nn.Module):
    def __init__(self, dim, heads=8):  # Increased heads
        super().__init__()
        self.heads = heads
        self.d_k = dim // heads
        self.scale = self.d_k ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim)
        
        # Fusion gate for adaptive weighting
        self.fusion_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        
    def forward(self, query, key_value, pe=None):
        if pe is not None:
            query = query + pe
            key_value = key_value + pe
            
        q = self.norm_q(query)
        kv_input = self.norm_kv(key_value)
        q = self.q_proj(q)
        kv = self.kv_proj(kv_input)
        
        B, N, C = q.shape
        H = self.heads
        q = q.view(B, N, H, self.d_k).transpose(1, 2)
        kv = kv.view(B, -1, H, 2*self.d_k).transpose(1, 2)
        k, v = kv.chunk(2, dim=-1)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        
        # Enhanced fusion with adaptive gating
        fusion_input = torch.cat([query, out], dim=-1)
        fusion_weight = self.fusion_gate(fusion_input)
        fused_out = fusion_weight * query + (1 - fusion_weight) * out
        
        return self.norm_out(self.out_proj(fused_out))

# --- Enhanced Self-Attention ---
class EnhancedSelfAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.d_k = dim // heads
        self.scale = self.d_k ** -0.5
        self.to_qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.norm = nn.InstanceNorm2d(dim, affine=True)
        
    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        
        q = q.view(B, self.heads, self.d_k, H * W)
        k = k.view(B, self.heads, self.d_k, H * W)
        v = v.view(B, self.heads, self.d_k, H * W)
        
        attn = torch.einsum('bhcn,bhcm->bhnm', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bhnm,bhcm->bhcn', attn, v)
        out = out.contiguous().view(B, C, H, W)
        
        # Add residual connection
        out = out + residual
        return self.norm(self.out_proj(out))

class FinalImprovedUNet_FusionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # Enhanced encoders with SE blocks
        self.rgb_enc1 = DoubleConv(3, 64)
        self.rgb_enc2 = DoubleConv(64, 128)
        self.rgb_enc3 = DoubleConv(128, 256)
        self.rgb_enc4 = DoubleConv(256, 512)  # Deeper encoding
        
        self.th_enc1 = DoubleConv(3, 64)
        self.th_enc2 = DoubleConv(64, 128)
        self.th_enc3 = DoubleConv(128, 256)
        self.th_enc4 = DoubleConv(256, 512)  # Deeper encoding
        
        self.pool = nn.MaxPool2d(2)
        
        # Enhanced bottleneck
        self.bottleneck_conv_rgb = DoubleConv(512, 512)
        self.bottleneck_conv_th = DoubleConv(512, 512)
        
        # Multiple cross-attention stages
        self.cross_attn1 = EnhancedCrossAttentionBlock(dim=512, heads=8)
        self.cross_attn2 = EnhancedCrossAttentionBlock(dim=512, heads=8)
        
        # Enhanced self-attention
        self.self_attn = EnhancedSelfAttention(dim=512, heads=8)
        
        # Enhanced skip fusion blocks
        self.fuse1 = EnhancedFuseBlock(64, 64, 64)
        self.fuse2 = EnhancedFuseBlock(128, 128, 128)
        self.fuse3 = EnhancedFuseBlock(256, 256, 256)
        self.fuse4 = EnhancedFuseBlock(512, 512, 512)
        
        # SE blocks for channel attention
        self.se1 = SEBlock(64)
        self.se2 = SEBlock(128)
        self.se3 = SEBlock(256)
        self.se4 = SEBlock(512)
        
        # Enhanced decoder
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(256 + 256, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(128 + 128, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(64 + 64, 64)
        
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(32 + 64, 32)
        
        # Enhanced final output
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=1)
        )
        
        # Residual connection with original inputs
        self.residual_fusion = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=1)
        )

    def forward(self, x_rgb, x_th):
        # Store original inputs
        orig_rgb = x_rgb
        orig_th = x_th
        
        # Enhanced encoding with SE blocks
        r1 = self.rgb_enc1(x_rgb)
        r2 = self.rgb_enc2(self.pool(r1))
        r3 = self.rgb_enc3(self.pool(r2))
        r4 = self.rgb_enc4(self.pool(r3))
        
        t1 = self.th_enc1(x_th)
        t2 = self.th_enc2(self.pool(t1))
        t3 = self.th_enc3(self.pool(t2))
        t4 = self.th_enc4(self.pool(t3))
        
        # Enhanced bottleneck processing
        r_bottleneck = self.bottleneck_conv_rgb(self.pool(r4))
        t_bottleneck = self.bottleneck_conv_th(self.pool(t4))
        
        # Apply SE blocks
        r_bottleneck = self.se4(r_bottleneck)
        t_bottleneck = self.se4(t_bottleneck)
        
        # Multiple cross-attention stages
        B, C, H, W = r_bottleneck.shape
        r_flat = r_bottleneck.view(B, C, -1).permute(0, 2, 1)
        t_flat = t_bottleneck.view(B, C, -1).permute(0, 2, 1)
        pe = get_2d_sincos_pos_embed(C, H, W, device=r_flat.device)
        
        # First cross-attention: RGB queries Thermal
        fused_rgb = self.cross_attn1(r_flat, t_flat, pe=pe)
        fused_rgb_map = fused_rgb.permute(0, 2, 1).view(B, C, H, W)
        
        # Second cross-attention: Thermal queries RGB
        fused_th = self.cross_attn2(t_flat, r_flat, pe=pe)
        fused_th_map = fused_th.permute(0, 2, 1).view(B, C, H, W)
        
        # Combine both fusion results
        combined_fused = (fused_rgb_map + fused_th_map) / 2
        
        # Self-attention refinement
        refined = self.self_attn(combined_fused)
        
        # Enhanced decoder with skip fusion and SE blocks
        d4 = self.up4(refined)
        r3_c = center_crop(r3, d4)
        t3_c = center_crop(t3, d4)
        f3_c = self.fuse3(r3_c, t3_c)
        f3_c = self.se3(f3_c)
        d4 = self.dec4(torch.cat([d4, f3_c], dim=1))
        
        d3 = self.up3(d4)
        r2_c = center_crop(r2, d3)
        t2_c = center_crop(t2, d3)
        f2_c = self.fuse2(r2_c, t2_c)
        f2_c = self.se2(f2_c)
        d3 = self.dec3(torch.cat([d3, f2_c], dim=1))
        
        d2 = self.up2(d3)
        r1_c = center_crop(r1, d2)
        t1_c = center_crop(t1, d2)
        f1_c = self.fuse1(r1_c, t1_c)
        f1_c = self.se1(f1_c)
        d2 = self.dec2(torch.cat([d2, f1_c], dim=1))
        
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, f1_c], dim=1))
        
        # Enhanced final output
        main_output = torch.sigmoid(self.final_conv(d1))
        
        # Residual connection with original inputs
        residual_input = torch.cat([orig_rgb, orig_th], dim=1)
        residual_output = torch.sigmoid(self.residual_fusion(residual_input))
        
        # Adaptive combination
        alpha = 0.8  # Weight for main output
        final_output = alpha * main_output + (1 - alpha) * residual_output
        
        return final_output