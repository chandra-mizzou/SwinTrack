import torch
import torch.nn as nn
import torch.nn.functional as F

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

# IMPROVED: Enhanced Cross-Attention with better fusion strategy
class ImprovedCrossAttentionBlock(nn.Module):
    def __init__(self, dim, heads=8):  # Increased heads for better attention
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.dim = dim

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        # IMPROVED: Add fusion gate for adaptive weighting
        self.fusion_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def forward(self, query, key_value):
        B, N, C = query.shape
        H = self.heads

        q = self.q_proj(query).view(B, N, H, C // H).transpose(1, 2)
        k = self.k_proj(key_value).view(B, N, H, C // H).transpose(1, 2)
        v = self.v_proj(key_value).view(B, N, H, C // H).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        
        # IMPROVED: Adaptive fusion instead of simple attention
        fusion_input = torch.cat([query, out], dim=-1)
        fusion_weight = self.fusion_gate(fusion_input)
        fused_out = fusion_weight * query + (1 - fusion_weight) * out

        return self.out_proj(fused_out)

# IMPROVED: Enhanced Self-Attention with residual connections
class ImprovedSelfAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.dim = dim

        self.to_qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)
        
        # IMPROVED: Add layer normalization
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        q = q.view(B, self.heads, C // self.heads, H * W)
        k = k.view(B, self.heads, C // self.heads, H * W)
        v = v.view(B, self.heads, C // self.heads, H * W)

        attn = torch.einsum('bhcn,bhcm->bhnm', q, k) * self.scale
        attn = attn.softmax(dim=-1)

        out = torch.einsum('bhnm,bhcm->bhcn', attn, v)
        out = out.contiguous().view(B, C, H, W)
        out = self.out_proj(out)
        
        # IMPROVED: Residual connection
        return out + residual

# IMPROVED: Enhanced UNet-Fusion Transformer
class ImprovedUNet_FusionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        
        # RGB and Thermal Encoders with skip connections
        self.rgb_enc1 = DoubleConv(3, 64)
        self.rgb_enc2 = DoubleConv(64, 128)
        self.rgb_enc3 = DoubleConv(128, 256)
        self.rgb_enc4 = DoubleConv(256, 512)  # Added deeper encoding

        self.th_enc1 = DoubleConv(3, 64)
        self.th_enc2 = DoubleConv(64, 128)
        self.th_enc3 = DoubleConv(128, 256)
        self.th_enc4 = DoubleConv(256, 512)  # Added deeper encoding

        self.pool = nn.MaxPool2d(2)

        # IMPROVED: Enhanced bottleneck with multiple fusion stages
        self.bottleneck_conv_rgb = DoubleConv(512, 512)
        self.bottleneck_conv_th = DoubleConv(512, 512)
        
        # Multiple cross-attention layers for better fusion
        self.cross_attn1 = ImprovedCrossAttentionBlock(dim=512, heads=8)
        self.cross_attn2 = ImprovedCrossAttentionBlock(dim=512, heads=8)
        
        # Enhanced self-attention
        self.self_attn = ImprovedSelfAttention(dim=512, heads=8)

        # IMPROVED: Enhanced decoder with more skip connections
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(256 + 256, 256)  # Skip connection from enc3

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(128 + 128, 128)  # Skip connection from enc2

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(64 + 64, 64)  # Skip connection from enc1

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(32, 32)

        # IMPROVED: Enhanced final output with residual connection
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=1)
        )
        
        # IMPROVED: Add feature fusion module
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, padding=1),  # RGB + Thermal input
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=1)
        )

    def forward(self, x_rgb, x_th):
        # IMPROVED: Store original inputs for residual connection
        orig_rgb = x_rgb
        orig_th = x_th

        # Encoding RGB
        r1 = self.rgb_enc1(x_rgb)
        r2 = self.rgb_enc2(self.pool(r1))
        r3 = self.rgb_enc3(self.pool(r2))
        r4 = self.rgb_enc4(self.pool(r3))  # Deeper encoding

        # Encoding Thermal
        t1 = self.th_enc1(x_th)
        t2 = self.th_enc2(self.pool(t1))
        t3 = self.th_enc3(self.pool(t2))
        t4 = self.th_enc4(self.pool(t3))  # Deeper encoding

        # IMPROVED: Enhanced bottleneck processing
        r_bottleneck = self.bottleneck_conv_rgb(self.pool(r4))
        t_bottleneck = self.bottleneck_conv_th(self.pool(t4))

        # IMPROVED: Multiple cross-attention stages for better fusion
        B, C, H, W = r_bottleneck.shape
        
        # First cross-attention: RGB queries Thermal
        r_flat = r_bottleneck.view(B, C, -1).permute(0, 2, 1)
        t_flat = t_bottleneck.view(B, C, -1).permute(0, 2, 1)
        
        fused_rgb = self.cross_attn1(r_flat, t_flat)
        fused_rgb_map = fused_rgb.permute(0, 2, 1).view(B, C, H, W)
        
        # Second cross-attention: Thermal queries RGB
        fused_th = self.cross_attn2(t_flat, r_flat)
        fused_th_map = fused_th.permute(0, 2, 1).view(B, C, H, W)
        
        # IMPROVED: Combine both fusion results
        combined_fused = (fused_rgb_map + fused_th_map) / 2
        
        # Self-attention refinement
        refined = self.self_attn(combined_fused)

        # IMPROVED: Enhanced decoder with skip connections
        d4 = self.up4(refined)
        r3_c = center_crop(r3, d4)
        d4 = self.dec4(torch.cat([d4, r3_c], dim=1))

        d3 = self.up3(d4)
        r2_c = center_crop(r2, d3)
        d3 = self.dec3(torch.cat([d3, r2_c], dim=1))

        d2 = self.up2(d3)
        r1_c = center_crop(r1, d2)
        d2 = self.dec2(torch.cat([d2, r1_c], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(d1)

        # IMPROVED: Enhanced final output
        main_output = torch.sigmoid(self.final_conv(d1))
        
        # IMPROVED: Add residual connection with original inputs
        residual_input = torch.cat([orig_rgb, orig_th], dim=1)
        residual_output = torch.sigmoid(self.feature_fusion(residual_input))
        
        # IMPROVED: Adaptive combination of main and residual outputs
        alpha = 0.7  # Weight for main output
        final_output = alpha * main_output + (1 - alpha) * residual_output
        
        return final_output