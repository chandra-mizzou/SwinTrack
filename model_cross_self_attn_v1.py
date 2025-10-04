import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),  # Replaced BatchNorm2d
            nn.ReLU(inplace=False),  # Avoid in-place ops in mixed precision

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),  # Replaced BatchNorm2d
            nn.ReLU(inplace=False),  # Avoid in-place ops in mixed precision
        )

    def forward(self, x):
        return self.double_conv(x)


def center_crop(enc_feature, target_feature):
    _, _, h, w = target_feature.size()
    _, _, H, W = enc_feature.size()
    dh = (H - h) // 2
    dw = (W - w) // 2
    return enc_feature[:, :, dh:dh+h, dw:dw+w]

# ----------- Lightweight Attention Modules -----------

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, query, key_value):
        B, N, C = query.shape
        H = self.heads

        q = self.q_proj(query).view(B, N, H, C // H).transpose(1, 2)
        kv = self.kv_proj(key_value).view(B, -1, H, 2 * (C // H)).transpose(1, 2)
        k, v = kv.chunk(2, dim=-1)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, C)

        return self.out_proj(out)

class LightweightSelfAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        q = q.view(B, self.heads, C // self.heads, H * W)
        k = k.view(B, self.heads, C // self.heads, H * W)
        v = v.view(B, self.heads, C // self.heads, H * W)

        attn = torch.einsum('bhcn,bhcm->bhnm', q, k) * self.scale
        attn = attn.softmax(dim=-1)

        out = torch.einsum('bhnm,bhcm->bhcn', attn, v)
        out = out.contiguous().view(B, C, H, W)

        return self.out_proj(out)

# ----------- UNet-Fusion Transformer -----------

class UNet_FusionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        
        # RGB and Thermal Encoders
        self.rgb_enc1 = DoubleConv(3, 64)
        self.rgb_enc2 = DoubleConv(64, 128)
        self.rgb_enc3 = DoubleConv(128, 256)

        self.th_enc1 = DoubleConv(3, 64)
        self.th_enc2 = DoubleConv(64, 128)
        self.th_enc3 = DoubleConv(128, 256)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck_conv = DoubleConv(256, 256)

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

        # Final conv
        self.final_conv = nn.Conv2d(32, 3, kernel_size=1)

    def forward(self, x_rgb, x_th):

        # Encoding RGB
        r1 = self.rgb_enc1(x_rgb)
        r2 = self.rgb_enc2(self.pool(r1))
        r3 = self.rgb_enc3(self.pool(r2))

        # Encoding Thermal
        t1 = self.th_enc1(x_th)
        t2 = self.th_enc2(self.pool(t1))
        t3 = self.th_enc3(self.pool(t2))

        # Bottleneck
        r3_pooled = self.pool(r3)
        t3_pooled = self.pool(t3)

        r_bottleneck = self.bottleneck_conv(r3_pooled)
        t_bottleneck = self.bottleneck_conv(t3_pooled)

        # Flatten for Cross-Attention
        B, C, H, W = r_bottleneck.shape
        r_flat = r_bottleneck.view(B, C, -1).permute(0, 2, 1)
        t_flat = t_bottleneck.view(B, C, -1).permute(0, 2, 1)

        fused = self.cross_attn(r_flat, t_flat)
        fused_map = fused.permute(0, 2, 1).view(B, C, H, W)

        # --- Downsample before Self Attention to save memory ---
        fused_map_down = F.avg_pool2d(fused_map, kernel_size=2)  # Downsample by 2

        refined_down = self.self_attn(fused_map_down)

        # Upsample back to original spatial resolution
        refined = F.interpolate(refined_down, size=(H, W), mode='bilinear', align_corners=False)

        # Decoder
        d3 = self.up3(refined)
        r3_c = center_crop(r3, d3)
        d3 = self.dec3(torch.cat([d3, r3_c], dim=1))

        d2 = self.up2(d3)
        r2_c = center_crop(r2, d2)
        d2 = self.dec2(torch.cat([d2, r2_c], dim=1))

        d1 = self.up1(d2)
        r1_c = center_crop(r1, d1)
        d1 = self.dec1(torch.cat([d1, r1_c], dim=1))

        out = torch.sigmoid(self.final_conv(d1))
        return out