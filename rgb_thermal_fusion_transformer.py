import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings

# Suppress interpolation warnings
warnings.filterwarnings("ignore", message=".*interpolate.*recompute_scale_factor.*")

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer attention"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.w_o(context)
        
        return output, attention_weights

class CrossAttentionBlock(nn.Module):
    """Cross attention between RGB and Thermal features"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, rgb_features, thermal_features):
        # Cross attention: RGB queries, Thermal keys/values
        cross_out, _ = self.cross_attention(rgb_features, thermal_features, thermal_features)
        rgb_features = self.norm1(rgb_features + self.dropout(cross_out))
        
        # Cross attention: Thermal queries, RGB keys/values
        cross_out, _ = self.cross_attention(thermal_features, rgb_features, rgb_features)
        thermal_features = self.norm1(thermal_features + self.dropout(cross_out))
        
        # Feed forward
        rgb_ff = self.feed_forward(rgb_features)
        rgb_features = self.norm2(rgb_features + self.dropout(rgb_ff))
        
        thermal_ff = self.feed_forward(thermal_features)
        thermal_features = self.norm2(thermal_features + self.dropout(thermal_ff))
        
        return rgb_features, thermal_features

class SelfAttentionBlock(nn.Module):
    """Self attention for feature refinement"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, features):
        # Self attention
        self_out, _ = self.self_attention(features, features, features)
        features = self.norm1(features + self.dropout(self_out))
        
        # Feed forward
        ff_out = self.feed_forward(features)
        features = self.norm2(features + self.dropout(ff_out))
        
        return features

class RGBThermalFusionTransformer(nn.Module):
    """RGB-Thermal Fusion Model with Transformer Attention"""
    def __init__(self, input_channels=3, d_model=256, num_heads=8, num_cross_attention=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Encoder for RGB
        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, d_model, 3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True)
        )
        
        # Encoder for Thermal
        self.thermal_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, d_model, 3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True)
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Cross attention blocks (mid-level)
        self.cross_attention_blocks = nn.ModuleList([
            CrossAttentionBlock(d_model, num_heads, dropout)
            for _ in range(num_cross_attention)
        ])
        
        # Self attention block (before final fusion)
        self.self_attention = SelfAttentionBlock(d_model, num_heads, dropout)
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(d_model * 2, d_model, 1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model, d_model, 3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(d_model, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, input_channels, 1)
        )
        
    def forward(self, rgb, thermal):
        batch_size, channels, height, width = rgb.shape
        
        # Encode RGB and Thermal
        rgb_features = self.rgb_encoder(rgb)  # [B, d_model, H, W]
        thermal_features = self.thermal_encoder(thermal)  # [B, d_model, H, W]
        
        # Reshape for transformer: [B, H*W, d_model]
        rgb_features = rgb_features.view(batch_size, self.d_model, -1).transpose(1, 2)
        thermal_features = thermal_features.view(batch_size, self.d_model, -1).transpose(1, 2)
        
        # Add positional encoding
        rgb_features = self.pos_encoding(rgb_features)
        thermal_features = self.pos_encoding(thermal_features)
        
        # Cross attention blocks (mid-level)
        for cross_attn in self.cross_attention_blocks:
            rgb_features, thermal_features = cross_attn(rgb_features, thermal_features)
        
        # Self attention (before final fusion)
        rgb_features = self.self_attention(rgb_features)
        thermal_features = self.self_attention(thermal_features)
        
        # Reshape back to spatial: [B, d_model, H, W]
        rgb_features = rgb_features.transpose(1, 2).view(batch_size, self.d_model, height, width)
        thermal_features = thermal_features.transpose(1, 2).view(batch_size, self.d_model, height, width)
        
        # Feature fusion
        fused_features = torch.cat([rgb_features, thermal_features], dim=1)
        fused_features = self.feature_fusion(fused_features)
        
        # Decode to final output
        output = self.decoder(fused_features)
        
        # Ensure output matches input size
        if output.shape[-2:] != (height, width):
            output = F.interpolate(output, size=(height, width), mode='bilinear', align_corners=False, recompute_scale_factor=True)
        
        return torch.sigmoid(output)

# Test the model
if __name__ == "__main__":
    # Create model
    model = RGBThermalFusionTransformer(
        input_channels=3,
        d_model=256,
        num_heads=8,
        num_cross_attention=2,
        dropout=0.1
    )
    
    # Test with sample inputs
    rgb = torch.randn(2, 3, 256, 256)
    thermal = torch.randn(2, 3, 256, 256)
    
    # Forward pass
    output = model(rgb, thermal)
    
    print(f"Input RGB shape: {rgb.shape}")
    print(f"Input Thermal shape: {thermal.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test with different sizes
    rgb_large = torch.randn(1, 3, 512, 512)
    thermal_large = torch.randn(1, 3, 512, 512)
    output_large = model(rgb_large, thermal_large)
    print(f"Large input RGB shape: {rgb_large.shape}")
    print(f"Large output shape: {output_large.shape}")