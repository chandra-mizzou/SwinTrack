import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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

class MemoryEfficientAttention(nn.Module):
    """Memory-efficient multi-head attention with reduced dimensions"""
    def __init__(self, d_model, num_heads, dropout=0.1, chunk_size=64):
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
        
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.shape
        
        # Linear projections
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Process in chunks to save memory
        output_chunks = []
        for i in range(0, seq_len, self.chunk_size):
            end_i = min(i + self.chunk_size, seq_len)
            Q_chunk = Q[:, :, i:end_i, :]
            
            # Attention scores
            scores = torch.matmul(Q_chunk, K.transpose(-2, -1)) / self.scale
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
                
            attention_weights = F.softmax(scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            
            # Apply attention to values
            context = torch.matmul(attention_weights, V)
            output_chunks.append(context)
        
        # Concatenate chunks
        output = torch.cat(output_chunks, dim=2)
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Final linear projection
        output = self.w_o(output)
        
        return output, None  # Return None for attention weights to save memory

class CrossAttentionBlock(nn.Module):
    """Cross attention between RGB and Thermal features"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.cross_attention = MemoryEfficientAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 2),  # Reduced from 4x to 2x
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
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
        self.self_attention = MemoryEfficientAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 2),  # Reduced from 4x to 2x
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
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

class MemoryEfficientRGBThermalFusion(nn.Module):
    """Memory-efficient RGB-Thermal Fusion Model with Transformer Attention"""
    def __init__(self, input_channels=3, d_model=128, num_heads=4, num_cross_attention=1, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Encoder for RGB
        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, d_model, 3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True)
        )
        
        # Encoder for Thermal
        self.thermal_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, d_model, 3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True)
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Cross attention blocks (mid-level) - Reduced to 1
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
            nn.Conv2d(d_model, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, input_channels, 1)
        )
        
    def forward(self, rgb, thermal):
        batch_size, channels, height, width = rgb.shape
        
        # Encode RGB and Thermal
        rgb_features = self.rgb_encoder(rgb)  # [B, d_model, H, W]
        thermal_features = self.thermal_encoder(thermal)  # [B, d_model, H, W]
        
        # Downsample for attention to save memory
        if height > 64 or width > 64:
            scale_factor = min(64 / height, 64 / width)
            new_h, new_w = int(height * scale_factor), int(width * scale_factor)
            rgb_features = F.interpolate(rgb_features, size=(new_h, new_w), mode='bilinear', align_corners=False)
            thermal_features = F.interpolate(thermal_features, size=(new_h, new_w), mode='bilinear', align_corners=False)
            height, width = new_h, new_w
        
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
        
        # Upsample back to original size if needed
        if rgb_features.shape[-2:] != (rgb.shape[-2], rgb.shape[-1]):
            rgb_features = F.interpolate(rgb_features, size=(rgb.shape[-2], rgb.shape[-1]), mode='bilinear', align_corners=False)
            thermal_features = F.interpolate(thermal_features, size=(rgb.shape[-2], rgb.shape[-1]), mode='bilinear', align_corners=False)
        
        # Feature fusion
        fused_features = torch.cat([rgb_features, thermal_features], dim=1)
        fused_features = self.feature_fusion(fused_features)
        
        # Decode to final output
        output = self.decoder(fused_features)
        
        # Ensure output matches input size
        if output.shape[-2:] != (rgb.shape[-2], rgb.shape[-1]):
            output = F.interpolate(output, size=(rgb.shape[-2], rgb.shape[-1]), mode='bilinear', align_corners=False)
        
        return torch.sigmoid(output)

# Test the model
if __name__ == "__main__":
    # Create model
    model = MemoryEfficientRGBThermalFusion(
        input_channels=3,
        d_model=128,  # Reduced from 256
        num_heads=4,  # Reduced from 8
        num_cross_attention=1,  # Reduced from 2
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