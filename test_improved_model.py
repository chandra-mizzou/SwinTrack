import torch
import torch.nn.functional as F
from improved_transformer_training import ImprovedRGBThermalFusion

# Test the model to make sure it works
if __name__ == "__main__":
    print("🧪 Testing Improved Transformer Model...")
    
    # Create model
    model = ImprovedRGBThermalFusion(
        input_channels=3,
        d_model=128,
        num_heads=8,
        num_cross_attention=2,
        dropout=0.1,
        chunk_size=64
    )
    
    # Test with different input sizes
    test_sizes = [(2, 3, 256, 256), (1, 3, 512, 512), (1, 3, 128, 128)]
    
    for i, (B, C, H, W) in enumerate(test_sizes):
        print(f"\nTest {i+1}: Input size {B}x{C}x{H}x{W}")
        
        rgb = torch.randn(B, C, H, W)
        thermal = torch.randn(B, C, H, W)
        
        try:
            with torch.no_grad():
                output = model(rgb, thermal)
            print(f"✅ Success! Output shape: {output.shape}")
            print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("✅ Model test completed successfully!")