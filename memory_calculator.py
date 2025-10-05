import torch

def calculate_memory_usage(batch_size, height, width, channels=3):
    """Calculate approximate GPU memory usage for training"""
    
    print(f"🔍 Memory Usage Calculation for {batch_size}x{height}x{width} images:")
    print("=" * 60)
    
    # Input tensors (RGB + Thermal + GT)
    input_memory = batch_size * height * width * channels * 3 * 4  # 3 images, float32
    print(f"📊 Input tensors: {input_memory / 1024**3:.2f} GB")
    
    # Model parameters (approximate)
    model_params = 15_000_000  # ~15M parameters
    model_memory = model_params * 4  # float32
    print(f"📊 Model parameters: {model_memory / 1024**3:.2f} GB")
    
    # Intermediate activations (approximate)
    # UNet has multiple levels, estimate for bottleneck
    bottleneck_h = height // 8  # After 3 pooling layers
    bottleneck_w = width // 8
    activation_memory = batch_size * 256 * bottleneck_h * bottleneck_w * 4  # 256 channels
    print(f"📊 Bottleneck activations: {activation_memory / 1024**3:.2f} GB")
    
    # Attention computation (most memory intensive)
    attention_memory = batch_size * (bottleneck_h * bottleneck_w) ** 2 * 4  # O(N²) complexity
    print(f"📊 Attention computation: {attention_memory / 1024**3:.2f} GB")
    
    # Gradients (same as parameters)
    gradient_memory = model_memory
    print(f"📊 Gradients: {gradient_memory / 1024**3:.2f} GB")
    
    # Optimizer states (Adam stores momentum and variance)
    optimizer_memory = model_memory * 2
    print(f"📊 Optimizer states: {optimizer_memory / 1024**3:.2f} GB")
    
    # Total estimated memory
    total_memory = (input_memory + model_memory + activation_memory + 
                   attention_memory + gradient_memory + optimizer_memory)
    
    print(f"\n🎯 TOTAL ESTIMATED MEMORY: {total_memory / 1024**3:.2f} GB")
    
    # Memory recommendations
    print(f"\n💡 Memory Recommendations:")
    if total_memory / 1024**3 < 8:
        print("✅ Should work on 8GB GPU")
    elif total_memory / 1024**3 < 16:
        print("✅ Should work on 16GB GPU")
    elif total_memory / 1024**3 < 24:
        print("⚠️ Needs 24GB+ GPU")
    else:
        print("❌ Needs very large GPU or gradient checkpointing")
    
    return total_memory / 1024**3

def test_different_sizes():
    """Test memory usage for different image sizes"""
    
    print("🚀 Memory Usage Analysis for Different Image Sizes")
    print("=" * 70)
    
    sizes_to_test = [
        (256, 256, 2),   # Small, batch 2
        (512, 512, 2),   # Medium, batch 2  
        (1024, 1024, 1), # Large, batch 1
        (1024, 1024, 2), # Large, batch 2
        (1536, 1536, 1), # Very large, batch 1
    ]
    
    for height, width, batch_size in sizes_to_test:
        print(f"\n📊 Testing {height}x{width} with batch_size={batch_size}:")
        memory_gb = calculate_memory_usage(batch_size, height, width)
        print("-" * 50)

if __name__ == "__main__":
    test_different_sizes()
    
    print(f"\n🔧 Tips for Training at 1024x1024:")
    print("=" * 50)
    print("1. Use batch_size=1 for 1024x1024")
    print("2. Enable gradient checkpointing if needed")
    print("3. Use mixed precision training (already enabled)")
    print("4. Consider reducing model depth if memory issues")
    print("5. Monitor GPU memory usage during training")