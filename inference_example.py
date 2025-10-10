import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from best_model_for_training import UNet_FusionTransformer
import numpy as np

def test_model_output_sizes():
    """Test the model with different input sizes to show output behavior"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet_FusionTransformer()
    
    # Load trained model (if available)
    try:
        model.load_state_dict(torch.load('/usr/mvl2/cvrfr/KeyPonFuse_V1/Codes/models_22aug25/model_training/trained_models/UNet_native_resolution.pth', map_location=device))
        print("✅ Loaded trained model")
    except:
        print("⚠️ Using untrained model for demonstration")
    
    model = model.to(device)
    model.eval()
    
    # Test different input sizes
    test_sizes = [
        (256, 256),    # Small
        (512, 512),    # Medium
        (800, 600),    # Rectangular
        (1024, 768),   # Large
        (640, 480),    # Another rectangular
    ]
    
    print("🔍 Testing Model Output Sizes:")
    print("=" * 50)
    
    with torch.no_grad():
        for h, w in test_sizes:
            # Create dummy inputs
            rgb_input = torch.randn(1, 3, h, w).to(device)
            thermal_input = torch.randn(1, 3, h, w).to(device)
            
            # Forward pass
            output = model(rgb_input, thermal_input)
            
            # Check output size
            output_h, output_w = output.shape[2], output.shape[3]
            
            print(f"Input:  {h:4d} x {w:4d} → Output: {output_h:4d} x {output_w:4d} ✅")
            
            # Verify they match
            if output_h == h and output_w == w:
                print(f"         Perfect size match! ✓")
            else:
                print(f"         ❌ Size mismatch!")

def inference_with_real_images(rgb_path, thermal_path, output_path):
    """Example of inference with real images"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet_FusionTransformer()
    
    # Load trained model
    try:
        model.load_state_dict(torch.load('/usr/mvl2/cvrfr/KeyPonFuse_V1/Codes/models_22aug25/model_training/trained_models/UNet_native_resolution.pth', map_location=device))
        print("✅ Loaded trained model")
    except:
        print("⚠️ Using untrained model")
    
    model = model.to(device)
    model.eval()
    
    # Load images
    rgb_img = Image.open(rgb_path).convert('RGB')
    thermal_img = Image.open(thermal_path).convert('RGB')
    
    print(f"📊 Input RGB size: {rgb_img.size}")
    print(f"📊 Input Thermal size: {thermal_img.size}")
    
    # Transform (same as training)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    rgb_tensor = transform(rgb_img).unsqueeze(0).to(device)
    thermal_tensor = transform(thermal_img).unsqueeze(0).to(device)
    
    print(f"📊 RGB tensor shape: {rgb_tensor.shape}")
    print(f"📊 Thermal tensor shape: {thermal_tensor.shape}")
    
    # Inference
    with torch.no_grad():
        fused_output = model(rgb_tensor, thermal_tensor)
    
    print(f"📊 Output tensor shape: {fused_output.shape}")
    print(f"📊 Output size: {fused_output.shape[2]} x {fused_output.shape[3]}")
    
    # Convert back to image
    fused_output = fused_output.squeeze(0).cpu()
    fused_output = (fused_output + 1) / 2  # Denormalize from [-1,1] to [0,1]
    fused_output = torch.clamp(fused_output, 0, 1)
    
    # Convert to PIL Image
    to_pil = transforms.ToPILImage()
    fused_img = to_pil(fused_output)
    
    print(f"📊 Final output size: {fused_img.size}")
    
    # Save result
    fused_img.save(output_path)
    print(f"💾 Saved fused image to: {output_path}")

def explain_output_behavior():
    """Explain how the model handles different input sizes"""
    
    print("🔍 Model Output Size Behavior Explanation:")
    print("=" * 60)
    print()
    print("1. 📏 INPUT SIZE DETERMINES OUTPUT SIZE:")
    print("   • If input is 800x600 → output is 800x600")
    print("   • If input is 1024x768 → output is 1024x768")
    print("   • If input is 256x256 → output is 256x256")
    print()
    print("2. 🔄 NO FIXED OUTPUT SIZE:")
    print("   • Unlike models trained on fixed 256x256")
    print("   • This model adapts to any input size")
    print("   • Perfect for real-world applications")
    print()
    print("3. ⚡ MEMORY CONSIDERATIONS:")
    print("   • Larger inputs = more GPU memory needed")
    print("   • Very large images (>2048px) may cause OOM")
    print("   • Consider tiling for extremely large images")
    print()
    print("4. 🎯 PRACTICAL USAGE:")
    print("   • Load your actual images (any size)")
    print("   • Model outputs same size as input")
    print("   • No need to resize or crop")
    print("   • Perfect for batch processing different sizes")

if __name__ == "__main__":
    print("🚀 Testing Native Resolution Model Output Sizes")
    print("=" * 60)
    
    # Test with dummy data
    test_model_output_sizes()
    
    print("\n" + "=" * 60)
    
    # Explain behavior
    explain_output_behavior()
    
    print("\n" + "=" * 60)
    print("💡 To test with real images, use:")
    print("inference_with_real_images('path/to/rgb.jpg', 'path/to/thermal.jpg', 'output.jpg')")