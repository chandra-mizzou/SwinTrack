import os
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

def analyze_dataset_quality(folder1, folder2, folder3, num_samples=10):
    """Analyze the quality of your dataset"""
    
    print("🔍 Dataset Quality Analysis")
    print("=" * 50)
    
    # Get sample files
    filenames = sorted(os.listdir(folder1))[:num_samples]
    
    # Transform for analysis
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    rgb_values = []
    thermal_values = []
    gt_values = []
    
    print(f"📊 Analyzing {len(filenames)} sample images...")
    
    for i, filename in enumerate(filenames):
        try:
            # Load images
            rgb_path = os.path.join(folder1, filename)
            thermal_path = os.path.join(folder2, filename)
            gt_path = os.path.join(folder3, filename)
            
            rgb_img = Image.open(rgb_path).convert('RGB')
            thermal_img = Image.open(thermal_path).convert('RGB')
            gt_img = Image.open(gt_path).convert('RGB')
            
            # Transform
            rgb_tensor = transform(rgb_img)
            thermal_tensor = transform(thermal_img)
            gt_tensor = transform(gt_img)
            
            # Calculate statistics
            rgb_mean = rgb_tensor.mean().item()
            thermal_mean = thermal_tensor.mean().item()
            gt_mean = gt_tensor.mean().item()
            
            rgb_values.append(rgb_mean)
            thermal_values.append(thermal_mean)
            gt_values.append(gt_mean)
            
            print(f"Sample {i+1}: RGB={rgb_mean:.3f}, Thermal={thermal_mean:.3f}, GT={gt_mean:.3f}")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    # Calculate overall statistics
    print(f"\n📊 Overall Statistics:")
    print(f"RGB mean: {np.mean(rgb_values):.3f} ± {np.std(rgb_values):.3f}")
    print(f"Thermal mean: {np.mean(thermal_values):.3f} ± {np.std(thermal_values):.3f}")
    print(f"GT mean: {np.mean(gt_values):.3f} ± {np.std(gt_values):.3f}")
    
    # Check if GT is just average of inputs
    rgb_thermal_avg = [(r + t) / 2 for r, t in zip(rgb_values, thermal_values)]
    gt_correlation = np.corrcoef(gt_values, rgb_thermal_avg)[0, 1]
    
    print(f"\n🔍 Ground Truth Analysis:")
    print(f"GT vs (RGB+Thermal)/2 correlation: {gt_correlation:.3f}")
    
    if gt_correlation > 0.95:
        print("❌ WARNING: GT appears to be just average of inputs!")
        print("   This explains why the model isn't learning.")
    elif gt_correlation > 0.8:
        print("⚠️  WARNING: GT is highly correlated with input average")
        print("   This may limit learning effectiveness.")
    else:
        print("✅ GT appears to be independent of input average")
    
    return rgb_values, thermal_values, gt_values

def test_simple_fusion_loss():
    """Test if the loss function is working correctly"""
    
    print(f"\n🧪 Testing Loss Function:")
    print("=" * 30)
    
    # Create dummy data
    batch_size = 2
    height, width = 256, 256
    
    # Create synthetic data
    rgb = torch.randn(batch_size, 3, height, width) * 0.5  # [-1, 1] range
    thermal = torch.randn(batch_size, 3, height, width) * 0.5
    gt = torch.randn(batch_size, 3, height, width) * 0.5
    
    # Test different scenarios
    scenarios = [
        ("Random inputs", rgb, thermal, gt),
        ("Identical inputs", rgb, rgb, rgb),
        ("GT = RGB", rgb, thermal, rgb),
        ("GT = Thermal", rgb, thermal, thermal),
        ("GT = Average", rgb, thermal, (rgb + thermal) / 2),
    ]
    
    for name, r, t, g in scenarios:
        # Calculate basic losses
        l1_loss = F.l1_loss(r, g).item()
        mse_loss = F.mse_loss(r, g).item()
        
        print(f"{name:15s}: L1={l1_loss:.3f}, MSE={mse_loss:.3f}")
    
    print(f"\n💡 Expected L1 loss for random data: ~0.67")
    print(f"💡 Your training loss of 2.8+ suggests serious issues!")

def suggest_improvements():
    """Suggest improvements for training"""
    
    print(f"\n💡 Suggested Improvements:")
    print("=" * 40)
    
    print("1. 🎯 GROUND TRUTH ISSUES:")
    print("   • Your GT may still be averaged inputs")
    print("   • Try using proper fusion algorithms as GT")
    print("   • Or use unsupervised/self-supervised approaches")
    
    print("\n2. 🔧 LOSS FUNCTION ISSUES:")
    print("   • Current loss weights may be wrong")
    print("   • Try simpler loss functions first")
    print("   • Use only L1 + SSIM initially")
    
    print("\n3. 📊 TRAINING STRATEGY:")
    print("   • Start with smaller images (256x256)")
    print("   • Use higher learning rate initially")
    print("   • Try different optimizers (Adam vs AdamW)")
    
    print("\n4. 🎨 DATA PREPROCESSING:")
    print("   • Check if normalization is correct")
    print("   • Verify image loading is working")
    print("   • Try different augmentation strategies")

if __name__ == "__main__":
    # Define paths
    folder1 = '/usr/mvl2/cvrfr/KeyPonFuse_V1/Codes/models_22aug25/training_datasets_full/training_dataset_20k/vi'
    folder2 = '/usr/mvl2/cvrfr/KeyPonFuse_V1/Codes/models_22aug25/training_datasets_full/training_dataset_20k/ir'
    folder3 = '/usr/mvl2/cvrfr/KeyPonFuse_V1/Codes/models_22aug25/model_outputs/SGT_DIrect_Fusion'
    
    # Analyze dataset
    rgb_vals, thermal_vals, gt_vals = analyze_dataset_quality(folder1, folder2, folder3)
    
    # Test loss function
    test_simple_fusion_loss()
    
    # Suggest improvements
    suggest_improvements()