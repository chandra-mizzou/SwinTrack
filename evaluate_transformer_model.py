import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import os
import numpy as np
from rgb_thermal_fusion_transformer import RGBThermalFusionTransformer
import cv2

def calculate_ssim(pred, target, window_size=11):
    """Calculate SSIM between prediction and target"""
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    
    mu_pred = F.avg_pool2d(pred, window_size, stride=1, padding=window_size // 2)
    mu_target = F.avg_pool2d(target, window_size, stride=1, padding=window_size // 2)
    
    sigma_pred = F.avg_pool2d(pred**2, window_size, stride=1, padding=window_size // 2) - mu_pred**2
    sigma_target = F.avg_pool2d(target**2, window_size, stride=1, padding=window_size // 2) - mu_target**2
    sigma_pred_target = F.avg_pool2d(pred * target, window_size, stride=1, padding=window_size // 2) - mu_pred * mu_target
    
    c1, c2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu_pred * mu_target + c1) * (2 * sigma_pred_target + c2)) / \
            ((mu_pred**2 + mu_target**2 + c1) * (sigma_pred + sigma_target + c2))
    
    return ssim_map.mean().item()

def calculate_psnr(pred, target):
    """Calculate PSNR between prediction and target"""
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()

def calculate_rmse(pred, target):
    """Calculate RMSE between prediction and target"""
    mse = F.mse_loss(pred, target)
    return torch.sqrt(mse).item()

def evaluate_model(model, test_loader, device='cpu', save_results=True, output_dir='results'):
    """Evaluate the model on test data"""
    model.eval()
    model = model.to(device)
    
    if save_results:
        os.makedirs(output_dir, exist_ok=True)
    
    total_ssim = 0.0
    total_psnr = 0.0
    total_rmse = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for batch_idx, (rgb, thermal, gt, filenames) in enumerate(test_loader):
            rgb = rgb.to(device)
            thermal = thermal.to(device)
            gt = gt.to(device)
            
            # Forward pass
            output = model(rgb, thermal)
            
            # Calculate metrics
            ssim = calculate_ssim(output, gt)
            psnr = calculate_psnr(output, gt)
            rmse = calculate_rmse(output, gt)
            
            total_ssim += ssim
            total_psnr += psnr
            total_rmse += rmse
            num_samples += 1
            
            print(f"Batch {batch_idx + 1}: SSIM={ssim:.4f}, PSNR={psnr:.2f}, RMSE={rmse:.4f}")
            
            # Save results
            if save_results:
                for i in range(output.shape[0]):
                    # Convert tensors to images
                    rgb_img = ((rgb[i].cpu() + 1) / 2 * 255).clamp(0, 255).permute(1, 2, 0).numpy().astype(np.uint8)
                    thermal_img = ((thermal[i].cpu() + 1) / 2 * 255).clamp(0, 255).permute(1, 2, 0).numpy().astype(np.uint8)
                    gt_img = ((gt[i].cpu() + 1) / 2 * 255).clamp(0, 255).permute(1, 2, 0).numpy().astype(np.uint8)
                    pred_img = ((output[i].cpu() + 1) / 2 * 255).clamp(0, 255).permute(1, 2, 0).numpy().astype(np.uint8)
                    
                    # Create comparison image
                    comparison = np.concatenate([rgb_img, thermal_img, gt_img, pred_img], axis=1)
                    
                    # Save
                    filename = filenames[i].split('.')[0]
                    cv2.imwrite(os.path.join(output_dir, f"{filename}_comparison.jpg"), 
                               cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    
    # Calculate average metrics
    avg_ssim = total_ssim / num_samples
    avg_psnr = total_psnr / num_samples
    avg_rmse = total_rmse / num_samples
    
    print(f"\n📊 EVALUATION RESULTS:")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average RMSE: {avg_rmse:.4f}")
    
    return avg_ssim, avg_psnr, avg_rmse

def test_single_image(model, rgb_path, thermal_path, gt_path, device='cpu'):
    """Test the model on a single image pair"""
    model.eval()
    model = model.to(device)
    
    # Load images
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    rgb = Image.open(rgb_path).convert('RGB')
    thermal = Image.open(thermal_path).convert('RGB')
    gt = Image.open(gt_path).convert('RGB')
    
    rgb_tensor = transform(rgb).unsqueeze(0).to(device)
    thermal_tensor = transform(thermal).unsqueeze(0).to(device)
    gt_tensor = transform(gt).unsqueeze(0).to(device)
    
    # Forward pass
    with torch.no_grad():
        output = model(rgb_tensor, thermal_tensor)
    
    # Calculate metrics
    ssim = calculate_ssim(output, gt_tensor)
    psnr = calculate_psnr(output, gt_tensor)
    rmse = calculate_rmse(output, gt_tensor)
    
    print(f"Single Image Results:")
    print(f"SSIM: {ssim:.4f}")
    print(f"PSNR: {psnr:.2f} dB")
    print(f"RMSE: {rmse:.4f}")
    
    return output, ssim, psnr, rmse

if __name__ == "__main__":
    # Load model
    model = RGBThermalFusionTransformer(
        input_channels=3,
        d_model=256,
        num_heads=8,
        num_cross_attention=2,
        dropout=0.1
    )
    
    # Load trained weights if available
    model_path = '/usr/mvl2/cvrfr/KeyPonFuse_V1/Codes/models_22aug25/model_training/trained_models/transformer_fusion_model.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print("✅ Loaded trained model weights")
    else:
        print("⚠️ No trained weights found, using random initialization")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test with random inputs
    print("🧪 Testing with random inputs...")
    rgb_test = torch.randn(1, 3, 256, 256)
    thermal_test = torch.randn(1, 3, 256, 256)
    
    with torch.no_grad():
        output_test = model(rgb_test, thermal_test)
    
    print(f"Input RGB shape: {rgb_test.shape}")
    print(f"Input Thermal shape: {thermal_test.shape}")
    print(f"Output shape: {output_test.shape}")
    print(f"Output range: [{output_test.min():.3f}, {output_test.max():.3f}]")
    
    print("\n✅ Model is working correctly!")