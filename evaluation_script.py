import os
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2
from improved_model import ImprovedUNet_FusionTransformer
from model_cross_self_attn_v1 import UNet_FusionTransformer

def calculate_metrics(pred, target):
    """Calculate SSIM, PSNR, and RMSE metrics"""
    # Convert to numpy arrays
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    # Ensure values are in [0, 1] range
    pred_np = np.clip(pred_np, 0, 1)
    target_np = np.clip(target_np, 0, 1)
    
    # Convert to uint8 for skimage metrics
    pred_uint8 = (pred_np * 255).astype(np.uint8)
    target_uint8 = (target_np * 255).astype(np.uint8)
    
    # Calculate metrics for each channel and average
    ssim_values = []
    psnr_values = []
    rmse_values = []
    
    for i in range(pred_np.shape[1]):  # For each channel
        pred_ch = pred_uint8[0, i, :, :]
        target_ch = target_uint8[0, i, :, :]
        
        # SSIM
        ssim_val = ssim(target_ch, pred_ch, data_range=255)
        ssim_values.append(ssim_val)
        
        # PSNR
        psnr_val = psnr(target_ch, pred_ch, data_range=255)
        psnr_values.append(psnr_val)
        
        # RMSE
        rmse_val = np.sqrt(np.mean((pred_np[0, i, :, :] - target_np[0, i, :, :]) ** 2))
        rmse_values.append(rmse_val)
    
    return {
        'SSIM': np.mean(ssim_values),
        'PSNR': np.mean(psnr_values),
        'RMSE': np.mean(rmse_values)
    }

def evaluate_model(model_path, test_folder_vi, test_folder_ir, test_folder_gt, 
                   output_folder, model_type='original'):
    """Evaluate model performance on test dataset"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    if model_type == 'improved':
        model = ImprovedUNet_FusionTransformer()
    else:
        model = UNet_FusionTransformer()
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Transform for evaluation
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Get test files
    test_files = sorted(os.listdir(test_folder_vi))
    
    all_metrics = []
    
    print(f"Evaluating {len(test_files)} test images...")
    
    with torch.no_grad():
        for i, filename in enumerate(test_files):
            try:
                # Load images
                vi_path = os.path.join(test_folder_vi, filename)
                ir_path = os.path.join(test_folder_ir, filename)
                gt_path = os.path.join(test_folder_gt, filename)
                
                vi_img = Image.open(vi_path).convert('RGB')
                ir_img = Image.open(ir_path).convert('RGB')
                gt_img = Image.open(gt_path).convert('RGB')
                
                # Transform
                vi_tensor = transform(vi_img).unsqueeze(0).to(device)
                ir_tensor = transform(ir_img).unsqueeze(0).to(device)
                gt_tensor = transform(gt_img).unsqueeze(0).to(device)
                
                # Forward pass
                fused_output = model(vi_tensor, ir_tensor)
                
                # Denormalize for evaluation
                fused_denorm = denormalize_tensor(fused_output)
                gt_denorm = denormalize_tensor(gt_tensor)
                
                # Calculate metrics
                metrics = calculate_metrics(fused_denorm, gt_denorm)
                all_metrics.append(metrics)
                
                # Save fused image
                save_fused_image(fused_denorm, filename, output_folder)
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(test_files)} images")
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
    
    # Calculate average metrics
    avg_metrics = {
        'SSIM': np.mean([m['SSIM'] for m in all_metrics]),
        'PSNR': np.mean([m['PSNR'] for m in all_metrics]),
        'RMSE': np.mean([m['RMSE'] for m in all_metrics])
    }
    
    print(f"\n=== Evaluation Results ===")
    print(f"Average SSIM: {avg_metrics['SSIM']:.4f}")
    print(f"Average PSNR: {avg_metrics['PSNR']:.4f}")
    print(f"Average RMSE: {avg_metrics['RMSE']:.4f}")
    
    return avg_metrics, all_metrics

def denormalize_tensor(tensor):
    """Denormalize tensor from ImageNet normalization"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(tensor.device)
    return torch.clamp(tensor * std + mean, 0, 1)

def save_fused_image(tensor, filename, output_folder):
    """Save fused image tensor to file"""
    # Convert tensor to PIL Image
    tensor_cpu = tensor.squeeze(0).cpu()
    tensor_cpu = torch.clamp(tensor_cpu, 0, 1)
    
    # Convert to PIL Image
    to_pil = transforms.ToPILImage()
    img = to_pil(tensor_cpu)
    
    # Save
    output_path = os.path.join(output_folder, f"fused_{filename}")
    img.save(output_path)

def compare_models():
    """Compare original and improved models"""
    
    # Define paths
    test_folder_vi = '/usr/mvl2/cvrfr/KeyPonFuse_V1/Codes/models_22aug25/training_datasets_full/training_dataset_20k/vi'
    test_folder_ir = '/usr/mvl2/cvrfr/KeyPonFuse_V1/Codes/models_22aug25/training_datasets_full/training_dataset_20k/ir'
    test_folder_gt = '/usr/mvl2/cvrfr/KeyPonFuse_V1/Codes/models_22aug25/model_outputs/SGT_DIrect_Fusion'
    
    # Model paths
    original_model_path = '/usr/mvl2/cvrfr/KeyPonFuse_V1/Codes/models_22aug25/model_training/trained_models/UNet_cross_self_attnTransformerSGTbase_fullres.pth'
    improved_model_path = '/usr/mvl2/cvrfr/KeyPonFuse_V1/Codes/models_22aug25/model_training/trained_models/UNet_cross_self_attnTransformerSGTbase_fullres_improved.pth'
    
    # Output folders
    original_output = '/usr/mvl2/cvrfr/KeyPonFuse_V1/Datasets/Training_Dataset_full/training_dataset_20k/test_ops_original'
    improved_output = '/usr/mvl2/cvrfr/KeyPonFuse_V1/Datasets/Training_Dataset_full/training_dataset_20k/test_ops_improved'
    
    os.makedirs(original_output, exist_ok=True)
    os.makedirs(improved_output, exist_ok=True)
    
    print("=== Evaluating Original Model ===")
    original_metrics, _ = evaluate_model(
        original_model_path, test_folder_vi, test_folder_ir, test_folder_gt,
        original_output, model_type='original'
    )
    
    print("\n=== Evaluating Improved Model ===")
    improved_metrics, _ = evaluate_model(
        improved_model_path, test_folder_vi, test_folder_ir, test_folder_gt,
        improved_output, model_type='improved'
    )
    
    print("\n=== Comparison Results ===")
    print(f"{'Metric':<10} {'Original':<12} {'Improved':<12} {'Improvement':<12}")
    print("-" * 50)
    
    for metric in ['SSIM', 'PSNR', 'RMSE']:
        orig_val = original_metrics[metric]
        impr_val = improved_metrics[metric]
        
        if metric == 'RMSE':
            improvement = ((orig_val - impr_val) / orig_val) * 100
            improvement_str = f"{improvement:+.2f}%"
        else:
            improvement = ((impr_val - orig_val) / orig_val) * 100
            improvement_str = f"{improvement:+.2f}%"
        
        print(f"{metric:<10} {orig_val:<12.4f} {impr_val:<12.4f} {improvement_str:<12}")

if __name__ == "__main__":
    compare_models()