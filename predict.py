# -*- coding: utf-8 -*-
import torch
import argparse
from pathlib import Path
from models.network import MLP_MFF
from PIL import Image
import cv2
import os
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms
import math

def parse_args():
    """Argument parsing"""
    parser = argparse.ArgumentParser(description='MLP-MFF inference script')
    parser.add_argument('--model_path', 
                        default=r'model.pth',
                        help='Path to the trained model weights file (.pth file)',
                        type=str)
    parser.add_argument('--input_dir', 
                        default=r'path\to\dataset',
                        help='Input data directory, should contain two subfolders A and B',
                        type=str)
    parser.add_argument('--output_dir', 
                        default=r'path\to\save',
                        help='Output directory to save results',
                        type=str)
    parser.add_argument('--force_size', 
                        default=None,
                        help='Force resize input images, format: width,height (e.g., 512,512). If not specified, automatically adjust to multiples of 32',
                        type=str)
    parser.add_argument('--device', 
                        default='auto',
                        help='Device to run on: auto/cuda/cpu',
                        choices=['auto', 'cuda', 'cpu'])
    return parser.parse_args()

def fuse_RGB_channels(img_A_path, img_B_path, fused_Y_path, save_path):
    """Fuse RGB channels using only CV2"""
    # Read source images
    img_A = cv2.imread(img_A_path)  # BGR format
    img_B = cv2.imread(img_B_path)  # BGR format

    # Record original size
    original_h, original_w = img_A.shape[:2]

    # Convert to YCrCb color space
    ycrcb_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2YCrCb)
    ycrcb_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2YCrCb)

    # Split channels
    _, Cr1, Cb1 = cv2.split(ycrcb_A)  # Note: CV2 YCrCb order is Y, Cr, Cb
    _, Cr2, Cb2 = cv2.split(ycrcb_B)

    # Ensure all channels have the same size
    # Read the fused Y channel from path (assume it is a grayscale image file)
    fused_Y = cv2.imread(fused_Y_path, cv2.IMREAD_GRAYSCALE)
    if fused_Y is None:
        raise ValueError(f"Cannot read fused Y channel image from path '{fused_Y_path}'")
    if fused_Y.shape[:2] != (original_h, original_w):
        fused_Y = cv2.resize(fused_Y, (original_w, original_h))

    # Fuse Cr and Cb channels (using weighted fusion)
    weights_Cr1 = np.abs(Cr1.astype(np.float32) - 128)
    weights_Cr2 = np.abs(Cr2.astype(np.float32) - 128)
    weights_sum_Cr = weights_Cr1 + weights_Cr2

    weights_Cb1 = np.abs(Cb1.astype(np.float32) - 128)
    weights_Cb2 = np.abs(Cb2.astype(np.float32) - 128)
    weights_sum_Cb = weights_Cb1 + weights_Cb2

    # Avoid division by zero
    Cr = np.where(weights_sum_Cr == 0, 128,
                  (Cr1.astype(np.float32) * weights_Cr1 + Cr2.astype(np.float32) * weights_Cr2) / weights_sum_Cr)
    Cb = np.where(weights_sum_Cb == 0, 128,
                  (Cb1.astype(np.float32) * weights_Cb1 + Cb2.astype(np.float32) * weights_Cb2) / weights_sum_Cb)

    # Merge channels
    fused_ycrcb = cv2.merge([fused_Y, np.uint8(Cr), np.uint8(Cb)])

    # Convert back to BGR color space
    fused_bgr = cv2.cvtColor(fused_ycrcb, cv2.COLOR_YCrCb2BGR)

    # Save result
    cv2.imwrite(save_path, fused_bgr)

def load_model(model_path, device):
    """Load trained model"""
    print(f"Loading model: {model_path}")
    
    # Initialize model
    model = MLP_MFF()
    
    # Load weights
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise

def inference_on_dataset(model, input_dir, output_dir, device, force_size=None):
    """
    Run inference on dataset

    Args:
        model: Trained PyTorch model
        input_dir: Input data directory, contains subfolders A and B
        output_dir: Output directory to save results
        device: Device to run on
        force_size: Force resize (width, height) or None
    """
    transform = transforms.Compose([transforms.ToTensor()])

    # Check input directory structure
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Get A and B subdirectories
    sub_dirs = sorted([d for d in input_path.iterdir() if d.is_dir()])
    if len(sub_dirs) < 2:
        raise ValueError(f"Input directory should contain at least two subdirectories, currently only: {[d.name for d in sub_dirs]}")
    
    dir_A = sub_dirs[0]
    dir_B = sub_dirs[1]
    print(f"Processing directories: A={dir_A.name}, B={dir_B.name}")

    # Get image file lists
    img_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    img_files_A = sorted([f for f in dir_A.iterdir() 
                         if f.is_file() and f.suffix.lower() in img_extensions])
    
    if not img_files_A:
        raise ValueError(f"No image files found in directory {dir_A}")

    # Create output directories
    output_path = Path(output_dir)
    y_output_dir = output_path / 'y'
    rgb_output_dir = output_path / 'rgb'
    y_output_dir.mkdir(parents=True, exist_ok=True)
    rgb_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directories:")
    print(f"  Y channel: {y_output_dir}")
    print(f"  RGB result: {rgb_output_dir}")

    successful_count = 0
    failed_count = 0

    with torch.no_grad():
        for img_file_A in tqdm(img_files_A, desc='Inference progress'):
            # Find corresponding B image
            img_file_B = dir_B / img_file_A.name
            
            if not img_file_B.exists():
                print(f"Warning: Corresponding image {img_file_A.name} not found in directory B, skipping")
                failed_count += 1
                continue

            try:
                # Read images and convert to YCbCr color space
                img_A_pil = Image.open(img_file_A).convert('YCbCr')
                img_B_pil = Image.open(img_file_B).convert('YCbCr')

                # Extract Y channel
                img_A_Y = img_A_pil.split()[0]
                img_B_Y = img_B_pil.split()[0]

                # Record original size
                original_w, original_h = img_A_Y.size

                # Determine target size
                if force_size:
                    target_w, target_h = force_size
                else:
                    # Adjust to multiples of 32
                    target_w = math.ceil(original_w / 32) * 32
                    target_h = math.ceil(original_h / 32) * 32

                # Resize Y channel images
                img_A_Y_resized = img_A_Y.resize((target_w, target_h), Image.Resampling.LANCZOS)
                img_B_Y_resized = img_B_Y.resize((target_w, target_h), Image.Resampling.LANCZOS)

                # Convert to tensor
                img_A_tensor = transform(img_A_Y_resized).unsqueeze(0).to(device)
                img_B_tensor = transform(img_B_Y_resized).unsqueeze(0).to(device)

                # Model inference
                fused_output = model(img_A_tensor, img_B_tensor)

                # Process output
                fused_np = fused_output.squeeze().cpu().numpy()
                
                # Normalize to 0-255
                min_val, max_val = np.min(fused_np), np.max(fused_np)
                if max_val > min_val:
                    fused_np_normalized = (fused_np - min_val) / (max_val - min_val)
                else:
                    fused_np_normalized = np.zeros_like(fused_np)
                
                fused_np_scaled = np.clip(fused_np_normalized * 255, 0, 255).astype(np.uint8)
                fused_pil = Image.fromarray(fused_np_scaled, mode='L')
                
                # Resize back to original size
                fused_pil_resized = fused_pil.resize((original_w, original_h), Image.Resampling.LANCZOS)

                # Save Y channel result
                y_save_path = y_output_dir / f"{img_file_A.stem}.png"
                fused_pil_resized.save(y_save_path)

                # Fuse RGB channels and save
                rgb_save_path = rgb_output_dir / f"{img_file_A.stem}.png"
                fuse_RGB_channels(str(img_file_A), str(img_file_B), str(y_save_path), str(rgb_save_path))

                successful_count += 1

            except Exception as e:
                print(f"Error processing image {img_file_A.name}: {e}")
                failed_count += 1

    print(f"\nInference completed!")
    print(f"Successfully processed: {successful_count} images")
    print(f"Failed: {failed_count} images")
    print(f"Results saved in: {output_dir}")

def main():
    """Main function"""
    args = parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")

    # Parse force_size argument
    force_size = None
    if args.force_size:
        try:
            width, height = map(int, args.force_size.split(','))
            force_size = (width, height)
            print(f"Force resizing images to: {force_size}")
        except ValueError:
            print("Error: force_size format should be 'width,height', e.g. '512,512'")
            return

    try:
        # Load model
        model = load_model(args.model_path, device)
        
        # Run inference
        inference_on_dataset(
            model=model,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            device=device,
            force_size=force_size
        )
        
    except Exception as e:
        print(f"Error during inference: {e}")
        return

if __name__ == '__main__':
    main()