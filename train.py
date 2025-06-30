# -*- coding: utf-8 -*-
import torch
import argparse
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tools.config_dir import config_dir
from Dataloader import myImageFloder, dataloader
from utils import test_on_mff_dataset
import os
from tqdm import tqdm
from models.network import MLP_MFF
from PIL import Image
import cv2
import numpy as np
import torchvision.transforms as transforms
import math
import torch.nn as nn

def parse_args():
    """Argument parsing"""
    parser = argparse.ArgumentParser(description='MLP-MFF')
    parser.add_argument('--save_name', default='train_runs_mff', help='Name of the save folder')
    parser.add_argument('--task', default='mff', help='mff,mef,nir,med')
    parser.add_argument('--mff_datapath',
                        default=Path(
                            '/path/to/train/dataset'),
                        help='Path to multi-focus fusion training dataset',
                        type=Path)
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--input_size', type=int, default=256, help='Input image size')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--model_save_freq', type=int, default=1, help='Model save frequency')
    parser.add_argument('--resume', default=False, help='Resume training or not')
    return parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def train_step(model, batch_data, device, criterion, task):
    """Single training step"""
    if task =='mff':
        imgL, imgR, img_label = [x.to(device) for x in batch_data]
        output = model(imgL, imgR)
        # Calculate loss
        loss = criterion(output, img_label)
        return loss, output

def validate_step(model, batch_data, device, criterion, task):
    """Single validation step"""
    if task == 'mff':
        with torch.no_grad():
            return train_step(model, batch_data, device, criterion, task)

def save_model(model, save_path):
    """Save model parameters"""
    # If training with multiple GPUs, save the state_dict from module
    state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save(state_dict, save_path)

def train_epoch(model, train_loader, optimizer, device, criterion, epoch, writer, task):
    """Train for one epoch"""
    model.train()
    total_loss = 0

    # Modify tqdm progress bar to show loss value
    pbar = tqdm(train_loader, desc=f'Training Epoch {epoch}')
    for batch_idx, batch_data in enumerate(pbar):
        loss, _ = train_step(model, batch_data, device, criterion, task)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Calculate and display current average loss
        current_avg_loss = total_loss / (batch_idx + 1)

        # Update progress bar info, show current batch loss and average loss
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{current_avg_loss:.4f}'
        })

    # Calculate and record average loss for the whole epoch
    avg_loss = total_loss / len(train_loader)
    writer.add_scalar('Train Loss', avg_loss, epoch)
    
    return avg_loss

def validate(model, val_loader, device, criterion, epoch, writer, model_save_path, task):
    """Validate the model"""
    model.eval()
    total_loss = 0

    for batch_data in tqdm(val_loader, desc=f'Validation Epoch {epoch}'):
        loss, _ = validate_step(model, batch_data, device, criterion, task)
        total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    writer.add_scalar('Validation Loss', avg_loss, epoch)
    
    # Write validation results to txt file
    with open(os.path.join(model_save_path,'validation_results.txt'), 'a') as f:
        f.write(f'Epoch {epoch}: Validation Loss = {avg_loss:.4f}\n')
    
    return avg_loss

def main():
    """Main function"""
    # Parse arguments and set random seed
    args = parse_args()
    set_seed(args.seed)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if args.task=='mff':
        # Load data
        train_left_img, train_right_img, test_left_img, test_right_img, \
            label_train_fusion_img, label_val_fusion_img = dataloader(args.mff_datapath)

    # Create data loaders
    train_loader = DataLoader(
        myImageFloder(train_left_img, train_right_img, label_train_fusion_img, augment=True, input_size=args.input_size),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        myImageFloder(test_left_img, test_right_img, label_val_fusion_img, augment=False, input_size=args.input_size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Initialize model
    model = MLP_MFF().to(device)

    # Set optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0001
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)

    def define_loss(task, device):
        if task == 'mff':
            from loss.loss_mff_with_lable import End2End_loss
            loss_fn = End2End_loss().to(device)
        return loss_fn

    # Set loss function
    criterion = define_loss(args.task, device)

    # Set up logging
    model_save_path = config_dir(resume=args.resume,subdir_name=args.save_name)
    writer = SummaryWriter(log_dir=model_save_path)

    # Training loop
    best_loss = float('inf')
    
    # Track best SSIM value for each dataset
    best_ssim_values = {
        'Lytro': {'value': 0.0, 'epoch': -1},
        'MFFW': {'value': 0.0, 'epoch': -1},
        'MFI-WHU': {'value': 0.0, 'epoch': -1}
    }

    for epoch in range(args.epochs):
        # Training phase
        train_loss = train_epoch(model, train_loader, optimizer, device, criterion, epoch, writer, args.task)
        print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, LR = {scheduler.get_last_lr()[0]:.6f}')

        # Validation phase
        val_loss = validate(model, val_loader, device, criterion, epoch, writer, model_save_path, args.task)
        print(f'Epoch {epoch}: Validation Loss = {val_loss:.4f}')

        # Test before saving model
        if (epoch + 1) % args.model_save_freq == 0:
            # Save model
            save_path = os.path.join(model_save_path, f'model_epoch_{epoch}.pth')
            save_model(model, save_path)
            # If best validation loss, save an extra copy
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_path = Path(model_save_path) / 'best_model.pth'
                save_model(model, best_model_path)
                print(f'Saved best validation loss model, Validation Loss: {val_loss:.4f}')

        # Update learning rate
        scheduler.step()

    writer.close()
    print('Training finished!')


if __name__ == '__main__':
    main()