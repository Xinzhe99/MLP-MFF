# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import glob
import os
import torch

def dataloader(filepath):
    # Support both .jpg and .png extensions
    train_sourceA_path = [os.path.join(filepath, 'train', 'sourceA', '*.jpg'),
                         os.path.join(filepath, 'train', 'sourceA', '*.png')]
    train_sourceB_path = [os.path.join(filepath, 'train', 'sourceB', '*.jpg'),
                         os.path.join(filepath, 'train', 'sourceB', '*.png')]
    test_sourceA_path = [os.path.join(filepath, 'test', 'sourceA', '*.jpg'),
                        os.path.join(filepath, 'test', 'sourceA', '*.png')]
    test_sourceB_path = [os.path.join(filepath, 'test', 'sourceB', '*.jpg'),
                        os.path.join(filepath, 'test', 'sourceB', '*.png')]
    train_groundtruth_path = [os.path.join(filepath, 'train', 'groundtruth', '*.jpg'),
                             os.path.join(filepath, 'train', 'groundtruth', '*.png')]
    test_groundtruth_path = [os.path.join(filepath, 'test', 'groundtruth', '*.jpg'),
                            os.path.join(filepath, 'test', 'groundtruth', '*.png')]

    # Collect all files with both extensions
    train_left_img = []
    for path in train_sourceA_path:
        train_left_img.extend(glob.glob(path))
    
    train_right_img = []
    for path in train_sourceB_path:
        train_right_img.extend(glob.glob(path))
    
    test_left_img = []
    for path in test_sourceA_path:
        test_left_img.extend(glob.glob(path))
    
    test_right_img = []
    for path in test_sourceB_path:
        test_right_img.extend(glob.glob(path))
    
    label_train_img = []
    for path in train_groundtruth_path:
        label_train_img.extend(glob.glob(path))
    
    label_val_img = []
    for path in test_groundtruth_path:
        label_val_img.extend(glob.glob(path))

    # Sort all lists to ensure consistent ordering
    train_left_img.sort()
    train_right_img.sort()
    test_left_img.sort()
    test_right_img.sort()
    label_train_img.sort()
    label_val_img.sort()

    return train_left_img, train_right_img, test_left_img, test_right_img, label_train_img, label_val_img

def default_loader(path):
    return Image.open(path).convert('YCbCr').split()[0]  # Return PIL Image instead of numpy array

def get_transform(augment=False, input_size=256):
    transforms_list = []
    
    if augment:
        # 几何变换
        transforms_list.extend([
            transforms.RandomHorizontalFlip(p=0.33),
            transforms.RandomVerticalFlip(p=0.33),
            transforms.RandomRotation(degrees=(-10, 10)),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ])
    
    # 最终变换（确保固定尺寸）
    transforms_list.extend([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ])
    
    return transforms.Compose(transforms_list)

class myImageFloder(data.Dataset):
    def __init__(self, left, right, label_img, augment, input_size, loader=default_loader):
        self.left = left
        self.right = right
        self.label_img = label_img
        self.augment = augment
        self.loader = loader
        self.input_size = input_size
        
    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]
        label_img = self.label_img[index]
        left_img = self.loader(left)
        right_img = self.loader(right)
        label_img = self.loader(label_img)
        
        if self.augment:
            # Generate a new random seed for this sample
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
            transform = get_transform(augment=True, input_size=self.input_size)
            
            # 应用相同的变换到所有图像
            torch.manual_seed(seed)
            left_img = transform(left_img)
            torch.manual_seed(seed)
            right_img = transform(right_img)
            torch.manual_seed(seed)
            label_img = transform(label_img)
        else:
            transform = get_transform(augment=False, input_size=self.input_size)
            left_img = transform(left_img)
            right_img = transform(right_img)
            label_img = transform(label_img)

        return left_img, right_img, label_img

    def __len__(self):
        return len(self.left)