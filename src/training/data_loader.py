# src/training/data_loader.py
import os
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DefectDataset(Dataset):
    """Custom dataset for defect detection"""
    
    def __init__(self, data_path: Path, split: str = "train", transform=None):
        self.data_path = Path(data_path)
        self.split = split
        self.transform = transform
        self.images = self._load_image_paths()
        
    def _load_image_paths(self) -> List[Tuple[str, int]]:
        """Load image paths and labels"""
        images = []
        split_path = self.data_path / self.split
        
        # Load normal images (label 0)
        good_path = split_path / "good"
        if good_path.exists():
            for img_path in good_path.glob("*.jpg"):
                images.append((str(img_path), 0))
            for img_path in good_path.glob("*.png"):
                images.append((str(img_path), 0))
                
        # Load defective images (label 1) - only for test/val splits
        if self.split in ["test", "val"]:
            defective_path = split_path / "defective"
            if defective_path.exists():
                for img_path in defective_path.glob("*.jpg"):
                    images.append((str(img_path), 1))
                for img_path in defective_path.glob("*.png"):
                    images.append((str(img_path), 1))
                    
        return images
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
            
        return {
            "image": image,
            "label": label,
            "image_path": img_path
        }

def get_train_transform(image_size: Tuple[int, int]):
    """Get training data transformations"""
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(p=0.2),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

def get_test_transform(image_size: Tuple[int, int]):
    """Get test data transformations"""
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])