"""
Dataset classes for document scanning and enhancement.
Handles loading images and annotations from COCO JSON format.
"""

import os
import json
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .degradation import DegradationPipeline, create_degradation_pipeline


class DocumentScanningDataset(Dataset):
    """
    PyTorch Dataset for document scanning with corner detection labels.
    Supports COCO JSON annotation format from Roboflow.
    """
    
    def __init__(self, 
                 root_dir: str,
                 annotation_file: str,
                 image_size: Tuple[int, int] = (512, 512),
                 transform: Optional[Callable] = None,
                 use_degradation: bool = False,
                 degradation_prob: float = 0.5,
                 seed: Optional[int] = None):
        """
        Initialize the dataset.
        
        Args:
            root_dir: Root directory containing images
            annotation_file: Path to COCO JSON annotation file
            image_size: Target image size (height, width)
            transform: Optional Albumentations transform
            use_degradation: Whether to apply online degradations
            degradation_prob: Probability of applying degradation
            seed: Random seed for reproducibility
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.transform = transform
        self.use_degradation = use_degradation
        self.degradation_prob = degradation_prob
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Build image info dictionary
        self.images = {img['id']: img for img in self.annotations['images']}
        self.image_ids = list(self.images.keys())
        
        # Build annotation index by image_id
        self.annotations_by_image = {}
        for ann in self.annotations['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations_by_image:
                self.annotations_by_image[img_id] = []
            self.annotations_by_image[img_id].append(ann)
        
        # Initialize degradation pipeline if needed
        self.degradation_pipeline = None
        if use_degradation:
            self.degradation_pipeline = create_degradation_pipeline(seed=seed)
        
        # Set seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
    
    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.image_ids)
    
    def _load_image(self, image_id: int) -> np.ndarray:
        """
        Load an image from disk.
        
        Args:
            image_id: ID of the image to load
            
        Returns:
            Loaded image as numpy array (H, W, C) in BGR format
        """
        img_info = self.images[image_id]
        filename = img_info['file_name']
        filepath = os.path.join(self.root_dir, filename)
        
        image = cv2.imread(filepath)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {filepath}")
        
        return image
    
    def _extract_corners(self, annotations: List[Dict]) -> np.ndarray:
        """
        Extract corner keypoints from annotations.
        
        Args:
            annotations: List of annotation dictionaries
            
        Returns:
            Array of shape (4, 2) with corner coordinates [x, y]
        """
        corners = np.zeros((4, 2), dtype=np.float32)
        
        for ann in annotations:
            if 'keypoints' in ann and len(ann['keypoints']) >= 8:
                keypoints = ann['keypoints']
                # Extract 4 corners: [x0, y0, x1, y1, x2, y2, x3, y3]
                for i in range(4):
                    corners[i, 0] = keypoints[i * 3]  # x coordinate
                    corners[i, 1] = keypoints[i * 3 + 1]  # y coordinate
                break
        
        return corners
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing:
                - 'image': Input image tensor
                - 'corners': Ground truth corner coordinates (4, 2)
                - 'image_id': Image ID
                - 'original_size': Original image size (H, W)
                - 'degraded': Whether degradation was applied (bool)
        """
        image_id = self.image_ids[idx]
        
        # Load image
        image = self._load_image(image_id)
        original_h, original_w = image.shape[:2]
        
        # Get annotations
        annotations = self.annotations_by_image.get(image_id, [])
        corners = self._extract_corners(annotations)
        
        # Scale corners to target size
        scale_x = self.image_size[1] / original_w
        scale_y = self.image_size[0] / original_h
        corners_scaled = corners.copy()
        corners_scaled[:, 0] *= scale_x
        corners_scaled[:, 1] *= scale_y
        
        degraded = False
        
        # Apply degradation if enabled
        if self.use_degradation and self.degradation_pipeline is not None:
            if np.random.rand() < self.degradation_prob:
                image = self.degradation_pipeline.apply_random_degradation(image)
                degraded = True
        
        # Resize image
        image = cv2.resize(image, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_LINEAR)
        
        # Apply transforms if provided
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        # Convert to tensor if not already done by transform
        if not isinstance(image, np.ndarray):
            pass  # Already a tensor
        else:
            image = image.astype(np.float32) / 255.0
            image = np.transpose(image, (2, 0, 1))  # HWC to CHW
            image = np.ascontiguousarray(image)
        
        return {
            'image': image,
            'corners': corners_scaled,
            'image_id': image_id,
            'original_size': (original_h, original_w),
            'degraded': degraded
        }


class DocumentEnhancementDataset(Dataset):
    """
    PyTorch Dataset for document enhancement task.
    Creates pairs of degraded and clean images for training enhancement networks.
    """
    
    def __init__(self,
                 root_dir: str,
                 annotation_file: str,
                 image_size: Tuple[int, int] = (512, 512),
                 transform: Optional[Callable] = None,
                 degradation_types: Optional[List[str]] = None,
                 seed: Optional[int] = None):
        """
        Initialize the enhancement dataset.
        
        Args:
            root_dir: Root directory containing images
            annotation_file: Path to COCO JSON annotation file
            image_size: Target image size (height, width)
            transform: Optional Albumentations transform
            degradation_types: List of degradation types to apply
            seed: Random seed for reproducibility
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.transform = transform
        self.degradation_types = degradation_types
        
        # Load base dataset
        self.base_dataset = DocumentScanningDataset(
            root_dir=root_dir,
            annotation_file=annotation_file,
            image_size=image_size,
            use_degradation=False,
            seed=seed
        )
        
        # Initialize degradation pipeline
        self.degradation_pipeline = create_degradation_pipeline(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
    
    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing:
                - 'degraded_image': Degraded input image tensor
                - 'clean_image': Clean target image tensor
                - 'corners': Ground truth corner coordinates
                - 'image_id': Image ID
        """
        # Get clean image
        sample = self.base_dataset[idx]
        clean_image = sample['image'].copy()
        
        # Create degraded version
        if isinstance(sample['image'], np.ndarray):
            # Convert from CHW to HWC and scale to 0-255
            image_hwc = np.transpose(sample['image'], (1, 2, 0))
            image_uint8 = (image_hwc * 255).astype(np.uint8)
            
            # Apply specific degradation or random
            if self.degradation_types:
                degradation_func = getattr(
                    self.degradation_pipeline,
                    f'apply_{self.degradation_types[np.random.randint(len(self.degradation_types))]}'
                )
                degraded_image = degradation_func(image_uint8)
            else:
                degraded_image = self.degradation_pipeline.apply_random_degradation(image_uint8)
            
            # Convert back to CHW float [0, 1]
            degraded_image = degraded_image.astype(np.float32) / 255.0
            degraded_image = np.transpose(degraded_image, (2, 0, 1))
            degraded_image = np.ascontiguousarray(degraded_image)
        else:
            # If already tensor, apply degradation differently
            degraded_image = clean_image
        
        return {
            'degraded_image': degraded_image,
            'clean_image': clean_image,
            'corners': sample['corners'],
            'image_id': sample['image_id']
        }


def get_train_transform(image_size: Tuple[int, int] = (512, 512)) -> Callable:
    """
    Create training data augmentation pipeline.
    
    Args:
        image_size: Target image size
        
    Returns:
        Albumentations composition
    """
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10),
        ], p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_val_transform(image_size: Tuple[int, int] = (512, 512)) -> Callable:
    """
    Create validation data transformation pipeline.
    
    Args:
        image_size: Target image size
        
    Returns:
        Albumentations composition
    """
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def create_dataloaders(root_dir: str,
                       annotation_file: str,
                       batch_size: int = 16,
                       image_size: Tuple[int, int] = (512, 512),
                       train_split: float = 0.8,
                       use_degradation: bool = True,
                       num_workers: int = 4,
                       seed: int = 42) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        root_dir: Root directory containing images
        annotation_file: Path to COCO JSON annotation file
        batch_size: Batch size
        image_size: Target image size
        train_split: Fraction of data for training
        use_degradation: Whether to apply online degradations
        num_workers: Number of data loading workers
        seed: Random seed
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Set seeds
    np.random.seed(seed)
    
    # Create datasets
    train_dataset = DocumentScanningDataset(
        root_dir=root_dir,
        annotation_file=annotation_file,
        image_size=image_size,
        transform=get_train_transform(image_size),
        use_degradation=use_degradation,
        degradation_prob=0.7,
        seed=seed
    )
    
    val_dataset = DocumentScanningDataset(
        root_dir=root_dir,
        annotation_file=annotation_file,
        image_size=image_size,
        transform=get_val_transform(image_size),
        use_degradation=False,
        seed=seed
    )
    
    # Split data
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(train_split * dataset_size))
    
    np.random.shuffle(indices)
    train_indices = indices[:split]
    val_indices = indices[split:]
    
    # Create subset datasets
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


# Import torch at module level for Subset
import torch
