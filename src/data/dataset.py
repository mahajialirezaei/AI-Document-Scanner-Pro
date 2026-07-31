import os
import json
import cv2
import numpy as np
import torch
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from torch.utils.data import Dataset, DataLoader
from .degradation import create_degradation_pipeline

class SyntheticDocumentDataset(Dataset):
    def __init__(self, 
                 clean_scans_paths: List[str], 
                 backgrounds_dir: str, 
                 image_size: Tuple[int, int] = (512, 512),
                 use_degradation: bool = True,
                 seed: Optional[int] = None):
        
        self.clean_scans = clean_scans_paths
        self.backgrounds = list(Path(backgrounds_dir).glob("*.jpg"))
        self.image_size = image_size
        self.use_degradation = use_degradation
        
        self.degradation_pipeline = create_degradation_pipeline(seed=seed) if use_degradation else None
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def __len__(self) -> int:
        return len(self.clean_scans)

    def _generate_random_corners(self, bg_w: int, bg_h: int) -> np.ndarray:
        margin_x, margin_y = bg_w // 5, bg_h // 5
        
        tl = [random.randint(0, margin_x), random.randint(0, margin_y)]
        tr = [random.randint(bg_w - margin_x, bg_w), random.randint(0, margin_y)]
        br = [random.randint(bg_w - margin_x, bg_w), random.randint(bg_h - margin_y, bg_h)]
        bl = [random.randint(0, margin_x), random.randint(bg_h - margin_y, bg_h)]
        
        return np.float32([tl, tr, br, bl])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        scan_path = self.clean_scans[idx]
        bg_path = random.choice(self.backgrounds)
        
        clean_scan = cv2.imread(str(scan_path))
        bg_image = cv2.imread(str(bg_path))
        
        bg_image = cv2.resize(bg_image, (self.image_size[1], self.image_size[0]))
        clean_scan = cv2.resize(clean_scan, (self.image_size[1], self.image_size[0]))
        scan_h, scan_w = clean_scan.shape[:2]
        bg_h, bg_w = bg_image.shape[:2]

        src_pts = np.float32([[0, 0], [scan_w, 0], [scan_w, scan_h], [0, scan_h]])
        dst_pts = self._generate_random_corners(bg_w, bg_h)
        
        H = cv2.getPerspectiveTransform(src_pts, dst_pts)

        warped_scan = cv2.warpPerspective(clean_scan, H, (bg_w, bg_h))
        
        mask = np.ones((scan_h, scan_w), dtype=np.uint8) * 255
        warped_mask = cv2.warpPerspective(mask, H, (bg_w, bg_h))
        
        composite = bg_image.copy()
        composite[warped_mask == 255] = warped_scan[warped_mask == 255]

        if self.use_degradation and self.degradation_pipeline:
            degraded_composite = self.degradation_pipeline.apply_random_degradation(composite)
        else:
            degraded_composite = composite.copy()

        flat_pts = np.float32([[0, 0], [self.image_size[1], 0], [self.image_size[1], self.image_size[0]], [0, self.image_size[0]]])
        H_inv = cv2.getPerspectiveTransform(dst_pts, flat_pts)
        
        rectified_degraded = cv2.warpPerspective(degraded_composite, H_inv, (self.image_size[1], self.image_size[0]))
        
        degraded_composite_tensor = torch.from_numpy(degraded_composite.astype(np.float32) / 255.0).permute(2, 0, 1)
        rectified_degraded_tensor = torch.from_numpy(rectified_degraded.astype(np.float32) / 255.0).permute(2, 0, 1)
        clean_target_tensor = torch.from_numpy(clean_scan.astype(np.float32) / 255.0).permute(2, 0, 1)
        
        corners_normalized = dst_pts.copy()
        corners_normalized[:, 0] /= bg_w
        corners_normalized[:, 1] /= bg_h

        return {
            'raw_photo': degraded_composite_tensor,
            'corners': torch.from_numpy(corners_normalized),
            'rectified_input': rectified_degraded_tensor,
            'clean_target': clean_target_tensor
        }

class RealEvaluationDataset(Dataset):
    def __init__(self, 
                 real_photos_dir: str, 
                 annotation_file: str, 
                 image_size: Tuple[int, int] = (512, 512)):
        
        self.root_dir = real_photos_dir
        self.image_size = image_size
        
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
            
        self.images_info = {img['id']: img for img in self.annotations['images']}
        self.image_ids = list(self.images_info.keys())
        
        self.annotations_by_image = {}
        for ann in self.annotations['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations_by_image:
                self.annotations_by_image[img_id] = []
            self.annotations_by_image[img_id].append(ann)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_id = self.image_ids[idx]
        img_info = self.images_info[image_id]
        
        filepath = os.path.join(self.root_dir, img_info['file_name'])
        image = cv2.imread(filepath)
        original_h, original_w = image.shape[:2]
        
        image_resized = cv2.resize(image, (self.image_size[1], self.image_size[0]))
        
        ann = self.annotations_by_image.get(image_id, [{}])[0]
        keypoints = ann.get('keypoints', [0]*12)
        
        corners = np.zeros((4, 2), dtype=np.float32)
        for i in range(4):
            corners[i, 0] = keypoints[i * 3]
            corners[i, 1] = keypoints[i * 3 + 1]
            
        corners[:, 0] = (corners[:, 0] * (self.image_size[1] / original_w)) / self.image_size[1]
        corners[:, 1] = (corners[:, 1] * (self.image_size[0] / original_h)) / self.image_size[0]

        image_tensor = torch.from_numpy(image_resized.astype(np.float32) / 255.0).permute(2, 0, 1)

        return {
            'raw_photo': image_tensor,
            'corners': torch.from_numpy(corners),
            'original_shape': (original_h, original_w),
            'filename': img_info['file_name']
        }