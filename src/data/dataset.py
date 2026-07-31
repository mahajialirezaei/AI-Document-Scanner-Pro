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
    """
    Synthetic dataset for document scanning enhancement.
    
    Generates training samples on-the-fly by:
    1. Selecting a clean scan and random background
    2. Generating 4 random corner points on the background
    3. Warping the clean scan onto the background using perspective transform
    4. Applying photometric degradations (shadows, blur, noise, etc.)
    5. Warping back the degraded composite to get rectified input
    6. Returning the original clean scan as target
    
    The 4 random corner points serve as ground-truth labels for corner detection.
    """
    def __init__(self, 
                 clean_scans_paths: List[str], 
                 backgrounds_dir: str, 
                 image_size: Tuple[int, int] = (512, 512),
                 use_degradation: bool = True,
                 seed: Optional[int] = None,
                 freeze_data: bool = False,
                 num_samples: Optional[int] = None):
        
        self.clean_scans = clean_scans_paths
        self.backgrounds = list(Path(backgrounds_dir).glob("*.jpg"))
        self.image_size = image_size
        self.use_degradation = use_degradation
        self.freeze_data = freeze_data
        
        self.degradation_pipeline = create_degradation_pipeline(seed=seed) if use_degradation else None
        
        # Set seeds for reproducibility
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Pre-generate and cache data for validation/test sets
        self._cached_data = None
        if freeze_data:
            self._cache_all_samples(num_samples)

    def _generate_random_corners(self, bg_w: int, bg_h: int) -> np.ndarray:
        """Generate 4 random corner points within reasonable bounds on the background."""
        margin_x, margin_y = bg_w // 5, bg_h // 5
        
        tl = [random.randint(0, margin_x), random.randint(0, margin_y)]
        tr = [random.randint(bg_w - margin_x, bg_w), random.randint(0, margin_y)]
        br = [random.randint(bg_w - margin_x, bg_w), random.randint(bg_h - margin_y, bg_h)]
        bl = [random.randint(0, margin_x), random.randint(bg_h - margin_y, bg_h)]
        
        return np.float32([tl, tr, br, bl])

    def _generate_single_sample(self, idx: int, rng_state: Optional[dict] = None) -> Dict[str, Any]:
        """Generate a single synthetic sample."""
        # Restore RNG state if provided (for frozen datasets)
        if rng_state is not None:
            random.setstate(rng_state['random'])
            np.random.set_state(rng_state['numpy'])
        
        scan_path = self.clean_scans[idx % len(self.clean_scans)]
        bg_path = random.choice(self.backgrounds)
        
        clean_scan = cv2.imread(str(scan_path))
        clean_scan = cv2.cvtColor(clean_scan, cv2.COLOR_BGR2RGB)

        bg_image = cv2.imread(str(bg_path))
        bg_image = cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)
        
        # Resize to target image size
        bg_image = cv2.resize(bg_image, (self.image_size[1], self.image_size[0]))
        clean_scan = cv2.resize(clean_scan, (self.image_size[1], self.image_size[0]))
        scan_h, scan_w = clean_scan.shape[:2]
        bg_h, bg_w = bg_image.shape[:2]

        # Source points: corners of the flat scan
        src_pts = np.float32([[0, 0], [scan_w, 0], [scan_w, scan_h], [0, scan_h]])
        # Destination points: random corners on background (these are the ground-truth labels)
        dst_pts = self._generate_random_corners(bg_w, bg_h)
        
        # Compute homography from flat scan to warped position
        H = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # Warp the clean scan onto the background
        warped_scan = cv2.warpPerspective(clean_scan, H, (bg_w, bg_h))
        
        # Create mask for blending
        mask = np.ones((scan_h, scan_w), dtype=np.uint8) * 255
        warped_mask = cv2.warpPerspective(mask, H, (bg_w, bg_h))
        
        # Composite: place warped scan onto background
        composite = bg_image.copy()
        composite[warped_mask == 255] = warped_scan[warped_mask == 255]

        # Apply photometric degradations (blur, noise, shadows, etc.)
        if self.use_degradation and self.degradation_pipeline:
            degraded_composite = self.degradation_pipeline.apply_random_degradation(composite)
        else:
            degraded_composite = composite.copy()

        # Calculate inverse homography to warp back to flat rectangle
        flat_pts = np.float32([[0, 0], [self.image_size[1], 0], 
                               [self.image_size[1], self.image_size[0]], [0, self.image_size[0]]])
        H_inv = cv2.getPerspectiveTransform(dst_pts, flat_pts)
        
        # Warp-back: rectify the degraded composite to align with clean target
        rectified_degraded = cv2.warpPerspective(degraded_composite, H_inv, 
                                                  (self.image_size[1], self.image_size[0]))
        
        # Convert to tensors
        degraded_composite_tensor = torch.from_numpy(degraded_composite.astype(np.float32) / 255.0).permute(2, 0, 1)
        rectified_degraded_tensor = torch.from_numpy(rectified_degraded.astype(np.float32) / 255.0).permute(2, 0, 1)
        clean_target_tensor = torch.from_numpy(clean_scan.astype(np.float32) / 255.0).permute(2, 0, 1)
        
        # Normalize corner coordinates to [0, 1] range
        corners_normalized = dst_pts.copy()
        corners_normalized[:, 0] /= bg_w
        corners_normalized[:, 1] /= bg_h

        return {
            'raw_photo': degraded_composite_tensor,
            'corners': torch.from_numpy(corners_normalized),
            'rectified_input': rectified_degraded_tensor,
            'clean_target': clean_target_tensor
        }

    def _cache_all_samples(self, num_samples: Optional[int] = None) -> None:
        """Pre-generate all samples for frozen validation/test sets."""
        total_samples = num_samples if num_samples is not None else len(self)
        
        # Save RNG state for each sample to ensure deterministic retrieval
        self._cached_data = []
        
        for i in range(total_samples):
            # Capture RNG state before generating each sample
            rng_state = {
                'random': random.getstate(),
                'numpy': np.random.get_state()
            }
            sample = self._generate_single_sample(i, rng_state)
            self._cached_data.append({
                'sample': sample,
                'rng_state': rng_state
            })

    def __len__(self) -> int:
        if self.freeze_data and self._cached_data is not None:
            return len(self._cached_data)
        return len(self.clean_scans)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # For frozen datasets, retrieve pre-generated sample
        if self.freeze_data and self._cached_data is not None:
            cached = self._cached_data[idx % len(self._cached_data)]
            # Restore RNG state and regenerate to ensure exact same sample
            random.setstate(cached['rng_state']['random'])
            np.random.set_state(cached['rng_state']['numpy'])
            return self._generate_single_sample(idx, cached['rng_state'])
        
        # For training, generate fresh sample on-the-fly
        return self._generate_single_sample(idx)

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