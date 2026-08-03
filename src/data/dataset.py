import os
import json
import re
import cv2
import numpy as np
import torch
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from torch.utils.data import Dataset, DataLoader
from .degradation import create_degradation_pipeline

def _order_points_polar(pts: np.ndarray) -> np.ndarray:
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    sorted_idx = np.argsort(angles)
    return pts[sorted_idx]

class SyntheticDocumentDataset(Dataset):
    """
    Synthetic dataset for document scanning enhancement.
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
        self.num_samples = num_samples
        
        self.degradation_pipeline = create_degradation_pipeline(seed=seed) if use_degradation else None
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            
        self._cached_data = None
        if freeze_data:
            self._cache_all_samples(num_samples)

    def _generate_random_corners(self, bg_w: int, bg_h: int) -> np.ndarray:
        scale = random.uniform(0.5, 0.85)
        doc_w = bg_w * scale
        doc_h = bg_h * scale
        
        corners_3d = np.array([
            [-doc_w/2, -doc_h/2, 0],
            [ doc_w/2, -doc_h/2, 0],
            [ doc_w/2,  doc_h/2, 0],
            [-doc_w/2,  doc_h/2, 0]
        ], dtype=np.float32)
        
        pitch = np.radians(random.uniform(-45, 45))
        yaw = np.radians(random.uniform(-35, 35))
        roll = np.radians(random.uniform(-15, 15))
        
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)]
        ])
        Ry = np.array([
            [np.cos(yaw), 0, np.sin(yaw)],
            [0, 1, 0],
            [-np.sin(yaw), 0, np.cos(yaw)]
        ])
        Rz = np.array([
            [np.cos(roll), -np.sin(roll), 0],
            [np.sin(roll), np.cos(roll), 0],
            [0, 0, 1]
        ])
        
        R = Rz @ Ry @ Rx
        rotated_3d = corners_3d @ R.T
        
        z_distance = random.uniform(doc_w * 1.0, doc_w * 2.5)
        
        cx = random.uniform(doc_w / 2, bg_w - (doc_w / 2))
        cy = random.uniform(doc_h / 2, bg_h - (doc_h / 2))
        
        focal_length = z_distance
        
        projected_2d = np.zeros((4, 2), dtype=np.float32)
        for i in range(4):
            x, y, z = rotated_3d[i]
            z_translated = z + z_distance
            
            proj_x = x * (focal_length / z_translated)
            proj_y = y * (focal_length / z_translated)
            
            projected_2d[i, 0] = proj_x + cx
            projected_2d[i, 1] = proj_y + cy
            
        return projected_2d

    def _add_distractors(self, img: np.ndarray, corners: np.ndarray) -> np.ndarray:
        result = img.copy()
        
        if random.random() < 0.15:
            edge_idx = random.choice([(3, 0), (1, 2)])
            pt1, pt2 = corners[edge_idx[0]], corners[edge_idx[1]]
            
            vec = pt2 - pt1
            length = np.linalg.norm(vec)
            if length > 0:
                dir_vec = vec / length
                normal = np.array([-dir_vec[1], dir_vec[0]])
                center = np.mean(corners, axis=0)
                midpoint = (pt1 + pt2) / 2.0
                if np.dot(normal, midpoint - center) < 0:
                    normal = -normal
                    
                offset_dist = random.uniform(5, 20)
                pt1_offset = pt1 + normal * offset_dist
                pt2_offset = pt2 + normal * offset_dist
                
                num_rings = int(length // random.uniform(15, 30))
                for i in range(1, max(2, num_rings)):
                    t = i / num_rings
                    ring_pt = pt1_offset * (1 - t) + pt2_offset * t
                    
                    color = (random.randint(10, 50), random.randint(10, 50), random.randint(10, 50))
                    radius = random.randint(4, 12)
                    cv2.circle(result, (int(ring_pt[0]), int(ring_pt[1])), radius, color, -1)
                    
        if random.random() < 0.15:
            edge_idx = random.choice([(0, 1), (1, 2), (2, 3), (3, 0)])
            pt1, pt2 = corners[edge_idx[0]], corners[edge_idx[1]]
            
            vec = pt2 - pt1
            length = np.linalg.norm(vec)
            if length > 0:
                dir_vec = vec / length
                normal = np.array([-dir_vec[1], dir_vec[0]])
                
                center = np.mean(corners, axis=0)
                midpoint = (pt1 + pt2) / 2.0
                if np.dot(normal, midpoint - center) < 0:
                    normal = -normal
                    
                offset_dist = random.uniform(5, 30)
                pt1_offset = pt1 + normal * offset_dist
                pt2_offset = pt2 + normal * offset_dist
                
                pt1_ext = pt1_offset - dir_vec * random.uniform(10, 50)
                pt2_ext = pt2_offset + dir_vec * random.uniform(10, 50)
                
                color = (random.randint(0, 40), random.randint(0, 40), random.randint(0, 40))
                thickness = random.randint(10, 60)
                cv2.line(result, (int(pt1_ext[0]), int(pt1_ext[1])), 
                         (int(pt2_ext[0]), int(pt2_ext[1])), color, thickness)
                         
        if random.random() < 0.15:
            h, w = result.shape[:2]
            for _ in range(random.randint(1, 3)):
                pt1 = (random.randint(0, w), random.randint(0, h))
                pt2 = (random.randint(0, w), random.randint(0, h))
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                thickness = random.randint(2, 12)
                cv2.line(result, pt1, pt2, color, thickness)

        return result

    def _generate_single_sample(self, idx: int, rng_state: Optional[dict] = None) -> Dict[str, Any]:
        if rng_state is not None:
            random.setstate(rng_state['random'])
            np.random.set_state(rng_state['numpy'])
            
        scan_path = self.clean_scans[idx % len(self.clean_scans)]
        bg_path = random.choice(self.backgrounds)
        
        clean_scan = cv2.imread(str(scan_path))
        
        if self.use_degradation and self.degradation_pipeline:
            clean_scan = self.degradation_pipeline.apply_ink_simulation(clean_scan)
            
        clean_scan = cv2.cvtColor(clean_scan, cv2.COLOR_BGR2RGB)
        bg_image = cv2.imread(str(bg_path))
        bg_image = cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)
        
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

        if self.use_degradation:
            composite = self._add_distractors(composite, dst_pts)

        if self.use_degradation and self.degradation_pipeline:
            degraded_composite = self.degradation_pipeline.apply_random_degradation(composite)
        else:
            degraded_composite = composite.copy()

        flat_pts = np.float32([[0, 0], [self.image_size[1], 0], 
                               [self.image_size[1], self.image_size[0]], [0, self.image_size[0]]])
        H_inv = cv2.getPerspectiveTransform(dst_pts, flat_pts)
        
        if degraded_composite is None or degraded_composite.size == 0:
            degraded_composite = composite.copy()
            
        if H_inv is None:
            H_inv = np.eye(3, dtype=np.float32)

        rectified_degraded = cv2.warpPerspective(degraded_composite, H_inv, 
                                                 (self.image_size[1], self.image_size[0]))
        
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

    def _cache_all_samples(self, num_samples: Optional[int] = None) -> None:
        total_samples = num_samples if num_samples is not None else len(self)
        self._cached_data = []
        
        for i in range(total_samples):
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
        if self.num_samples is not None:
            return self.num_samples
        return len(self.clean_scans)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.freeze_data and self._cached_data is not None:
            cached = self._cached_data[idx % len(self._cached_data)]
            return cached['sample']
            
        return self._generate_single_sample(idx)


class RealDocumentDataset(Dataset):
    """
    Dataset for real document photos and their corresponding reference scans.
    """
    def __init__(self, 
                 real_photos_dir: str,
                 scanned_photos_dir: str,
                 annotation_file: str, 
                 image_size: Tuple[int, int] = (512, 512)):
        
        self.real_photos_dir = real_photos_dir
        self.scanned_photos_dir = scanned_photos_dir
        self.image_size = image_size
        
        with open(annotation_file, 'r') as f:
            self.annotations_data = json.load(f)
            
        annotations_by_image = {}
        for ann in self.annotations_data.get('annotations', []):
            img_id = ann['image_id']
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)

        self.valid_data = []
        
        for img_info in self.annotations_data.get('images', []):
            file_name = img_info['file_name']
            img_id = img_info['id']
            
            match = re.search(r'^(\d+)_', file_name)
            if not match:
                continue
                
            doc_number = match.group(1)
            scan_file_name = f"{doc_number}.jpg"
            
            raw_path = os.path.join(self.real_photos_dir, file_name)
            scan_path = os.path.join(self.scanned_photos_dir, scan_file_name)
            
            if os.path.exists(raw_path) and os.path.exists(scan_path) and img_id in annotations_by_image:
                ann = annotations_by_image[img_id][0]
                
                if 'segmentation' in ann and len(ann['segmentation']) > 0:
                    self.valid_data.append({
                        'raw_path': raw_path,
                        'scan_path': scan_path,
                        'segmentation': ann['segmentation'][0],
                        'original_w': img_info['width'],
                        'original_h': img_info['height']
                    })

    def __len__(self) -> int:
        return len(self.valid_data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        data = self.valid_data[idx]
        
        raw_bgr = cv2.imread(data['raw_path'])
        scan_bgr = cv2.imread(data['scan_path'])
        
        if raw_bgr is None or scan_bgr is None:
            raise ValueError(f"Failed to load image pair for index {idx}")
            
        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        scan_rgb = cv2.cvtColor(scan_bgr, cv2.COLOR_BGR2RGB)
        
        original_h, original_w = raw_rgb.shape[:2]
        
        corners_original = np.array(data['segmentation'], dtype=np.float32).reshape(4, 2)
        corners_original = _order_points_polar(corners_original)
        
        corners_norm = corners_original.copy()
        corners_norm[:, 0] /= original_w
        corners_norm[:, 1] /= original_h
        
        raw_resized = cv2.resize(raw_rgb, (self.image_size[1], self.image_size[0]))
        scan_resized = cv2.resize(scan_rgb, (self.image_size[1], self.image_size[0]))
        
        raw_tensor = torch.from_numpy(raw_resized.astype(np.float32) / 255.0).permute(2, 0, 1)
        scan_tensor = torch.from_numpy(scan_resized.astype(np.float32) / 255.0).permute(2, 0, 1)
        corners_tensor = torch.from_numpy(corners_norm)
        
        return {
            'raw_photo': raw_tensor,       
            'clean_target': scan_tensor,   
            'corners': corners_tensor      
        }