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

def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Deterministic Top/Bottom -> Left/Right sorting.
    Guarantees [Top-Left, Top-Right, Bottom-Right, Bottom-Left] order.
    """
    if len(pts) != 4:
        return pts
        
    y_sorted = pts[np.argsort(pts[:, 1])]
    top_half = y_sorted[:2, :]
    bottom_half = y_sorted[2:, :]
    
    tl, tr = top_half[np.argsort(top_half[:, 0])]
    bl, br = bottom_half[np.argsort(bottom_half[:, 0])]
    
    return np.array([tl, tr, br, bl], dtype=np.float32)

class SyntheticDocumentDataset(Dataset):
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
        tl_x = random.uniform(0, bg_w * 0.3)
        tl_y = random.uniform(0, bg_h * 0.3)
        
        tr_x = random.uniform(bg_w * 0.7, bg_w)
        tr_y = random.uniform(0, bg_h * 0.3)
        
        br_x = random.uniform(bg_w * 0.7, bg_w)
        br_y = random.uniform(bg_h * 0.7, bg_h)
        
        bl_x = random.uniform(0, bg_w * 0.3)
        bl_y = random.uniform(bg_h * 0.7, bg_h)
        
        perspective_type = random.random()
        if perspective_type < 0.25:
            tl_x = random.uniform(bg_w * 0.25, bg_w * 0.45)
            tr_x = random.uniform(bg_w * 0.55, bg_w * 0.75)
        elif perspective_type < 0.5:
            bl_x = random.uniform(bg_w * 0.25, bg_w * 0.45)
            br_x = random.uniform(bg_w * 0.55, bg_w * 0.75)
            
        pts = np.array([[tl_x, tl_y], [tr_x, tr_y], [br_x, br_y], [bl_x, bl_y]], dtype=np.float32)
        return pts

    def _add_binding_artifacts(self, img: np.ndarray, probability: float = 0.25) -> Tuple[np.ndarray, np.ndarray]:
        h, w = img.shape[:2]
        hole_mask = np.ones((h, w), dtype=np.uint8) * 255
        
        if not self.use_degradation or random.random() > probability:
            return img, hole_mask
            
        art_type = random.choice(['spiral', 'gutter_shadow'])
        side = random.choice(['left', 'right'])
        
        result = img.copy()
        
        if art_type == 'spiral':
            num_holes = random.randint(20, 40)
            y_steps = np.linspace(10, h - 10, num_holes)
            x_pos = random.randint(15, 35) if side == 'left' else w - random.randint(15, 35)
            
            for y in y_steps:
                radius = random.randint(4, 9)
                cv2.circle(hole_mask, (x_pos, int(y)), radius, 0, -1)
                cv2.line(result, (x_pos, int(y)), (x_pos + random.randint(-20, 20), int(y)), (120, 120, 120), 4)
                
        elif art_type == 'gutter_shadow':
            shadow_width = random.randint(60, 150)
            gradient = np.linspace(0.2, 1.0, shadow_width).reshape(1, shadow_width, 1)
            
            if side == 'left':
                result[:, :shadow_width] = np.clip(result[:, :shadow_width] * gradient, 0, 255).astype(np.uint8)
            else:
                gradient = np.flip(gradient, axis=1)
                result[:, -shadow_width:] = np.clip(result[:, -shadow_width:] * gradient, 0, 255).astype(np.uint8)
                
        return result, hole_mask

    def _add_adjacent_page(self, img: np.ndarray, corners: np.ndarray) -> np.ndarray:
        if random.random() > 0.3:
            return img
        
        result = img.copy()
        is_left = random.random() > 0.5
        
        edge_idx = (3, 0) if is_left else (1, 2)
        pt1, pt2 = corners[edge_idx[0]], corners[edge_idx[1]]
        
        vec = pt2 - pt1
        normal_len = np.linalg.norm(vec)
        if normal_len == 0: return result
        
        normal = np.array([-vec[1], vec[0]]) if is_left else np.array([vec[1], -vec[0]])
        normal = normal / normal_len
        
        page_width = random.uniform(100, 300)
        pt3 = pt2 + normal * page_width
        pt4 = pt1 + normal * page_width
        
        page_pts = np.array([pt1, pt2, pt3, pt4], dtype=np.int32)
        
        page_color = (random.randint(220, 245), random.randint(225, 250), random.randint(230, 255))
        cv2.fillPoly(result, [page_pts], page_color)
        
        for _ in range(random.randint(15, 30)):
            t1, t2 = random.uniform(0.05, 0.95), random.uniform(0.05, 0.95)
            line_pt1 = pt1 * (1 - t1) + pt4 * t1
            line_pt2 = pt2 * (1 - t2) + pt3 * t2
            
            line_color = (random.randint(30, 80), random.randint(30, 80), random.randint(30, 80))
            cv2.line(result, tuple(line_pt1.astype(int)), tuple(line_pt2.astype(int)), 
                     line_color, random.randint(2, 4))
                     
        cv2.line(result, tuple(pt1.astype(int)), tuple(pt2.astype(int)), (20, 20, 20), random.randint(8, 15))
            
        return result

    def _add_distractors(self, img: np.ndarray, corners: np.ndarray) -> np.ndarray:
        result = img.copy()
        
        if random.random() < 0.2:
            overlay = np.zeros_like(result)
            corner_idx = random.randint(0, 3)
            c_pt = corners[corner_idx]
            
            poly_pts = []
            for _ in range(random.randint(3, 5)):
                offset_x = random.uniform(-60, 60)
                offset_y = random.uniform(-60, 60)
                poly_pts.append([c_pt[0] + offset_x, c_pt[1] + offset_y])
                
            color = (random.randint(10, 200), random.randint(10, 200), random.randint(10, 200))
            cv2.fillPoly(overlay, [np.array(poly_pts, dtype=np.int32)], color)
            overlay = cv2.GaussianBlur(overlay, (9, 9), 0)
            
            mask = np.any(overlay > 0, axis=2)[..., None]
            result = np.where(mask, overlay, result)

        if random.random() < 0.15:
            overlay = np.zeros_like(result)
            edge_idx = random.choice([(0, 1), (1, 2), (2, 3), (3, 0)])
            pt1, pt2 = corners[edge_idx[0]], corners[edge_idx[1]]
            t = random.uniform(0.1, 0.9)
            finger_center = pt1 * (1 - t) + pt2 * t
            skin_color = (random.randint(110, 160), random.randint(140, 190), random.randint(190, 230))
            cv2.circle(overlay, tuple(finger_center.astype(int)), random.randint(30, 50), skin_color, -1)
            
            overlay = cv2.GaussianBlur(overlay, (15, 15), 5)
            mask = np.any(overlay > 0, axis=2)[..., None]
            result = np.where(mask, overlay, result)
            
        return result

    def _add_3d_shadow_distractors(self, img: np.ndarray, corners: np.ndarray) -> np.ndarray:
        if random.random() > 0.3:
            return img
        
        result = img.copy()
        for _ in range(random.randint(1, 2)):
            overlay = np.zeros_like(result)
            shadow_overlay = np.zeros_like(result)
            
            edge_idx = random.choice([(0, 1), (1, 2), (2, 3), (3, 0)])
            pt1, pt2 = corners[edge_idx[0]], corners[edge_idx[1]]
            
            t = random.uniform(0.1, 0.9)
            center = pt1 * (1 - t) + pt2 * t
            
            poly_pts = []
            for _ in range(random.randint(4, 7)):
                offset_x = random.uniform(-90, 90)
                offset_y = random.uniform(-90, 90)
                poly_pts.append([center[0] + offset_x, center[1] + offset_y])
            poly = np.array(poly_pts, dtype=np.int32)
            
            shadow_offset = np.array([random.uniform(8, 20), random.uniform(8, 20)])
            shadow_poly = poly + shadow_offset.astype(np.int32)
            
            cv2.fillPoly(shadow_overlay, [shadow_poly], (15, 15, 15))
            shadow_overlay = cv2.GaussianBlur(shadow_overlay, (25, 25), 12)
            
            obj_color = (random.randint(30, 200), random.randint(30, 200), random.randint(30, 200))
            cv2.fillPoly(overlay, [poly], obj_color)
            
            mask_shadow = np.any(shadow_overlay > 0, axis=2)[..., None]
            result = np.where(mask_shadow, (shadow_overlay * 0.6 + result * 0.4).astype(np.uint8), result)
            
            mask_obj = np.any(overlay > 0, axis=2)[..., None]
            result = np.where(mask_obj, overlay, result)
            
        return result

    def _add_camouflage_polygons(self, img: np.ndarray, corners: np.ndarray) -> np.ndarray:
        if random.random() > 0.25:
            return img
            
        result = img.copy()
        overlay = np.zeros_like(result)
        
        edge_idx = random.choice([(0, 1), (1, 2), (2, 3), (3, 0)])
        pt1, pt2 = corners[edge_idx[0]], corners[edge_idx[1]]
        t = random.uniform(0.2, 0.8)
        center = pt1 * (1 - t) + pt2 * t
        
        poly_pts = []
        for _ in range(random.randint(4, 8)):
            offset_x = random.uniform(-110, 110)
            offset_y = random.uniform(-110, 110)
            poly_pts.append([center[0] + offset_x, center[1] + offset_y])
            
        poly = np.array(poly_pts, dtype=np.int32)
        camo_color = (random.randint(220, 255), random.randint(220, 255), random.randint(220, 255))
        cv2.fillPoly(overlay, [poly], camo_color)
        
        overlay = cv2.GaussianBlur(overlay, (5, 5), 2)
        mask = np.any(overlay > 0, axis=2)[..., None]
        result = np.where(mask, overlay, result)
        return result

    def _add_clothing_occlusions(self, img: np.ndarray, corners: np.ndarray) -> np.ndarray:
        if random.random() > 0.10:
            return img
            
        result = img.copy()
        overlay = np.zeros_like(result)
        
        color = random.choice([(40, 200, 240), (200, 100, 40), (120, 120, 120), (50, 50, 150)])
        
        edge_idx = random.choice([(1, 2), (2, 3)])
        pt1, pt2 = corners[edge_idx[0]], corners[edge_idx[1]]
        
        vec = pt2 - pt1
        length = np.linalg.norm(vec)
        if length == 0: return img
        
        normal = np.array([-vec[1], vec[0]]) / length
        center_doc = np.mean(corners, axis=0)
        edge_center = (pt1 + pt2) / 2.0
        
        if np.dot(normal, edge_center - center_doc) < 0:
            normal = -normal
            
        t = random.uniform(0.2, 0.8)
        edge_pt = pt1 * (1 - t) + pt2 * t
        
        dist_outside = random.uniform(100, 180)
        sleeve_center = edge_pt + normal * dist_outside
        
        poly_pts = []
        for _ in range(random.randint(5, 8)):
            radius = dist_outside + random.uniform(10, 30)
            angle = random.uniform(0, 2 * np.pi)
            poly_pts.append([sleeve_center[0] + radius * np.cos(angle), 
                             sleeve_center[1] + radius * np.sin(angle)])
            
        poly = np.array(poly_pts, dtype=np.int32)
        cv2.fillPoly(overlay, [poly], color)
        
        overlay = cv2.GaussianBlur(overlay, (21, 21), 10)
        mask = np.any(overlay > 0, axis=2)[..., None]
        result = np.where(mask, overlay, result)
        
        return result

    def _generate_single_sample(self, idx: int, rng_state: Optional[dict] = None) -> Dict[str, Any]:
        if rng_state is not None:
            random.setstate(rng_state['random'])
            np.random.set_state(rng_state['numpy'])
            
        scan_path = self.clean_scans[idx % len(self.clean_scans)]
        bg_path = random.choice(self.backgrounds)
        
        clean_scan = cv2.imread(str(scan_path))
        
        clean_scan = cv2.resize(clean_scan, (self.image_size[1], self.image_size[0]))
        
        clean_scan, hole_mask = self._add_binding_artifacts(clean_scan, probability=0.30)
        
        if self.use_degradation and self.degradation_pipeline:
            bgra = cv2.cvtColor(clean_scan, cv2.COLOR_BGR2BGRA)
            bgra[:, :, 3] = hole_mask
            bgra = self.degradation_pipeline.apply_elastic_transform(bgra)
            
            clean_scan = np.ascontiguousarray(bgra[:, :, :3])
            hole_mask = np.ascontiguousarray(bgra[:, :, 3])
            
            clean_scan = self.degradation_pipeline.apply_ink_simulation(clean_scan)
            
        clean_scan = cv2.cvtColor(clean_scan, cv2.COLOR_BGR2RGB)
        scan_h, scan_w = clean_scan.shape[:2]
        
        bg_image = cv2.imread(str(bg_path))
        bg_image = cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)
        bg_image = cv2.resize(bg_image, (self.image_size[1], self.image_size[0]))
        bg_h, bg_w = bg_image.shape[:2]

        src_pts = np.float32([[0, 0], [scan_w, 0], [scan_w, scan_h], [0, scan_h]])
        dst_pts = self._generate_random_corners(bg_w, bg_h)
        
        H = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped_scan = cv2.warpPerspective(clean_scan, H, (bg_w, bg_h))
        
        warped_mask = cv2.warpPerspective(hole_mask, H, (bg_w, bg_h), flags=cv2.INTER_NEAREST)
        
        composite = bg_image.copy()
        composite[warped_mask == 255] = warped_scan[warped_mask == 255]

        if self.use_degradation:
            composite = self._add_adjacent_page(composite, dst_pts)
            composite = self._add_distractors(composite, dst_pts)
            composite = self._add_3d_shadow_distractors(composite, dst_pts)
            composite = self._add_camouflage_polygons(composite, dst_pts)
            composite = self._add_clothing_occlusions(composite, dst_pts)

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
            rng_state = {'random': random.getstate(), 'numpy': np.random.get_state()}
            sample = self._generate_single_sample(i, rng_state)
            self._cached_data.append({'sample': sample, 'rng_state': rng_state})

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
    def __init__(self, real_photos_dir: str, scanned_photos_dir: str, annotation_file: str, image_size: Tuple[int, int] = (512, 512)):
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
            if not match: continue
                
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

    def __len__(self) -> int: return len(self.valid_data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        data = self.valid_data[idx]

        raw_bgr = cv2.imread(data['raw_path'])
        scan_bgr = cv2.imread(data['scan_path'])
        
        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        scan_rgb = cv2.cvtColor(scan_bgr, cv2.COLOR_BGR2RGB)
        
        original_h, original_w = raw_rgb.shape[:2]

        corners_original = np.array(data['segmentation'], dtype=np.float32).reshape(4, 2)
        corners_original = order_points(corners_original)
        
        corners_norm = corners_original.copy()
        corners_norm[:, 0] /= original_w
        corners_norm[:, 1] /= original_h
        
        raw_resized = cv2.resize(raw_rgb, (self.image_size[1], self.image_size[0]))
        scan_resized = cv2.resize(scan_rgb, (self.image_size[1], self.image_size[0]))
        
        raw_tensor = torch.from_numpy(raw_resized.astype(np.float32) / 255.0).permute(2, 0, 1)
        scan_tensor = torch.from_numpy(scan_resized.astype(np.float32) / 255.0).permute(2, 0, 1)
        corners_tensor = torch.from_numpy(corners_norm)
        
        return {'raw_photo': raw_tensor, 'clean_target': scan_tensor, 'corners': corners_tensor}