"""
Degradation pipeline for document images using OpenCV.
Implements various degradation functions to simulate real-world document conditions,
including 3D physical paper curl (Elastic Transform).
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import random


class DegradationPipeline:
    """
    A pipeline for applying various degradations to document images.
    Simulates real-world conditions like blur, noise, perspective distortion, color temp, and physical curl.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the degradation pipeline.
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            
    def apply_ink_simulation(self, image: np.ndarray) -> np.ndarray:
        """
        Simulate different real-world ink types (blue ballpoint, faded black) 
        by targeting dark pixels.
        """
        degraded = image.copy()
        
        gray = cv2.cvtColor(degraded, cv2.COLOR_BGR2GRAY)
        mask = gray < 130
        
        if not np.any(mask):
            return degraded
            
        ink_colors = [
            (160, 80, 20),   # Standard Blue pen
            (190, 110, 40),  # Light blue pen
            (110, 110, 110), # Faded grey/black
            (130, 50, 20),   # Navy blue
            (150, 40, 100)   # Purple-ish pen
        ]
        chosen_ink = np.array(random.choice(ink_colors), dtype=np.float32)
        
        fade_factor = random.uniform(0.4, 0.85)
        
        for c in range(3):
            degraded[mask, c] = np.clip(
                degraded[mask, c] * (1 - fade_factor) + chosen_ink[c] * fade_factor, 
                0, 255
            ).astype(np.uint8)
            
        return degraded
    
    def apply_color_temperature(self, image: np.ndarray) -> np.ndarray:
        """
        Simulate warm (tungsten/yellow) or cool (shade/blue) lighting 
        by scaling color channels. Fast and lightweight.
        """
        degraded = image.copy().astype(np.float32)
        
        # 0: Blue, 1: Green, 2: Red in OpenCV (BGR)
        is_warm = random.random() > 0.5
        
        if is_warm:
            # Warm lighting: Boost Red/Green, Reduce Blue
            degraded[:, :, 0] *= random.uniform(0.7, 0.95)  # Reduce B
            degraded[:, :, 1] *= random.uniform(0.95, 1.05) # Slightly alter G
            degraded[:, :, 2] *= random.uniform(1.05, 1.3)  # Boost R
        else:
            # Cool lighting: Boost Blue, Reduce Red
            degraded[:, :, 0] *= random.uniform(1.05, 1.3)  # Boost B
            degraded[:, :, 1] *= random.uniform(0.95, 1.05) # Slightly alter G
            degraded[:, :, 2] *= random.uniform(0.7, 0.95)  # Reduce R
            
        return np.clip(degraded, 0, 255).astype(np.uint8)

    def apply_motion_blur(self, image: np.ndarray, kernel_size: int = 15, angle: Optional[float] = None) -> np.ndarray:
        """Apply motion blur to simulate camera shake."""
        if angle is None:
            angle = random.uniform(0, 360)
        
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        end_x = int(center + kernel_size / 2 * np.cos(np.radians(angle)))
        end_y = int(center + kernel_size / 2 * np.sin(np.radians(angle)))
        
        end_x = max(0, min(kernel_size - 1, end_x))
        end_y = max(0, min(kernel_size - 1, end_y))
        
        cv2.line(kernel, (center, center), (end_x, end_y), 1.0, 1)
        kernel /= np.sum(kernel)
        
        degraded = cv2.filter2D(image, -1, kernel)
        return degraded
    
    def apply_gaussian_blur(self, image: np.ndarray, kernel_size: int = 5, sigma: float = 0) -> np.ndarray:
        """Apply Gaussian blur to simulate out-of-focus capture."""
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        degraded = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        return degraded
    
    def apply_gaussian_noise(self, image: np.ndarray, mean: float = 0, std: float = 25) -> np.ndarray:
        """Add Gaussian noise to simulate sensor noise."""
        noise = np.random.normal(mean, std, image.shape).astype(np.float32)
        degraded = image.astype(np.float32) + noise
        degraded = np.clip(degraded, 0, 255).astype(np.uint8)
        return degraded
    
    def apply_poisson_noise(self, image: np.ndarray) -> np.ndarray:
        """Add Poisson noise to simulate photon shot noise."""
        normalized = image.astype(np.float32) / 255.0
        noisy = np.random.poisson(normalized * 255) / 255.0
        degraded = np.clip(noisy * 255, 0, 255).astype(np.uint8)
        return degraded
    
    def apply_jpeg_compression(self, image: np.ndarray, quality: int = 50) -> np.ndarray:
        """Simulate artifacts from image compression."""
        quality = max(1, min(100, quality))
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        
        success, encoded = cv2.imencode('.jpg', image, encode_param)
        if not success:
            return image
            
        degraded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if degraded is None:
            return image
            
        return degraded
    
    def apply_brightness_change(self, image: np.ndarray, factor: Optional[float] = None) -> np.ndarray:
        """Simulate over or under exposure."""
        if factor is None:
            factor = random.uniform(0.6, 1.15)
        
        degraded = image.astype(np.float32) * factor
        degraded = np.clip(degraded, 0, 255).astype(np.uint8)
        return degraded
    
    def apply_contrast_change(self, image: np.ndarray, factor: Optional[float] = None) -> np.ndarray:
        """Simulate washed out or harsh lighting contrasts."""
        if factor is None:
            factor = random.uniform(0.7, 1.2)
        
        mean = np.mean(image, axis=(0, 1), keepdims=True)
        degraded = (image - mean) * factor + mean
        degraded = np.clip(degraded, 0, 255).astype(np.uint8)
        return degraded

    def apply_salt_pepper_noise(self, image: np.ndarray, salt_prob: float = 0.003, pepper_prob: float = 0.003) -> np.ndarray:
        """Add salt-and-pepper noise to simulate dead pixels or dust."""
        degraded = image.copy()
        h, w = degraded.shape[:2]
        
        num_salt = int(h * w * salt_prob)
        coords = [np.random.randint(0, i, num_salt) for i in (h, w)]
        if len(degraded.shape) == 3:
            degraded[coords[0], coords[1], :] = 255
        else:
            degraded[coords[0], coords[1]] = 255
        
        num_pepper = int(h * w * pepper_prob)
        coords = [np.random.randint(0, i, num_pepper) for i in (h, w)]
        if len(degraded.shape) == 3:
            degraded[coords[0], coords[1], :] = 0
        else:
            degraded[coords[0], coords[1]] = 0
        
        return degraded

    def apply_shadow(self, image: np.ndarray, num_shadows: int = 1) -> np.ndarray:
        """Add soft shadow effects to simulate uneven lighting."""
        degraded = image.copy().astype(np.float32)
        h, w = degraded.shape[:2]
        
        for _ in range(num_shadows):
            num_points = random.randint(3, 6)
            points = np.random.randint(0, max(h, w), (num_points, 2)).astype(np.int32)
            points[:, 0] = np.clip(points[:, 0], 0, w - 1)
            points[:, 1] = np.clip(points[:, 1], 0, h - 1)
            
            mask = np.zeros((h, w), dtype=np.float32)
            cv2.fillPoly(mask, [points], 1.0)
            
            mask = cv2.GaussianBlur(mask, (151, 151), 0)
            
            shadow_intensity = random.uniform(0.6, 0.85)
            
            if len(degraded.shape) == 3:
                for c in range(degraded.shape[2]):
                    degraded[:, :, c] = degraded[:, :, c] * (1.0 - mask * (1.0 - shadow_intensity))
            else:
                degraded = degraded * (1.0 - mask * (1.0 - shadow_intensity))
        
        return np.clip(degraded, 0, 255).astype(np.uint8)
    
    def apply_resolution_loss(self, image: np.ndarray, scale_factor: Optional[float] = None) -> np.ndarray:
        """Apply resolution loss by downscaling and upscaling to simulate camera distance."""
        h, w = image.shape[:2]
        
        if scale_factor is None:
            scale_factor = random.uniform(2, 4)
        
        new_h, new_w = int(h / scale_factor), int(w / scale_factor)
        downscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        upscaled = cv2.resize(downscaled, (w, h), interpolation=cv2.INTER_LINEAR)
        
        return upscaled
        
    def apply_elastic_transform(self, image: np.ndarray, alpha: Optional[float] = None, sigma: Optional[float] = None) -> np.ndarray:
        """
        Apply elastic transformation to simulate 3D paper curl and page warping.
        Creates a randomized smooth displacement field to warp the image non-linearly.
        """
        shape = image.shape[:2]
        
        if alpha is None:
            alpha = shape[1] * random.uniform(0.015, 0.035)
        if sigma is None:
            sigma = shape[1] * random.uniform(0.05, 0.1)
            
        dx = cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
        dy = cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
        
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        map_x = np.float32(x + dx)
        map_y = np.float32(y + dy)
        
        degraded = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        return degraded
    
    def apply_random_degradation(self, image: np.ndarray, 
                                  num_degradations: int = 4) -> np.ndarray:
        """Apply a random combination of degradations."""
        degradation_funcs = [
            lambda x: self.apply_motion_blur(x),
            lambda x: self.apply_gaussian_blur(x),
            lambda x: self.apply_gaussian_noise(x),
            lambda x: self.apply_salt_pepper_noise(x),
            lambda x: self.apply_jpeg_compression(x, quality=random.randint(30, 80)),
            lambda x: self.apply_brightness_change(x),
            lambda x: self.apply_contrast_change(x),
            lambda x: self.apply_color_temperature(x),
            lambda x: self.apply_shadow(x, num_shadows=random.randint(1, 2)),
            lambda x: self.apply_resolution_loss(x),
            lambda x: self.apply_elastic_transform(x)
        ]

        num_to_apply = min(num_degradations, len(degradation_funcs))
        selected_funcs = random.sample(degradation_funcs, num_to_apply)
        
        degraded = image.copy()
        for func in selected_funcs:
            degraded = func(degraded)
            
        return degraded


def create_degradation_pipeline(seed: Optional[int] = None) -> DegradationPipeline:
    return DegradationPipeline(seed=seed)