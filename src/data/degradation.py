"""
Degradation pipeline for document images using OpenCV.
Implements various degradation functions to simulate real-world document conditions.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import random


class DegradationPipeline:
    """
    A pipeline for applying various degradations to document images.
    Simulates real-world conditions like blur, noise, perspective distortion, etc.
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
    
    def apply_motion_blur(self, image: np.ndarray, kernel_size: int = 15, angle: Optional[float] = None) -> np.ndarray:
        """
        Apply motion blur to simulate camera shake or movement.
        
        Args:
            image: Input image (H, W, C)
            kernel_size: Size of the motion blur kernel
            angle: Angle of motion in degrees (random if None)
            
        Returns:
            Motion blurred image
        """
        if angle is None:
            angle = random.uniform(0, 360)
        
        # Create motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        end_x = int(center + kernel_size / 2 * np.cos(np.radians(angle)))
        end_y = int(center + kernel_size / 2 * np.sin(np.radians(angle)))
        
        # Clamp to kernel bounds
        end_x = max(0, min(kernel_size - 1, end_x))
        end_y = max(0, min(kernel_size - 1, end_y))
        
        cv2.line(kernel, (center, center), (end_x, end_y), 1.0, 1)
        kernel /= np.sum(kernel)
        
        # Apply convolution
        degraded = cv2.filter2D(image, -1, kernel)
        return degraded
    
    def apply_gaussian_blur(self, image: np.ndarray, kernel_size: int = 5, sigma: float = 0) -> np.ndarray:
        """
        Apply Gaussian blur to simulate out-of-focus capture.
        
        Args:
            image: Input image (H, W, C)
            kernel_size: Size of the Gaussian kernel (must be odd)
            sigma: Standard deviation of Gaussian (0 = auto-calculate)
            
        Returns:
            Gaussian blurred image
        """
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        degraded = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        return degraded
    
    def apply_gaussian_noise(self, image: np.ndarray, mean: float = 0, std: float = 25) -> np.ndarray:
        """
        Add Gaussian noise to simulate sensor noise.
        
        Args:
            image: Input image (H, W, C)
            mean: Mean of the Gaussian noise
            std: Standard deviation of the Gaussian noise
            
        Returns:
            Noisy image
        """
        noise = np.random.normal(mean, std, image.shape).astype(np.float32)
        degraded = image.astype(np.float32) + noise
        degraded = np.clip(degraded, 0, 255).astype(np.uint8)
        return degraded
    
    def apply_salt_pepper_noise(self, image: np.ndarray, salt_prob: float = 0.01, pepper_prob: float = 0.01) -> np.ndarray:
        """
        Add salt-and-pepper noise to simulate dead pixels or dust.
        
        Args:
            image: Input image (H, W, C)
            salt_prob: Probability of white pixels
            pepper_prob: Probability of black pixels
            
        Returns:
            Image with salt-and-pepper noise
        """
        degraded = image.copy()
        h, w = degraded.shape[:2]
        
        # Salt noise
        num_salt = int(h * w * salt_prob)
        coords = [np.random.randint(0, i, num_salt) for i in (h, w)]
        if len(degraded.shape) == 3:
            degraded[coords[0], coords[1], :] = 255
        else:
            degraded[coords[0], coords[1]] = 255
        
        # Pepper noise
        num_pepper = int(h * w * pepper_prob)
        coords = [np.random.randint(0, i, num_pepper) for i in (h, w)]
        if len(degraded.shape) == 3:
            degraded[coords[0], coords[1], :] = 0
        else:
            degraded[coords[0], coords[1]] = 0
        
        return degraded
    
    def apply_poisson_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Add Poisson noise to simulate photon shot noise.
        
        Args:
            image: Input image (H, W, C)
            
        Returns:
            Image with Poisson noise
        """
        # Normalize to [0, 1]
        normalized = image.astype(np.float32) / 255.0
        
        # Apply Poisson noise
        noisy = np.random.poisson(normalized * 255) / 255.0
        degraded = np.clip(noisy * 255, 0, 255).astype(np.uint8)
        return degraded
    
    def apply_jpeg_compression(self, image: np.ndarray, quality: int = 50) -> np.ndarray:
        """
        Simulate JPEG compression artifacts.
        
        Args:
            image: Input image (H, W, C)
            quality: JPEG quality factor (1-100)
            
        Returns:
            Compressed image
        """
        quality = max(1, min(100, quality))
        
        # Encode to JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', image, encode_param)
        
        # Decode from JPEG
        degraded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return degraded
    
    def apply_brightness_change(self, image: np.ndarray, factor: Optional[float] = None) -> np.ndarray:
        """
        Change brightness to simulate varying lighting conditions.
        
        Args:
            image: Input image (H, W, C)
            factor: Brightness multiplier (<1 darker, >1 brighter). Random if None.
            
        Returns:
            Brightness-adjusted image
        """
        if factor is None:
            factor = random.uniform(0.5, 1.5)
        
        degraded = image.astype(np.float32) * factor
        degraded = np.clip(degraded, 0, 255).astype(np.uint8)
        return degraded
    
    def apply_contrast_change(self, image: np.ndarray, factor: Optional[float] = None) -> np.ndarray:
        """
        Change contrast to simulate poor lighting conditions.
        
        Args:
            image: Input image (H, W, C)
            factor: Contrast multiplier (<1 lower contrast, >1 higher). Random if None.
            
        Returns:
            Contrast-adjusted image
        """
        if factor is None:
            factor = random.uniform(0.5, 1.5)
        
        mean = np.mean(image, axis=(0, 1), keepdims=True)
        degraded = (image - mean) * factor + mean
        degraded = np.clip(degraded, 0, 255).astype(np.uint8)
        return degraded
    
    def apply_shadow(self, image: np.ndarray, num_shadows: int = 1) -> np.ndarray:
        """
        Add shadow effects to simulate uneven lighting.
        
        Args:
            image: Input image (H, W, C)
            num_shadows: Number of shadow regions to add
            
        Returns:
            Image with shadows
        """
        degraded = image.copy()
        h, w = degraded.shape[:2]
        
        for _ in range(num_shadows):
            # Random shadow polygon
            num_points = random.randint(3, 6)
            points = np.random.randint(0, max(h, w), (num_points, 2)).astype(np.int32)
            points[:, 0] = np.clip(points[:, 0], 0, w - 1)
            points[:, 1] = np.clip(points[:, 1], 0, h - 1)
            
            # Create shadow mask
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [points], 255)
            
            # Apply shadow (darken region)
            shadow_intensity = random.uniform(0.3, 0.7)
            if len(degraded.shape) == 3:
                for c in range(degraded.shape[2]):
                    degraded[:, :, c] = np.where(mask > 0, 
                                                  degraded[:, :, c] * shadow_intensity, 
                                                  degraded[:, :, c])
            else:
                degraded = np.where(mask > 0, 
                                    degraded * shadow_intensity, 
                                    degraded)
        
        degraded = np.clip(degraded, 0, 255).astype(np.uint8)
        return degraded
    
    def apply_resolution_loss(self, image: np.ndarray, scale_factor: Optional[float] = None) -> np.ndarray:
        """
        Apply resolution loss by downscaling and upscaling to simulate camera distance.
        
        Args:
            image: Input image (H, W, C)
            scale_factor: Scale factor for downscaling (between 2 and 4). Random if None.
            
        Returns:
            Image with resolution loss
        """
        h, w = image.shape[:2]
        
        if scale_factor is None:
            scale_factor = random.uniform(2, 4)
        
        # Downscale
        new_h, new_w = int(h / scale_factor), int(w / scale_factor)
        downscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Upscale back to original dimensions
        upscaled = cv2.resize(downscaled, (w, h), interpolation=cv2.INTER_LINEAR)
        
        return upscaled
    
    def apply_random_degradation(self, image: np.ndarray, 
                                  num_degradations: int = 3) -> np.ndarray:
        """
        Apply a random combination of degradations.
        
        Args:
            image: Input image (H, W, C)
            num_degradations: Number of degradation types to apply
            
        Returns:
            Degraded image
        """
        degradation_funcs = [
            lambda x: self.apply_motion_blur(x),
            lambda x: self.apply_gaussian_blur(x),
            lambda x: self.apply_gaussian_noise(x),
            lambda x: self.apply_salt_pepper_noise(x),
            lambda x: self.apply_jpeg_compression(x, quality=random.randint(30, 80)),
            lambda x: self.apply_brightness_change(x),
            lambda x: self.apply_contrast_change(x),
            lambda x: self.apply_shadow(x, num_shadows=random.randint(1, 2)),
            lambda x: self.apply_resolution_loss(x),
        ]
        
        # Select random degradations
        selected = random.sample(degradation_funcs, min(num_degradations, len(degradation_funcs)))
        
        degraded = image.copy()
        for func in selected:
            degraded = func(degraded)
        
        return degraded


def create_degradation_pipeline(seed: Optional[int] = None) -> DegradationPipeline:
    """
    Factory function to create a degradation pipeline.
    
    Args:
        seed: Random seed for reproducibility
        
    Returns:
        Configured DegradationPipeline instance
    """
    return DegradationPipeline(seed=seed)
