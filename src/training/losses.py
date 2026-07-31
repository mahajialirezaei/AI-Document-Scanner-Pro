import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_ssim(x: torch.Tensor, y: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """
    Compute Structural Similarity Index (SSIM) between two images.
    
    Args:
        x: First image tensor (B, C, H, W)
        y: Second image tensor (B, C, H, W)
        window_size: Size of the Gaussian window for SSIM computation
        
    Returns:
        SSIM value (scalar)
    """
    # Parameters for SSIM
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Create Gaussian window
    def create_window(window_size: int, sigma: float = 1.5) -> torch.Tensor:
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        gaussian = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        window = gaussian / gaussian.sum()
        return window.view(1, 1, -1, 1) * window.view(1, 1, 1, -1)
    
    window = create_window(window_size).to(x.device)
    window = window.repeat(x.shape[1], 1, 1, 1)
    
    # Compute means
    mu_x = F.conv2d(x, window, padding=window_size // 2, groups=x.shape[1])
    mu_y = F.conv2d(y, window, padding=window_size // 2, groups=y.shape[1])
    
    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y
    
    # Compute variances and covariance
    sigma_x_sq = F.conv2d(x ** 2, window, padding=window_size // 2, groups=x.shape[1]) - mu_x_sq
    sigma_y_sq = F.conv2d(y ** 2, window, padding=window_size // 2, groups=y.shape[1]) - mu_y_sq
    sigma_xy = F.conv2d(x * y, window, padding=window_size // 2, groups=x.shape[1]) - mu_xy
    
    # Compute SSIM
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
    
    return ssim_map.mean()


class SobelLoss(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.mean(dim=1, keepdim=True)
        grad_x = F.conv2d(x, self.kernel_x, padding=1)
        grad_y = F.conv2d(x, self.kernel_y, padding=1)
        grad = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        return grad

class EnhancementLoss(nn.Module):
    def __init__(self, l1_weight=1.0, edge_weight=0.1, ssim_weight=0.5):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.sobel = SobelLoss()
        self.l1_weight = l1_weight
        self.edge_weight = edge_weight
        self.ssim_weight = ssim_weight

    def forward(self, pred, target):
        loss = self.l1_weight * self.l1(pred, target)
        
        if self.edge_weight > 0:
            pred_edge = self.sobel(pred)
            target_edge = self.sobel(target)
            loss += self.edge_weight * self.l1(pred_edge, target_edge)
        
        if self.ssim_weight > 0:
            # SSIM is a similarity measure, so we use 1 - SSIM as loss
            ssim_loss = 1 - compute_ssim(pred, target)
            loss += self.ssim_weight * ssim_loss
        
        return loss

class CornerLoss(nn.Module):
    def __init__(self, type='l1'):
        super().__init__()
        self.loss = nn.L1Loss() if type == 'l1' else nn.MSELoss()

    def forward(self, pred, target):
        return self.loss(pred, target)

class HeatmapLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred_heatmaps, target_heatmaps):
        return self.mse(pred_heatmaps, target_heatmaps)
