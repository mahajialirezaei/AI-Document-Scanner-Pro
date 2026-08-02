import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_ssim(x: torch.Tensor, y: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """
    Compute Structural Similarity Index (SSIM) between two images with numerical stability.
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    def create_window(window_size: int, sigma: float = 1.5) -> torch.Tensor:
        coords = torch.arange(window_size, dtype=torch.float32, device=x.device) - window_size // 2
        gaussian = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        window = gaussian / gaussian.sum()
        return window.view(1, 1, -1, 1) * window.view(1, 1, 1, -1)
    
    window = create_window(window_size)
    window = window.repeat(x.shape[1], 1, 1, 1)
    
    mu_x = F.conv2d(x, window, padding=window_size // 2, groups=x.shape[1])
    mu_y = F.conv2d(y, window, padding=window_size // 2, groups=y.shape[1])
    
    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y
    
    sigma_x_sq = F.conv2d(x ** 2, window, padding=window_size // 2, groups=x.shape[1]) - mu_x_sq
    sigma_y_sq = F.conv2d(y ** 2, window, padding=window_size // 2, groups=y.shape[1]) - mu_y_sq
    sigma_xy = F.conv2d(x * y, window, padding=window_size // 2, groups=x.shape[1]) - mu_xy
    
    # Clamp variances to prevent negative values due to floating point inaccuracies
    sigma_x_sq = torch.clamp(sigma_x_sq, min=0.0)
    sigma_y_sq = torch.clamp(sigma_y_sq, min=0.0)
    
    numerator = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
    
    ssim_map = numerator / (denominator + 1e-8)
    
    return ssim_map.mean()


class SobelLoss(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)

    def forward(self, x):
        x = x.mean(dim=1, keepdim=True)
        grad_x = F.conv2d(x, self.kernel_x, padding=1)
        grad_y = F.conv2d(x, self.kernel_y, padding=1)
        # Using clamp and stable hypot to prevent NaN gradients at zero
        grad = torch.sqrt(torch.clamp(grad_x**2 + grad_y**2, min=1e-8))
        return grad


class EnhancementLoss(nn.Module):
    def __init__(self, l1_weight=1.0, edge_weight=0.5, ssim_weight=1.0, text_weight=15.0, color_weight=0.2):
        super().__init__()
        self.sobel = SobelLoss()
        self.l1_weight = l1_weight
        self.edge_weight = edge_weight
        self.ssim_weight = ssim_weight
        self.text_weight = text_weight
        self.color_weight = color_weight

    def forward(self, pred, target):
        loss = 0.0
        
        # Ensure inputs are safely clamped
        pred = torch.clamp(pred, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)
        
        grayscale_target = target.mean(dim=1, keepdim=True)
        weight_map = 1.0 + (self.text_weight * torch.pow(torch.clamp(1.0 - grayscale_target, min=0.0), 3))
        
        l1_error = F.smooth_l1_loss(pred, target, reduction='none', beta=0.1)
        weighted_l1 = (l1_error * weight_map).mean()
        
        loss += self.l1_weight * weighted_l1
        
        if self.edge_weight > 0:
            pred_edge = self.sobel(pred)
            target_edge = self.sobel(target)
            edge_error = F.smooth_l1_loss(pred_edge, target_edge, beta=0.1).mean()
            loss += self.edge_weight * edge_error
        
        if self.ssim_weight > 0:
            ssim_loss = 1.0 - compute_ssim(pred, target)
            loss += self.ssim_weight * ssim_loss
            
        if self.color_weight > 0 and pred.shape[1] == 3:
            rg_diff_pred = pred[:, 0, :, :] - pred[:, 1, :, :]
            rb_diff_pred = pred[:, 0, :, :] - pred[:, 2, :, :]
            gb_diff_pred = pred[:, 1, :, :] - pred[:, 2, :, :]
            
            rg_diff_target = target[:, 0, :, :] - target[:, 1, :, :]
            rb_diff_target = target[:, 0, :, :] - target[:, 2, :, :]
            gb_diff_target = target[:, 1, :, :] - target[:, 2, :, :]
            
            color_loss = F.smooth_l1_loss(rg_diff_pred, rg_diff_target) + \
                         F.smooth_l1_loss(rb_diff_pred, rb_diff_target) + \
                         F.smooth_l1_loss(gb_diff_pred, gb_diff_target)
                         
            loss += self.color_weight * color_loss
        
        return loss


class CornerLoss(nn.Module):
    def __init__(self, type='l1'):
        super().__init__()
        self.loss = nn.L1Loss() if type == 'l1' else nn.MSELoss()

    def forward(self, pred, target):
        return self.loss(pred, target)


class HeatmapLoss(nn.Module):
    def __init__(self, alpha=100.0, beta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred_heatmaps, target_heatmaps):
        mse = (pred_heatmaps - target_heatmaps) ** 2
        weights = (self.alpha * target_heatmaps) + self.beta
        weighted_loss = mse * weights
        return weighted_loss.mean()