import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, l1_weight=1.0, edge_weight=0.1):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.sobel = SobelLoss()
        self.l1_weight = l1_weight
        self.edge_weight = edge_weight

    def forward(self, pred, target):
        loss = self.l1_weight * self.l1(pred, target)
        if self.edge_weight > 0:
            pred_edge = self.sobel(pred)
            target_edge = self.sobel(target)
            loss += self.edge_weight * self.l1(pred_edge, target_edge)
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
