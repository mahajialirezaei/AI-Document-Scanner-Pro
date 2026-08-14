import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout_rate=0.0):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout_rate=dropout_rate)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True, dropout_rate=0.0):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, dropout_rate=dropout_rate)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout_rate=dropout_rate)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class EnhancementUNet(nn.Module):
    """Task 1: Enhancement (U-Net) with Selective Regularization"""
    def __init__(self, n_channels=3, n_classes=3, bilinear=False, dropout_rate=0.0):
        super(EnhancementUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Early layers: No dropout to preserve low-level features and geometric structure
        self.inc = DoubleConv(n_channels, 64, dropout_rate=0.0)
        self.down1 = Down(64, 128, dropout_rate=0.0)
        self.down2 = Down(128, 256, dropout_rate=0.0)
        
        # Deep layers / Bottleneck: Apply dropout for semantic regularization
        self.down3 = Down(256, 512, dropout_rate=dropout_rate)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, dropout_rate=dropout_rate)
        self.up1 = Up(1024, 512 // factor, bilinear, dropout_rate=dropout_rate)
        
        # Late layers: No dropout
        self.up2 = Up(512, 256 // factor, bilinear, dropout_rate=0.0)
        self.up3 = Up(256, 128 // factor, bilinear, dropout_rate=0.0)
        self.up4 = Up(128, 64, bilinear, dropout_rate=0.0)
        self.outc = OutConv(64, n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return self.sigmoid(logits)

class CornerRegressionModel(nn.Module):
    """Task 2: Corner Approach A - Regression"""
    def __init__(self, n_channels=3, dropout_rate=0.0):
        super(CornerRegressionModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 8),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class SoftArgmax2D(nn.Module):
    def __init__(self, train_beta=300.0, eval_beta=10000.0):
        super(SoftArgmax2D, self).__init__()
        self.train_beta = train_beta
        self.eval_beta = eval_beta

    def forward(self, x):
        """
        Args:
            x: heatmaps (B, C, H, W)
        Returns:
            coords: (B, C, 2) in [0, 1] range
        """
        B, C, H, W = x.size()
        x_flat = x.view(B, C, -1)
        
        current_beta = self.train_beta if self.training else self.eval_beta
        
        x_flat_max, _ = torch.max(x_flat, dim=-1, keepdim=True)
        weights = F.softmax(current_beta * (x_flat - x_flat_max), dim=-1)
        
        indices = torch.arange(H * W).to(x.device).float()
        idx_y = indices // W
        idx_x = indices % W
        
        expected_y = (weights * idx_y).sum(dim=-1) / max(1, H - 1)
        expected_x = (weights * idx_x).sum(dim=-1) / max(1, W - 1)
        
        return torch.stack([expected_x, expected_y], dim=-1)

class AddCoords(nn.Module):
    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        batch_size, _, y_dim, x_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        xx_channel = xx_channel.to(input_tensor.device)
        yy_channel = yy_channel.to(input_tensor.device)

        ret = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret

class CornerHeatmapModel(nn.Module):
    """Task 2: Corner Approach B - Heatmap (With CoordConv)"""
    def __init__(self, n_channels=3, n_classes=4, bilinear=False, dropout_rate=0.0):
        super(CornerHeatmapModel, self).__init__()
        self.addcoords = AddCoords(with_r=False)
        self.unet = EnhancementUNet(n_channels + 2, n_classes, bilinear, dropout_rate)
        self.soft_argmax = SoftArgmax2D()

    def forward(self, x):
        x_with_coords = self.addcoords(x) 
        heatmaps = self.unet(x_with_coords) 
        coords = self.soft_argmax(heatmaps) 
        coords = coords.view(coords.size(0), -1)
        return coords, heatmaps