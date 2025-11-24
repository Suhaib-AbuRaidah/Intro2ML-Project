import torch
import torch.nn as nn
from .image_encoder import ResidualBlock


class NormalEncoder(nn.Module):
    """
    Surface-normal feature extractor.
    keeps the channel budget lower (normals are already geometry-heavy) 
    -> outputs stride-4 maps aligned with the original image coordinates

    Input tensors come from Stage-1 normal estimation and follow `[3, H, W]`
    (or `[B, 3, H, W]`) with values in `[-1, 1]`.
    """

    def __init__(self, out_channels: int = 64):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, normals: torch.Tensor) -> torch.Tensor:
        """
        Args:
            normals: `[3, H, W]` or `[B, 3, H, W]` tensor of unit normals.
        Returns:
            `[C_norm, H/4, W/4]` or `[B, C_norm, H/4, W/4]` stride-4 features.
        """
        squeeze_batch = normals.dim() == 3
        if squeeze_batch:
            normals = normals.unsqueeze(0)

        feats = self.block3(self.block2(self.block1(normals)))
        return feats.squeeze(0) if squeeze_batch else feats

class NormalEncoder_Res(nn.Module):
    """
    starts with only 16 channels 
    """
    def __init__(self, out_channels: int = 64):
        super().__init__()
        # --1-- downsample by 2 -> H/2 
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        # --2-- downsample by 2 -> H/4
        self.layer1 = ResidualBlock(16, 32, stride=2)

        # --3-- feature refinement at H/4
        self.layer2 = ResidualBlock(32, 32, stride=1)

        # --4-- projection
        self.final = nn.Sequential(
            nn.Conv2d(32, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, normals: torch.Tensor) -> torch.Tensor:
        squeeze_batch = False
        if normals.dim() == 3:
            normals = normals.unsqueeze(0)
            squeeze_batch = True
        
        x = self.stem(normals) # [B, 16, H/2, W/2]
        x = self.layer1(x) # [B, 32, H/4, W/4]
        x = self.layer2(x) # [B, 32, H/4, W/4]
        x = self.final(x) # [B, out, H/4, W/4]

        return x.squeeze(0) if squeeze_batch else x