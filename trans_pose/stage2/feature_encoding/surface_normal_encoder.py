import torch
import torch.nn as nn


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