import torch
import torch.nn as nn
import torch.nn.functional as F

# lightweight 3-layer CNN with bacth normalzer and ReLU activations 
# accept single image [3xHxW] or batch of images [Bx3xHxW]
# return feature map [CxH'xW'] or [BxCxH'xW'] with stride 4 and H'=H/4, W'=W/4
class ImageEncoder(nn.Module):
    def __init__(self, out_channels: int = 128):
        super().__init__()
        # apply 3x3 conv with stride 2 ++ BN and ReLU 
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        # final conv layer to get desired output channels
        # stride 1 to keep spatial resolution and output C channels
        self.block3 = nn.Sequential(
            nn.Conv2d(128, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb: `[3, H, W]` or `[B, 3, H, W]` float tensor (already normalized
                 upstream by Stage-1 transforms).
        Returns:
            Feature map with spatial resolution reduced by 4 (stride-4 output).
        """
        # squeeze batch dimension if input was single image
        squeeze_batch = rgb.dim() == 3
        if squeeze_batch:
            rgb = rgb.unsqueeze(0)

        feats = self.block3(self.block2(self.block1(rgb)))
        return feats.squeeze(0) if squeeze_batch else feats



# maps original pixel coordinates to encoder space and normalizes them 
# returns [*, N, C] descsriptors  
def sample_2d_features(
    feature_map: torch.Tensor,
    uv: torch.Tensor,
    downsample: int = 4,
) -> torch.Tensor:
    """
    Bilinearly samples encoder features at pixel locations defined in the
    original Stage-1 image coordinate frame.

    Args:
        feature_map: `[C, H', W']` or `[B, C, H', W']` tensor from ImageEncoder.
        uv: `[N, 2]` or `[B, N, 2]` tensor of pixel coordinates (u, v) at the
            input resolution.
        downsample: encoder stride to map pixels into feature-map space.

    Returns:
        Sampled descriptors shaped `[N, C]` or `[B, N, C]`.
    """    
    squeeze_batch = feature_map.dim() == 3
    if squeeze_batch:
        feature_map = feature_map.unsqueeze(0)
    if uv.dim() == 2:
        uv = uv.unsqueeze(0)

    B, N, _ = uv.shape
    device = feature_map.device
    uv = uv.to(device).float() / downsample  # ← ADD .float()

    h, w = feature_map.shape[-2:]
    if h < 2 or w < 2:
        raise ValueError("Feature map too small for sampling.")

    # Normalize coordinates to [-1, 1] for grid_sample (x = u, y = v)
    x_norm = (uv[..., 0] / (w - 1)) * 2 - 1
    y_norm = (uv[..., 1] / (h - 1)) * 2 - 1
    grid = torch.stack([x_norm, y_norm], dim=-1).view(B, N, 1, 2)

    sampled = F.grid_sample(
        feature_map,
        grid,
        mode="bilinear",
        padding_mode="zeros",  # ← Consider "border" instead
        align_corners=True,
    )  # [B, C, N, 1]
    sampled = sampled.squeeze(-1).permute(0, 2, 1).contiguous()  # [B, N, C]

    if squeeze_batch:
        sampled = sampled.squeeze(0)
    return sampled