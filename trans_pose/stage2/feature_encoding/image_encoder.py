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
        feature_map: `[B, C, H', W']` tensor from ImageEncoder.
        uv: `[B, N, 2]` tensor of pixel coordinates (u, v) at the
            input resolution.
        downsample: encoder stride to map pixels into feature-map space.

    Returns:
        Sampled descriptors shaped `[B, N, C]`.
    """    
    B, C, H, W = feature_map.shape
    _, N, _ = uv.shape

    # scale UV to feature map 
    uv_scaled = uv.float() / downsample 
    u_clamped = torch.clamp(uv_scaled[:, :, 0], 0, W - 1)
    v_clamped = torch.clamp(uv_scaled[:, :, 1], 0, H - 1)

    # normalize to [-1, 1] for grid_sample
    grid = torch.zeros(B, N, 1, 2, device=feature_map.device)
    grid[:, :, 0, 0] = 2.0 * u_clamped / (W - 1) - 1.0  # u
    grid[:, :, 0, 1] = 2.0 * v_clamped / (H - 1) - 1.0  # v


    # sample features: [B, C, H, W] --> grid_sample --> [B, C, N, 1]
    sampled = F.grid_sample(
        feature_map,
        grid,
        mode="bilinear",
        padding_mode="zeros",  
        align_corners=True,
    )  # [B, C, N, 1]
    
    # reshape to [B, N, C]
    return sampled.squeeze(-1).transpose(1, 2)

# for image encoder; good to start with 32 channels instead of 64
# --> 5 deep layers but better than normal CNN
class ResidualBlock(nn.Module):
    """ 
    a lighter residual block 
    input -> [Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN] + Shortcut -> ReLU
    """

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        # use shortcut to match dimensions if stride>1 or channel changes
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out
    

class ImageEncoder_Res(nn.Module):
    """ 
    Stem(s2) -> ResBlock(s2) -> ResBlock(s1) -> Project
    stride = 4 overall
    """
    def __init__(self, out_channels: int = 128):
        super().__init__()
        # --1-- downsample by 2 -> H/2 
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # --2-- downsample by 2 -> H/4
        self.layer1 = ResidualBlock(32, 64, stride=2)

        # --3-- feature refinement at H/4
        self.layer2 = ResidualBlock(64, 64, stride=1)

        # --4-- projection
        self.final = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        squeeze_batch = False
        if rgb.dim() == 3:
            rgb = rgb.unsqueeze(0)
            squeeze_batch = True
        
        x = self.stem(rgb) # [B, 32, H/2, W/2]
        x = self.layer1(x) # [B, 64, H/4, W/4]
        x = self.layer2(x) # [B, 64, H/4, W/4]
        x = self.final(x) # [B, out, H/4, W/4]

        return x.squeeze(0) if squeeze_batch else x

