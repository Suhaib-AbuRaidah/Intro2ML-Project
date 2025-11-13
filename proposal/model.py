import torch
import torch.nn as nn
import torchvision.models as models

class PoseNet(nn.Module):
    def __init__(self, pretrained=True, num_objects=4, use_depth=True):
        super().__init__()
        self.use_depth = use_depth

        # Base encoder (ResNet18)
        self.rgb_encoder = models.resnet18(pretrained=pretrained)
        self.rgb_encoder.fc = nn.Identity()

        if use_depth:
            # Copy structure for depth but single channel
            self.depth_encoder = models.resnet18(pretrained=False)
            self.depth_encoder.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            self.depth_encoder.fc = nn.Identity()
            fusion_dim = 512 * 2
        else:
            self.depth_encoder = None
            fusion_dim = 512

        # Regression heads
        self.fc_rot = nn.Linear(fusion_dim, num_objects * 4)
        self.fc_trans = nn.Linear(fusion_dim, num_objects * 3)

    def forward(self, rgb, depth=None):
        feat_rgb = self.rgb_encoder(rgb)
        if self.use_depth and depth is not None:
            feat_depth = self.depth_encoder(depth)
            feat = torch.cat([feat_rgb, feat_depth], dim=1)
        else:
            feat = feat_rgb

        rot = self.fc_rot(feat).view(-1, 4, 4)
        trans = self.fc_trans(feat).view(-1, 4, 3)
        rot = torch.nn.functional.normalize(rot, dim=-1)
        return rot, trans
