# goal: concatenate featurs all together
# but we need to align them together 
# image/normal featyre are 2D grids and point features are sparse 3D points 
#
# given that the build_instance points can return the point cloud with 
# its corresponding 2D pixel from RGB image 

# Image Encoder:	in:[3, 480, 640]	out:[128, 120, 160]	ENTIRE scene
# Normal Encoder:	in:[3, 480, 640]	out:[64, 120, 160]	ENTIRE scene
# Point Sampling:	in:uv [N, 2]	out:[N, 128], [N, 64]	Only object pixels
# PointNet:	in:[N, 3]	out:[N, 512]	Only object 3D points
import torch
import torch.nn as nn
from typing import Tuple

from .image_encoder import ImageEncoder, sample_2d_features
from .surface_normal_encoder import NormalEncoder
from .depth_encoder import PointNetBackbone, build_instance_points


class DenseFusion(nn.Module):
    """
    Multi-modal feature fusion for 6D pose estimation.
    
    Flow:
    1. Encode FULL RGB image → [128, H/4, W/4]
    2. Encode FULL normal map → [64, H/4, W/4]
    3. Build point cloud from depth+mask → [N, 3] + uv [N, 2]
    4. Sample RGB/normal features at uv locations → [N, 128], [N, 64]
    5. Encode point cloud → [N, 512]
    6. Concatenate all → [N, 704]
    """
    def __init__(
        self,
        image_channels: int = 128,
        normal_channels: int = 64,
        pointnet_channels: int = 512,
        num_samples: int = 4096,
    ):
        super().__init__()
        self.num_samples = num_samples
        
        self.image_encoder = ImageEncoder(out_channels=image_channels)
        self.normal_encoder = NormalEncoder(out_channels=normal_channels)
        self.point_encoder = PointNetBackbone()
        
        self.total_dim = image_channels + normal_channels + pointnet_channels
    def forward(
        self,
        rgb: torch.Tensor,
        depth_c: torch.Tensor,
        normals: torch.Tensor,
        mask_inst: torch.Tensor,
        intrinsics: Tuple[float, float, float, float],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Handle batching
        squeeze_batch = rgb.dim() == 3
        if squeeze_batch:
            rgb = rgb.unsqueeze(0)
            depth_c = depth_c.unsqueeze(0)
            normals = normals.unsqueeze(0)
            mask_inst = mask_inst.unsqueeze(0)
        
        # batch_size=1
        if rgb.shape[0] != 1:
            raise NotImplementedError("Batch size > 1 not yet supported. Process samples individually.")
        
        # Remove batch dimension for processing
        rgb = rgb.squeeze(0)
        depth_c = depth_c.squeeze(0)
        normals = normals.squeeze(0)
        mask_inst = mask_inst.squeeze(0)
        
        # Encode FULL images
        feats_rgb = self.image_encoder(rgb)        # [128, H/4, W/4]
        feats_normal = self.normal_encoder(normals) # [64, H/4, W/4]
        
        # Build point cloud from masked depth
        points_xyz, uv = build_instance_points(
            depth_c=depth_c,
            mask_inst=mask_inst,
            intrinsics=intrinsics,
            num_samples=self.num_samples,
        )  # points_xyz: [N, 3], uv: [N, 2]
        
        # Sample 2D features AT the point locations
        rgb_pts = sample_2d_features(feats_rgb, uv, downsample=4)      # [N, 128]
        normal_pts = sample_2d_features(feats_normal, uv, downsample=4) # [N, 64]
        
        # Encode 3D geometry
        point_feats = self.point_encoder(points_xyz)  # [N, 512]
        
        # Concatenate
        fused_features = torch.cat([
            point_feats,   # [N, 512]
            rgb_pts,       # [N, 128]
            normal_pts,    # [N, 64]
        ], dim=-1)  # [N, 704]
        
        return fused_features, points_xyz
# model = DenseFusion(image_channels=128, normal_channels=64, pointnet_channels=512, num_samples=4096)
