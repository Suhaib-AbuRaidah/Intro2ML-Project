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
import torch.nn.functional as F
from typing import Tuple

from .image_encoder import ImageEncoder, sample_2d_features, ImageEncoder_Res
from .surface_normal_encoder import NormalEncoder, NormalEncoder_Res
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
        pointnet_channels: int = 256,
        num_samples: int = 4096,
    ):
        super().__init__()
        self.num_samples = num_samples
        
        self.image_encoder = ImageEncoder(out_channels=image_channels)
        self.normal_encoder = NormalEncoder(out_channels=normal_channels)
        self.point_encoder = PointNetBackbone(out_dim=pointnet_channels)
        
        self.total_dim = image_channels + normal_channels + pointnet_channels
    def forward(
        self,
        rgb: torch.Tensor,
        depth_c: torch.Tensor,
        normals: torch.Tensor,
        o_mask: torch.Tensor,    # This is the instance mask
        intrinsics: Tuple[float, float, float, float],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            rgb: [B, 3, H, W]
            depth_c: [B, 1, H, W]
            normals: [B, 3, H, W]
            o_mask: [B, 1, H, W] (instance mask with object IDs)
            intrinsics: (fx, fy, cx, cy)
        Returns:
            fused_features: [B, N, total_dim]
            points_xyz: [B, N, 3]
        """
        # Handle unbatched input [C, H, W] -> [1, C, H, W]
        if rgb.dim() == 3:
            rgb = rgb.unsqueeze(0)
            depth_c = depth_c.unsqueeze(0)
            normals = normals.unsqueeze(0)
            o_mask = o_mask.unsqueeze(0) # Changed binary_mask to o_mask

        # encode 2D images (vectorized over batch)
        feats_rgb = self.image_encoder(rgb)        # [B, 128, H/4, W/4]
        feats_normal = self.normal_encoder(normals) # [B, 64, H/4, W/4]

        # build point clouds(looping over batch because point sampling is random/variable)
        points_list = []
        uv_list = []

        # how to define B
        # ghina - review again
        B = rgb.shape[0] 

        for i in range(B):
            # build instance points hanfles [1, H, W] inputs 
            points_xyz, uv = build_instance_points(
                depth_c=depth_c[i:i+1], # This is fine
                mask_inst=o_mask[i:i+1], # Pass the instance mask here
                intrinsics=intrinsics,
                num_samples=self.num_samples,
            )  # points_xyz: [N, 3], uv: [N, 2]
            points_list.append(points_xyz)
            uv_list.append(uv)

        points_xyz = torch.stack(points_list, dim=0)  # [B, N, 3]
        uv = torch.stack(uv_list, dim=0)              # [B, N, 2]

        # sample 2D features at 3D point projections
        rgb_pts = sample_2d_features(feats_rgb, uv, downsample=4)      # [B, N, 128]
        normal_pts = sample_2d_features(feats_normal, uv, downsample=4) # [B, N, 64]


        # encode 3D geometry (vectorized over batch)
        point_feats, trans_feat = self.point_encoder(points_xyz)  # [B, N, 256] 
        # is it 256 or 512? -- ghina - review again 
        fused_features = torch.cat([
            point_feats,   # [B, N, 256]
            rgb_pts,       # [B, N, 128]
            normal_pts,    # [B, N, 64]
        ], dim=-1)  # [B, N, 448]

        return fused_features, points_xyz, trans_feat

# model = DenseFusion(image_channels=128, normal_channels=64, pointnet_channels=512, num_samples=4096)

class DenseFusion_Res(nn.Module):
    def __init__(
        self,
        image_channels: int = 128,
        normal_channels: int = 64,
        pointnet_channels: int = 256,
        num_samples: int = 4096,
    ):
        
        super().__init__()
        self.num_samples = num_samples

        self.image_encoder = ImageEncoder_Res(out_channels= image_channels)
        self.normal_encoder = NormalEncoder_Res(out_channels= normal_channels)
        self.point_encoder = PointNetBackbone(out_dim= pointnet_channels)

        self.total_dim = image_channels + normal_channels + pointnet_channels

    def forward(
        self,
        rgb: torch.Tensor,
        depth_c: torch.Tensor,
        normals: torch.Tensor,
        o_mask: torch.Tensor,    # This is the instance mask
        intrinsics: Tuple[float, float, float, float],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
        # Handle unbatched input [C, H, W] -> [1, C, H, W]
        if rgb.dim() == 3:
            rgb = rgb.unsqueeze(0)
            depth_c = depth_c.unsqueeze(0)
            normals = normals.unsqueeze(0)
            o_mask = o_mask.unsqueeze(0) 
 
        feats_rgb = self.image_encoder(rgb)        # [B, 128, H/4, W/4]
        feats_normal = self.normal_encoder(normals) # [B, 64, H/4, W/4]

        # build point clouds(looping over batch because point sampling is random/variable)
        points_list = []
        uv_list = [] 
        B = rgb.shape[0] # read batch size from input tensor 

        for i in range(B): 
            points_xyz, uv = build_instance_points(
                depth_c=depth_c[i:i+1], # This is fine
                mask_inst=o_mask[i:i+1], # Pass the instance mask here
                intrinsics=intrinsics,
                num_samples=self.num_samples,
            )  # points_xyz: [N, 3], uv: [N, 2]
            points_list.append(points_xyz)
            uv_list.append(uv)

        points_xyz = torch.stack(points_list, dim=0)  
        uv = torch.stack(uv_list, dim=0)

        rgb_pts = sample_2d_features(feats_rgb, uv, downsample=4)
        normal_pts = sample_2d_features(feats_normal, uv, downsample=4)

        point_feats, trans_feat = self.point_encoder(points_xyz)

        fused_features = torch.cat([
            point_feats,
            rgb_pts,
            normal_pts,
        ], dim=-1)

        return fused_features, points_xyz, trans_feat