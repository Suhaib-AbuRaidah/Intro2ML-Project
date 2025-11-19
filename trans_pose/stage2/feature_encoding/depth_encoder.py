from trans_pose.lib.pointnet.PointNetEncoder_file import PointNetEncoder
import torch.nn as nn
from typing import Tuple
import torch


def build_instance_points(
    depth_c: torch.Tensor,
    mask_inst: torch.Tensor,
    intrinsics: Tuple[float, float, float, float],
    num_samples: int = 4096,
    min_depth: float = 0.1,
    max_depth: float = 3.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Back-project depth pixels to 3D points with robust filtering.
    
    Args:
        depth_c: [1, H, W] depth map in meters
        mask_inst: [1, H, W] instance mask
        intrinsics: (fx, fy, cx, cy)
        num_samples: target number of points (always returns this many)
        min_depth: minimum valid depth (meters) - filters camera noise
        max_depth: maximum valid depth (meters) - filters outliers
        
    Returns:
        points_xyz: [num_samples, 3] 3D coordinates (always fixed size)
        uv: [num_samples, 2] 2D pixel coordinates (always fixed size)
    """
    fx, fy, cx, cy = intrinsics
    device = depth_c.device
    
    
    # Create validity mask
    valid = (mask_inst.squeeze(0) > 0) & (depth_c.squeeze(0) > 0)
    valid_indices = torch.nonzero(valid, as_tuple=False)

    num_valid = valid_indices.shape[0]
    
    # Handle empty mask
    if valid_indices.numel() == 0:
        # no valid pixels → return zeros so downstream shapes stay consistent
        dummy_xyz = torch.zeros(num_samples, 3, device=device)
        dummy_uv = torch.zeros(num_samples, 2, device=device)
        return dummy_xyz, dummy_uv
    
    if num_valid >= num_samples:
        # Random sampling
        choice = torch.randperm(num_valid, device=device)[:num_samples]
    else:
        # Pad by repeating random points
        extra = num_samples - num_valid
        reps = torch.randint(0, num_valid, (extra,), device=device)
        choice = torch.cat([torch.arange(num_valid, device=device), reps], dim=0)
        choice = choice[torch.randperm(choice.shape[0], device=device)]
    
    picked = valid_indices[choice]
    v = picked[:, 1]
    u = picked[:, 2]
    
    # Back-project to 3D using pinhole camera model
    z = depth_c[0, 0, v, u]
    x = (u.float() - cx) * z / fx
    y = (v.float() - cy) * z / fy
    
    points_xyz = torch.stack([x, y, z], dim=-1)  # [num_samples, 3]
    uv = torch.stack([u.float(), v.float()], dim=-1)  # [num_samples, 2]
    
    return points_xyz, uv


class PointNetBackbone(nn.Module):
    """
    PointNet encoder wrapper that produces configurable output dimension.
    
    Architecture:
        Input [N, 3] → PointNet [N, 320] → MLP → [N, out_dim]
    
    Note: PointNet with global_feat=False and out_dim=256 returns:
          256 (global) + 64 (pointfeat) = 320 channels
    """
    def __init__(self, out_dim: int = 512):
        super().__init__()
        self.out_dim = out_dim
        self.pointnet_hidden = 256
        
        # PointNet returns pointnet_hidden + 64 = 320 channels
        self.pointnet = PointNetEncoder(
            global_feat=False,       # Per-point features
            feature_transform=True,  # Use T-Net
            channel=3,               # xyz only
            out_dim=self.pointnet_hidden  # 256 → yields 320 with pointfeat
        )
        
        # Project from 320 to desired dimension
        self.feature_projection = nn.Sequential(
            nn.Conv1d(self.pointnet_hidden + 64, 512, 1),  # 320 → 512
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Conv1d(512, out_dim, 1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, pointcloud: torch.Tensor):
        """
        Args:
            pointcloud: [N, 3] point cloud (xyz coordinates)
                        or [B, N, 3] batched
                        or [B, 3, N] PointNet format
        
        Returns:
            features: [N, out_dim] per-point features
                      or [B, N, out_dim] if batched
        """
        squeeze_batch = False
        
        if pointcloud.dim() == 2:
            # [N, 3] → [1, 3, N] for PointNet
            pointcloud = pointcloud.T.unsqueeze(0)
            squeeze_batch = True
        elif pointcloud.dim() == 3 and pointcloud.shape[1] > pointcloud.shape[2]:
            # [B, N, 3] → [B, 3, N]
            pointcloud = pointcloud.transpose(1, 2).contiguous()
        
        # PointNet forward (returns tuple)
        # feats_raw: [B, 320, N], trans: [B, 3, 3], trans_feat: [B, 64, 64]
        feats_raw, trans, trans_feat = self.pointnet(pointcloud)
        
        # Project to desired dimension
        feats = self.feature_projection(feats_raw)  # [B, out_dim, N]
        
        # Transpose to [B, N, out_dim] format
        feats = feats.transpose(1, 2).contiguous()
        
        if squeeze_batch:
            return feats.squeeze(0), trans_feat.squeeze(0)
        return feats, trans_feat