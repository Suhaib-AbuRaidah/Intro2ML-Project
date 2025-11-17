# we are using for first training PointNet instead of PointNet++
# todo: check differences later and see if worth using PointNet++
from trans_pose.lib.pointnet.PointNetEncoder_file import PointNetEncoder
import torch.nn as nn
from typing import Tuple
import torch


def build_instance_points(
    depth_c: torch.Tensor,
    mask_inst: torch.Tensor,
    intrinsics: Tuple[float, float, float, float],
    num_samples: int = 4096,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Back-project depth pixels to 3D points.
    
    Args:
        depth_c: [1, H, W] depth map
        mask_inst: [1, H, W] instance mask
        intrinsics: (fx, fy, cx, cy)
        num_samples: number of points to sample
        
    Returns:
        points_xyz: [N, 3] 3D coordinates
        uv: [N, 2] 2D pixel coordinates
    """
    fx, fy, cx, cy = intrinsics
    device = depth_c.device
    mask = mask_inst.squeeze(0)
    valid_indices = torch.nonzero(mask > 0, as_tuple=False)
    
    if valid_indices.shape[0] == 0:
        # Return dummy point (will be masked in loss)
        dummy_xyz = torch.zeros(1, 3, device=device)
        dummy_uv = torch.zeros(1, 2, device=device)
        return dummy_xyz, dummy_uv
        
    if valid_indices.shape[0] > num_samples:
        sampled_idx = torch.randperm(valid_indices.shape[0], device=device)[:num_samples]
        valid_indices = valid_indices[sampled_idx]
    
    v = valid_indices[:, 0]
    u = valid_indices[:, 1]
    
    z = depth_c[0, v, u]
    x = (u.float() - cx) * z / fx
    y = (v.float() - cy) * z / fy
    points_xyz = torch.stack([x, y, z], dim=1)
    uv = torch.stack([u.float(), v.float()], dim=1)
    
    return points_xyz, uv


class PointNetBackbone(nn.Module):
    """
    PointNet encoder wrapper that produces configurable output dimension.
    
    Architecture:
        Input [N, 3] → PointNet [N, 1088] → MLP → [N, out_dim]
    
    Note: gnn_PointNetEncoder.py cannot be modified, so we handle its
          fixed 1088-dim output here.
    """
    def __init__(self, out_dim: int = 512):
        """
        Args:
            out_dim: Desired output feature dimension per point.
                     Common choices: 256, 512, 1024
        """
        super().__init__()
        self.out_dim = out_dim
        
        # PointNet with fixed output (returns 1088 when global_feat=False)
        self.pointnet = PointNetEncoder(
            global_feat=False,       
            feature_transform=True,  
            channel=3,      
            out_dim=1024       
        )
        # When global_feat=False, actual output is 1024 + 64 = 1088


        # # Disable BN inside STN when batch size = 1
        # for m in self.pointnet.modules():
        #     if isinstance(m, nn.BatchNorm1d):
        #         m.track_running_stats = False
        #         m.affine = True  # allow learning gamma/beta
        #         m.eval()

        #         def _disable_bn(module, x, y=None):
        #             module.training = False
        #         m.register_forward_pre_hook(_disable_bn)


        # Project from PointNet's 1088 to desired dimension
        self.feature_projection = nn.Sequential(
            nn.Conv1d(1088, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Conv1d(512, out_dim, 1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, pointcloud: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pointcloud: [N, 3] point cloud (xyz coordinates)
                        or [B, N, 3] batched
                        or [B, 3, N] PointNet format
        
        Returns:
            features: [N, out_dim] per-point features
                      or [B, N, out_dim] if batched
        """
        # Handle different input formats
        squeeze_batch = False
        
        if pointcloud.dim() == 2:
            # [N, 3] → [1, 3, N] for PointNet
            pointcloud = pointcloud.T.unsqueeze(0)
            squeeze_batch = True
        elif pointcloud.dim() == 3 and pointcloud.shape[1] > pointcloud.shape[2]:
            # [B, N, 3] → [B, 3, N]
            pointcloud = pointcloud.transpose(1, 2).contiguous()
        
        # PointNet forward (returns tuple)
        # feats: [B, 1088, N], trans: [B, 3, 3], trans_feat: [B, 64, 64]
        feats, trans, trans_feat = self.pointnet(pointcloud)
        
        # Project to desired dimension
        feats = self.feature_projection(feats)  # [B, out_dim, N]
        
        # Transpose to [B, N, out_dim] format
        feats = feats.transpose(1, 2).contiguous()
        
        # Remove batch dim if input was unbatched
        if squeeze_batch:
            feats = feats.squeeze(0)  # [N, out_dim]
        
        return feats