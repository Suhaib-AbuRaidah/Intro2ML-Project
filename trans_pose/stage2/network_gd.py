"""
Multi-Object 6D Pose Estimation Network

Key Differences from network.py:
1. PointSegHead outputs NUM_CLASSES (multi-class segmentation)
2. Uses forward_multi() from DenseFusion (returns point labels)
3. Handles multiple objects in one forward pass
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from trans_pose.stage2.feature_encoding.dense_fusion import DenseFusion


class PointSegHeadMulti(nn.Module):
    """
    Multi-class segmentation head.
    
    Classifies each point as background (0) or one of NUM_OBJECTS object IDs.
    
    Args:
        in_dim: Input feature dimension (e.g., 448)
        num_classes: Number of classes (60 objects + 1 background = 61)
    
    Returns:
        logits: [B, N, num_classes] raw class scores
    """
    def __init__(self, in_dim: int, num_classes: int = 61):
        super().__init__()
        self.num_classes = num_classes
        
        self.mlp = nn.Sequential(
            nn.Conv1d(in_dim, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            # MULTI-CLASS: Output num_classes channels
            nn.Conv1d(128, num_classes, 1)
        )

    def forward(self, x):
        """
        Args:
            x: [B, N, F] fused features
        Returns:
            logits: [B, N, num_classes]
        """
        # x: [B, N, F] -> [B, F, N]
        x = x.permute(0, 2, 1)
        out = self.mlp(x)
        # out: [B, num_classes, N] -> [B, N, num_classes]
        return out.permute(0, 2, 1)


class OffsetHead(nn.Module):
    """
    Offset prediction head (SAME as original).
    
    Predicts 3D offsets from each point to K keypoints.
    These offsets are object-agnostic (just 3D vectors).
    The segmentation head tells us which object each point belongs to.
    """
    def __init__(self, in_dim: int, num_keypoints: int):
        super().__init__()
        self.num_keypoints = num_keypoints

        self.mlp = nn.Sequential(
            nn.Conv1d(in_dim, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv1d(128, num_keypoints * 3, 1)
        )

    def forward(self, x):
        """
        Args:
            x: [B, N, F]
        Returns:
            offsets: [B, N, K, 3]
        """
        B, N, _ = x.shape
        x = x.permute(0, 2, 1)
        out = self.mlp(x)
        # out: [B, K*3, N] -> [B, N, K*3]
        out = out.permute(0, 2, 1)
        return out.reshape(B, N, self.num_keypoints, 3)


class TransPoseNetworkMulti(nn.Module):
    """
    Multi-Object 6D Pose Estimation Network.
    
    Processes ALL objects in a scene simultaneously.
    
    Args:
        img_outdim: RGB encoder output channels
        normals_outdim: Normal encoder output channels
        points_outdim: PointNet output channels
        num_keypoints: Number of keypoints per object
        num_classes: Total number of classes (60 objects + 1 background)
    """
    def __init__(
        self,
        img_outdim: int = 128,
        normals_outdim: int = 64,
        points_outdim: int = 256,
        num_keypoints: int = 10,
        num_classes: int = 61,  # 60 objects + background
        **kwargs
    ):
        super().__init__()
        
        # Feature fusion (uses forward_multi)
        self.features = DenseFusion(
            image_channels=img_outdim,
            normal_channels=normals_outdim,
            pointnet_channels=points_outdim,
            num_samples=4096
        )
        
        feature_outdim = img_outdim + normals_outdim + points_outdim
        
        # Multi-class segmentation head
        self.seg_head = PointSegHeadMulti(feature_outdim, num_classes=num_classes)
        
        # Offset head (same as before)
        self.offset_head = OffsetHead(feature_outdim, num_keypoints)
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, img, normals, depth_c, o_mask, intrinsics):
        """
        Forward pass for multi-object pose estimation.
        
        Args:
            img: [B, 3, H, W] RGB images
            normals: [B, 3, H, W] surface normals
            depth_c: [B, 1, H, W] depth maps
            o_mask: [B, 1, H, W] instance masks with object IDs
            intrinsics: (fx, fy, cx, cy) camera intrinsics
        
        Returns:
            seg_logits: [B, N, num_classes] segmentation logits
            offsets: [B, N, K, 3] offset predictions
            points: [B, N, 3] 3D point cloud
            trans_feat: [B, 64, 64] PointNet transform
            point_labels: [B, N] ground truth object ID for each point
        """
        # Use forward_multi to get point labels
        features, points, trans_feat, point_labels = self.features.forward_multi(
            img, depth_c, normals, o_mask, intrinsics
        )
        
        # Predict segmentation and offsets
        seg_logits = self.seg_head(features)  # [B, N, num_classes]
        offsets = self.offset_head(features)  # [B, N, K, 3]
        
        return seg_logits, offsets, points, trans_feat, point_labels