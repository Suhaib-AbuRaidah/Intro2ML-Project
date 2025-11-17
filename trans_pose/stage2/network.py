import torch
import torch.nn as nn
import torch.nn.functional as F

from trans_pose.stage2.feature_encoding.dense_fusion import DenseFusion

class PointSegHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Linear(128, num_classes)          
        )

    def forward(self, x):
        B, N, F = x.shape
        x = x.reshape(B * N, F)
        out = self.mlp(x)
        return out.reshape(B, N, -1)   # class per point

class OffsetHead(nn.Module):
    def __init__(self, in_dim, num_keypoints):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_keypoints * 3)
        )

    def forward(self, x):
        # x: (B, N, F)
        B, N, F = x.shape
        out = self.mlp(x)             # (B, N, K*3)
        return out.reshape(B, N, -1, 3)  # (B, N, K, 3)
    
class TransPoseNetwork(nn.Module):
    def __init__(self, img_outdim=128,normals_outdim=128,points_outdim=256,num_classes=4, num_keypoints=10):
        super().__init__()
        self.features = DenseFusion(image_channels=img_outdim,
                                    normal_channels=normals_outdim,
                                    pointnet_channels=points_outdim,
                                    num_samples=1024)
        
        feature_outdim = img_outdim + normals_outdim + points_outdim
        self.seg_head = PointSegHead(feature_outdim, num_classes)
        self.offset_head = OffsetHead(feature_outdim, num_keypoints)

    def forward(self, img,normals,depth_c,mask_inst,intrinsics):
        
        # points: (B, N, 3) or (B, N, feat)
        features,points= self.features(img,normals,depth_c,mask_inst,intrinsics)  # (B, N, F)
        features=features.unsqueeze(0)
        seg_logits = self.seg_head(features)  # (B, N, C)
        offsets = self.offset_head(features)  # (B, N, K, 3)
        return seg_logits, offsets,points