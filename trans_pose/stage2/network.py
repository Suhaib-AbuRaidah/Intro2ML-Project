import torch
import torch.nn as nn
import torch.nn.functional as F

from trans_pose.stage2.feature_encoding.dense_fusion import DenseFusion

class PointSegHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_dim, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Conv1d(128, num_classes, 1)
        )

    def forward(self, x):
        # x: [B, N, F] -> [B, F, N]
        x = x.permute(0, 2, 1)
        out = self.mlp(x)
        # out: [B, C, N] -> [B, N, C]
        return out.permute(0, 2, 1)


class OffsetHead(nn.Module):
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
        # x: [B, N, F]
        B, N, _ = x.shape
        x = x.permute(0, 2, 1)
        out = self.mlp(x)
        # out: [B, K*3, N] -> [B, N, K*3]
        out = out.permute(0, 2, 1)
        return out.reshape(B, N, self.num_keypoints, 3) # [B, N, K, 3]


class PointSegHead_Large(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_dim, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, num_classes, 1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.mlp(x)
        return out.permute(0, 2, 1)



class OffsetHead_Large(nn.Module):
    def __init__(self, in_dim: int, num_keypoints: int):
        super().__init__()
        self.num_keypoints = num_keypoints

        self.mlp = nn.Sequential(
            nn.Conv1d(in_dim, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, num_keypoints * 3, 1)
        )

    def forward(self, x):
        B, N, _ = x.shape
        x = x.permute(0, 2, 1)
        out = self.mlp(x)
        out = out.permute(0, 2, 1)
        return out.reshape(B, N, self.num_keypoints, 3)



class TransPoseNetwork(nn.Module):
    def __init__(self, img_outdim=128,normals_outdim=64,points_outdim=256,num_classes=4, num_keypoints=10):
        super().__init__()
        self.features = DenseFusion(image_channels=img_outdim,
                                    normal_channels=normals_outdim,
                                    pointnet_channels=points_outdim,
                                    num_samples=4096)
        
        feature_outdim = img_outdim + normals_outdim + points_outdim
        self.seg_head = PointSegHead_Large(feature_outdim, num_classes)
        self.offset_head = OffsetHead_Large(feature_outdim, num_keypoints)

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

    def forward(self, img,normals,depth_c,mask_inst,intrinsics):
        
        # points: (B, N, 3) or (B, N, feat)
        features, points, trans_feat= self.features(img, depth_c, normals, mask_inst, intrinsics)  # (B, N, F)
        # features=features.unsqueeze(0)
        seg_logits = self.seg_head(features)  # (B, N, C)
        offsets = self.offset_head(features)  # (B, N, K, 3)
        return seg_logits, offsets, points, trans_feat