import torch
import torch.nn as nn

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
        return out.reshape(B, N, -1)   # logits per point
