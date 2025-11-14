import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractor(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )

    def forward(self, x):
        # x: (B, N, in_dim)
        B, N, F = x.shape
        out = self.mlp(x.reshape(B * N, F))  # (B*N, out_dim)
        return out.reshape(B, N, -1)  # (B, N, out_dim)