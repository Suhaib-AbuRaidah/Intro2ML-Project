import torch
from torch import nn
from torch.nn import functional as F

def votes_from_offsets(points, offsets):
    """
    points: (B, N, 3)
    offsets: (B, N, K, 3)
    returns votes: (B, N, K, 3)
    """
    return points.unsqueeze(2) + offsets 