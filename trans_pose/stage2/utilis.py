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


def mean_shift_clustering(votes, mask=None, bandwidth=0.05, num_iters=10, eps=1e-3):
    """
    Mean-shift clustering per (B, K). For each batch and keypoint, cluster N votes using
    gaussian kernel weights. Returns cluster centers (B, K, 3).
    votes: (B, N, K, 3)
    mask: optional (B, N) boolean: if provided, only consider masked points (object instance)
    bandwidth: gaussian kernel bandwidth (sigma)
    num_iters: iterations of mean-shift
    """
    B, N, K, _ = votes.shape
    device = votes.device
    centers = votes.mean(dim=1)  # (B, K, 3) initial guess: mean across points

    # If mask provided, set votes of masked-out points to large distance (or zero weight)
    if mask is not None:
        mask = mask.unsqueeze(2).float()  # (B, N, 1)
        # For numerical stability, multiply votes by mask (won't remove bias fully),
        # better to use weights in kernel.
    for it in range(num_iters):
        # compute gaussian weights: w_ij = exp(-||v_ij - c_j||^2 / (2*sigma^2))
        # votes: (B,N,K,3), centers: (B,K,3) -> expand centers to (B,1,K,3)
        c = centers.unsqueeze(1)  # (B,1,K,3)
        diff = votes - c          # (B,N,K,3)
        dist2 = (diff ** 2).sum(dim=-1)  # (B,N,K)
        weights = torch.exp(-dist2 / (2 * bandwidth * bandwidth))  # (B,N,K)
        if mask is not None:
            weights = weights * mask  # zero out votes from masked-out points

        # weighted mean
        numerator = (weights.unsqueeze(-1) * votes).sum(dim=1)   # (B,K,3)
        denom = weights.sum(dim=1).unsqueeze(-1) + 1e-8         # (B,K,1)
        new_centers = numerator / denom
        shift = torch.norm(new_centers - centers, dim=-1).max()
        centers = new_centers
        if shift.item() < eps:
            break
    return centers  # (B,K,3)


def rigid_transform_3D(A, B):
    """
    Computes rigid transform T that aligns A → B using SVD.
    A, B: (K,3)
    Returns: R (3×3), t (3)
    """
    centroid_A = A.mean(0)
    centroid_B = B.mean(0)
    
    AA = A - centroid_A
    BB = B - centroid_B

    H = AA.T @ BB
    U, S, Vt = torch.linalg.svd(H)
    R = Vt.T @ U.T

    # reflection fix
    if torch.det(R) < 0:
        Vt[-1,:] *= -1
        R = Vt.T @ U.T

    t = centroid_B - R @ centroid_A

    return R, t
