import torch
import torch.nn as nn
from typing import Tuple

try:
    from trans_pose.lib.pointnet2 import pointnet2_modules as pnet2_modules
except ImportError as exc:  # pragma: no cover - informative message for missing build
    pnet2_modules = None
    PN2_IMPORT_ERROR = exc
else:
    PN2_IMPORT_ERROR = None


def build_instance_points(
    depth_c: torch.Tensor,
    mask_inst: torch.Tensor,
    intrinsics: Tuple[float, float, float, float],
    normals: torch.Tensor = None,
    num_samples: int = 4096,
):
    """
    Samples per-instance point clouds aligned with Stage-1 outputs.

    Args:
        depth_c: completed depth `[1, H, W]`.
        mask_inst: binary mask `[1, H, W]` for the instance of interest.
        intrinsics: (fx, fy, cx, cy) camera intrinsics.
        normals: optional surface normal map `[3, H, W]`.
        num_samples: number of pixels/points to return.

    Returns:
        points_xyz: `[N, 3]` back-projected 3D coordinates.
        uv: `[N, 2]` pixel coordinates at the original resolution.
        normals_pix: `[N, 3]` sampled normals (zeros if normals is None).
    """
    fx, fy, cx, cy = intrinsics
    device = depth_c.device

    valid = ((mask_inst > 0.5) & (depth_c > 0)).squeeze(0)
    valid_idx = valid.nonzero(as_tuple=False)  # [M, 2] -> (v, u)
    if valid_idx.numel() == 0:
        raise ValueError("No valid pixels inside instance mask.")

    num_valid = valid_idx.shape[0]
    if num_valid >= num_samples:
        choice = torch.randperm(num_valid, device=device)[:num_samples]
    else:
        extra = num_samples - num_valid
        reps = torch.randint(0, num_valid, (extra,), device=device)
        choice = torch.cat([torch.arange(num_valid, device=device), reps], dim=0)
        choice = choice[torch.randperm(choice.shape[0], device=device)]

    picked = valid_idx[choice]
    v = picked[:, 0]
    u = picked[:, 1]
    z = depth_c[0, v, u]
    x = (u.float() - cx) * z / fx
    y = (v.float() - cy) * z / fy

    points_xyz = torch.stack([x, y, z], dim=-1)
    uv = torch.stack([u.float(), v.float()], dim=-1)

    if normals is not None:
        normals_pix = normals[:, v, u].permute(1, 0).contiguous()
    else:
        normals_pix = torch.zeros_like(points_xyz)

    return points_xyz, uv, normals_pix


class PointNetLight(nn.Module):
    """
    PointNet-style encoder for extended point sets (XYZ plus optional cues).
    """

    def __init__(self, in_dim: int = 3, feature_dim: int = 64):
        super().__init__()
        self.local_mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
        )
        self.output_mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: `[N, d_in]` or `[B, N, d_in]` extended point cloud.
        Returns:
            `[N, C_3D]` or `[B, N, C_3D]` point-wise descriptors.
        """
        squeeze_batch = points.dim() == 2
        if squeeze_batch:
            points = points.unsqueeze(0)

        local = self.local_mlp(points)
        global_feat = local.max(dim=1, keepdim=True)[0]
        global_tiled = global_feat.expand(-1, local.shape[1], -1)

        fused = torch.cat([local, global_tiled], dim=-1)
        feats = self.output_mlp(fused)
        return feats.squeeze(0) if squeeze_batch else feats


class PointNet2Backbone(nn.Module):
    """
    Wrapper around PointNet++ set abstraction / feature propagation blocks.
    Expects the compiled pointnet2 extension to be available under
    `trans_pose.lib.pointnet2`.
    """

    def __init__(self, in_feat_dim: int = 6, out_dim: int = 128):
        super().__init__()
        if pnet2_modules is None:
            raise ImportError(
                "PointNet++ modules not found. Build the extension via "
                "`python setup.py install` inside the project root. "
                f"Original import error: {PN2_IMPORT_ERROR}"
            )

        self.sa1 = pnet2_modules.PointnetSAModuleMSG(
            npoint=1024,
            radii=[0.05, 0.1],
            nsamples=[32, 64],
            mlps=[[in_feat_dim, 32, 32, 64], [in_feat_dim, 64, 64, 128]],
            use_xyz=True,
        )
        self.sa2 = pnet2_modules.PointnetSAModuleMSG(
            npoint=256,
            radii=[0.1, 0.2],
            nsamples=[64, 128],
            mlps=[[64 + 128, 128, 128, 256], [64 + 128, 128, 196, 256]],
            use_xyz=True,
        )
        self.sa3 = pnet2_modules.PointnetSAModule(
            npoint=None,
            radius=None,
            nsample=None,
            mlp=[256 + 256, 256, 512, 1024],
            use_xyz=True,
        )

        self.fp3 = pnet2_modules.PointnetFPModule(mlp=[256 + 1024, 512, 256])
        self.fp2 = pnet2_modules.PointnetFPModule(mlp=[192 + 256, 256, 128])
        self.fp1 = pnet2_modules.PointnetFPModule(mlp=[in_feat_dim + 128, 128, out_dim])

    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: `[B, N, 3]` point coordinates.
            features: `[B, C_in, N]` extra per-point descriptors (e.g., RGB+normals).
        Returns:
            `[B, N, out_dim]` point-wise descriptors.
        """
        if pnet2_modules is None:
            raise ImportError(
                "PointNet++ modules not available. Ensure the extension is built."
            )

        l0_xyz, l0_feat = xyz, features
        l1_xyz, l1_feat = self.sa1(l0_xyz, l0_feat)
        l2_xyz, l2_feat = self.sa2(l1_xyz, l1_feat)
        l3_xyz, l3_feat = self.sa3(l2_xyz, l2_feat)

        l2_feat = self.fp3(l2_xyz, l3_xyz, l2_feat, l3_feat)
        l1_feat = self.fp2(l1_xyz, l2_xyz, l1_feat, l2_feat)
        l0_feat = self.fp1(l0_xyz, l1_xyz, l0_feat, l1_feat)

        return l0_feat.transpose(1, 2).contiguous()
