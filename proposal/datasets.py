import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from scipy.spatial.transform import Rotation as Rscipy

class TransCG_D435_Dataset(Dataset):
    """
    Loads only D435 camera data (rgb1, depth1, mask1) and corrected_pose/*.npy
    Folder example:
      root/
        scene1/0/
        scene1/1/
        scene2/0/
        ...
    """
    def __init__(self, root_dir, object_ids=[0, 32, 42, 47], use_depth=True):
        self.root_dir = root_dir
        self.object_ids = object_ids
        self.use_depth = use_depth
        self.samples = []

        # Recursively search all corrected_pose directories
        pose_dirs = sorted(glob.glob(os.path.join(root_dir, "**/corrected_pose"), recursive=True))
        if not pose_dirs:
            raise RuntimeError(f"No corrected_pose directories found under {root_dir}")

        for pose_dir in pose_dirs:
            scene_path = os.path.dirname(pose_dir)

            rgb_path = os.path.join(scene_path, "rgb1.png")
            depth_path = os.path.join(scene_path, "depth1-gt.png")
            mask_path = os.path.join(scene_path, "depth1-gt-mask.png")

            # Skip if RGB is missing (required)
            if not os.path.exists(rgb_path):
                continue

            pose_paths = []
            for oid in object_ids:
                pose_file = os.path.join(pose_dir, f"{oid}.npy")
                if os.path.exists(pose_file):
                    pose_paths.append(pose_file)

            if pose_paths:
                self.samples.append({
                    "rgb": rgb_path,
                    "depth": depth_path if os.path.exists(depth_path) else None,
                    "mask": mask_path if os.path.exists(mask_path) else None,
                    "poses": pose_paths
                })

        if not self.samples:
            raise RuntimeError(f"No valid D435 samples found in {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # RGB
        rgb = cv2.cvtColor(cv2.imread(s["rgb"]), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = torch.from_numpy(rgb).permute(2, 0, 1)

        # Depth
        depth = None
        if s["depth"] and self.use_depth:
            d = cv2.imread(s["depth"], cv2.IMREAD_UNCHANGED).astype(np.float32)
            d = d / (d.max() + 1e-8)
            depth = torch.from_numpy(d[None, ...])

        # Mask
        mask = None
        if s["mask"]:
            m = cv2.imread(s["mask"], cv2.IMREAD_GRAYSCALE)
            m = (m > 0).astype(np.float32)
            mask = torch.from_numpy(m[None, ...])

        # Load multiple object poses (each 4x4)
        poses = []
        quats = []
        trans = []
        for p in s["poses"]:
            pose = np.load(p).astype(np.float32)
            if pose.shape != (4, 4):
                continue
            R = pose[:3, :3]
            t = pose[:3, 3]
            q = Rscipy.from_matrix(R).as_quat()  # xyzw
            poses.append(pose)
            quats.append(q)
            trans.append(t)

        poses = torch.from_numpy(np.stack(poses)) if poses else None
        quats = torch.from_numpy(np.stack(quats)) if quats else None
        trans = torch.from_numpy(np.stack(trans)) if trans else None

        return {
            "rgb": rgb,
            "depth": depth,
            "mask": mask,
            "poses": poses,
            "quat": quats,
            "trans": trans,
            "path": s["rgb"]
        }

if __name__ == "__main__":
    root = "/home/suhaib/6DPOSE/data/transcg-data-1/transcg"
    ds = TransCG_D435_Dataset(root)
    print("Total D435 samples:", len(ds))
    s = ds[1]
    print("RGB:", s["rgb"].shape)
    print("Depth:", s["depth"].shape)
    print("Mask:", s["mask"].shape)
    print("Poses:", s["poses"].shape)
    print("Quats:", s["quat"].shape)
