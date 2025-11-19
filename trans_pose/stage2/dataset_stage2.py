import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
import json
import glob

class Stage2Dataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        """
        Args:
            root_dir: Path to the dataset (e.g., .../transcg-data-1/transcg)
            transforms: Optional transforms for data augmentation
        """
        self.root_dir = root_dir
        self.transforms = transforms
        
        # Get all scene directories
        self.scenes = sorted(glob.glob(os.path.join(root_dir, "*")))
        self.data_list = []
        
        # Camera intrinsics (Hardcoded for TransCG dataset or loaded from file)
        # fx, fy, cx, cy
        self.camera_intrisics = np.array([
            [525.0, 0.0, 319.5],
            [0.0, 525.0, 239.5],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        # Index all frames
        for scene in self.scenes:
            rgb_files = sorted(glob.glob(os.path.join(scene, "rgb", "*.png")))
            for rgb_path in rgb_files:
                frame_id = os.path.basename(rgb_path).split('.')[0]
                self.data_list.append({
                    'scene': scene,
                    'frame_id': frame_id,
                    'rgb_path': rgb_path,
                    'depth_path': os.path.join(scene, "depth", f"{frame_id}.png"),
                    'mask_path': os.path.join(scene, "mask", f"{frame_id}.png"),
                    'meta_path': os.path.join(scene, "meta", f"{frame_id}.json") # Assuming keypoints are here
                })

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        # 1. Load RGB
        # Requirement: Return [0, 1] float tensor
        rgb = cv2.imread(item['rgb_path'])
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = rgb.astype(np.float32) / 255.0  # Normalize to [0, 1]
        
        # 2. Load Depth
        # Requirement: Metric depth in meters. DO NOT normalize by max().
        # TransCG usually stores depth as uint16 in millimeters.
        depth = cv2.imread(item['depth_path'], cv2.IMREAD_UNCHANGED)
        depth = depth.astype(np.float32) / 1000.0  # Convert mm to meters
        
        # 3. Load Mask (Instance Segmentation)
        mask = cv2.imread(item['mask_path'], cv2.IMREAD_UNCHANGED)
        
        # 4. Load Surface Normals (Placeholder or Pre-computed)
        # If you don't have pre-computed normals, you might need to compute them from depth
        # or load them if Stage 1 saved them. 
        # For now, let's assume we compute them or load a placeholder.
        # Here is a simple placeholder (random) or computation from depth:
        sn = self.compute_normals(depth)
        
        # 5. Load Keypoints/Meta
        keypoints = {}
        if os.path.exists(item['meta_path']):
            with open(item['meta_path'], 'r') as f:
                meta = json.load(f)
                # Parse meta to get keypoints per object ID
                # Structure depends on your specific json format
                # Example: keypoints = {'1': [[x,y,z], ...], '2': ...}
                pass 

        # Convert to Tensors
        # RGB: [H, W, 3] -> [3, H, W]
        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()
        
        # Depth: [H, W] -> [1, H, W]
        depth_tensor = torch.from_numpy(depth).unsqueeze(0).float()
        
        # Mask: [H, W] -> [1, H, W]
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        
        # Normals: [H, W, 3] -> [3, H, W]
        sn_tensor = torch.from_numpy(sn).permute(2, 0, 1).float()

        return {
            'rgb': rgb_tensor,
            'depth': depth_tensor,
            'mask': mask_tensor,
            'sn': sn_tensor,
            'keypoints': keypoints, # Raw dict, handled in collate_fn
            'intrinsics': self.camera_intrisics
        }

    def compute_normals(self, depth):
        """
        Simple approximation of surface normals from depth map.
        """
        zy, zx = np.gradient(depth)
        normal = np.dstack((-zx, -zy, np.ones_like(depth)))
        n = np.linalg.norm(normal, axis=2)
        normal[:, :, 0] /= n
        normal[:, :, 1] /= n
        normal[:, :, 2] /= n
        return normal

def collate_fn(batch):
    """
    Custom collate function to handle dictionary lists (keypoints).
    """
    rgb = torch.stack([item['rgb'] for item in batch])
    depth = torch.stack([item['depth'] for item in batch])
    mask = torch.stack([item['mask'] for item in batch])
    sn = torch.stack([item['sn'] for item in batch])
    
    # Keypoints are variable length (per object), keep as list of dicts
    keypoints = [item['keypoints'] for item in batch]
    
    return {
        'rgb': rgb,
        'depth': depth,
        'mask': mask,
        'sn': sn,
        'keypoints': keypoints
    }