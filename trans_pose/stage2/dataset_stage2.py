import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
import json
import glob

class Stage2Dataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        
        self.scenes = sorted(glob.glob(os.path.join(root_dir, "*")))
        self.data_list = []
        
        # fx, fy, cx, cy
        self.camera_intrisics = np.array([
            [525.0, 0.0, 319.5],
            [0.0, 525.0, 239.5],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

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
                    'meta_path': os.path.join(scene, "meta", f"{frame_id}.json")
                })

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        # 1. RGB [0, 1]
        rgb = cv2.imread(item['rgb_path'])
        if rgb is None:
            raise FileNotFoundError(f"Image not found: {item['rgb_path']}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = rgb.astype(np.float32) / 255.0
        
        # 2. Depth (Meters)
        depth = cv2.imread(item['depth_path'], cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"Depth not found: {item['depth_path']}")
        depth = depth.astype(np.float32) / 1000.0
        
        # 3. Mask
        mask = cv2.imread(item['mask_path'], cv2.IMREAD_UNCHANGED)
        if mask is None:
            # Fallback for missing mask: zeros
            mask = np.zeros_like(depth, dtype=np.uint8)
        
        # 4. Normals
        sn = self.compute_normals(depth)
        
        # 5. Keypoints (FIXED PARSING)
        keypoints = {}
        if os.path.exists(item['meta_path']):
            with open(item['meta_path'], 'r') as f:
                meta = json.load(f)
                # Logic: Iterate over objects in meta and extract keypoints
                # Adjust 'objects' key based on your actual JSON structure
                # Case A: JSON is a list of objects
                if isinstance(meta, list):
                    objects = meta
                # Case B: JSON has an 'objects' key
                elif 'objects' in meta:
                    objects = meta['objects']
                else:
                    objects = []

                for obj in objects:
                    # We need 'class_id' (or 'obj_id') and 'keypoints'
                    # Adjust keys 'class_id' and 'keypoints' to match your JSON
                    oid = obj.get('class_id', obj.get('obj_id', None))
                    kpts = obj.get('keypoints', obj.get('kpts', None))
                    
                    if oid is not None and kpts is not None:
                        # Ensure kpts is a list of 3D points
                        keypoints[str(oid)] = kpts

        # Tensors
        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()
        depth_tensor = torch.from_numpy(depth).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        sn_tensor = torch.from_numpy(sn).permute(2, 0, 1).float()

        return {
            'rgb': rgb_tensor,
            'depth': depth_tensor,
            'mask': mask_tensor,
            'sn': sn_tensor,
            'keypoints': keypoints,
            'intrinsics': self.camera_intrisics
        }

    def compute_normals(self, depth):
        zy, zx = np.gradient(depth)
        normal = np.dstack((-zx, -zy, np.ones_like(depth)))
        n = np.linalg.norm(normal, axis=2)
        # Avoid div by zero
        n[n == 0] = 1.0
        normal[:, :, 0] /= n
        normal[:, :, 1] /= n
        normal[:, :, 2] /= n
        return normal

def collate_fn(batch):
    rgb = torch.stack([item['rgb'] for item in batch])
    depth = torch.stack([item['depth'] for item in batch])
    mask = torch.stack([item['mask'] for item in batch])
    sn = torch.stack([item['sn'] for item in batch])
    keypoints = [item['keypoints'] for item in batch]
    
    return {
        'rgb': rgb,
        'depth': depth,
        'mask': mask,
        'sn': sn,
        'keypoints': keypoints
    }