import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
import json
import glob
from pathlib import Path

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

        for scene_dir_name in self.scenes:
            # Resolve the full path to the current scene directory
            scene_path = Path(scene_dir_name)
            
            # 3. Find the subfolders inside the scene (e.g., '0' or '1')
            # Use glob to find all directories one level deep inside the scene_path
            subfolder_paths = [p for p in scene_path.iterdir() if p.is_dir()]

            if not subfolder_paths:
                # Add print here if you want to know which scenes are skipped
                # print(f"Warning: Skipping scene {scene_path.name}. No subfolders found.")
                continue

            # Iterate through all subfolders found (e.g., '0', '1', '2'...)
            for subfolder_path in subfolder_paths:
                # Now search for RGB files inside the 'rgb' directory of the subfolder
                # subfolder_path / "rgb" / "*.png"
                rgb_files = sorted(glob.glob(os.path.join(subfolder_path, "rgb1.png")))
                
                if rgb_files:
                    print(f" Found {len(rgb_files)} files in: {subfolder_path}")
                else:
                    print(f"Found 0 files in: {subfolder_path}")

                for rgb_path_str in rgb_files:
                    rgb_path = Path(rgb_path_str)
                    frame_id = rgb_path.stem # Gets filename without extension (e.g., '00000')
                    
                    # Construct other paths relative to the subfolder_path
                    self.data_list.append({
                        # Store the path to the current subfolder as the scene/base for retrieval
                        'base_path': subfolder_path, 
                        'frame_id': frame_id,
                        'rgb_path': rgb_path_str, # Store as string for easy os.path usage, or keep as Path
                        'depth_path': os.path.join(subfolder_path, "depth1.png"),
                        'mask_path': os.path.join(subfolder_path, "depth1-gt-mask.png"),
                        'meta_path': os.path.join(subfolder_path, "meta", f"{frame_id}.json")
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