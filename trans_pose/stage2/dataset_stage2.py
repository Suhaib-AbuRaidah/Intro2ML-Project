import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
import json
import glob
from pathlib import Path

class Stage2Dataset(Dataset):
    def __init__(self, root_dir, keypoints_dir, camera_id='1', num_keypoints=10, transforms=None, target_size=None):
        """
        Args:
            root_dir: Path to 'tanscg-data-2'
            keypoints_dir: Path to 'tanscg-data-2/keypoints'
            camera_id: '1' for D435.
            target_size (tuple, optional): (width, height) to resize images to.
        """
        self.root_dir = root_dir
        self.camera_id = str(camera_id)
        self.num_keypoints = num_keypoints
        self.transforms = transforms
        self.target_size = target_size # (W, H)
        
        # 1. Load Canonical Keypoints
        self.canonical_kpts = self._load_all_keypoints(keypoints_dir)
        
        # 2. Index Scenes
        self.data_list = []
        self.scenes = sorted(glob.glob(os.path.join(root_dir, "scene*")))
        
        print(f"Found {len(self.scenes)} scenes. Indexing D435 frames...")
        
        for scene_path in self.scenes:
            meta_path = os.path.join(scene_path, "metadata.json")
            if not os.path.exists(meta_path): continue
                
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                
            # Get valid perspective folders (e.g. 0, 1, 2...)
            valid_folders = meta.get("D435_valid_perspective_list", [])
            scene_objects = meta.get("model_list", [])
            
            for folder_num in valid_folders:
                # The folder name is the perspective number (e.g. "0")
                perspective_folder = os.path.join(scene_path, str(folder_num))
                if not os.path.isdir(perspective_folder): continue
                
                # STRICT FILE NAMING: Always use camera_id.png (e.g. "1.png")
                rgb_filename = f"rgb{self.camera_id}.png"
                depth_filename = f"depth{self.camera_id}.png"
                mask_filename = f"depth{self.camera_id}-gt-mask.png"
                pose_filename = f"{self.camera_id}.npy"
                
                rgb_path = os.path.join(perspective_folder, rgb_filename)
                depth_path = os.path.join(perspective_folder, depth_filename)
                mask_path = os.path.join(perspective_folder, mask_filename)
                
                # Pose is inside a folder 'corrected_pose'
                pose_path = os.path.join(perspective_folder, "corrected_pose", pose_filename)
                
                # Only add if essential files exist
                if os.path.exists(rgb_path) and os.path.exists(depth_path):
                    self.data_list.append({
                        'rgb_path': rgb_path,
                        'depth_path': depth_path,
                        'mask_path': mask_path,
                        'pose_path': pose_path,
                        'model_list': scene_objects
                    })
        
        print(f"Indexed {len(self.data_list)} valid samples.")

        # Intrinsics (D435)
        # image is downsampled by 2 from 1280x720 to 640x360 so we modify cx, fy, cy and fx accordingly -- ghina 
        self.camera_intrisics = np.array([
            [463.58, 0.0, 325.66],
            [0.0, 463.69, 174.81],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)


    def _load_all_keypoints(self, kpts_dir):
        kpts_map = {}
        if not os.path.exists(kpts_dir): return kpts_map
        files = glob.glob(os.path.join(kpts_dir, "*.npz"))
        for f in files:
            try:
                fname = os.path.basename(f)
                obj_id = int(fname.split('-')[0])
                data = np.load(f)
                pts = data['points'] if 'points' in data else data[list(data.keys())[0]]
                if len(pts) > self.num_keypoints:
                    indices = np.linspace(0, len(pts)-1, self.num_keypoints, dtype=int)
                    pts = pts[indices]
                kpts_map[obj_id] = pts.astype(np.float32)
            except: pass
        return kpts_map



    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        # 1. RGB
        rgb = cv2.imread(item['rgb_path'])
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # 2. Depth (mm -> meters)
        depth = cv2.imread(item['depth_path'], cv2.IMREAD_UNCHANGED)
        depth = depth.astype(np.float32) / 1000.0

        # 3. Mask (depth-gt-mask)
        mask = cv2.imread(item['mask_path'], cv2.IMREAD_UNCHANGED)
        if mask is None: mask = np.zeros_like(depth, dtype=np.uint8)

        # --- RESIZING ---
        if self.target_size:
            # Resize with correct interpolation
            # For RGB, use INTER_LINEAR. For depth/mask, use INTER_NEAREST.
            rgb = cv2.resize(rgb, self.target_size, interpolation=cv2.INTER_LINEAR)
            depth = cv2.resize(depth, self.target_size, interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
        # --- END RESIZING ---

        # 4. Normals (Computed on the fly from the potentially resized depth map)
        sn = self.compute_normals(depth)
        
        # 5. Pose & Keypoints
        # Note: Keypoints are in world coordinates, so they are not affected by image resizing.
        target_keypoints = {}
        if os.path.exists(item['pose_path']):
            try:
                pose_data = np.load(item['pose_path'], allow_pickle=True)
                poses_dict = {}
                if isinstance(pose_data, dict): poses_dict = pose_data
                elif pose_data.shape == (): poses_dict = pose_data.item()
                
                for obj_id in item['model_list']:
                    if obj_id in poses_dict and obj_id in self.canonical_kpts:
                        pose = poses_dict[obj_id]
                        kpts_can = self.canonical_kpts[obj_id]
                        
                        # R * K.T + t
                        t_vec = pose[:3, 3]
                        if np.linalg.norm(t_vec) > 50.0: t_vec /= 1000.0 # mm fix
                        
                        kpts_world = (pose[:3, :3] @ kpts_can.T).T + t_vec
                        target_keypoints[str(obj_id)] = kpts_world.tolist()
            except: pass

        return {
            'rgb': torch.from_numpy(rgb).permute(2, 0, 1).float(),
            'depth': torch.from_numpy(depth).unsqueeze(0).float(),
            'mask': torch.from_numpy(mask).unsqueeze(0).float(),
            'sn': torch.from_numpy(sn).permute(2, 0, 1).float(),
            'keypoints': target_keypoints,
            'intrinsics': self.camera_intrisics
        }

    def compute_normals(self, depth):
        zy, zx = np.gradient(depth)
        normal = np.dstack((-zx, -zy, np.ones_like(depth)))
        n = np.linalg.norm(normal, axis=2)
        n[n == 0] = 1.0
        normal[:, :, 0] /= n
        normal[:, :, 1] /= n
        normal[:, :, 2] /= n
        return normal

def collate_fn(batch):
    return {
        'rgb': torch.stack([item['rgb'] for item in batch]),
        'depth': torch.stack([item['depth'] for item in batch]),
        'mask': torch.stack([item['mask'] for item in batch]),
        'sn': torch.stack([item['sn'] for item in batch]),
        'keypoints': [item['keypoints'] for item in batch]
    }