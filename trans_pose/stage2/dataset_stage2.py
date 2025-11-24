import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
import json
import glob
from pathlib import Path

class Stage2Dataset(Dataset):
    def __init__(self, root_dir, keypoints_dir, camera_id='1', num_keypoints=10, transforms=None, target_size=None, use_gt_normals=True, augment=False):
        """
        Args:
            root_dir: Path to 'tanscg-data-2'
            keypoints_dir: Path to 'tanscg-data-2/keypoints'
            camera_id: '1' for D435.
            target_size (tuple, optional): (width, height) to resize images to.
            use_gt_normals (bool): If True, load GT normals instead of computing from depth
        """
        self.root_dir = root_dir
        self.camera_id = str(camera_id)
        self.num_keypoints = num_keypoints
        self.transforms = transforms
        self.target_size = target_size # (W, H)
        self.use_gt_normals = use_gt_normals
        self.augment = augment

        
        # 1. Load Canonical Keypoints
        self.canonical_kpts = self._load_all_keypoints(keypoints_dir)
        
        # 2. Index Scenes
        self.data_list = []
        self.scenes = sorted(glob.glob(os.path.join(root_dir, "scene*")))
        
        print(f"Found {len(self.scenes)} scenes. Indexing D435 frames (AUTO-DETECT)...")
        print(f"Using {'GT' if use_gt_normals else 'computed'} surface normals")
        
        for scene_path in self.scenes:
            meta_path = os.path.join(scene_path, "metadata.json")
            if not os.path.exists(meta_path): 
                continue
                
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            
            scene_objects = meta.get("model_list", [])
            
            # AUTO-DETECT: Scan all numeric folders for rgb1.png
            valid_folders = self._get_valid_perspectives(scene_path)
            
            for folder_num in valid_folders:
                perspective_folder = os.path.join(scene_path, str(folder_num))
                
                # File paths
                rgb_filename = f"rgb{self.camera_id}.png"
                depth_filename = f"depth{self.camera_id}.png"
                mask_filename = f"depth{self.camera_id}-gt-mask-corrected.png"  # Use CORRECTED mask
                normal_filename = f"depth{self.camera_id}-gt-sn.png"  # GT normals
                pose_filename = f"{self.camera_id}.npy"
                
                rgb_path = os.path.join(perspective_folder, rgb_filename)
                depth_path = os.path.join(perspective_folder, depth_filename)
                mask_path = os.path.join(perspective_folder, mask_filename)
                normal_path = os.path.join(perspective_folder, normal_filename)
                pose_path = os.path.join(perspective_folder, "corrected_pose", pose_filename)
                corrected_pose_dir = os.path.join(perspective_folder, "corrected_pose")  # <-- ADD

                # Check essential files
                if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
                    continue

                # If using GT normals, check if file exists
                if self.use_gt_normals and not os.path.exists(normal_path):
                    continue

                self.data_list.append({
                    'rgb_path': rgb_path,
                    'depth_path': depth_path,
                    'mask_path': mask_path,
                    'normal_path': normal_path,
                    'pose_path': pose_path,             # kept for backward compatibility
                    'corrected_pose_dir': corrected_pose_dir,  # <-- NEW entry
                    'model_list': scene_objects
                })
        
        print(f"Indexed {len(self.data_list)} valid samples.")

        # Intrinsics (D435) - scaled for 640x360
        self.camera_intrisics = np.array([
            [463.58, 0.0, 325.66],
            [0.0, 463.69, 174.81],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        self._dbg_pose_prints = 0

    def _get_valid_perspectives(self, scene_path):
        """
        Auto-detect valid perspectives by checking for rgb1.png.
        
        Returns:
            List of perspective indices (as integers)
        """
        valid_perspectives = []
        
        for item in os.listdir(scene_path):
            item_path = os.path.join(scene_path, item)
            
            # Skip non-directories
            if not os.path.isdir(item_path):
                continue
            
            # Try to parse as perspective number
            try:
                persp_idx = int(item)
            except ValueError:
                continue  # Not a numeric folder
            
            # Check if rgb1.png exists
            rgb1_path = os.path.join(item_path, "rgb1.png")
            if os.path.exists(rgb1_path):
                valid_perspectives.append(persp_idx)
        
        return sorted(valid_perspectives)

    def _load_all_keypoints(self, kpts_dir):
        """Load canonical keypoints for all objects."""
        kpts_map = {}
        if not os.path.exists(kpts_dir):
            print(f" Keypoints directory not found: {kpts_dir}")
            return kpts_map
        
        files = glob.glob(os.path.join(kpts_dir, "*.npz"))
        print(f" Loading keypoints from {len(files)} files...")
        
        loaded_count = 0
        failed_files = []
        
        def _extract_array(obj):
            """Recursively extract a numeric ndarray from nested containers/dicts/object-arrays."""
            # Direct numpy array
            if isinstance(obj, np.ndarray):
                # object dtype -> unwrap
                if obj.dtype == object:
                    if obj.shape == ():
                        return _extract_array(obj.item())
                    # if it's a 1D object array of ndarrays, try to vstack
                    if len(obj) > 0 and isinstance(obj[0], np.ndarray):
                        try:
                            return np.vstack([_extract_array(x) for x in obj])
                        except Exception:
                            return np.asarray([_extract_array(x) for x in obj])
                    # fallback: try to convert to numeric array
                    try:
                        return np.asarray(obj)
                    except Exception:
                        return None
                else:
                    return obj
            # Dict -> search values
            if isinstance(obj, dict):
                for v in obj.values():
                    res = _extract_array(v)
                    if res is not None:
                        return res
                return None
            # list/tuple -> search elements
            if isinstance(obj, (list, tuple)):
                for v in obj:
                    res = _extract_array(v)
                    if res is not None:
                        return res
                return None
            # scalar numeric
            if isinstance(obj, (int, float)):
                return np.array([obj], dtype=np.float32)
            return None
        
        for f in files:
            try:
                fname = os.path.basename(f)
                obj_id = int(fname.split('-')[0])
                obj_id_str = str(obj_id)
                
                # allow_pickle needed for object arrays / pickled dicts
                data = np.load(f, allow_pickle=True)
                
                # choose best candidate value
                pts_candidate = None
                if 'points' in data:
                    pts_candidate = data['points']
                elif 'arr_0' in data:
                    pts_candidate = data['arr_0']
                elif len(data.keys()) > 0:
                    pts_candidate = data[list(data.keys())[0]]
                
                if pts_candidate is None:
                    print(f"    {fname}: No data keys found")
                    failed_files.append((fname, "No data keys"))
                    continue
                
                # Extract numeric ndarray even if nested inside dict/object array
                pts = _extract_array(pts_candidate)
                if pts is None:
                    failed_files.append((fname, "Could not extract numeric array"))
                    continue
                
                # Ensure shape is (N,3). Try to repair common mis-shapes.
                pts = np.asarray(pts, dtype=np.float32)
                if pts.ndim == 1:
                    # if length divisible by 3, reshape
                    if pts.size % 3 == 0:
                        pts = pts.reshape(-1, 3)
                    else:
                        failed_files.append((fname, f"1D array with length {pts.size} not divisible by 3"))
                        continue
                if pts.ndim == 2 and pts.shape[1] != 3:
                    # if flattened as (3,N) or (N,3) swapped, try transpose if reasonable
                    if pts.ndim == 2 and pts.shape[0] == 3 and pts.shape[1] > 3:
                        pts = pts.T
                    else:
                        failed_files.append((fname, f"Array shape {pts.shape} incompatible (expect Nx3)"))
                        continue
                
                # Subsample if needed
                if len(pts) > self.num_keypoints:
                    indices = np.linspace(0, len(pts)-1, self.num_keypoints, dtype=int)
                    pts = pts[indices]
                
                kpts_map[obj_id_str] = pts
                loaded_count += 1
                
                if loaded_count <= 3:
                    print(f"    {fname}: ID={obj_id_str}, shape={pts.shape}")
                
            except Exception as e:
                failed_files.append((fname if 'fname' in locals() else os.path.basename(f), str(e)))
        
        print(f" Loaded {loaded_count}/{len(files)} keypoint sets")
        
        if failed_files:
            print(f" Failed to load {len(failed_files)} files:")
            for fname, error in failed_files[:10]:
                print(f"   - {fname}: {error}")
        
        return kpts_map

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        # 1. RGB
        rgb = cv2.imread(item['rgb_path'])
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # --- Lightweight photometric augmentation (training only) ---
        if self.augment:
            # brightness / contrast multiplier
            alpha = 1.0 + (np.random.rand() - 0.5) * 0.4   # ~[0.8,1.2]
            rgb = np.clip(rgb * alpha, 0.0, 1.0)
            # subtle color jitter per channel
            jitter = (np.random.rand(3) - 0.5) * 0.08
            rgb = np.clip(rgb + jitter.reshape(1,1,3), 0.0, 1.0)
            # gaussian noise
            noise = np.random.normal(0.0, 0.01, size=rgb.shape).astype(np.float32)
            rgb = np.clip(rgb + noise, 0.0, 1.0)
        
        # 2. Depth (mm -> meters)
        depth = cv2.imread(item['depth_path'], cv2.IMREAD_UNCHANGED)
        depth = depth.astype(np.float32) / 1000.0

        # # 3. Mask (corrected instance mask)
        # mask = cv2.imread(item['mask_path'], cv2.IMREAD_UNCHANGED)
        # if mask is None: 
        #     mask = np.zeros_like(depth, dtype=np.uint8)

        # 3. Mask (corrected instance mask) - try fallbacks if missing
        mask = cv2.imread(item['mask_path'], cv2.IMREAD_UNCHANGED)
        if mask is None: 
            mask = np.zeros_like(depth, dtype=np.uint8)
        mask = None
        try_paths = []
        if item.get('mask_path'):
            try_paths.append(item['mask_path'])
        # fallback: common uncorrected name
        base = os.path.splitext(os.path.basename(item['depth_path']))[0]
        parent = os.path.dirname(item['depth_path'])
        try_paths.append(os.path.join(parent, base + "-gt-mask.png"))
        # fallback: any gt-mask file in the perspective folder
        glob_masks = glob.glob(os.path.join(parent, "*gt-mask*.png"))
        try_paths.extend(glob_masks)

        for p in try_paths:
            if p and os.path.exists(p):
                mask = cv2.imread(p, cv2.IMREAD_UNCHANGED)
                if mask is not None:
                    break

        if mask is None:
            # final fallback: empty mask (log once)
            if self._dbg_pose_prints < 3:
                print(f"DEBUG: mask missing for {item['depth_path']}, using empty mask")
                self._dbg_pose_prints += 1
            mask = np.zeros_like(depth, dtype=np.uint8)


        # 4. Surface Normals (GT or computed)
        if self.use_gt_normals and os.path.exists(item['normal_path']):
            # Load GT normals
            sn = cv2.imread(item['normal_path'], cv2.IMREAD_UNCHANGED)
            if sn is None:
                # defensive: fallback to computed normals if file unreadable
                sn = self.compute_normals(depth)
            else:
                if sn.dtype == np.uint16:
                    sn = (sn.astype(np.float32) / 32767.5) - 1.0
                else:
                    sn = sn.astype(np.float32)
        else:
            # Compute from depth (fallback)
            sn = self.compute_normals(depth)

        # --- RESIZING ---
        if self.target_size:
            rgb = cv2.resize(rgb, self.target_size, interpolation=cv2.INTER_LINEAR)
            depth = cv2.resize(depth, self.target_size, interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
            sn = cv2.resize(sn, self.target_size, interpolation=cv2.INTER_LINEAR)

        # 5. Keypoints (world coordinates)
        target_keypoints = {}
        corrected_dir = item.get('corrected_pose_dir', None)

        # Try per-object files inside corrected_pose first (e.g. corrected_pose/4.npy)
        if corrected_dir is not None and os.path.isdir(corrected_dir):
            for obj_id in item['model_list']:
                obj_id_str = str(obj_id)
                # skip if canonical keypoints missing
                if obj_id_str not in self.canonical_kpts:
                    continue

                # candidate file names: 4.npy / 4.npz
                npy_file = os.path.join(corrected_dir, f"{obj_id_str}.npy")
                npz_file = os.path.join(corrected_dir, f"{obj_id_str}.npz")

                pose = None
                try:
                    if os.path.exists(npy_file):
                        pose = np.load(npy_file, allow_pickle=True)
                    elif os.path.exists(npz_file):
                        d = np.load(npz_file, allow_pickle=True)
                        # pick common keys or first entry
                        if 'pose' in d:
                            pose = d['pose']
                        elif 'arr_0' in d:
                            pose = d['arr_0']
                        elif len(getattr(d, 'files', [])) > 0:
                            pose = d[d.files[0]]
                        else:
                            # if loading gives a scalar object (e.g. a dict)
                            try:
                                pose = d.item()
                            except Exception:
                                pose = None
                    else:
                        continue
                except Exception:
                    continue

                if pose is None:
                    continue

                # normalize to ndarray and 4x4
                pose = np.asarray(pose)
                if pose.size == 16:
                    pose = pose.reshape(4, 4)
                if pose.shape != (4, 4):
                    continue

                kpts_can = self.canonical_kpts[obj_id_str]
                t_vec = pose[:3, 3]
                if np.linalg.norm(t_vec) > 50.0:
                    t_vec = t_vec / 1000.0

                kpts_world = (pose[:3, :3] @ kpts_can.T).T + t_vec
                target_keypoints[obj_id_str] = kpts_world.tolist()

        else:
            # fallback: existing code that tries single per-perspective file
            if os.path.exists(item['pose_path']):
                try:
                    pose_data = np.load(item['pose_path'], allow_pickle=True)
                    poses_dict = {}
                    if isinstance(pose_data, dict):
                        poses_dict = pose_data
                    elif pose_data.shape == ():
                        poses_dict = pose_data.item()
                    else:
                        try:
                            poses_dict = {k: pose_data[k] for k in pose_data.files}
                        except Exception:
                            poses_dict = {}

                    for obj_id in item['model_list']:
                        obj_id_str = str(obj_id)
                        if obj_id in poses_dict:
                            pose = poses_dict[obj_id]
                        elif obj_id_str in poses_dict:
                            pose = poses_dict[obj_id_str]
                        else:
                            continue

                        if obj_id_str not in self.canonical_kpts:
                            continue

                        pose = np.asarray(pose)
                        if pose.size == 16:
                            pose = pose.reshape(4, 4)
                        if pose.shape != (4, 4):
                            continue

                        kpts_can = self.canonical_kpts[obj_id_str]
                        t_vec = pose[:3, 3]
                        if np.linalg.norm(t_vec) > 50.0:
                            t_vec = t_vec / 1000.0

                        kpts_world = (pose[:3, :3] @ kpts_can.T).T + t_vec
                        target_keypoints[obj_id_str] = kpts_world.tolist()
                except Exception as e:
                    print(f"WARNING: Failed to load pose {item['pose_path']}: {e}")
                    pass

        return {
            'rgb': torch.from_numpy(rgb).permute(2, 0, 1).float(),
            'depth': torch.from_numpy(depth).unsqueeze(0).float(),
            'mask': torch.from_numpy(mask).unsqueeze(0).float(),
            'sn': torch.from_numpy(sn).permute(2, 0, 1).float(),
            'keypoints': target_keypoints,
            'intrinsics': self.camera_intrisics
        }
    
    def compute_normals(self, depth):
        """Fallback: compute normals from depth."""
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