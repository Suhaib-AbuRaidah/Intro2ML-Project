import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
import sys
from torchvision import transforms as T
sys.path.append("/home/suhaib/ML_Project")

import torch

def scale_intrinsics(K, old_width, old_height, new_width, new_height):
    """
    Scale a 3x3 camera intrinsic matrix K from (old_width, old_height)
    to (new_width, new_height).

    Args:
        K: torch.Tensor or numpy array, shape (3,3)
        old_width, old_height: original image size
        new_width, new_height: target image size

    Returns:
        K_new: scaled intrinsic matrix, same type as input
    """

    sx = new_width / old_width
    sy = new_height / old_height

    K_new = K.copy() if hasattr(K, "copy") else K.clone()

    K_new[0, 0] *= sx   # fx
    K_new[1, 1] *= sy   # fy
    K_new[0, 2] *= sx   # cx
    K_new[1, 2] *= sy   # cy

    return K_new

class Stage2Dataset(Dataset):
	def __init__(self, root_dir, transforms=None):
		"""
		root_dir: path to scenes (e.g., ~/ML_Project/data/transcg-data-1/transcg)
		"""
		self.root_dir = root_dir
		self.transforms = transforms
		self.samples = []
		camera_int_dir = root_dir.replace("transcg-data-1/transcg","transcg-info/transcg/camera_intrinsics")
		camera_intri_file = os.path.join(camera_int_dir, "1-camIntrinsics-D435.npy")
		self.camera_intrisics = np.load(camera_intri_file)
		self.camera_intrisics = scale_intrinsics(self.camera_intrisics, 1280, 720, 256, 256)
		
		keypoints_dir = root_dir.replace("transcg-data-1/transcg","transcg-info/transcg/keypoints")
		self.keypoints = {}
		self.center= {}
		for file in sorted(os.listdir(keypoints_dir)):
			file_id = file[0:2] if file[1].isdigit() else file[0]
			keypoints_path = os.path.join(keypoints_dir, file)
			data = np.load(keypoints_path, allow_pickle=True)
			obj = data["arr_0"].item()     # extract Python dict
			kpts = obj["keypoints"]
			center = obj["center"]
			self.keypoints[file_id] = kpts
			self.center[file_id] = center

		# Traverse all scenes
		for scene_name in sorted(os.listdir(root_dir)):
			scene_path = os.path.join(root_dir, scene_name)
			if not os.path.isdir(scene_path):
				continue

			# scene subfolders (0,1,2,...)
			for sub_name in sorted(os.listdir(scene_path)):
				sub_path = os.path.join(scene_path, sub_name)
				if not os.path.isdir(sub_path) or sub_name == "metadata.json":
					continue

				# Look for rgb1.png
				for fname in os.listdir(sub_path):
					if fname.startswith("rgb1") and fname.endswith(".png"):
						idx = fname.replace("rgb", "").replace(".png", "")
						
						rgb_path = os.path.join(sub_path, f"rgb{idx}.png")
						sn_path = os.path.join(sub_path, f"depth{idx}-gt-sn.png")
						depth_path = os.path.join(sub_path, f"depth{idx}-gt.png")
						mask_path = os.path.join(sub_path, f"depth{idx}-gt-mask.png")

						corrected_pose_dir = os.path.join(sub_path, "corrected_pose")

						if (
							os.path.exists(sn_path)
							and os.path.exists(depth_path)
							and os.path.exists(mask_path)
							and os.path.isdir(corrected_pose_dir)
						):
							self.samples.append(
								dict(
									rgb=rgb_path,
									sn=sn_path,
									depth=depth_path,
									mask=mask_path,
									pose_dir=corrected_pose_dir,
								)
							)

		if len(self.samples) == 0:
			raise RuntimeError(f"No valid samples found in {root_dir}")

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		item = self.samples[idx]

		rgb = Image.open(item["rgb"]).convert("RGB")
		sn = Image.open(item["sn"]).convert("RGB")
		depth = Image.open(item["depth"])    # depth is single channel
		mask = Image.open(item["mask"]).convert("L")

		target_size = (256, 256)
		rgb = rgb.resize(target_size, Image.BILINEAR)
		sn = sn.resize(target_size, Image.BILINEAR)
		depth = depth.resize(target_size, Image.BILINEAR)
		mask = mask.resize(target_size, Image.NEAREST)

		depth = np.array(depth).astype(np.float32)/1000
		depth = torch.from_numpy(depth).unsqueeze(0)

		# Load all corrected poses
		pose_dict = {}
		zero_kepoints_dict = {}
		center_dict = {}
		target_keypoints = {}
		for f in sorted(os.listdir(item["pose_dir"])):
			if f.endswith(".npy"):
				obj_id = f.replace(".npy", "")
				pose_path = os.path.join(item["pose_dir"], f)
				pose_dict[obj_id] = np.load(pose_path)
				zero_kepoints_dict[obj_id] = self.keypoints[obj_id]
				center_dict[obj_id] = self.center[obj_id]

				pose = pose_dict[obj_id]
				kpts_can = zero_kepoints_dict[obj_id]
	                        
				# R * K.T + t
				t_vec = pose[:3, 3]
				if np.linalg.norm(t_vec) > 50.0: t_vec /= 1000.0 # mm fix
				
				kpts_world = (pose[:3, :3] @ kpts_can.T).T + t_vec
				target_keypoints[str(obj_id)] = kpts_world.tolist()

		if self.transforms:
			# apply transforms
			rgb = self.transforms(rgb).float()
			sn = self.transforms(sn).float()
			# depth = self.transforms(depth).float()
			mask = self.transforms(mask).float()

		return {
			"rgb": rgb,
			"sn": sn,
			"depth": depth,
			"mask": mask,
			"poses": pose_dict,
			"zero_keypoints": zero_kepoints_dict,
			"centers": center_dict,
			"target_keypoints": target_keypoints
		}
	
def collate_fn(batch):
    out = {}

    # Non-dict items (rgb, sn, depth, mask) → stack normally
    for key in ["rgb", "sn", "depth", "mask"]:
        out[key] = torch.stack([item[key] for item in batch], dim=0)

    # Dict items → keep as a list of dicts (no merging, no stacking)
    for key in ["poses", "zero_keypoints", "centers", "target_keypoints"]:
        out[key] = [item[key] for item in batch]

    return out

if __name__=="__main__":
	from torchvision import transforms as T
	root_dir = "./data/transcg-data-1/transcg"
	dataset = Stage2Dataset(root_dir, transforms=T.ToTensor())
	print(len(dataset))
	dataloader = torch.utils.data.DataLoader(
	dataset, batch_size=8, drop_last=True, shuffle=False, collate_fn=collate_fn)
	for data in dataloader:
			# print(data["rgb"].shape)
			# print(data["sn"].shape)
			print(data["depth"].shape)
			img = data['depth'][7]
			# # img = cv2.cvtColor(img.permute(1,2,0).cpu().numpy(), cv2.COLOR_GRAY2BGR)
			img=img.permute(1,2,0).cpu().numpy()
			print(f"img shape: {img.shape},min: {img.min()}, max: {img.max()}")
			cv2.imshow("img", img)
			if cv2.waitKey(0)==ord('q'):
				cv2.destroyAllWindows()			
			# print(data["mask"].shape)
			# x=[i.keys() for i in data['poses']]
			# print(x)
			# print(data["keypoints"])
			# print(data["centers"])
			break