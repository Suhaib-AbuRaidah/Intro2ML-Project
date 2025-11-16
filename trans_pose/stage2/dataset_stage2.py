import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
import sys
from torchvision import transforms as T
sys.path.append("/home/suhaib/ML_Project")
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
		depth = Image.open(item["depth"]).convert("L")     # depth is single channel
		mask = Image.open(item["mask"]).convert("L")

		# Load all corrected poses
		pose_dict = {}
		kepoints_dict = {}
		center_dict = {}
		for f in sorted(os.listdir(item["pose_dir"])):
			if f.endswith(".npy"):
				obj_id = f.replace(".npy", "")
				pose_path = os.path.join(item["pose_dir"], f)
				pose_dict[obj_id] = np.load(pose_path)
				kepoints_dict[obj_id] = self.keypoints[obj_id]
				center_dict[obj_id] = self.center[obj_id]

		if self.transforms:
			# apply transforms
			rgb = self.transforms(rgb).float()
			sn = self.transforms(sn).float()
			depth = self.transforms(depth).float()
			mask = self.transforms(mask).float()

		return {
			"rgb": rgb,
			"sn": sn,
			"depth": depth,
			"mask": mask,
			"poses": pose_dict,
			"keypoints": kepoints_dict,
			"centers": center_dict
		}
	
def collate_fn(batch):
    out = {}

    # Non-dict items (rgb, sn, depth, mask) → stack normally
    for key in ["rgb", "sn", "depth", "mask"]:
        out[key] = torch.stack([item[key] for item in batch], dim=0)

    # Dict items → keep as a list of dicts (no merging, no stacking)
    for key in ["poses", "keypoints", "centers"]:
        out[key] = [item[key] for item in batch]

    return out

if __name__=="__main__":
	from torchvision import transforms as T
	root_dir = "./data/transcg-data-1/transcg"
	dataset = Stage2Dataset(root_dir, transforms=T.ToTensor())
	print(len(dataset))
	dataloader = torch.utils.data.DataLoader(
	dataset, batch_size=2, drop_last=True, shuffle=False, collate_fn=collate_fn)
	for data in dataloader:
			print(data["rgb"].shape)
			print(data["sn"].shape)
			print(data["depth"].shape)
			print(data["mask"].shape)
			print(data["poses"])
			print(data["keypoints"])
			print(data["centers"].keys())
			break