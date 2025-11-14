import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class Stage2_dataset(Dataset):
	def __init__(self, root_dir):
		"""
		Args:
		root_dir (str): root path of scenes, e.g., ~/6DPOSE/data/transcg-data-1/transcg
		transforms (callable): torchvision-style transforms to apply to RGB and target
		"""
		self.root_dir = root_dir
		self.samples = []

		# collect all scenes and rgb-depth pairs
		for scene_name in sorted(os.listdir(root_dir)):
			scene_path = os.path.join(root_dir, scene_name)
			if not os.path.isdir(scene_path):
				continue

			for scene_name in sorted(os.listdir(scene_path)):
				scene_path_2 = os.path.join(scene_path, scene_name)
				if not os.path.isdir(scene_path_2) or scene_name=="metadata.json":
					continue    

				for fname in os.listdir(scene_path_2):
					if fname.startswith("rgb") and fname.endswith(".png"):
						idx = fname.replace("rgb", "").replace(".png", "")
						rgb_path = os.path.join(scene_path_2, fname)
						sn_path = os.path.join(scene_path_2, f"depth{idx}-gt-sn.png")
						if os.path.exists(sn_path):
							self.samples.append((rgb_path, sn_path))

		if len(self.samples) == 0:
			raise RuntimeError(f"No valid samples found in {root_dir}")

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		rgb_path, sn_path = self.samples[idx]
		rgb = Image.open(rgb_path).convert("RGB")
		sn = Image.open(sn_path).convert("RGB")  # stored as RGB normal map

		if self.transforms:
			rgb,sn = self.transforms(rgb, sn)

		# convert normal map [0,1] → [-1,1]
		sn = sn.float() * 2 - 1
		return rgb, sn

