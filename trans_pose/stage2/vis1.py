import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import sys
sys.path.append("/home/suhaib/ML_Project")
import cv2

# root_dir = "./data/transcg-data-1/transcg"

depth_gt = cv2.imread("./data/transcg-data-1/transcg/scene1/11/depth1-gt-mask.png", cv2.IMREAD_UNCHANGED)*255
print(f"img shape: {depth_gt.shape},min: {depth_gt.min()}, max: {depth_gt.max()},mean: {depth_gt.mean()}")

cv2.imshow("depth",depth_gt)
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()