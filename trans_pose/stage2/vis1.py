import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import sys
sys.path.append("/home/suhaib/ML_Project")
import cv2

# root_dir = "./data/transcg-data-1/transcg"

depth_gt = cv2.imread("./data/transcg-data-1/transcg/scene1/1/depth1-gt-mask.png",cv2.IMREAD_UNCHANGED)
print(depth_gt.max(),depth_gt.min())
cv2.imshow("depth",depth_gt*255.0)
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()