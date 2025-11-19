import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as F
import sys
sys.path.append("/home/suhaib/ML_Project")
import trans_pose.stage1.surface_normals.transforms as T

from trans_pose.stage1.surface_normals.deeplabv3 import build_model
from trans_pose.stage1.surface_normals.surface_normals_dataset import TransCGSurfaceNormalDataset
from trans_pose.stage1.surface_normals import utils

def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    header = f"Epoch: [{epoch}]"
    i=0
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)


        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        #lr_scheduler.step()

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        i+=1

class NormalCriterion(nn.Module):
	def __init__(self):
		super(NormalCriterion, self).__init__()
		self.cosine_similarity = torch.nn.CosineSimilarity(dim=1)

	def forward(self, output, target):
		norm = torch.linalg.vector_norm(target, dim=1)
		norm_nonzero = torch.nonzero(norm, as_tuple=True)
		output_nonzero = output[norm_nonzero[0],:,norm_nonzero[1],norm_nonzero[2]]
		target_nonzero = target[norm_nonzero[0],:,norm_nonzero[1],norm_nonzero[2]]
		
		return torch.mean(1-self.cosine_similarity(output_nonzero, target_nonzero))


def get_transform(train):
	transforms = []
	transforms.append(T.PILToTensor())
	transforms.append(T.ConvertImageDtype(torch.float))
	transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
	if train:
		transforms.append(T.RandomHorizontalFlip(0.5))
	return T.Compose(transforms)


def show(imgs):
	if not isinstance(imgs, list):
		imgs = [imgs]
	fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
	for i, img in enumerate(imgs):
		img = img.detach()
		img = F.to_pil_image(img)
		axs[0, i].imshow(np.asarray(img))
		axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
	plt.show()

def main(save_dir="./trans_pose/stage1/surface_normals/pretrained_models"):
	# train on the GPU or on the CPU, if a GPU is not available
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	# device = torch.device('cpu')
	# use our dataset and defined transformations
	# dataset = SurfaceNormalDataset(transforms=get_transform(train=True))
	# dataset_test = SurfaceNormalDataset(transforms=get_transform(train=False))

	dataset = TransCGSurfaceNormalDataset(
		root_dir="F:/ML-Dataset/train",
		transforms=get_transform(train=True)
	)

	# dataset_test = TransCGSurfaceNormalDataset(
	# 	root_dir="/home/suhaib/6DPOSE/data/transcg-data-1/transcg",
	# 	transforms=get_transform(train=False)
	# )

	# split the dataset in train and test set
	indices = torch.randperm(len(dataset)).tolist()
	dataset = torch.utils.data.Subset(dataset, indices[:-50])
	# dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

	# define training and validation data loaders
	data_loader = torch.utils.data.DataLoader(
		dataset, batch_size=2, drop_last=True, shuffle=True, num_workers=4)

	# data_loader_test = torch.utils.data.DataLoader(
	# 	dataset_test, batch_size=1, shuffle=False, num_workers=4)
	criterion = NormalCriterion()

	# get the model using our helper function
	model = build_model()


	# move model to the right device
	model.to(device)

	# construct an optimizer
	params = [p for p in model.parameters() if p.requires_grad]
	optimizer = torch.optim.SGD(params, lr=0.005,
								momentum=0.9, weight_decay=0.0005)
	# and a learning rate scheduler
	lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
												   step_size=3,
												   gamma=0.1)

	# let's train it for 10 epochs
	num_epochs = 10
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	import time
	current_time = time.strftime("%Y%m%d-%H%M%S")
	log_dir = os.path.join(save_dir, current_time)
	torch.save(model.state_dict(), os.path.join(save_dir,"deeplabv3_0.pt"))
	for epoch in range(num_epochs):
		# train for one epoch, printing every 10 iterations
		train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, print_freq=100)
		# update the learning rate
		torch.save(model.state_dict(), os.path.join(save_dir,"deeplabv3_"+str(epoch)+".pt"))
		lr_scheduler.step()


if __name__=="__main__":
	main()