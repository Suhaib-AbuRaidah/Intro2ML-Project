from torchviz import make_dot
import torch
import sys
import torch
from torch import nn
import torch.functional as f
from torch.utils.tensorboard import SummaryWriter
from torchview import draw_graph
sys.path.append('/home/suhaib/6DPOSE')
from model import PoseNet
import os
import argparse
import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rscipy
from torch.utils.data import DataLoader

from datasets import TransCG_D435_Dataset
from model import PoseNet
from utils import draw_pose_on_rgb
import cv2


p = argparse.ArgumentParser()
p.add_argument("--data", default="/home/suhaib/6DPOSE/data/transcg-data-1/transcg", help="root folder for transcg data")
p.add_argument("--model", default="./pre_trained_models/2025-10-16 15:17:10/chkpt_best_model_val.pth", help="path to model.pth")
p.add_argument("--out", default="pred_poses", help="output folder")
p.add_argument("--bs", type=int, default=1)
p.add_argument("--use_depth", action="store_true", default=True)
p.add_argument("--num_objects", type=int, default=4)
args = p.parse_args()

os.makedirs(args.out, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

ds = TransCG_D435_Dataset(args.data, object_ids=None if args.num_objects is None else [0,32,42,47][:args.num_objects], use_depth=args.use_depth)
loader = DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=4, collate_fn=lambda x: x)

model = PoseNet(pretrained=False, num_objects=args.num_objects, use_depth=args.use_depth).to(device)

sample = ds[212]
# collate_fn returned list of samples; handle arbitrary batch size
rgb = sample["rgb"].unsqueeze(0).to(device)           # (1,3,H,W)
if args.use_depth and sample.get("depth") is not None:
    depth = sample["depth"].unsqueeze(0).to(device)  # (1,1,H,W)


model = PoseNet(pretrained=False, num_objects=args.num_objects, use_depth=args.use_depth).to(device)

torch.onnx.export(
    model,
    (rgb, depth),
    "pose_net.onnx",
    input_names=["rgb", "depth"],
    output_names=["rot", "trans"],
    opset_version=12,
)

rot, trans = model(rgb, depth)

# Combine outputs into one tensor for visualization
out = torch.cat([rot.view(rot.size(0), -1), trans.view(trans.size(0), -1)], dim=1)

# Create the graph
dot = make_dot(out, params=dict(model.named_parameters()), show_attrs=False)
dot.format = "pdf"  # or 'svg'
dot.render("pose_net_graph")




gc = PoseNet(pretrained=False, num_objects=args.num_objects, use_depth=args.use_depth).to(device)
# gc = experimentnn()

# graph = draw_graph(gc, input_data=(x), expand_nested=True,hide_module_functions= False, hide_inner_tensors= False, roll= True)

graph = draw_graph(gc, input_data=(rgb,depth), expand_nested=False,hide_module_functions= True, hide_inner_tensors= True, roll= True)
graph.visual_graph.render("graph_convolution_architecture", format="svg",)



# gc = GraphConvolution(in_features=16, out_features=16)
# # gc = experimentnn()
# writer = SummaryWriter("runs/gc_demo")

# x = torch.randn(10, 16)
# adj = torch.eye(10)
# h0 = torch.randn(10, 16)

# # Convert scalars to tensors
# lamda = torch.tensor([0.5])
# alpha = torch.tensor([0.1])
# l = torch.tensor([1.0])
# writer.add_graph(gc, (x))
# writer.add_graph(gc, (x, adj, h0, lamda, alpha, l))
# writer.close()