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

def main():
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
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    with torch.no_grad():
        sample = ds[212]
        # collate_fn returned list of samples; handle arbitrary batch size
        rgb = sample["rgb"].unsqueeze(0).to(device)           # (1,3,H,W)
        depth = None
        if args.use_depth and sample.get("depth") is not None:
            depth = sample["depth"].unsqueeze(0).to(device)  # (1,1,H,W)

        rot_pred, trans_pred = model(rgb, depth)  # rot: (1, n_obj, 4), trans: (1, n_obj, 3)
        rot_pred = rot_pred.cpu().numpy()[0]      # (n_obj,4)
        trans_pred = trans_pred.cpu().numpy()[0]  # (n_obj,3)

        # sample naming
        base = os.path.splitext(os.path.basename(sample["path"]))[0]

        # object id ordering from dataset
        obj_ids = ds.object_ids
        poses = []
        for i, q in enumerate(rot_pred):
            # ensure quaternion is (x,y,z,w) as used by scipy
            q = q / (np.linalg.norm(q) + 1e-8)
            Rm = Rscipy.from_quat(q).as_matrix()
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = Rm
            T[:3, 3] = trans_pred[i]
            poses.append(T)

        print(f"Saved {len(rot_pred)} poses for {base}")
        k= np.load("/home/suhaib/6DPOSE/data/transcg-info/transcg/camera_intrinsics/1-camIntrinsics-D435.npy")
        img = draw_pose_on_rgb(sample["rgb"].numpy().transpose(1,2,0), poses, k)

        cv2.imshow("img", img)
        print("Press (q) to quit")
        if cv2.waitKey(0)==ord('q'):
            cv2.destroyAllWindows()




if __name__ == "__main__":
    main()
