import torch
from datasets import TransCG_D435_Dataset
from model import PoseNet
import numpy as np
import argparse
import matplotlib.pyplot as plt
from utils import plot_histogram, plot_pose_errors

def angular_error(q1, q2):
    q1 = q1 / (np.linalg.norm(q1) + 1e-8)
    q2 = q2 / (np.linalg.norm(q2) + 1e-8)
    dot = abs(np.dot(q1, q2))
    dot = np.clip(dot, -1.0, 1.0)
    return 2 * np.arccos(dot) * 180.0 / np.pi

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/suhaib/6DPOSE/data/transcg-data-1/transcg")
    parser.add_argument("--model", default="pre_trained_models/2025-10-16 15:17:10/chkpt_best_model_val.pth")
    parser.add_argument("--use_depth", default=True, action="store_true")
    parser.add_argument("--num_objects", type=int, default=4)
    parser.add_argument("--rot_thresh", type=float, default=5.0)
    parser.add_argument("--trans_thresh", type=float, default=0.05)
    parser.add_argument("--save_figs", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = TransCG_D435_Dataset(args.data, use_depth=args.use_depth)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)

    model = PoseNet(pretrained=False, num_objects=args.num_objects, use_depth=args.use_depth).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    rot_errs, trans_errs = [], []
    correct = 0
    total = 0

    with torch.no_grad():
        for s in loader:
            rgb = s["rgb"].to(device).float()
            depth = s["depth"].to(device).float() if args.use_depth and s["depth"] is not None else None

            pq, pt = model(rgb, depth)
            pq = pq.cpu().numpy()[0]
            pt = pt.cpu().numpy()[0]

            gt_q = s["quat"].numpy().squeeze()
            gt_t = s["trans"].numpy().squeeze()

            for i in range(pq.shape[0]):
                r_err = angular_error(pq[i], gt_q[i])
                t_err = np.linalg.norm(pt[i] - gt_t[i])
                rot_errs.append(r_err)
                trans_errs.append(t_err)
                if r_err < args.rot_thresh and t_err < args.trans_thresh:
                    correct += 1
                total += 1

    rot_errs, trans_errs = np.array(rot_errs), np.array(trans_errs)
    acc = 100 * correct / total

    print("Rotation error (deg): mean={:.2f}, median={:.2f}".format(rot_errs.mean(), np.median(rot_errs)))
    print("Translation error (m): mean={:.4f}, median={:.4f}".format(trans_errs.mean(), np.median(trans_errs)))
    print("Pose accuracy (R<{:.1f}°, T<{:.2f}m): {:.2f}%".format(args.rot_thresh, args.trans_thresh, acc))

    plot_histogram(rot_errs, trans_errs, acc, args.rot_thresh)
    plot_pose_errors(rot_errs, trans_errs)

if __name__ == "__main__":
    main()
