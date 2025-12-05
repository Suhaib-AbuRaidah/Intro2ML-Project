import os
import json
import torch
import numpy as np
from torchvision import transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
sys.path.append("/home/suhaib/ML_Project")

from trans_pose.stage2.dataset2_stage2 import Stage2Dataset, collate_fn
from trans_pose.stage2.network import TransPoseNetwork
from trans_pose.stage2.utilis import rigid_transform_3D
from sklearn.metrics import accuracy_score

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT = "checkpoints1/best_model.pth"
BATCH_SIZE = 8
NUM_WORKERS = 4


def compute_pose_error(pred_pose, gt_pose):
    """
    pred_pose, gt_pose: (4,4)
    returns:
        rot_error_deg (float)
        trans_error (float L2)
    """
    R_pred = pred_pose[:3, :3]
    R_gt   = gt_pose[:3, :3]
    t_pred = pred_pose[:3, 3]
    t_gt   = gt_pose[:3, 3]

    # rotation error (geodesic)
    R_diff = R_pred.T @ R_gt
    trace_val = np.clip((np.trace(R_diff) - 1) / 2.0, -1.0, 1.0)
    rot_error = np.degrees(np.arccos(trace_val))
    trans_error = np.linalg.norm(t_pred - t_gt)

    return rot_error, trans_error


def evaluate():
    print(f"Loading model from {CHECKPOINT}")
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)

    params = {
        "img_outdim": 128,
        "normals_outdim": 128,
        "points_outdim": 256,
        "num_classes": 4,
        "num_keypoints": 10,
    }
    model = TransPoseNetwork(**params).float().to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Test dataset
    test_dataset = Stage2Dataset(
        root_dir="/home/suhaib/ML_Project/data/transcg-data-1/transcg",
        transforms=T.ToTensor(),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        shuffle=False, collate_fn=collate_fn,
        num_workers=NUM_WORKERS
    )

    # evaluation accumulators
    all_seg_pred = []
    all_seg_gt = []
    all_kp_errors = []
    all_rot_errors = []
    all_trans_errors = []
    all_offset_mae = []

    intrinsics = test_dataset.camera_intrisics
    intrinsics_tuple = (
        intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    )
    count=0
    num_obj=0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            rgb   = batch["rgb"].to(DEVICE)
            sn    = batch["sn"].to(DEVICE)
            depth = batch["depth"].to(DEVICE)
            o_mask = batch["mask"].to(DEVICE)

            poses_list        = batch["poses"]
            zero_kpts_list    = batch["zero_keypoints"]
            target_kpts_list  = batch["target_keypoints"]

            seg_logits, offsets, points, trans_feat = model(
                rgb, sn, depth, o_mask, intrinsics_tuple
            )

            seg_pred = seg_logits.argmax(dim=-1)

            # iterate per sample
            B = seg_pred.shape[0]
            for b in range(B):
                pts_b      = points[b]
                seg_b_pred = seg_pred[b].cpu().numpy()

                # === segmentation ===
                # rebuild GT from nearest keypoint distances (same logic as training)
                object_ids = sorted([int(k) for k in poses_list[b].keys()])
                all_kpts = []
                for oid in object_ids:
                    all_kpts.append(torch.tensor(
                        target_kpts_list[b][str(oid)], device=DEVICE
                    ))
                all_kpts = torch.stack(all_kpts, dim=0)  # (O,K,3)
                N = pts_b.shape[0]
                O, K, _ = all_kpts.shape

                d = torch.cdist(
                    pts_b, all_kpts.reshape(O*K, 3)
                ).reshape(N, O, K)
                seg_b_gt = d.min(dim=2).values.argmin(dim=1).cpu().numpy()

                all_seg_pred.extend(seg_b_pred.tolist())
                all_seg_gt.extend(seg_b_gt.tolist())

                # === offsets MAE ===
                offsets_b = offsets[b].cpu().numpy()
                # reconstruct offset GT
                offsets_gt = np.zeros_like(offsets_b)
                for o_i, oid in enumerate(object_ids):
                    mask = seg_b_gt == o_i
                    if mask.sum() == 0:
                        continue
                    kpts = np.array(target_kpts_list[b][str(oid)])
                    pts_obj = pts_b[mask].cpu().numpy()
                    offsets_gt[mask] = kpts[None] - pts_obj[:, None]
                mae = np.abs(offsets_b - offsets_gt).mean()
                all_offset_mae.append(mae)

                # === keypoint/pose errors ===
                # mean-shift predicted votes → centers_b already computed in training logic
                votes_b = offsets[b] + pts_b[:, None, :]  # (N,K,3)
                centers = []
                for o_i, oid in enumerate(object_ids):
                    mask = seg_b_pred == o_i
                    pts_votes = votes_b[mask]
                    if pts_votes.shape[0] == 0:
                        ctr = votes_b.mean(dim=0).cpu().numpy()
                    else:
                        ctr = pts_votes.mean(dim=0).cpu().numpy()
                    centers.append(ctr)
                centers = np.stack(centers, axis=0)  # (O,K,3)

                for o_i, oid in enumerate(object_ids):
                    pred_kp = centers[o_i]
                    zero_kp = np.array(zero_kpts_list[b][str(oid)])
                    gt_pose = np.array(poses_list[b][str(oid)])

                    pred_pose = rigid_transform_3D(
                        torch.tensor(zero_kp).float(),
                        torch.tensor(pred_kp).float()
                    ).cpu().numpy()

                    rot_err, trans_err = compute_pose_error(pred_pose, gt_pose)
                    num_obj+=1
                    if rot_err<10:
                        count+=1
                    all_rot_errors.append(rot_err)
                    all_trans_errors.append(trans_err)

                    kp_err = np.linalg.norm(pred_kp - np.array(target_kpts_list[b][str(oid)]), axis=1).mean()
                    all_kp_errors.append(kp_err)

    metrics = {
        "segmentation_accuracy": accuracy_score(all_seg_gt, all_seg_pred),
        "offset_mae": float(np.mean(all_offset_mae)),
        "mean_keypoint_error": float(np.mean(all_kp_errors)),
        "mean_rotation_error_deg": float(np.mean(all_rot_errors)),
        "mean_translation_error": float(np.mean(all_trans_errors)),
        "num_obj": num_obj,
        "count_rot_err_less_than_10": count
    }

    print(json.dumps(metrics, indent=4))

    with open("test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)



    import matplotlib.pyplot as plt

    # ---------------------------
    # 1. ROTATION ERROR THRESHOLD PLOTS
    # ---------------------------

    rot_errs = np.array(all_rot_errors)
    trans_errs = np.array(all_trans_errors)

    rot_thresholds = np.array([1, 3, 5, 10, 15, 20, 30])  # degrees
    trans_thresholds = np.array([0.005, 0.01, 0.02, 0.05, 0.1])  # meters

    rot_percentages = [(rot_errs < t).mean() * 100 for t in rot_thresholds]
    trans_percentages = [(trans_errs < t).mean() * 100 for t in trans_thresholds]

    rot_mean = rot_errs.mean()
    trans_mean = trans_errs.mean()

    # --- Plot rotation ---
    plt.figure(figsize=(16,10))
    plt.plot(rot_thresholds, rot_percentages, marker='o')
    plt.axvline(rot_mean, color='red', linestyle='--', linewidth=2,
                label=f"Mean = {rot_mean:.2f}°")
    plt.xlabel("Rotation Error Threshold (degrees)", fontsize=18)
    plt.ylabel("Percentage of Objects (%)", fontsize=18)
    plt.title("Rotation Error Distribution", fontsize=22)
    plt.grid(True)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig("rotation_error_thresholds.svg")
    plt.close()

    # --- Plot translation ---
    plt.figure(figsize=(16,10))
    plt.plot(trans_thresholds, trans_percentages, marker='o')
    plt.axvline(trans_mean, color='red', linestyle='--', linewidth=2,
                label=f"Mean = {trans_mean:.4f} m")
    plt.xlabel("Translation Error Threshold (meters)", fontsize=18)
    plt.ylabel("Percentage of Objects (%)", fontsize=18)
    plt.title("Translation Error Distribution", fontsize=22)
    plt.grid(True)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig("translation_error_thresholds.svg")
    plt.close()

    # ---------------------------
    # 2. SEGMENTATION METRICS PLOT
    # ---------------------------
    from sklearn.metrics import confusion_matrix

    y_true = np.array(all_seg_gt)
    y_pred = np.array(all_seg_pred)

    num_classes = len(np.unique(y_true))

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    iou_per_class = []

    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = tp + fp + fn
        iou = tp / denom if denom > 0 else 0
        iou_per_class.append(iou)

    # --- Plot segmentation results ---
    plt.figure(figsize=(16,10))
    plt.bar(np.arange(num_classes), iou_per_class)
    plt.xlabel("Class ID", fontsize=18)
    plt.ylabel("IoU", fontsize=18)
    plt.ylim(0, 1)
    plt.title("Per-Class IoU (Segmentation)", fontsize=22)
    plt.grid(True, axis='y')
    plt.xticks(np.arange(num_classes), fontsize=14)
    plt.tight_layout()
    plt.savefig("segmentation_iou.svg")
    plt.close()


if __name__ == "__main__":
    evaluate()
