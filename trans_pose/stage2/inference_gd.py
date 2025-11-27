# note GD: NEEDS MODIFICATION AND CORRECTION
import sys, os
from pathlib import Path
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from trans_pose.stage2.dataset_stage2 import Stage2Dataset
from trans_pose.stage2.network_gd import TransPoseNetworkMulti
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    # try common keys
    state = None
    for k in ("model_state_dict", "state_dict", "model"):
        if isinstance(ckpt, dict) and k in ckpt:
            state = ckpt[k]
            break
    if state is None:
        state = ckpt if isinstance(ckpt, dict) else None
    if state is None:
        raise RuntimeError(f"Could not find model state in {ckpt_path}")
    # if keys have "module." prefix, adjust
    try:
        model.load_state_dict(state)
    except RuntimeError:
        # strip module. if present
        new_state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(new_state)
    return model

def estimate_pose_kabsch(src_pts, dst_pts):
    """Estimate rigid transform T (4x4) mapping src_pts -> dst_pts using Kabsch.
    src_pts, dst_pts: (K,3) numpy arrays (matching order)"""
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    assert src.shape == dst.shape and src.shape[1] == 3
    c_src = src.mean(axis=0)
    c_dst = dst.mean(axis=0)
    src_c = src - c_src
    dst_c = dst - c_dst
    H = src_c.T @ dst_c
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = c_dst - R @ c_src
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R.astype(np.float32)
    T[:3, 3] = t.astype(np.float32)
    return T
# ...existing code...

def visualize_segmentation_and_keypoints(rgb_np, seg_pred, pred_poses_dict, obj_ids, out_dir, sample_idx):
    """
    Create side-by-side visualization of segmentation + keypoints.
    
    Args:
        rgb_np: [H, W, 3] uint8 RGB image
        seg_pred: [N] predicted class IDs
        pred_poses_dict: {obj_id_str: {'T': T, 'pred_kpts': [...], ...}}
        obj_ids: list of object IDs to visualize
        out_dir: output directory
        sample_idx: sample index (for filename)
    """
    import matplotlib.pyplot as plt
    
    H, W = rgb_np.shape[:2]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel 1: Segmentation mask overlay
    seg_visual = np.zeros((H, W, 3), dtype=np.uint8)
    colors = plt.cm.tab20(np.linspace(0, 1, len(obj_ids)))
    
    for i, obj_id in enumerate(obj_ids):
        mask = (seg_pred == int(obj_id))
        color = (colors[i, :3] * 255).astype(np.uint8)
        seg_visual[mask] = color
    
    axes[0].imshow(rgb_np)
    axes[0].imshow(seg_visual, alpha=0.4)
    axes[0].set_title('Predicted Segmentation')
    axes[0].axis('off')
    
    # Panel 2: Keypoints visualization (scatter)
    axes[1].imshow(rgb_np)
    for obj_id_str, pose_info in pred_poses_dict.items():
        pred_kpts = pose_info['pred_kpts']
        if pred_kpts.shape[0] > 0:
            # Project keypoints to 2D if needed (if in 3D, use simple projection)
            u = pred_kpts[:, 0]
            v = pred_kpts[:, 1]
            axes[1].scatter(u, v, s=100, marker='o', edgecolors='white', linewidth=2)
            for i, (ui, vi) in enumerate(zip(u, v)):
                axes[1].text(ui+5, vi+5, str(i), color='yellow', fontsize=8)
    
    axes[1].set_title('Predicted Keypoints')
    axes[1].axis('off')
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, f'sample_{sample_idx}_visualization.png')
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    return out_path


def predict_and_save(args):
    device = torch.device(args.device)
    ds = Stage2Dataset(root_dir=args.data, keypoints_dir=args.kpts,
                       target_size=(640, 360), use_gt_normals=True)
    print(f"Indexed {len(ds)} samples, {len(ds.canonical_kpts)} canonical keypoint sets loaded")

    # model init (match training params)
    params = {
        "img_outdim": 128,
        "normals_outdim": 64,
        "points_outdim": 256,
        "num_keypoints": ds.num_keypoints if hasattr(ds, "num_keypoints") else 10,
        "num_classes": args.num_classes
    }
    model = TransPoseNetworkMulti(**params).to(device)
    model = load_checkpoint(model, args.model, device)
    model.eval()

    idx = args.index
    sample = ds[idx]
    rgb = sample['rgb'].unsqueeze(0).to(device)   # (1,3,H,W)
    sn = sample['sn'].unsqueeze(0).to(device)     # (1,3,H,W)
    depth = sample['depth'].unsqueeze(0).to(device) # (1,1,H,W)
    o_mask = sample['mask'].unsqueeze(0).to(device) # (1,1,H,W)
    intr = ds.camera_intrisics
    fx, fy = float(intr[0,0]), float(intr[1,1])
    cx, cy = float(intr[0,2]), float(intr[1,2])
    intrinsics_tuple = (fx, fy, cx, cy)

    with torch.no_grad():
        out = model(rgb, sn, depth, o_mask, intrinsics_tuple)
    # training_gd forward returns: seg_logits, offsets, points, trans_feat, point_labels
    seg_logits, offsets, points, trans_feat, _ = out

    # Shapes:
    # seg_logits: [B, N, num_classes]
    # offsets: [B, N, K, 3]
    # points: [B, N, 3]
    B, N, K3 = offsets.shape[0], offsets.shape[1], offsets.shape[2]
    B0 = 0
    seg_pred = seg_logits.argmax(dim=-1)[B0].cpu().numpy()  # [N]

    # for each predicted object (exclude background 0)
    unique_ids = np.unique(seg_pred)
    unique_ids = unique_ids[unique_ids != 0]

    poses = {}
    for obj_id in unique_ids:
        mask_idx = np.where(seg_pred == int(obj_id))[0]
        if mask_idx.size == 0:
            continue
        pts_obj = points[B0, mask_idx].cpu().numpy()          # [M,3]
        offs_obj = offsets[B0, mask_idx].cpu().numpy()       # [M,K,3]
        # predicted keypoints: average (points + offsets) over points
        pred_kpts = (pts_obj[:, None, :] + offs_obj).mean(axis=0)  # [K,3]

        obj_id_str = str(int(obj_id))
        if obj_id_str not in ds.canonical_kpts:
            print(f"Skipping object {obj_id_str}: canonical keypoints not found")
            continue
        can_kpts = ds.canonical_kpts[obj_id_str]  # (K,3)
        if can_kpts.shape[0] != pred_kpts.shape[0]:
            print(f"Keypoint count mismatch for object {obj_id_str}, skipping")
            continue

        T_est = estimate_pose_kabsch(can_kpts, pred_kpts)
        poses[obj_id_str] = {
            "T": T_est,
            "pred_kpts": pred_kpts,
            "canonical_kpts": can_kpts
        }
        print(f"Object {obj_id_str}: pose estimated. T:\n{T_est}")

    # save poses
    out_dir = args.outdir
    os.makedirs(out_dir, exist_ok=True)
    
    # --- NEW: Save visualization ---
    if args.visualize:
        rgb_np = sample['rgb'].permute(1, 2, 0).cpu().numpy()
        rgb_np = (rgb_np * 255).astype(np.uint8)
        vis_path = visualize_segmentation_and_keypoints(
            rgb_np, seg_pred, poses, unique_ids, out_dir, idx
        )
        print(f"Visualization saved to {vis_path}")
    
    base = f"sample_{idx}_poses.npy"
    save_path = os.path.join(out_dir, base)
    np.save(save_path, poses)
    print(f"Saved poses to {save_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to checkpoint (.pth) containing model_state_dict")
    p.add_argument("--data", required=True, help="Path to tanscg-data-2 root (train/valid folders inside)")
    p.add_argument("--kpts", required=True, help="Path to keypoints dir used by Stage2Dataset")
    p.add_argument("--index", type=int, default=0, help="Sample index in Stage2Dataset to run inference on")
    p.add_argument("--num_classes", type=int, default=61)
    p.add_argument("--outdir", default="inference_out")
    p.add_argument("--visualize", action="store_true", help="Save visualization PNG")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    predict_and_save(args)