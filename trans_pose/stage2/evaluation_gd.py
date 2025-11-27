"""
Comprehensive evaluation script for multi-object 6D pose estimation.
Computes:
  - Segmentation accuracy & per-class metrics
  - Rotation/translation errors (requires GT poses)
  - Visualizations (segmentation, keypoints)
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import json
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trans_pose.stage2.dataset_stage2 import Stage2Dataset
from trans_pose.stage2.network_gd import TransPoseNetworkMulti
from trans_pose.stage2.inference_gd import load_checkpoint, estimate_pose_kabsch


def compute_pose_errors(T_gt, T_pred):
    """
    Compute rotation and translation errors.
    
    Args:
        T_gt: [4, 4] ground truth pose
        T_pred: [4, 4] predicted pose
    
    Returns:
        rot_err_deg: rotation error in degrees
        trans_err_cm: translation error in cm
    """
    R_gt = T_gt[:3, :3]
    t_gt = T_gt[:3, 3]
    R_pred = T_pred[:3, :3]
    t_pred = T_pred[:3, 3]
    
    # Rotation error
    R_diff = R_pred @ R_gt.T
    trace = np.clip(np.trace(R_diff), -1, 3)
    angle_rad = np.arccos((trace - 1) / 2)
    rot_err_deg = np.degrees(angle_rad)
    
    # Translation error
    trans_err_cm = np.linalg.norm(t_gt - t_pred) * 100
    
    return rot_err_deg, trans_err_cm


def compute_segmentation_metrics(seg_pred, seg_gt, num_classes):
    """
    Compute per-class accuracy, precision, recall.
    
    Args:
        seg_pred: [N] predicted class IDs
        seg_gt: [N] ground truth class IDs
        num_classes: total number of classes
    
    Returns:
        dict with per-class and overall metrics
    """
    metrics = {
        'overall_acc': np.mean(seg_pred == seg_gt),
        'per_class': {}
    }
    
    for cls_id in range(num_classes):
        if cls_id == 0:  # skip background
            continue
        
        pred_mask = seg_pred == cls_id
        gt_mask = seg_gt == cls_id
        
        tp = np.sum(pred_mask & gt_mask)
        fp = np.sum(pred_mask & ~gt_mask)
        fn = np.sum(~pred_mask & gt_mask)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics['per_class'][cls_id] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn)
        }
    
    return metrics


def visualize_segmentation(rgb, seg_pred, seg_gt, obj_id, out_dir):
    """
    Visualize predicted vs GT segmentation overlayed on RGB.
    
    Args:
        rgb: [H, W, 3] uint8 RGB image
        seg_pred: [N] predicted class IDs (for sampled points, NOT full image)
        seg_gt: [N] GT class IDs
        obj_id: object ID
        out_dir: output directory for saving
    """
    os.makedirs(out_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original RGB
    axes[0].imshow(rgb)
    axes[0].set_title('RGB Input')
    axes[0].axis('off')
    
    # Show class distribution (histogram instead of image mask)
    axes[1].bar(['Pred', 'GT'], [
        (seg_pred == obj_id).sum(),
        (seg_gt == obj_id).sum()
    ])
    axes[1].set_title(f'Points for Object {obj_id}')
    axes[1].set_ylabel('# Points')
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, f'seg_comparison_obj_{obj_id}.png')
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    return out_path


def project_keypoints_to_2d(keypoints_3d, intrinsics):
    """
    Project 3D keypoints to 2D image space.
    
    Args:
        keypoints_3d: [K, 3] 3D keypoints in camera frame
        intrinsics: (fx, fy, cx, cy) camera intrinsics
    
    Returns:
        keypoints_2d: [K, 2] projected 2D keypoints (u, v)
    """
    fx, fy, cx, cy = intrinsics
    
    u = (keypoints_3d[:, 0] * fx / keypoints_3d[:, 2]) + cx
    v = (keypoints_3d[:, 1] * fy / keypoints_3d[:, 2]) + cy
    
    return np.stack([u, v], axis=-1)


def visualize_keypoints(rgb, pred_kpts_3d, can_kpts_3d, obj_id, intrinsics, out_dir):
    """
    Visualize predicted 3D keypoints projected to 2D.
    
    Args:
        rgb: [H, W, 3] uint8 RGB image
        pred_kpts_3d: [K, 3] predicted keypoints in 3D
        can_kpts_3d: [K, 3] canonical keypoints (for reference)
        obj_id: object ID
        intrinsics: (fx, fy, cx, cy)
        out_dir: output directory
    """
    os.makedirs(out_dir, exist_ok=True)
    H, W = rgb.shape[:2]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(rgb)
    
    # Project predicted keypoints to 2D
    if pred_kpts_3d.shape[0] > 0:
        pred_kpts_2d = project_keypoints_to_2d(pred_kpts_3d, intrinsics)
        
        u = np.clip(pred_kpts_2d[:, 0], 0, W - 1)
        v = np.clip(pred_kpts_2d[:, 1], 0, H - 1)
        ax.scatter(u, v, c='red', s=100, marker='o', label='Predicted KPts', zorder=5, edgecolors='white', linewidth=2)
        
        # Draw skeleton (simple line connecting keypoints)
        for i in range(len(pred_kpts_2d) - 1):
            ax.plot([u[i], u[i+1]], [v[i], v[i+1]], 'r-', linewidth=2, alpha=0.7)
        
        # Label keypoints
        for i, (ui, vi) in enumerate(zip(u, v)):
            ax.text(ui+10, vi-10, str(i), color='yellow', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    ax.set_title(f'Predicted Keypoints (Obj {obj_id})')
    ax.legend(fontsize=12)
    ax.axis('off')
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, f'keypoints_obj_{obj_id}.png')
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    return out_path


def find_gt_pose(sample, obj_id_str, args):
    """
    Locate and load GT pose for a sample/object.
    
    Tries multiple possible path formats:
    1. From sample metadata (if available)
    2. Construct from args.data and scene/perspective info
    3. Search directory tree
    
    Args:
        sample: dataset sample dict
        obj_id_str: object ID as string
        args: argument parser with data path
    
    Returns:
        T_gt: [4, 4] pose matrix or None
    """
    T_gt = None
    
    # Try method 1: sample provides direct path
    if 'gt_pose_path' in sample:
        gt_path = sample['gt_pose_path']
        if os.path.exists(gt_path):
            try:
                T_gt = np.load(gt_path)
                if T_gt.shape == (4, 4):
                    return T_gt
            except:
                pass
    
    # Try method 2: construct from scene/perspective indices
    if 'scene_idx' in sample and 'perspective_idx' in sample:
        scene_idx = sample['scene_idx']
        perspective_idx = sample['perspective_idx']
        
        # Try corrected_pose folder
        gt_path = os.path.join(
            args.data,
            f"scene{scene_idx}",
            str(perspective_idx),
            "corrected_pose",
            f"{obj_id_str}.npy"
        )
        if os.path.exists(gt_path):
            try:
                T_gt = np.load(gt_path)
                if T_gt.shape == (4, 4):
                    return T_gt
            except:
                pass
    
    # Try method 3: search for gt_poses in data directory
    # Look for patterns like: data/*/*/corrected_pose/{obj_id}.npy
    try:
        for root, dirs, files in os.walk(args.data):
            if 'corrected_pose' in root:
                gt_path = os.path.join(root, f"{obj_id_str}.npy")
                if os.path.exists(gt_path):
                    try:
                        T_gt = np.load(gt_path)
                        if T_gt.shape == (4, 4):
                            return T_gt
                    except:
                        pass
    except:
        pass
    
    return None


def evaluate_dataset(args):
    """Main evaluation loop over entire dataset."""
    device = torch.device(args.device)
    
    # Load dataset
    ds = Stage2Dataset(
        root_dir=args.data,
        keypoints_dir=args.kpts,
        target_size=(640, 360),
        use_gt_normals=True
    )
    print(f"✅ Loaded dataset: {len(ds)} samples, {len(ds.canonical_kpts)} canonical keypoint sets\n")
    
    # Initialize model
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
    print(f"✅ Model loaded from {args.model}\n")
    
    # Prepare output directory
    os.makedirs(args.outdir, exist_ok=True)
    viz_dir = os.path.join(args.outdir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Metrics storage
    seg_metrics = defaultdict(list)
    pose_errors = {'rot': [], 'trans': []}
    sample_results = []
    gt_poses_found = 0
    
    # Intrinsics
    intr = ds.camera_intrisics
    fx, fy = float(intr[0, 0]), float(intr[1, 1])
    cx, cy = float(intr[0, 2]), float(intr[1, 2])
    intrinsics_tuple = (fx, fy, cx, cy)
    
    print(f"Evaluating {len(ds)} samples...")
    print("="*80)
    
    with torch.no_grad():
        for idx in range(len(ds)):
            sample = ds[idx]
            
            rgb = sample['rgb'].unsqueeze(0).to(device)  # [1, 3, H, W]
            sn = sample['sn'].unsqueeze(0).to(device)
            depth = sample['depth'].unsqueeze(0).to(device)
            o_mask = sample['mask'].unsqueeze(0).to(device)
            
            # Forward pass
            seg_logits, offsets, points, trans_feat, point_labels = model(
                rgb, sn, depth, o_mask, intrinsics_tuple
            )
            
            # Get predictions
            seg_pred = seg_logits.argmax(dim=-1)[0].cpu().numpy()  # [N]
            point_labels_np = point_labels[0].cpu().numpy()  # [N] GT
            
            # Segmentation metrics (per-sample)
            sample_seg_metrics = compute_segmentation_metrics(
                seg_pred, point_labels_np, args.num_classes
            )
            seg_metrics['overall_acc'].append(sample_seg_metrics['overall_acc'])
            
            # Process each object in this sample
            unique_ids = np.unique(seg_pred)
            unique_ids = unique_ids[unique_ids != 0]
            
            sample_info = {
                'sample_idx': idx,
                'seg_acc': sample_seg_metrics['overall_acc'],
                'per_class_metrics': sample_seg_metrics['per_class'],
                'objects': {}
            }
            
            for obj_id in unique_ids:
                obj_id_str = str(int(obj_id))
                
                # Skip if no canonical keypoints
                if obj_id_str not in ds.canonical_kpts:
                    continue
                
                # Get points/offsets for this object
                mask_idx = np.where(seg_pred == int(obj_id))[0]
                if mask_idx.size == 0:
                    continue
                
                pts_obj = points[0, mask_idx].cpu().numpy()  # [M, 3]
                offs_obj = offsets[0, mask_idx].cpu().numpy()  # [M, K, 3]
                
                # Predict keypoints
                pred_kpts = (pts_obj[:, None, :] + offs_obj).mean(axis=0)  # [K, 3]
                
                can_kpts = ds.canonical_kpts[obj_id_str]  # [K, 3]
                
                if can_kpts.shape[0] != pred_kpts.shape[0]:
                    continue
                
                # Estimate pose
                T_pred = estimate_pose_kabsch(can_kpts, pred_kpts)
                
                # Try to load GT pose
                T_gt = find_gt_pose(sample, obj_id_str, args)
                if T_gt is not None:
                    gt_poses_found += 1
                
                # Compute errors (if GT available)
                obj_info = {
                    'obj_id': int(obj_id),
                    'pred_kpts': pred_kpts.tolist(),
                    'can_kpts': can_kpts.tolist(),
                    'T_pred': T_pred.tolist(),
                    'num_points': int(pts_obj.shape[0]),
                    'num_keypoints': int(pred_kpts.shape[0])
                }
                
                if T_gt is not None:
                    rot_err, trans_err = compute_pose_errors(T_gt, T_pred)
                    obj_info['rot_err_deg'] = float(rot_err)
                    obj_info['trans_err_cm'] = float(trans_err)
                    obj_info['T_gt'] = T_gt.tolist()
                    pose_errors['rot'].append(rot_err)
                    pose_errors['trans'].append(trans_err)
                
                sample_info['objects'][obj_id_str] = obj_info
                
                # Visualizations (optional, only first N samples to avoid clutter)
                if args.visualize and idx < args.max_vis:
                    rgb_np = sample['rgb'].permute(1, 2, 0).cpu().numpy()
                    rgb_np = (rgb_np * 255).astype(np.uint8)
                    
                    # Segmentation visualization
                    try:
                        visualize_segmentation(rgb_np, seg_pred, point_labels_np, int(obj_id), viz_dir)
                    except Exception as e:
                        pass
                    
                    # Keypoints visualization
                    try:
                        visualize_keypoints(rgb_np, pred_kpts, can_kpts, int(obj_id), intrinsics_tuple, viz_dir)
                    except Exception as e:
                        pass
            
            sample_results.append(sample_info)
            
            if (idx + 1) % 10 == 0:
                print(f"  [{idx+1}/{len(ds)}] Seg Acc: {sample_seg_metrics['overall_acc']:.4f}")
    
    # Print summary
    print("\n" + "="*80)
    print("📊 EVALUATION SUMMARY")
    print("="*80)
    
    print(f"\n🔷 Segmentation Metrics:")
    print(f"  Overall Accuracy: {np.mean(seg_metrics['overall_acc']):.4f} ± {np.std(seg_metrics['overall_acc']):.4f}")
    print(f"  Samples evaluated: {len(ds)}")
    
    if len(pose_errors['rot']) > 0:
        print(f"\n🎯 Pose Estimation Errors ({len(pose_errors['rot'])} instances):")
        print(f"  Rotation (deg):")
        print(f"    mean={np.mean(pose_errors['rot']):.2f}°")
        print(f"    median={np.median(pose_errors['rot']):.2f}°")
        print(f"    std={np.std(pose_errors['rot']):.2f}°")
        print(f"    min={np.min(pose_errors['rot']):.2f}°, max={np.max(pose_errors['rot']):.2f}°")
        print(f"  Translation (cm):")
        print(f"    mean={np.mean(pose_errors['trans']):.2f}cm")
        print(f"    median={np.median(pose_errors['trans']):.2f}cm")
        print(f"    std={np.std(pose_errors['trans']):.2f}cm")
        print(f"    min={np.min(pose_errors['trans']):.2f}cm, max={np.max(pose_errors['trans']):.2f}cm")
    else:
        print(f"\n⚠️  GT poses not found ({gt_poses_found} found)")
        print(f"  Expected format: data/scene*/perspective_idx/corrected_pose/{obj_id}.npy")
    
    # Save results
    results_file = os.path.join(args.outdir, 'evaluation_results.json')
    results_dict = {
        'timestamp': str(np.datetime64('now')),
        'model_path': args.model,
        'dataset': args.data,
        'seg_accuracy_mean': float(np.mean(seg_metrics['overall_acc'])),
        'seg_accuracy_std': float(np.std(seg_metrics['overall_acc'])),
        'pose_rot_err_mean': float(np.mean(pose_errors['rot'])) if len(pose_errors['rot']) > 0 else None,
        'pose_rot_err_median': float(np.median(pose_errors['rot'])) if len(pose_errors['rot']) > 0 else None,
        'pose_rot_err_min': float(np.min(pose_errors['rot'])) if len(pose_errors['rot']) > 0 else None,
        'pose_rot_err_max': float(np.max(pose_errors['rot'])) if len(pose_errors['rot']) > 0 else None,
        'pose_trans_err_mean': float(np.mean(pose_errors['trans'])) if len(pose_errors['trans']) > 0 else None,
        'pose_trans_err_median': float(np.median(pose_errors['trans'])) if len(pose_errors['trans']) > 0 else None,
        'pose_trans_err_min': float(np.min(pose_errors['trans'])) if len(pose_errors['trans']) > 0 else None,
        'pose_trans_err_max': float(np.max(pose_errors['trans'])) if len(pose_errors['trans']) > 0 else None,
        'total_samples': len(ds),
        'num_pose_errors': len(pose_errors['rot']),
        'gt_poses_found': gt_poses_found,
        'sample_results': sample_results
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n✅ Results saved to {results_file}")
    
    if args.visualize:
        print(f"✅ Visualizations saved to {viz_dir}")
    
    return sample_results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate multi-object 6D pose estimation")
    p.add_argument("--model", required=True, help="Path to checkpoint (.pth file)")
    p.add_argument("--data", required=True, help="Path to dataset root (valid or test folder)")
    p.add_argument("--kpts", required=True, help="Path to keypoints dir")
    p.add_argument("--num_classes", type=int, default=61)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--outdir", default="eval_results")
    p.add_argument("--visualize", action="store_true", help="Save visualizations (segmentation + keypoints)")
    p.add_argument("--max_vis", type=int, default=5, help="Max samples to visualize")
    args = p.parse_args()
    
    # --- Safety checks ---
    if not os.path.exists(args.model):
        print(f"❌ ERROR: Model checkpoint not found: {args.model}")
        sys.exit(1)
    
    if not os.path.exists(args.data):
        print(f"❌ ERROR: Data path not found: {args.data}")
        sys.exit(1)
    
    if not os.path.exists(args.kpts):
        print(f"❌ ERROR: Keypoints path not found: {args.kpts}")
        sys.exit(1)
    
    print("="*80)
    print("🚀 Starting Evaluation")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Output: {args.outdir}")
    print("="*80 + "\n")
    
    evaluate_dataset(args)