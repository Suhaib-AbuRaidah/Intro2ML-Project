"""
Complete inference pipeline for 6D pose estimation with error analysis.
Computes rotation/translation errors and generates visualizations.
"""
import sys
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import json
import os
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trans_pose.stage2.network_gd import TransPoseNetworkMulti
from trans_pose.stage2.dataset_stage2 import Stage2Dataset
from scipy.spatial.transform import Rotation as R


def sample_mask_at_points(o_mask, points_3d, intrinsics):
    """Sample mask object IDs at 3D point locations."""
    B, _, H, W = o_mask.shape
    fx, fy, cx, cy = intrinsics
    
    # Project to 2D
    u = (points_3d[..., 0] * fx / (points_3d[..., 2] + 1e-6)) + cx
    v = (points_3d[..., 1] * fy / (points_3d[..., 2] + 1e-6)) + cy
    
    # Normalize to [-1, 1] for grid_sample
    grid_x = 2.0 * u / (W - 1) - 1.0
    grid_y = 2.0 * v / (H - 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(2)  # [B, N, 1, 2]
    
    # Sample with nearest neighbor (preserve integer IDs)
    sampled = F.grid_sample(
        o_mask.float(), grid, 
        mode='nearest', padding_mode='zeros', align_corners=True
    )
    
    return sampled.squeeze(1).squeeze(-1).long()  # [B, N]


def rotation_error_degrees(R_pred, R_gt):
    """Compute rotation error in degrees between two rotation matrices."""
    R_diff = R_gt.T @ R_pred
    trace = np.trace(R_diff)
    trace = np.clip(trace, -1.0, 3.0)  # Numerical stability
    angle_rad = np.arccos((trace - 1.0) / 2.0)
    return np.degrees(angle_rad)


def translation_error_cm(t_pred, t_gt):
    """Compute translation error in centimeters."""
    return np.linalg.norm(t_pred - t_gt) * 100  # meters to cm


def kabsch_transform(points_src, points_dst):
    """
    Compute optimal rotation and translation using Kabsch algorithm.
    
    Args:
        points_src: [N, 3] source points (canonical keypoints)
        points_dst: [N, 3] destination points (predicted keypoints)
    
    Returns:
        R: [3, 3] rotation matrix
        t: [3] translation vector
    """
    # Center the points
    centroid_src = np.mean(points_src, axis=0)
    centroid_dst = np.mean(points_dst, axis=0)
    
    points_src_centered = points_src - centroid_src
    points_dst_centered = points_dst - centroid_dst
    
    # Compute covariance matrix
    H = points_src_centered.T @ points_dst_centered
    
    # SVD
    U, S, Vt = np.linalg.svd(H)
    
    # Compute rotation
    R = Vt.T @ U.T
    
    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute translation
    t = centroid_dst - R @ centroid_src
    
    return R, t


def run_inference_with_errors(model, dataset, device, num_samples=None):
    """
    Run inference on validation set and compute pose errors.
    
    Returns:
        results: dict with per-sample results and statistics
    """
    model.eval()
    
    all_rot_errors = []
    all_trans_errors = []
    per_object_errors = defaultdict(lambda: {'rot': [], 'trans': []})
    
    sample_results = []
    
    num_samples = len(dataset) if num_samples is None else min(num_samples, len(dataset))
    
    print(f"\n{'='*80}")
    print(f"Running inference on {num_samples} samples...")
    print(f"{'='*80}\n")
    
    with torch.no_grad():
        for idx in tqdm(range(num_samples), desc="Processing"):
            try:
                # Load sample
                sample = dataset[idx]
                
                # Prepare inputs
                rgb = sample['rgb'].unsqueeze(0).to(device)
                depth = sample['depth'].unsqueeze(0).to(device)
                sn = sample['sn'].unsqueeze(0).to(device)
                o_mask = sample['mask'].unsqueeze(0).to(device)
                
                intrinsics_raw = sample['intrinsics']
                kpts_dict = sample['keypoints']

                # Convert to list of floats
                if isinstance(intrinsics_raw, torch.Tensor):
                    intrinsics = intrinsics_raw.cpu().numpy().flatten().tolist()
                elif isinstance(intrinsics_raw, np.ndarray):
                    intrinsics = intrinsics_raw.flatten().tolist()
                elif isinstance(intrinsics_raw, (list, tuple)):
                    intrinsics = list(intrinsics_raw)
                else:
                    raise ValueError(f"Unknown intrinsics type: {type(intrinsics_raw)}")
                
                # Handle different formats
                if len(intrinsics) == 9:
                    # Full 3x3 matrix (flattened): [fx, 0, cx, 0, fy, cy, 0, 0, 1]
                    fx, _, cx, _, fy, cy, _, _, _ = intrinsics
                    intrinsics = [fx, fy, cx, cy]
                elif len(intrinsics) == 4:
                    # Already [fx, fy, cx, cy]
                    pass
                elif len(intrinsics) == 3:
                    # [fx, fy, cx] - duplicate cx as cy
                    intrinsics = [intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[2]]
                else:
                    raise ValueError(f"Unexpected intrinsics length: {len(intrinsics)}")
                
                # DEBUG first sample
                if idx == 0:
                    print(f"📐 Intrinsics: {intrinsics}")
                
                # Forward pass
                outputs = model(rgb, sn, depth, o_mask, intrinsics)

                if len(outputs) == 5:
                    seg_logits, offsets, points, trans_feat, point_labels = outputs
                elif len(outputs) == 3:
                    seg_logits, offsets, points = outputs
                    # Sample point labels from mask
                    point_labels = sample_mask_at_points(o_mask, points, intrinsics)
                else:
                    raise ValueError(f"Unexpected number of outputs: {len(outputs)}")
                
                # Move to CPU
                points = points[0].cpu().numpy()  # [N, 3]
                offsets = offsets[0].cpu().numpy()  # [N, K, 3]
                point_labels = point_labels[0].cpu().numpy()  # [N]
                
                # Get unique object IDs
                unique_objs = np.unique(point_labels)
                unique_objs = unique_objs[unique_objs > 0]  # Skip background
                
                sample_result = {
                    'sample_idx': idx,
                    'objects': {}
                }
                
                for obj_id in unique_objs:
                    obj_id_str = str(int(obj_id))
                    
                    # Get canonical keypoints
                    if obj_id_str not in dataset.canonical_kpts:
                        continue
                    
                    can_kpts = dataset.canonical_kpts[obj_id_str]  # [K, 3]
                    
                    # Get GT keypoints
                    if obj_id_str not in kpts_dict:
                        continue
                    
                    gt_kpts = np.array(kpts_dict[obj_id_str])  # [K, 3]
                    
                    # Get points and offsets for this object
                    mask_obj = (point_labels == obj_id)
                    if mask_obj.sum() == 0:
                        continue
                    
                    points_obj = points[mask_obj]  # [M, 3]
                    offsets_obj = offsets[mask_obj]  # [M, K, 3]
                    
                    # Predict keypoints (mean-shift: average offset predictions)
                    pred_kpts = (points_obj[:, None, :] + offsets_obj).mean(axis=0)  # [K, 3]
                    
                    # Compute GT pose (canonical -> GT)
                    R_gt, t_gt = kabsch_transform(can_kpts, gt_kpts)
                    T_gt = np.eye(4)
                    T_gt[:3, :3] = R_gt
                    T_gt[:3, 3] = t_gt
                    
                    # Compute predicted pose (canonical -> predicted)
                    R_pred, t_pred = kabsch_transform(can_kpts, pred_kpts)
                    T_pred = np.eye(4)
                    T_pred[:3, :3] = R_pred
                    T_pred[:3, 3] = t_pred
                    
                    # Compute errors
                    rot_err = rotation_error_degrees(R_pred, R_gt)
                    trans_err = translation_error_cm(t_pred, t_gt)
                    
                    # Store results
                    all_rot_errors.append(rot_err)
                    all_trans_errors.append(trans_err)
                    per_object_errors[obj_id_str]['rot'].append(rot_err)
                    per_object_errors[obj_id_str]['trans'].append(trans_err)
                    
                    sample_result['objects'][obj_id_str] = {
                        'rot_err_deg': float(rot_err),
                        'trans_err_cm': float(trans_err),
                        'num_points': int(mask_obj.sum()),
                        'num_keypoints': len(can_kpts),
                        'pred_kpts': pred_kpts.tolist(),
                        'gt_kpts': gt_kpts.tolist(),
                        'can_kpts': can_kpts.tolist(),
                        'T_gt': T_gt.tolist(),
                        'T_pred': T_pred.tolist()
                    }
                
                sample_results.append(sample_result)
                
            except Exception as e:
                print(f"\n⚠️  Error processing sample {idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Check if we got any results
    if len(all_rot_errors) == 0:
        print("\n❌ ERROR: No valid predictions! All samples failed.")
        return None
    
    # Compute statistics
    all_rot_errors = np.array(all_rot_errors)
    all_trans_errors = np.array(all_trans_errors)
    
    results = {
        'summary': {
            'num_samples': num_samples,
            'num_predictions': len(all_rot_errors),
            'rotation': {
                'mean': float(np.mean(all_rot_errors)),
                'median': float(np.median(all_rot_errors)),
                'std': float(np.std(all_rot_errors)),
                'min': float(np.min(all_rot_errors)),
                'max': float(np.max(all_rot_errors)),
                'outliers_90deg': int(np.sum(all_rot_errors > 90))
            },
            'translation': {
                'mean': float(np.mean(all_trans_errors)),
                'median': float(np.median(all_trans_errors)),
                'std': float(np.std(all_trans_errors)),
                'min': float(np.min(all_trans_errors)),
                'max': float(np.max(all_trans_errors)),
                'outliers_100cm': int(np.sum(all_trans_errors > 100))
            }
        },
        'per_object': {},
        'sample_results': sample_results,
        'all_rot_errors': all_rot_errors.tolist(),
        'all_trans_errors': all_trans_errors.tolist()
    }
    
    # Per-object statistics
    for obj_id, errors in per_object_errors.items():
        rot = np.array(errors['rot'])
        trans = np.array(errors['trans'])
        
        results['per_object'][obj_id] = {
            'count': len(rot),
            'rotation': {
                'mean': float(np.mean(rot)),
                'median': float(np.median(rot)),
                'std': float(np.std(rot))
            },
            'translation': {
                'mean': float(np.mean(trans)),
                'median': float(np.median(trans)),
                'std': float(np.std(trans))
            }
        }
    
    return results


def visualize_error_distributions(results, output_dir):
    """Create comprehensive error visualization plots."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    rot_errors = np.array(results['all_rot_errors'])
    trans_errors = np.array(results['all_trans_errors'])
    
    # ========================================
    # PLOT 1: Error Distributions (Histograms)
    # ========================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Rotation errors
    axes[0].hist(rot_errors, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(results['summary']['rotation']['mean'], 
                    color='red', linestyle='--', linewidth=2,
                    label=f"Mean: {results['summary']['rotation']['mean']:.2f}°")
    axes[0].axvline(results['summary']['rotation']['median'], 
                    color='green', linestyle='--', linewidth=2,
                    label=f"Median: {results['summary']['rotation']['median']:.2f}°")
    axes[0].set_xlabel('Rotation Error (degrees)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Rotation Error Distribution', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Translation errors
    axes[1].hist(trans_errors, bins=50, color='darkgreen', edgecolor='black', alpha=0.7)
    axes[1].axvline(results['summary']['translation']['mean'], 
                    color='red', linestyle='--', linewidth=2,
                    label=f"Mean: {results['summary']['translation']['mean']:.2f}cm")
    axes[1].axvline(results['summary']['translation']['median'], 
                    color='orange', linestyle='--', linewidth=2,
                    label=f"Median: {results['summary']['translation']['median']:.2f}cm")
    axes[1].set_xlabel('Translation Error (cm)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Translation Error Distribution', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_distributions.png'), dpi=150)
    plt.close()
    print(f"✅ Saved: error_distributions.png")
    
    # ========================================
    # PLOT 2: Per-Object Bar Charts
    # ========================================
    if len(results['per_object']) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        obj_ids = sorted(results['per_object'].keys(), key=lambda x: int(x))
        rot_data = [results['per_object'][oid]['rotation']['mean'] for oid in obj_ids]
        trans_data = [results['per_object'][oid]['translation']['mean'] for oid in obj_ids]
        
        # Rotation by object
        axes[0].bar(range(len(obj_ids)), rot_data, color='steelblue', edgecolor='black')
        axes[0].set_xticks(range(len(obj_ids)))
        axes[0].set_xticklabels(obj_ids, rotation=45, ha='right')
        axes[0].set_ylabel('Mean Rotation Error (degrees)', fontsize=12)
        axes[0].set_title('Mean Rotation Error by Object', fontsize=14)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Translation by object
        axes[1].bar(range(len(obj_ids)), trans_data, color='darkgreen', edgecolor='black')
        axes[1].set_xticks(range(len(obj_ids)))
        axes[1].set_xticklabels(obj_ids, rotation=45, ha='right')
        axes[1].set_ylabel('Mean Translation Error (cm)', fontsize=12)
        axes[1].set_title('Mean Translation Error by Object', fontsize=14)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'errors_by_object.png'), dpi=150)
        plt.close()
        print(f"✅ Saved: errors_by_object.png")
    
    # ========================================
    # PLOT 3: Rotation vs Translation Scatter
    # ========================================
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(rot_errors, trans_errors, c=trans_errors, 
                        cmap='coolwarm', s=50, alpha=0.6, edgecolors='black')
    
    # Add threshold lines
    ax.axhline(y=20, color='red', linestyle='--', linewidth=2, alpha=0.5, label='20cm threshold')
    ax.axvline(x=30, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='30° threshold')
    
    ax.set_xlabel('Rotation Error (degrees)', fontsize=12)
    ax.set_ylabel('Translation Error (cm)', fontsize=12)
    ax.set_title('Rotation vs Translation Errors', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Translation Error (cm)', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rot_vs_trans_scatter.png'), dpi=150)
    plt.close()
    print(f"✅ Saved: rot_vs_trans_scatter.png")
    
    # ========================================
    # PLOT 4: Cumulative Error Distribution
    # ========================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Rotation CDF
    rot_sorted = np.sort(rot_errors)
    rot_cdf = np.arange(1, len(rot_sorted) + 1) / len(rot_sorted) * 100
    axes[0].plot(rot_sorted, rot_cdf, linewidth=2, color='steelblue')
    axes[0].axvline(30, color='red', linestyle='--', label='30° threshold')
    axes[0].set_xlabel('Rotation Error (degrees)', fontsize=12)
    axes[0].set_ylabel('Cumulative Percentage (%)', fontsize=12)
    axes[0].set_title('Cumulative Rotation Error', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Translation CDF
    trans_sorted = np.sort(trans_errors)
    trans_cdf = np.arange(1, len(trans_sorted) + 1) / len(trans_sorted) * 100
    axes[1].plot(trans_sorted, trans_cdf, linewidth=2, color='darkgreen')
    axes[1].axvline(20, color='red', linestyle='--', label='20cm threshold')
    axes[1].set_xlabel('Translation Error (cm)', fontsize=12)
    axes[1].set_ylabel('Cumulative Percentage (%)', fontsize=12)
    axes[1].set_title('Cumulative Translation Error', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cumulative_errors.png'), dpi=150)
    plt.close()
    print(f"✅ Saved: cumulative_errors.png")


def print_summary(results):
    """Print formatted summary of results."""
    
    print(f"\n{'='*80}")
    print("INFERENCE RESULTS SUMMARY")
    print(f"{'='*80}")
    
    summary = results['summary']
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  Total samples: {summary['num_samples']}")
    print(f"  Total predictions: {summary['num_predictions']}")
    
    print(f"\n🔄 Rotation Error:")
    print(f"  Mean:   {summary['rotation']['mean']:.2f}°")
    print(f"  Median: {summary['rotation']['median']:.2f}°")
    print(f"  Std:    {summary['rotation']['std']:.2f}°")
    print(f"  Range:  [{summary['rotation']['min']:.2f}°, {summary['rotation']['max']:.2f}°]")
    print(f"  Outliers (>90°): {summary['rotation']['outliers_90deg']}")
    
    print(f"\n📏 Translation Error:")
    print(f"  Mean:   {summary['translation']['mean']:.2f}cm")
    print(f"  Median: {summary['translation']['median']:.2f}cm")
    print(f"  Std:    {summary['translation']['std']:.2f}cm")
    print(f"  Range:  [{summary['translation']['min']:.2f}cm, {summary['translation']['max']:.2f}cm]")
    print(f"  Outliers (>100cm): {summary['translation']['outliers_100cm']}")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="6D pose inference and error analysis")
    parser.add_argument('--model', required=True, help='Path to trained model checkpoint')
    parser.add_argument('--data', required=True, help='Path to validation dataset')
    parser.add_argument('--kpts', required=True, help='Path to canonical keypoints')
    parser.add_argument('--outdir', default='inference_results', help='Output directory')
    parser.add_argument('--num-samples', type=int, default=None, help='Number of samples to process')
    parser.add_argument('--device', default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    model = TransPoseNetworkMulti(
        img_outdim=128,
        normals_outdim=128,
        points_outdim=256,
        num_keypoints=10,
        num_classes=61
    ).to(device)
    
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Loaded model from {args.model}")
    
    # Load dataset
    print("Loading dataset...")
    dataset = Stage2Dataset(
        root_dir=args.data,
        keypoints_dir=args.kpts,
        target_size=(640, 360)
    )
    print(f"✅ Loaded {len(dataset)} samples")
    
    # Run inference
    results = run_inference_with_errors(model, dataset, device, args.num_samples)
    
    if results is None:
        print("\n❌ Inference failed! Check model outputs.")
        sys.exit(1)
    
    # Save results
    os.makedirs(args.outdir, exist_ok=True)
    results_file = os.path.join(args.outdir, 'inference_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Saved results to {results_file}")
    
    # Print summary
    print_summary(results)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    visualize_error_distributions(results, args.outdir)
    
    print(f"\n✅ All results saved to {args.outdir}/")