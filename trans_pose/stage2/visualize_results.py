"""
Comprehensive visualization script for 6D pose estimation results.
Creates plots comparing:
  - GT vs Predicted keypoints (3D scatter)
  - GT vs Predicted poses (coordinate frames)
  - Rotation/translation errors per object
  - Error distributions
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os
from collections import defaultdict

def plot_3d_keypoints(gt_kpts, pred_kpts, can_kpts, obj_id, ax):
    """Plot GT, Predicted, and Canonical keypoints in 3D."""
    # Canonical (reference)
    ax.scatter(can_kpts[:, 0], can_kpts[:, 1], can_kpts[:, 2], 
               c='green', marker='o', s=100, label='Canonical', alpha=0.7, edgecolors='darkgreen')
    
    # GT keypoints
    ax.scatter(gt_kpts[:, 0], gt_kpts[:, 1], gt_kpts[:, 2],
               c='blue', marker='s', s=80, label='GT', alpha=0.7, edgecolors='darkblue')
    
    # Predicted keypoints
    ax.scatter(pred_kpts[:, 0], pred_kpts[:, 1], pred_kpts[:, 2],
               c='red', marker='^', s=80, label='Predicted', alpha=0.7, edgecolors='darkred')
    
    # Connect keypoints with lines (skeleton)
    for i in range(len(gt_kpts) - 1):
        ax.plot([gt_kpts[i, 0], gt_kpts[i+1, 0]],
               [gt_kpts[i, 1], gt_kpts[i+1, 1]],
               [gt_kpts[i, 2], gt_kpts[i+1, 2]], 'b-', alpha=0.3, linewidth=1)
        
        ax.plot([pred_kpts[i, 0], pred_kpts[i+1, 0]],
               [pred_kpts[i, 1], pred_kpts[i+1, 1]],
               [pred_kpts[i, 2], pred_kpts[i+1, 2]], 'r--', alpha=0.3, linewidth=1)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Object {obj_id}: Keypoint Comparison')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)


def plot_pose_frames(T_gt, T_pred, ax, scale=0.05):
    """Plot GT and predicted pose coordinate frames."""
    # GT frame (blue)
    origin_gt = T_gt[:3, 3]
    axes_gt = T_gt[:3, :3]
    
    ax.quiver(origin_gt[0], origin_gt[1], origin_gt[2],
             axes_gt[0, 0]*scale, axes_gt[1, 0]*scale, axes_gt[2, 0]*scale,
             color='blue', arrow_length_ratio=0.2, linewidth=2, label='GT X')
    ax.quiver(origin_gt[0], origin_gt[1], origin_gt[2],
             axes_gt[0, 1]*scale, axes_gt[1, 1]*scale, axes_gt[2, 1]*scale,
             color='green', arrow_length_ratio=0.2, linewidth=2, label='GT Y')
    ax.quiver(origin_gt[0], origin_gt[1], origin_gt[2],
             axes_gt[0, 2]*scale, axes_gt[1, 2]*scale, axes_gt[2, 2]*scale,
             color='blue', arrow_length_ratio=0.2, linewidth=2, label='GT Z', alpha=0.5)
    
    # Predicted frame (red)
    origin_pred = T_pred[:3, 3]
    axes_pred = T_pred[:3, :3]
    
    ax.quiver(origin_pred[0], origin_pred[1], origin_pred[2],
             axes_pred[0, 0]*scale, axes_pred[1, 0]*scale, axes_pred[2, 0]*scale,
             color='red', arrow_length_ratio=0.2, linewidth=2, linestyle='--', label='Pred X')
    ax.quiver(origin_pred[0], origin_pred[1], origin_pred[2],
             axes_pred[0, 1]*scale, axes_pred[1, 1]*scale, axes_pred[2, 1]*scale,
             color='orange', arrow_length_ratio=0.2, linewidth=2, linestyle='--', label='Pred Y')
    ax.quiver(origin_pred[0], origin_pred[1], origin_pred[2],
             axes_pred[0, 2]*scale, axes_pred[1, 2]*scale, axes_pred[2, 2]*scale,
             color='red', arrow_length_ratio=0.2, linewidth=2, linestyle='--', label='Pred Z', alpha=0.5)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Pose Frames: GT vs Predicted')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)


def visualize_sample_results(results_file, sample_idx, output_dir):
    """Visualize results for one sample with GT and predictions."""
    with open(results_file) as f:
        results = json.load(f)
    
    sample = results['sample_results'][sample_idx]
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📊 Visualizing Sample {sample_idx}")
    print("="*80)
    
    for obj_id_str, obj_info in sample['objects'].items():
        if 'T_gt' not in obj_info:
            print(f"  ⚠️  Object {obj_id_str}: No GT pose available")
            continue
        
        pred_kpts = np.array(obj_info['pred_kpts'])
        can_kpts = np.array(obj_info['can_kpts'])
        T_gt = np.array(obj_info['T_gt'])
        T_pred = np.array(obj_info['T_pred'])
        
        # Transform canonical keypoints to GT and predicted poses
        gt_kpts_transformed = (T_gt[:3, :3] @ can_kpts.T).T + T_gt[:3, 3]
        pred_kpts_transformed = (T_pred[:3, :3] @ can_kpts.T).T + T_pred[:3, 3]
        
        # Create figure
        fig = plt.figure(figsize=(16, 6))
        
        # Plot 1: 3D Keypoints
        ax1 = fig.add_subplot(131, projection='3d')
        plot_3d_keypoints(gt_kpts_transformed, pred_kpts_transformed, can_kpts, obj_id_str, ax1)
        
        # Plot 2: Pose Frames
        ax2 = fig.add_subplot(132, projection='3d')
        plot_pose_frames(T_gt, T_pred, ax2, scale=0.05)
        
        # Plot 3: Error metrics text
        ax3 = fig.add_subplot(133)
        ax3.axis('off')
        
        rot_err = obj_info.get('rot_err_deg', np.nan)
        trans_err = obj_info.get('trans_err_cm', np.nan)
        
        error_text = f"""
OBJECT {obj_id_str}
{'='*40}

GT Pose:
  Translation: {T_gt[:3, 3]}
  Rotation (trace): {np.trace(T_gt[:3, :3]):.3f}

Predicted Pose:
  Translation: {T_pred[:3, 3]}
  Rotation (trace): {np.trace(T_pred[:3, :3]):.3f}

Errors:
  Rotation: {rot_err:.2f}°
  Translation: {trans_err:.2f}cm
  Keypoints: {obj_info['num_keypoints']}
  Points used: {obj_info['num_points']}

Status: {'✅ Good' if rot_err < 30 and trans_err < 20 else '⚠️  High error'}
        """
        ax3.text(0.1, 0.5, error_text, fontsize=11, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        out_path = os.path.join(output_dir, f'sample{sample_idx}_obj{obj_id_str}_comparison.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Object {obj_id_str}: rot_err={rot_err:.2f}°, trans_err={trans_err:.2f}cm")
        print(f"    → Saved to {out_path}")


def plot_error_distributions(results_file, output_dir):
    """Plot error distributions across all objects."""
    with open(results_file) as f:
        results = json.load(f)
    
    rot_errs = []
    trans_errs = []
    obj_errors = defaultdict(lambda: {'rot': [], 'trans': []})
    
    for sample in results['sample_results']:
        for obj_id_str, obj_info in sample['objects'].items():
            if 'rot_err_deg' not in obj_info:
                continue
            
            rot_err = obj_info['rot_err_deg']
            trans_err = obj_info['trans_err_cm']
            
            rot_errs.append(rot_err)
            trans_errs.append(trans_err)
            obj_errors[obj_id_str]['rot'].append(rot_err)
            obj_errors[obj_id_str]['trans'].append(trans_err)
    
    rot_errs = np.array(rot_errs)
    trans_errs = np.array(trans_errs)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Figure 1: Overall distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(rot_errs, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(np.mean(rot_errs), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rot_errs):.2f}°')
    axes[0].axvline(np.median(rot_errs), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(rot_errs):.2f}°')
    axes[0].set_xlabel('Rotation Error (degrees)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Rotation Error Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(trans_errs, bins=30, color='darkgreen', edgecolor='black', alpha=0.7)
    axes[1].axvline(np.mean(trans_errs), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(trans_errs):.2f}cm')
    axes[1].axvline(np.median(trans_errs), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(trans_errs):.2f}cm')
    axes[1].set_xlabel('Translation Error (cm)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Translation Error Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_distributions.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved error distributions to {output_dir}/error_distributions.png")
    
    # Figure 2: Per-object boxplot
    if len(obj_errors) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        obj_ids = sorted(obj_errors.keys())
        rot_data = [obj_errors[oid]['rot'] for oid in obj_ids if len(obj_errors[oid]['rot']) > 0]
        trans_data = [obj_errors[oid]['trans'] for oid in obj_ids if len(obj_errors[oid]['trans']) > 0]
        obj_ids_filtered = [oid for oid in obj_ids if len(obj_errors[oid]['rot']) > 0]
        
        axes[0].boxplot(rot_data, labels=obj_ids_filtered)
        axes[0].set_ylabel('Rotation Error (degrees)')
        axes[0].set_title('Rotation Error by Object')
        axes[0].grid(True, alpha=0.3)
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)
        
        axes[1].boxplot(trans_data, labels=obj_ids_filtered)
        axes[1].set_ylabel('Translation Error (cm)')
        axes[1].set_title('Translation Error by Object')
        axes[1].grid(True, alpha=0.3)
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'errors_by_object.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved per-object errors to {output_dir}/errors_by_object.png")


def plot_rotation_translation_scatter(results_file, output_dir):
    """Scatter plot of rotation vs translation errors."""
    with open(results_file) as f:
        results = json.load(f)
    
    rot_errs = []
    trans_errs = []
    colors = []
    
    for sample in results['sample_results']:
        for obj_id_str, obj_info in sample['objects'].items():
            if 'rot_err_deg' not in obj_info:
                continue
            
            rot_errs.append(obj_info['rot_err_deg'])
            trans_errs.append(obj_info['trans_err_cm'])
            colors.append(int(obj_id_str))
    
    rot_errs = np.array(rot_errs)
    trans_errs = np.array(trans_errs)
    colors = np.array(colors)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(rot_errs, trans_errs, c=colors, cmap='tab20', s=100, alpha=0.6, edgecolors='black')
    
    # Add threshold lines
    ax.axhline(y=20, color='red', linestyle='--', linewidth=2, alpha=0.5, label='20cm threshold')
    ax.axvline(x=30, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='30° threshold')
    
    ax.set_xlabel('Rotation Error (degrees)', fontsize=12)
    ax.set_ylabel('Translation Error (cm)', fontsize=12)
    ax.set_title('Rotation vs Translation Errors', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Object ID', fontsize=11)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'rot_vs_trans_scatter.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved scatter plot to {output_dir}/rot_vs_trans_scatter.png")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize pose estimation results")
    parser.add_argument('--results', required=True, help='Path to evaluation_results.json')
    parser.add_argument('--sample', type=int, default=0, help='Sample index to visualize')
    parser.add_argument('--outdir', default='visualization_output')
    parser.add_argument('--all-samples', action='store_true', help='Visualize all samples')
    parser.add_argument('--distributions', action='store_true', help='Plot error distributions')
    parser.add_argument('--scatter', action='store_true', help='Plot rot vs trans scatter')
    
    args = parser.parse_args()
    
    print("="*80)
    print("🎨 VISUALIZATION SCRIPT")
    print("="*80)
    
    if args.all_samples:
        with open(args.results) as f:
            results = json.load(f)
        
        for i in range(min(10, len(results['sample_results']))):
            visualize_sample_results(args.results, i, args.outdir)
    else:
        visualize_sample_results(args.results, args.sample, args.outdir)
    
    if args.distributions:
        plot_error_distributions(args.results, args.outdir)
    
    if args.scatter:
        plot_rotation_translation_scatter(args.results, args.outdir)
    
    print("\n✅ Visualization complete!")