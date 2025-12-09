"""
6D Pose Visualization Script

Visualizes predicted and ground truth 6D poses overlaid on RGB images.
Shows both bounding box projections and coordinate axes.

Usage:
    python visualize_6d_pose.py --model checkpoints_multi/best_model.pth \
                                 --data /path/to/valid \
                                 --kpts /path/to/keypoints \
                                 --num-samples 3 \
                                 --outdir pose_visualizations
"""
import sys
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import json
import os
from tqdm import tqdm
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trans_pose.stage2.network_gd import TransPoseNetworkMulti
from trans_pose.stage2.dataset_stage2 import Stage2Dataset
from trans_pose.stage2.inference_and_analysis import kabsch_transform, sample_mask_at_points
from trans_pose.stage2.utilis import mean_shift_clustering, votes_from_offsets


def draw_axis(img, R, t, K, scale=0.1):
    """
    Draw 3D coordinate axes on image.
    
    Args:
        img: RGB image
        R: [3, 3] rotation matrix
        t: [3] translation vector
        K: [3, 3] camera intrinsic matrix
        scale: axis length in meters
    
    Returns:
        img: image with axes drawn
    """
    # 3D axis points (origin + 3 directions)
    points_3d = np.array([
        [0, 0, 0],
        [scale, 0, 0],  # X-axis (red)
        [0, scale, 0],  # Y-axis (green)
        [0, 0, scale],  # Z-axis (blue)
    ], dtype=np.float32)
    
    # Transform to camera frame
    points_cam = (R @ points_3d.T).T + t
    
    # Project to 2D
    points_2d = (K @ points_cam.T).T
    points_2d = points_2d[:, :2] / points_2d[:, 2:3]
    points_2d = points_2d.astype(np.int32)
    
    origin = tuple(points_2d[0])
    
    # Draw axes
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR: Red, Green, Blue
    for i, color in enumerate(colors, start=1):
        end_point = tuple(points_2d[i])
        cv2.line(img, origin, end_point, color, 3)
        cv2.circle(img, end_point, 5, color, -1)
    
    return img


def draw_3d_bbox(img, bbox_corners, K, color=(0, 255, 0), thickness=2):
    """
    Draw 3D bounding box on image.
    
    Args:
        img: RGB image
        bbox_corners: [8, 3] 3D corners in camera frame
        K: [3, 3] camera intrinsic matrix
        color: BGR color tuple
        thickness: line thickness
    """
    # Project to 2D
    corners_2d = (K @ bbox_corners.T).T
    corners_2d = corners_2d[:, :2] / corners_2d[:, 2:3]
    corners_2d = corners_2d.astype(np.int32)
    
    # Define edges of bbox (12 edges connecting 8 corners)
    edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),  # Bottom face
        (4, 5), (5, 7), (7, 6), (6, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7),  # Vertical edges
    ]
    
    for i, j in edges:
        pt1 = tuple(corners_2d[i])
        pt2 = tuple(corners_2d[j])
        cv2.line(img, pt1, pt2, color, thickness)
    
    return img


def get_bbox_corners_from_keypoints(keypoints):
    """
    Compute 3D bounding box corners from keypoints.
    
    Args:
        keypoints: [K, 3] 3D keypoints
    
    Returns:
        corners: [8, 3] bbox corners
    """
    min_pt = keypoints.min(axis=0)
    max_pt = keypoints.max(axis=0)
    
    corners = np.array([
        [min_pt[0], min_pt[1], min_pt[2]],
        [max_pt[0], min_pt[1], min_pt[2]],
        [min_pt[0], max_pt[1], min_pt[2]],
        [max_pt[0], max_pt[1], min_pt[2]],
        [min_pt[0], min_pt[1], max_pt[2]],
        [max_pt[0], min_pt[1], max_pt[2]],
        [min_pt[0], max_pt[1], max_pt[2]],
        [max_pt[0], max_pt[1], max_pt[2]],
    ])
    
    return corners


def visualize_sample(sample_idx, model, dataset, device, output_dir, intrinsics_matrix):
    """
    Visualize one sample with GT and predicted 6D poses.
    
    Creates a grid showing:
    - Original RGB
    - GT pose overlay
    - Predicted pose overlay
    - Side-by-side comparison
    """
    print(f"\nProcessing sample {sample_idx}...")
    
    # Load sample
    sample = dataset[sample_idx]
    
    # Prepare inputs
    rgb = sample['rgb'].unsqueeze(0).to(device)
    depth = sample['depth'].unsqueeze(0).to(device)
    sn = sample['sn'].unsqueeze(0).to(device)
    o_mask = sample['mask'].unsqueeze(0).to(device)
    
    intrinsics_raw = sample['intrinsics']
    kpts_dict = sample['keypoints']
    
    # Fix intrinsics format
    if isinstance(intrinsics_raw, torch.Tensor):
        intrinsics = intrinsics_raw.cpu().numpy().flatten().tolist()
    elif isinstance(intrinsics_raw, np.ndarray):
        intrinsics = intrinsics_raw.flatten().tolist()
    elif isinstance(intrinsics_raw, (list, tuple)):
        intrinsics = list(intrinsics_raw)
    
    if len(intrinsics) == 9:
        fx, _, cx, _, fy, cy, _, _, _ = intrinsics
        intrinsics = [fx, fy, cx, cy]
    elif len(intrinsics) == 3:
        intrinsics = [intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[2]]
    
    fx, fy, cx, cy = intrinsics
    
    # Convert RGB tensor to numpy for visualization
    rgb_np = (sample['rgb'].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    rgb_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)  # For cv2 drawing
    
    # Forward pass
    with torch.no_grad():
        outputs = model(rgb, sn, depth, o_mask, intrinsics)
        
        if len(outputs) == 5:
            seg_logits, offsets, points, trans_feat, point_labels = outputs
        else:
            seg_logits, offsets, points = outputs
            point_labels = sample_mask_at_points(o_mask, points, intrinsics)
    
    # Move to CPU
    points = points[0].cpu().numpy()
    offsets = offsets[0].cpu().numpy()
    point_labels = point_labels[0].cpu().numpy()
    
    # Get unique objects
    unique_objs = np.unique(point_labels)
    unique_objs = unique_objs[unique_objs > 0]
    
    if len(unique_objs) == 0:
        print(f"  No objects detected in sample {sample_idx}")
        return False
    
    print(f"  Found {len(unique_objs)} objects: {list(unique_objs)}")
    
    # Create visualization canvases
    img_gt = rgb_np.copy()
    img_pred = rgb_np.copy()
    
    # Process each object
    for obj_id in unique_objs:
        obj_id_str = str(int(obj_id))
        
        # Check if canonical keypoints exist
        if obj_id_str not in dataset.canonical_kpts:
            print(f"  Skipping object {obj_id_str}: no canonical keypoints")
            continue
        
        can_kpts = dataset.canonical_kpts[obj_id_str]
        
        # Check if GT keypoints exist
        if obj_id_str not in kpts_dict:
            print(f"  Skipping object {obj_id_str}: no GT keypoints")
            continue
        
        gt_kpts = np.array(kpts_dict[obj_id_str])
        
        # Get points for this object
        mask_obj = (point_labels == obj_id)
        if mask_obj.sum() == 0:
            continue
        
        points_obj = points[mask_obj]
        offsets_obj = offsets[mask_obj]
        
        # Predict keypoints using mean-shift
        votes = points_obj[:, None, :] + offsets_obj  # [M, K, 3]
        votes_tensor = torch.tensor(votes, device=device).unsqueeze(0)  # [1, M, K, 3]
        pred_kpts = mean_shift_clustering(votes_tensor, bandwidth=0.05, num_iters=15)
        pred_kpts = pred_kpts[0].cpu().numpy()  # [K, 3]
        
        # Compute GT pose
        R_gt, t_gt = kabsch_transform(can_kpts, gt_kpts)
        
        # Compute predicted pose
        R_pred, t_pred = kabsch_transform(can_kpts, pred_kpts)
        
        # Draw on GT image
        draw_axis(img_gt, R_gt, t_gt, intrinsics_matrix, scale=0.05)
        
        # Draw GT bbox
        gt_bbox = get_bbox_corners_from_keypoints(gt_kpts)
        draw_3d_bbox(img_gt, gt_bbox, intrinsics_matrix, color=(0, 255, 0), thickness=2)
        
        # Draw on predicted image
        draw_axis(img_pred, R_pred, t_pred, intrinsics_matrix, scale=0.05)
        
        # Draw predicted bbox
        pred_bbox = get_bbox_corners_from_keypoints(pred_kpts)
        draw_3d_bbox(img_pred, pred_bbox, intrinsics_matrix, color=(0, 0, 255), thickness=2)
        
        # Add labels
        cv2.putText(img_gt, f"OBJ {obj_id_str}", (10, 30 + int(obj_id) * 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img_pred, f"OBJ {obj_id_str}", (10, 30 + int(obj_id) * 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Create comparison grid
    h, w = rgb_np.shape[:2]
    
    # Top row: Original + GT
    top_row = np.hstack([rgb_np, img_gt])
    
    # Bottom row: Predicted + Overlay (GT=green, Pred=red)
    img_overlay = img_gt.copy()
    img_overlay = cv2.addWeighted(img_overlay, 0.5, img_pred, 0.5, 0)
    bottom_row = np.hstack([img_pred, img_overlay])
    
    # Combine
    grid = np.vstack([top_row, bottom_row])
    
    # Add titles
    cv2.putText(grid, "Original", (w//4 - 50, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(grid, "Ground Truth", (w + w//4 - 80, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(grid, "Predicted", (w//4 - 50, h + 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(grid, "Overlay", (w + w//4 - 50, h + 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    
    # Save
    output_path = os.path.join(output_dir, f"sample_{sample_idx:04d}_pose_viz.png")
    cv2.imwrite(output_path, grid)
    print(f"  ✅ Saved: {output_path}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Visualize 6D pose predictions")
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--data', required=True, help='Path to validation dataset')
    parser.add_argument('--kpts', required=True, help='Path to canonical keypoints')
    parser.add_argument('--num-samples', type=int, default=3, help='Number of samples to visualize')
    parser.add_argument('--outdir', default='pose_visualizations', help='Output directory')
    parser.add_argument('--device', default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.outdir, exist_ok=True)
    
    print("="*80)
    print("6D POSE VISUALIZATION")
    print("="*80)
    
    # Load model
    print("\nLoading model...")
    model = TransPoseNetworkMulti(
        img_outdim=128,
        normals_outdim=128,
        points_outdim=256,
        num_keypoints=10,
        num_classes=61
    ).to(device)
    
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✅ Loaded: {args.model}")
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = Stage2Dataset(
        root_dir=args.data,
        keypoints_dir=args.kpts,
        target_size=(640, 360)
    )
    print(f"✅ Loaded {len(dataset)} samples")
    
    # Get camera intrinsics
    intrinsics_matrix = dataset.camera_intrisics
    print(f"\n📐 Camera Intrinsics:\n{intrinsics_matrix}")
    
    # Process samples
    print(f"\nVisualizing {args.num_samples} samples...")
    print("="*80)
    
    success_count = 0
    for idx in range(min(args.num_samples, len(dataset))):
        success = visualize_sample(idx, model, dataset, device, args.outdir, intrinsics_matrix)
        if success:
            success_count += 1
    
    print("\n" + "="*80)
    print(f"DONE: {success_count}/{args.num_samples} visualizations created")
    print(f"Output directory: {args.outdir}/")
    print("="*80)


if __name__ == "__main__":
    main()