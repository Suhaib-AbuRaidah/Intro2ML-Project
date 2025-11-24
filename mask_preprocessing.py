"""
Preprocessing script to create corrected instance masks.

This script:
1. Loads merged masks (values [0, 1]) 
2. Uses GT poses + .ply mesh bounding boxes to separate objects
3. Creates corrected masks (values [0, obj_id_1, obj_id_2, ...])
4. Saves as *-corrected.png files

Usage:
    python mask_preprocessing.py --scene scene11 --perspective 0  # Test on one
    python mask_preprocessing.py --all  # Process entire dataset
"""

import sys
import os
import argparse
import numpy as np
import cv2
import json
from pathlib import Path
from tqdm import tqdm
import glob

# Add project to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def find_ply_file(models_dir, obj_id):
    """
    Find .ply file for object ID.
    Handles naming like: 7-bottle8.ply, 4-bottle5.ply, etc.
    
    Returns:
        path to .ply file or None
    """
    # Try exact match first
    exact_path = os.path.join(models_dir, f"{obj_id}.ply")
    if os.path.exists(exact_path):
        return exact_path
    
    # Search for files starting with obj_id
    pattern = os.path.join(models_dir, f"{obj_id}-*.ply")
    matches = glob.glob(pattern)
    
    if len(matches) > 0:
        return matches[0]  # Return first match
    
    return None


def load_mesh_bbox(ply_path):
    """
    Load .ply mesh and compute 3D bounding box.
    
    Returns:
        bbox: dict with 'min' [x_min, y_min, z_min] and 'max' [x_max, y_max, z_max]
    """
    try:
        from plyfile import PlyData
        ply_data = PlyData.read(ply_path)
        vertices = ply_data['vertex']
        points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
        
        bbox = {
            'min': points.min(axis=0),
            'max': points.max(axis=0),
            'center': points.mean(axis=0)
        }
        return bbox
    except ImportError:
        print("ERROR: plyfile not installed. Install with: pip install plyfile")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR loading mesh {ply_path}: {e}")
        return None


def get_bbox_corners_3d(bbox):
    """Get 8 corners of 3D bounding box."""
    x_min, y_min, z_min = bbox['min']
    x_max, y_max, z_max = bbox['max']
    
    corners = np.array([
        [x_min, y_min, z_min],
        [x_max, y_min, z_min],
        [x_min, y_max, z_min],
        [x_max, y_max, z_min],
        [x_min, y_min, z_max],
        [x_max, y_min, z_max],
        [x_min, y_max, z_max],
        [x_max, y_max, z_max],
    ])
    return corners


def project_bbox_to_2d(bbox_3d, pose, intrinsics):
    """
    Project 3D bounding box to 2D image coordinates.
    
    Args:
        bbox_3d: [8, 3] array of 3D corners
        pose: [4, 4] transformation matrix
        intrinsics: (fx, fy, cx, cy)
    
    Returns:
        corners_2d: [8, 2] projected corners
        depth_expected: Average depth of object (for filtering)
    """
    fx, fy, cx, cy = intrinsics
    
    # Transform to world coordinates
    corners_world = (pose[:3, :3] @ bbox_3d.T).T + pose[:3, 3]
    
    # Project to 2D
    u = (corners_world[:, 0] * fx / corners_world[:, 2]) + cx
    v = (corners_world[:, 1] * fy / corners_world[:, 2]) + cy
    
    corners_2d = np.stack([u, v], axis=-1)  # [8, 2]
    
    # Average depth for filtering
    depth_expected = corners_world[:, 2].mean()
    
    return corners_2d, depth_expected


def create_convex_hull_mask(corners_2d, shape):
    """
    Create a mask from convex hull of projected 3D bbox corners.
    
    Args:
        corners_2d: [8, 2] array of 2D corner points
        shape: (H, W) image shape
    
    Returns:
        mask: Binary mask where convex hull = 1
    """
    H, W = shape
    mask = np.zeros((H, W), dtype=np.uint8)
    
    # Convert to integer coordinates
    points = corners_2d.astype(np.int32)
    
    # Compute convex hull
    hull = cv2.convexHull(points)
    
    # Fill convex hull
    cv2.fillConvexPoly(mask, hull, 1)
    
    return mask


def create_corrected_mask(scene_path, perspective_idx, models_dir, intrinsics, depth_threshold=0.15, visualize=False):
    """
    Create corrected instance mask for one perspective.
    
    Args:
        scene_path: Path to scene folder (e.g., "scene11")
        perspective_idx: Perspective number (e.g., 0)
        models_dir: Path to folder containing .ply files
        intrinsics: (fx, fy, cx, cy) camera intrinsics
        depth_threshold: Maximum depth difference (meters) for assigning pixels
        visualize: If True, show before/after comparison
    
    Returns:
        corrected_mask: numpy array with object IDs
        success: bool indicating if processing succeeded
    """
    perspective_folder = os.path.join(scene_path, str(perspective_idx))
    
    # Load metadata
    meta_path = os.path.join(scene_path, "metadata.json")
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    object_ids = meta.get('model_list', [])
    
    print(f"  Objects in scene: {object_ids}")
    
    # Load images
    gt_depth_path = os.path.join(perspective_folder, "depth1-gt.png")  # GT DEPTH!
    depth_path = os.path.join(perspective_folder, "depth1.png")  # Fallback to sensor depth
    mask_path = os.path.join(perspective_folder, "depth1-gt-mask.png")
    rgb_path = os.path.join(perspective_folder, "rgb1.png")
    
    if not os.path.exists(mask_path):
        print(f"  ERROR: Missing mask file in {perspective_folder}")
        return None, False
    
    # Try to load GT depth first, fallback to sensor depth
    if os.path.exists(gt_depth_path):
        print(f"  Using GT depth map: {gt_depth_path}")
        depth = cv2.imread(gt_depth_path, cv2.IMREAD_UNCHANGED)
        depth_meters = depth.astype(np.float32) / 1000.0
    elif os.path.exists(depth_path):
        print(f"  WARNING: GT depth not found, using sensor depth: {depth_path}")
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth_meters = depth.astype(np.float32) / 1000.0
    else:
        print(f"  ERROR: No depth file found in {perspective_folder}")
        return None, False
    
    old_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    
    # Load RGB for visualization
    rgb = None
    if os.path.exists(rgb_path):
        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    
    H, W = depth_meters.shape
    
    # Initialize corrected mask (all background)
    corrected_mask = np.zeros((H, W), dtype=np.uint8)
    
    # Load poses
    pose_folder = os.path.join(perspective_folder, "corrected_pose")
    if not os.path.exists(pose_folder):
        print(f"  ERROR: Pose folder not found: {pose_folder}")
        return None, False
    
    # Collect all objects with their data
    object_data = []
    
    for obj_id in object_ids:
        pose_path = os.path.join(pose_folder, f"{obj_id}.npy")
        
        if not os.path.exists(pose_path):
            print(f"    WARNING: Pose not found for object {obj_id}")
            continue
        
        # Find .ply file
        ply_path = find_ply_file(models_dir, obj_id)
        if ply_path is None:
            print(f"    WARNING: Mesh not found for object {obj_id}")
            continue
        
        # Load pose
        pose = np.load(pose_path, allow_pickle=True)
        if pose.shape != (4, 4):
            print(f"    WARNING: Invalid pose shape for object {obj_id}")
            continue
        
        # Load mesh bounding box
        bbox = load_mesh_bbox(ply_path)
        if bbox is None:
            continue
        
        # Get depth (Z translation)
        depth_obj = pose[2, 3]
        
        # Get 3D bbox corners
        bbox_corners = get_bbox_corners_3d(bbox)
        
        # Project to 2D
        corners_2d, depth_expected = project_bbox_to_2d(bbox_corners, pose, intrinsics)
        
        object_data.append({
            'obj_id': obj_id,
            'pose': pose,
            'bbox': bbox,
            'corners_2d': corners_2d,
            'depth': depth_obj,
            'depth_expected': depth_expected,
            'ply_path': ply_path
        })
    
    # Sort by depth (FARTHEST FIRST so closer objects overwrite)
    object_data.sort(key=lambda x: x['depth'], reverse=True)
    
    print(f"  Processing {len(object_data)} objects in depth order (farthest first):")
    
    # Store bbox info for visualization
    bbox_info = []
    
    # Track which objects got pixels assigned
    objects_with_pixels = {}  # obj_id -> pixel_count
    
    # Process each object in depth order
    objects_processed = 0
    for obj_info in object_data:
        obj_id = obj_info['obj_id']
        corners_2d = obj_info['corners_2d']
        depth_expected = obj_info['depth_expected']
        
        print(f"  Processing object {obj_id} (depth={obj_info['depth']:.3f}m)...")
        
        # Create convex hull mask for this object
        hull_mask = create_convex_hull_mask(corners_2d, (H, W))
        
        # Get bounding box for visualization
        u_min = int(np.floor(corners_2d[:, 0].min()))
        u_max = int(np.ceil(corners_2d[:, 0].max()))
        v_min = int(np.floor(corners_2d[:, 1].min()))
        v_max = int(np.ceil(corners_2d[:, 1].max()))
        
        bbox_info.append({
            'obj_id': obj_id,
            'u_min': u_min,
            'u_max': u_max,
            'v_min': v_min,
            'v_max': v_max,
            'corners_2d': corners_2d,
            'depth': obj_info['depth'],
            'hull_mask': hull_mask.copy()  # Store for fallback
        })
        
        # Assign pixels within convex hull
        pixels_assigned = 0
        
        # Only process pixels within the convex hull
        ys, xs = np.where(hull_mask > 0)
        
        for y, x in zip(ys, xs):
            # Check if pixel is marked as object in original mask
            if old_mask[y, x] == 0:
                continue
            
            # Check depth consistency WITH GT DEPTH (should be very accurate!)
            pixel_depth = depth_meters[y, x]
            if pixel_depth <= 0:
                continue
            
            if abs(pixel_depth - depth_expected) > depth_threshold:
                continue
            
            # Assign object ID
            corrected_mask[y, x] = obj_id
            pixels_assigned += 1
        
        objects_with_pixels[obj_id] = pixels_assigned
        print(f"    Assigned {pixels_assigned} pixels to object {obj_id}")
        objects_processed += 1
    
    # FALLBACK: Assign remaining unassigned pixels to objects with low coverage
    print(f"\n  Checking for objects with missing pixels...")
    
    # Count pixels in original mask per object's hull
    for bbox in bbox_info:
        obj_id = bbox['obj_id']
        hull_mask = bbox['hull_mask']
        
        # Count how many pixels in original mask fall in this object's hull
        pixels_in_hull = np.sum((old_mask > 0) & (hull_mask > 0))
        pixels_assigned = objects_with_pixels.get(obj_id, 0)
        
        coverage = pixels_assigned / max(pixels_in_hull, 1)
        
        print(f"  Object {obj_id}: {pixels_assigned}/{pixels_in_hull} pixels ({coverage*100:.1f}% coverage)")
        
        # If coverage is low, use fallback (no depth check)
        if coverage < 0.7 and pixels_in_hull > 100:  # Less than 70% coverage
            print(f"    FALLBACK: Object {obj_id} has low coverage, assigning remaining pixels...")
            
            # Find unassigned pixels in this hull
            unassigned_in_hull = (old_mask > 0) & (corrected_mask == 0) & (hull_mask > 0)
            
            if unassigned_in_hull.sum() > 0:
                # Assign ALL unassigned pixels in hull (no depth check in fallback)
                corrected_mask[unassigned_in_hull] = obj_id
                fallback_assigned = unassigned_in_hull.sum()
                print(f"    FALLBACK assigned {fallback_assigned} additional pixels to object {obj_id}")
                objects_with_pixels[obj_id] += fallback_assigned
    
    print(f"\n  Total objects processed: {objects_processed}/{len(object_ids)}")
    print(f"  Final pixel counts: {objects_with_pixels}")
    
    # Visualize if requested
    if visualize and rgb is not None:
        import matplotlib.pyplot as plt
        
        # Create figure with larger size
        fig = plt.figure(figsize=(24, 16))
        
        # 1. Original mask (top left)
        ax1 = plt.subplot(2, 2, 1)
        ax1.imshow(old_mask, cmap='gray')
        ax1.set_title(f"Original Mask\nValues: {np.unique(old_mask)}\nTotal pixels: {(old_mask > 0).sum()}", 
                     fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # 2. RGB with convex hull bounding boxes (top right)
        ax2 = plt.subplot(2, 2, 2)
        rgb_with_bbox = rgb.copy()
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
        for i, bbox in enumerate(bbox_info):
            color = colors[i % len(colors)]
            obj_id = bbox['obj_id']
            
            # Draw convex hull
            corners_2d = bbox['corners_2d'].astype(np.int32)
            hull = cv2.convexHull(corners_2d)
            cv2.polylines(rgb_with_bbox, [hull], True, color, 4)
            
            # Draw 3D bbox corners
            for corner in corners_2d:
                if 0 <= corner[0] < W and 0 <= corner[1] < H:
                    cv2.circle(rgb_with_bbox, tuple(corner.astype(int)), 8, color, -1)
            
            # Draw edges of 3D bbox
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
                (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
                (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
            ]
            for edge in edges:
                pt1 = tuple(corners_2d[edge[0]].astype(int))
                pt2 = tuple(corners_2d[edge[1]].astype(int))
                if (0 <= pt1[0] < W and 0 <= pt1[1] < H and 
                    0 <= pt2[0] < W and 0 <= pt2[1] < H):
                    cv2.line(rgb_with_bbox, pt1, pt2, color, 3)
            
            # Label with pixel count
            pixel_count = objects_with_pixels.get(obj_id, 0)
            cv2.putText(rgb_with_bbox, f"ID:{obj_id} ({pixel_count}px)", 
                       (bbox['u_min'], max(20, bbox['v_min']-15)),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        
        ax2.imshow(rgb_with_bbox)
        ax2.set_title("RGB + Convex Hull of 3D BBox", fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        # 3. Corrected mask (bottom left)
        ax3 = plt.subplot(2, 2, 3)
        ax3.imshow(corrected_mask, cmap='tab20', vmin=0, vmax=60)
        ax3.set_title(f"Corrected Mask\nValues: {np.unique(corrected_mask)}\nTotal pixels: {(corrected_mask > 0).sum()}", 
                     fontsize=14, fontweight='bold')
        ax3.axis('off')
        
        # 4. RGB overlay (bottom right)
        ax4 = plt.subplot(2, 2, 4)
        cmap = plt.colormaps.get_cmap('tab20')
        mask_colored = np.zeros_like(rgb)
        for i, mask_val in enumerate(np.unique(corrected_mask)):
            if mask_val == 0:
                continue
            color = cmap(i / max(len(object_ids), 1))[:3]
            mask_colored[corrected_mask == mask_val] = (np.array(color) * 255).astype(np.uint8)
        
        overlay = cv2.addWeighted(rgb, 0.6, mask_colored, 0.4, 0)
        ax4.imshow(overlay)
        ax4.set_title("RGB + Corrected Mask Overlay", fontsize=14, fontweight='bold')
        ax4.axis('off')
        
        plt.tight_layout()
        
        # Maximize window
        mng = plt.get_current_fig_manager()
        try:
            mng.window.state('zoomed')  # Windows
        except:
            try:
                mng.frame.Maximize(True)  # Alternative
            except:
                pass
        
        plt.show()
    
    success = objects_processed > 0
    return corrected_mask, success


def process_one_sample(scene_name, perspective_idx, data_root, models_dir, intrinsics, visualize=True, depth_threshold=0.15):
    """Process a single scene/perspective for testing."""
    scene_path = os.path.join(data_root, scene_name)
    
    if not os.path.exists(scene_path):
        print(f"ERROR: Scene not found: {scene_path}")
        return False
    
    print(f"\nProcessing: {scene_name}/perspective_{perspective_idx}")
    print("="*60)
    
    corrected_mask, success = create_corrected_mask(
        scene_path, perspective_idx, models_dir, intrinsics, 
        depth_threshold=depth_threshold, visualize=visualize
    )
    
    if not success:
        print("  FAILED: Could not create corrected mask")
        return False
    
    # Save corrected mask
    perspective_folder = os.path.join(scene_path, str(perspective_idx))
    output_path = os.path.join(perspective_folder, "depth1-gt-mask-corrected.png")
    cv2.imwrite(output_path, corrected_mask)
    
    print(f"\n  SUCCESS: Saved corrected mask to {output_path}")
    print(f"  Original mask values: [0, 1]")
    print(f"  Corrected mask values: {np.unique(corrected_mask)}")
    
    return True


def process_all_scenes(data_root, models_dir, intrinsics, depth_threshold=0.15):
    """Process entire dataset."""
    scenes = sorted([d for d in Path(data_root).glob("scene*") if d.is_dir()])
    
    print(f"Found {len(scenes)} scenes")
    print("="*60)
    
    total_processed = 0
    total_failed = 0
    
    for scene in tqdm(scenes, desc="Processing scenes"):
        scene_path = str(scene)
        scene_name = scene.name
        
        # Load metadata
        meta_path = os.path.join(scene_path, "metadata.json")
        if not os.path.exists(meta_path):
            continue
        
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        valid_perspectives = meta.get('D435_valid_perspective_list', [])
        
        for persp_idx in tqdm(valid_perspectives, desc=f"  {scene_name}", leave=False):
            corrected_mask, success = create_corrected_mask(
                scene_path, persp_idx, models_dir, intrinsics, 
                depth_threshold=depth_threshold, visualize=False
            )
            
            if success:
                # Save
                perspective_folder = os.path.join(scene_path, str(persp_idx))
                output_path = os.path.join(perspective_folder, "depth1-gt-mask-corrected.png")
                cv2.imwrite(output_path, corrected_mask)
                total_processed += 1
            else:
                total_failed += 1
    
    print("\n" + "="*60)
    print(f"SUMMARY: {total_processed} processed, {total_failed} failed")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Preprocess instance masks")
    parser.add_argument('--scene', type=str, default='scene4', help='Scene name (e.g., scene11)')
    parser.add_argument('--perspective', type=int, default=5, help='Perspective index')
    parser.add_argument('--all', action='store_true', help='Process all scenes')
    parser.add_argument('--depth-thresh', type=float, default=0.15, help='Depth threshold in meters')
    parser.add_argument('--data_root', type=str, 
                        default=r'C:\Users\user\Desktop\AUB\Intro2ML\Project\Intro2ML-Project\tanscg-data-2\train',
                        help='Path to training data')
    parser.add_argument('--models_dir', type=str,
                        default=r'C:\Users\user\Desktop\AUB\Intro2ML\Project\TransCG\models',
                        help='Path to .ply mesh files')
    parser.add_argument('--no-viz', action='store_true', help='Disable visualization')
    
    args = parser.parse_args()
    
    # Camera intrinsics (D435, original resolution 1280x720)
    intrinsics = (927.17, 927.37, 651.32, 349.62)
    
    print("="*60)
    print("MASK PREPROCESSING TOOL")
    print("="*60)
    print(f"Data root: {args.data_root}")
    print(f"Models dir: {args.models_dir}")
    print(f"Intrinsics: {intrinsics}")
    print(f"Depth threshold: {args.depth_thresh}m")
    print()
    
    # Check paths exist
    if not os.path.exists(args.data_root):
        print(f"ERROR: Data root not found: {args.data_root}")
        return
    
    if not os.path.exists(args.models_dir):
        print(f"ERROR: Models directory not found: {args.models_dir}")
        return
    
    if args.all:
        # Process entire dataset
        process_all_scenes(args.data_root, args.models_dir, intrinsics, args.depth_thresh)
    else:
        # Process single sample (for testing)
        process_one_sample(
            args.scene, 
            args.perspective, 
            args.data_root, 
            args.models_dir, 
            intrinsics,
            visualize=not args.no_viz,
            depth_threshold=args.depth_thresh
        )


if __name__ == "__main__":
    main()