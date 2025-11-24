import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import json

def visualize_scene(scene_path, perspective_idx=0):
    """
    Visualize a single scene to understand the data structure.
    """
    print(f"\n{'='*60}")
    print(f"Analyzing: {scene_path}")
    print(f"{'='*60}\n")
    
    # 1. Load metadata
    meta_path = os.path.join(scene_path, "metadata.json")
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    print("METADATA:")
    print(f"  Scene type: {meta.get('type', 'unknown')}")
    print(f"  Objects in scene (model_list): {meta.get('model_list', [])}")
    print(f"  Valid perspectives: {len(meta.get('D435_valid_perspective_list', []))}")
    
    # 2. Load perspective data
    perspective_folder = os.path.join(scene_path, str(perspective_idx))
    
    rgb_path = os.path.join(perspective_folder, "rgb1.png")
    depth_path = os.path.join(perspective_folder, "depth1.png")
    mask_path = os.path.join(perspective_folder, "depth1-gt-mask.png")
    
    print(f"\nCHECKING FILES:")
    print(f"  RGB exists: {os.path.exists(rgb_path)}")
    print(f"  Depth exists: {os.path.exists(depth_path)}")
    print(f"  Mask exists: {os.path.exists(mask_path)}")
    
    if not all([os.path.exists(p) for p in [rgb_path, depth_path, mask_path]]):
        print("\nERROR: Some files are missing!")
        return
    
    # Load images
    rgb = cv2.imread(rgb_path)
    if rgb is None:
        print(f"ERROR: Could not load RGB from {rgb_path}")
        return
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        print(f"ERROR: Could not load depth from {depth_path}")
        return
    depth_meters = depth.astype(np.float32) / 1000.0
    
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        print(f"ERROR: Could not load mask from {mask_path}")
        return
    
    print(f"\nIMAGE SHAPES:")
    print(f"  RGB: {rgb.shape}")
    print(f"  Depth: {depth.shape}")
    print(f"  Mask: {mask.shape}")
    
    # 3. Analyze mask
    print(f"\nMASK ANALYSIS:")
    unique_mask_values = np.unique(mask)
    print(f"  Unique mask values: {unique_mask_values}")
    print(f"  Number of objects (excluding background): {len(unique_mask_values) - 1}")
    
    for mask_val in unique_mask_values:
        pixel_count = np.sum(mask == mask_val)
        if mask_val == 0:
            print(f"    Mask ID {mask_val}: Background ({pixel_count} pixels)")
        else:
            print(f"    Mask ID {mask_val}: Object ({pixel_count} pixels)")
    
    # 4. Load pose data for each object in metadata
    print(f"\nPOSE DATA:")
    pose_folder = os.path.join(perspective_folder, "corrected_pose")
    print(f"  Pose folder: {pose_folder}")
    print(f"  Pose folder exists: {os.path.exists(pose_folder)}")
    
    poses_dict = {}
    object_ids_from_meta = meta.get('model_list', [])
    
    if os.path.exists(pose_folder):
        print(f"\n  Looking for pose files for objects: {object_ids_from_meta}")
        for obj_id in object_ids_from_meta:
            pose_path = os.path.join(pose_folder, f"{obj_id}.npy")
            print(f"    Checking: {pose_path} ... ", end="")
            
            if os.path.exists(pose_path):
                print("FOUND")
                try:
                    pose = np.load(pose_path, allow_pickle=True)
                    poses_dict[obj_id] = pose
                    print(f"      Pose shape: {pose.shape}")
                    print(f"      Translation (x, y, z): {pose[:3, 3]}")
                    print(f"      Rotation determinant: {np.linalg.det(pose[:3, :3]):.3f}")
                except Exception as e:
                    print(f"      ERROR loading: {e}")
            else:
                print("NOT FOUND")
    else:
        print(f"  WARNING: Pose folder does not exist!")
    
    print(f"\n  Summary: Found poses for {len(poses_dict)} objects: {list(poses_dict.keys())}")
    
    # 5. CREATE VISUALIZATION
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Scene: {os.path.basename(scene_path)}, Perspective: {perspective_idx}", fontsize=16, fontweight='bold')
    
    # RGB
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("RGB Image", fontsize=14)
    axes[0, 0].axis('off')
    
    # Depth
    depth_vis = axes[0, 1].imshow(depth_meters, cmap='jet', vmin=0, vmax=3)
    axes[0, 1].set_title(f"Depth Map (meters)\nRange: [{depth_meters.min():.2f}, {depth_meters.max():.2f}]", fontsize=14)
    axes[0, 1].axis('off')
    plt.colorbar(depth_vis, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # Mask (colored by ID)
    mask_vis = axes[0, 2].imshow(mask, cmap='tab20', vmin=0, vmax=max(unique_mask_values))
    axes[0, 2].set_title(f"Instance Mask\nUnique IDs: {unique_mask_values.tolist()}", fontsize=14)
    axes[0, 2].axis('off')
    plt.colorbar(mask_vis, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # Mask overlay on RGB
    mask_colored = np.zeros_like(rgb)
    cmap = plt.colormaps.get_cmap('tab20')  # Fixed for newer matplotlib
    for i, mask_val in enumerate(unique_mask_values):
        if mask_val == 0:
            continue
        color = cmap(i / len(unique_mask_values))[:3]
        mask_colored[mask == mask_val] = (np.array(color) * 255).astype(np.uint8)
    
    overlay = cv2.addWeighted(rgb, 0.6, mask_colored, 0.4, 0)
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title("RGB + Mask Overlay", fontsize=14)
    axes[1, 0].axis('off')
    
    # Individual objects (show first 2 non-background objects)
    non_bg_masks = unique_mask_values[unique_mask_values > 0]
    for i in range(min(2, len(non_bg_masks))):
        mask_val = non_bg_masks[i]
        obj_mask = (mask == mask_val).astype(np.uint8) * 255
        axes[1, i+1].imshow(obj_mask, cmap='gray')
        axes[1, i+1].set_title(f"Object Mask ID={mask_val}\n{np.sum(mask == mask_val)} pixels", fontsize=14)
        axes[1, i+1].axis('off')
    
    plt.tight_layout()
    save_path = f"scene_analysis_{os.path.basename(scene_path)}_persp{perspective_idx}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved as: {save_path}")
    plt.show()
    
    # 6. CRITICAL MAPPING ANALYSIS
    print(f"\n{'='*60}")
    print("CRITICAL: MASK ID -> OBJECT ID MAPPING")
    print(f"{'='*60}")
    print(f"\nFrom MASK image:")
    print(f"  Mask IDs (non-zero): {unique_mask_values[unique_mask_values > 0].tolist()}")
    print(f"\nFrom METADATA (model_list):")
    print(f"  Object IDs: {object_ids_from_meta}")
    print(f"\nFrom POSE files (found):")
    print(f"  Object IDs with poses: {list(poses_dict.keys())}")
    
    print(f"\n{'='*60}")
    print("HYPOTHESIS TESTING:")
    print(f"{'='*60}")
    
    mask_ids = sorted(unique_mask_values[unique_mask_values > 0].tolist())
    meta_ids = sorted(object_ids_from_meta)
    pose_ids = sorted(list(poses_dict.keys()))
    
    print(f"\nHypothesis 1: Mask IDs == Object IDs (Direct match)")
    print(f"  Mask IDs: {mask_ids}")
    print(f"  Object IDs: {meta_ids}")
    print(f"  Match: {mask_ids == meta_ids}")
    
    print(f"\nHypothesis 2: Mask IDs map by ORDER to metadata IDs")
    print(f"  Mask IDs (sorted): {mask_ids}")
    print(f"  Metadata IDs (sorted): {meta_ids}")
    if len(mask_ids) == len(meta_ids):
        print(f"  Counts match: YES")
        print(f"  Proposed mapping (by index):")
        for m_id, obj_id in zip(mask_ids, meta_ids):
            print(f"    Mask ID {m_id} -> Object ID {obj_id}")
    else:
        print(f"  Counts match: NO ({len(mask_ids)} mask IDs vs {len(meta_ids)} object IDs)")
    
    print(f"\nHypothesis 3: Metadata IDs match pose file names")
    print(f"  Metadata IDs: {meta_ids}")
    print(f"  Pose file IDs: {pose_ids}")
    print(f"  Match: {meta_ids == pose_ids}")
    
    print(f"\n{'='*60}")
    print("CONCLUSION:")
    print(f"{'='*60}")
    if mask_ids == meta_ids:
        print("MASK IDs DIRECTLY MATCH OBJECT IDs - Use mask values directly!")
    elif len(mask_ids) == len(meta_ids):
        print("MASK IDs are SEQUENTIAL - Map by index to metadata order")
    else:
        print("WARNING: Mismatch between mask and metadata!")


def main():
    # Path to your data
    data_root = r"C:\Users\user\Desktop\AUB\Intro2ML\Project\Intro2ML-Project\tanscg-data-2\train"
    
    # Find all scenes
    scenes = sorted([d for d in Path(data_root).glob("scene*") if d.is_dir()])
    
    if not scenes:
        print(f"ERROR: No scenes found in {data_root}")
        return
    
    print(f"Found {len(scenes)} scenes: {[s.name for s in scenes]}")
    
    # Visualize first 3 scenes
    for i, scene in enumerate(scenes[:3]):
        print(f"\n\n{'#'*60}")
        print(f"# SCENE {i+1}/{min(3, len(scenes))}")
        print(f"{'#'*60}")
        visualize_scene(str(scene), perspective_idx=0)
        
        if i < min(2, len(scenes) - 1):
            input("\nPress Enter to see next scene...")
    
    print("\n" + "="*60)
    print("DATA INSPECTION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()