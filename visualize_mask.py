"""
Visualize and compare mask quality for scene11.
Shows: RGB, Old Mask, New Mask, Overlay side-by-side.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import json
from pathlib import Path


def visualize_masks(scene_path, perspective_idx, save_dir=None):
    """Visualize RGB, old mask, new mask, and overlay."""
    perspective_folder = os.path.join(scene_path, str(perspective_idx))
    
    # Load images
    rgb_path = os.path.join(perspective_folder, "rgb1.png")
    old_mask_path = os.path.join(perspective_folder, "depth1-gt-mask.png")
    new_mask_path = os.path.join(perspective_folder, "depth1-gt-mask-corrected.png")
    depth_path = os.path.join(perspective_folder, "depth1-gt.png")
    
    if not os.path.exists(new_mask_path):
        print(f"WARNING: Corrected mask not found for perspective {perspective_idx}")
        return False
    
    # Load images
    rgb = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    
    old_mask = cv2.imread(old_mask_path, cv2.IMREAD_UNCHANGED)
    new_mask = cv2.imread(new_mask_path, cv2.IMREAD_UNCHANGED)
    
    depth = None
    if os.path.exists(depth_path):
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth = depth.astype(np.float32) / 1000.0  # mm to meters
    
    # Get unique object IDs
    old_ids = np.unique(old_mask)
    old_ids = old_ids[old_ids > 0]
    
    new_ids = np.unique(new_mask)
    new_ids = new_ids[new_ids > 0]
    
    # Create colored masks
    def colorize_mask(mask):
        """Colorize mask with random colors per object ID."""
        H, W = mask.shape
        colored = np.zeros((H, W, 3), dtype=np.uint8)
        
        unique_ids = np.unique(mask)
        unique_ids = unique_ids[unique_ids > 0]
        
        np.random.seed(42)  # Consistent colors
        for obj_id in unique_ids:
            color = np.random.randint(50, 255, 3)
            colored[mask == obj_id] = color
        
        return colored
    
    old_colored = colorize_mask(old_mask)
    new_colored = colorize_mask(new_mask)
    
    # Create overlays
    old_overlay = cv2.addWeighted(rgb, 0.6, old_colored, 0.4, 0)
    new_overlay = cv2.addWeighted(rgb, 0.6, new_colored, 0.4, 0)
    
    # Create figure
    fig = plt.figure(figsize=(20, 12))
    
    # Row 1: RGB and Depth
    ax1 = plt.subplot(3, 4, 1)
    ax1.imshow(rgb)
    ax1.set_title(f'RGB - Perspective {perspective_idx}', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    if depth is not None:
        ax2 = plt.subplot(3, 4, 2)
        im = ax2.imshow(depth, cmap='jet', vmin=0, vmax=2.0)
        ax2.set_title('Depth (GT)', fontsize=14, fontweight='bold')
        ax2.axis('off')
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    
    # Row 2: Old Mask
    ax3 = plt.subplot(3, 4, 5)
    ax3.imshow(old_colored)
    ax3.set_title(f'OLD Mask (IDs: {list(old_ids)})', fontsize=14, fontweight='bold', color='red')
    ax3.axis('off')
    
    ax4 = plt.subplot(3, 4, 6)
    ax4.imshow(old_overlay)
    ax4.set_title('OLD Overlay', fontsize=14, fontweight='bold', color='red')
    ax4.axis('off')
    
    # Row 3: New Mask
    ax5 = plt.subplot(3, 4, 9)
    ax5.imshow(new_colored)
    ax5.set_title(f'NEW Mask (IDs: {list(new_ids)})', fontsize=14, fontweight='bold', color='green')
    ax5.axis('off')
    
    ax6 = plt.subplot(3, 4, 10)
    ax6.imshow(new_overlay)
    ax6.set_title('NEW Overlay', fontsize=14, fontweight='bold', color='green')
    ax6.axis('off')
    
    # Statistics comparison
    ax7 = plt.subplot(3, 4, 7)
    ax7.axis('off')
    
    stats_text = f"""
    COMPARISON STATS
    ================
    
    OLD MASK:
    - Unique IDs: {len(old_ids)}
    - Object IDs: {list(old_ids)}
    - Total pixels: {(old_mask > 0).sum():,}
    
    NEW MASK:
    - Unique IDs: {len(new_ids)}
    - Object IDs: {list(new_ids)}
    - Total pixels: {(new_mask > 0).sum():,}
    
    PIXEL COUNTS:
    """
    
    for obj_id in set(list(old_ids) + list(new_ids)):
        old_count = (old_mask == obj_id).sum()
        new_count = (new_mask == obj_id).sum()
        
        if old_count > 0 or new_count > 0:
            change = ((new_count - old_count) / max(old_count, 1)) * 100
            stats_text += f"\n  Obj {obj_id:2d}: {old_count:6,} → {new_count:6,} ({change:+.1f}%)"
    
    ax7.text(0.1, 0.5, stats_text, fontsize=11, family='monospace', 
             verticalalignment='center')
    
    # Difference visualization
    ax8 = plt.subplot(3, 4, 11)
    
    # Show where masks differ
    diff = (old_mask > 0).astype(int) - (new_mask > 0).astype(int)
    # -1: pixel in old but not new (RED)
    #  0: same (BLACK)
    # +1: pixel in new but not old (GREEN)
    
    diff_colored = np.zeros((diff.shape[0], diff.shape[1], 3), dtype=np.uint8)
    diff_colored[diff == -1] = [255, 0, 0]    # Red: lost pixels
    diff_colored[diff == 1] = [0, 255, 0]     # Green: gained pixels
    diff_colored[diff == 0] = [50, 50, 50]    # Gray: unchanged
    
    ax8.imshow(diff_colored)
    ax8.set_title('Pixel Difference\n(Red=Lost, Green=Gained)', 
                  fontsize=14, fontweight='bold')
    ax8.axis('off')
    
    plt.tight_layout()
    
    # Save or show
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"scene11_persp{perspective_idx}_comparison.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    
    return True


def compare_scene11_random_samples(data_root, models_dir, num_samples=3):
    """Compare random perspectives from scene11."""
    scene_path = os.path.join(data_root, "scene11")
    
    if not os.path.exists(scene_path):
        print("ERROR: scene11 not found")
        return
    
    # Load metadata
    meta_path = os.path.join(scene_path, "metadata.json")
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    valid_perspectives = meta.get('D435_valid_perspective_list', [])
    
    print(f"Scene11 has {len(valid_perspectives)} valid perspectives")
    print(f"Valid perspective indices: {valid_perspectives}")
    
    # Sample random perspectives
    num_samples = min(num_samples, len(valid_perspectives))
    sampled = random.sample(valid_perspectives, num_samples)
    
    print(f"\nVisualizing {num_samples} random perspectives: {sampled}")
    print("="*60)
    
    save_dir = "mask_quality_check"
    
    for persp_idx in sampled:
        print(f"\nPerspective {persp_idx}:")
        success = visualize_masks(scene_path, persp_idx, save_dir)
        if not success:
            print(f"  Failed to visualize perspective {persp_idx}")


if __name__ == "__main__":
    data_root = r"C:\Users\user\Desktop\AUB\Intro2ML\Project\Intro2ML-Project\tanscg-data-2\train"
    models_dir = r"C:\Users\user\Desktop\AUB\Intro2ML\Project\TransCG\models"
    
    print("="*60)
    print("MASK QUALITY CHECK - Scene11")
    print("="*60)
    
    # Visualize 3 random perspectives
    compare_scene11_random_samples(data_root, models_dir, num_samples=3)
    
    print("\n" + "="*60)
    print("Check saved images in: mask_quality_check/")
    print("="*60)