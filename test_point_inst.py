import torch
import cv2
from trans_pose.stage2.feature_encoding.depth_encoder import build_instance_points_multi

# Load corrected instance mask (FIX THE PATH!)
mask_path = r"C:\Users\user\Desktop\AUB\Intro2ML\Project\Intro2ML-Project\tanscg-data-2\train\scene11\0\depth1-gt-mask-corrected.png"
mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

if mask is None:
    print(f"ERROR: Could not load mask from {mask_path}")
    print("Please check if:")
    print("  1. You ran mask_preprocessing.py on scene11")
    print("  2. The file depth1-gt-mask-corrected.png exists")
    exit(1)

mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()

# Load depth (GT depth for best results)
depth_path = r"C:\Users\user\Desktop\AUB\Intro2ML\Project\Intro2ML-Project\tanscg-data-2\train\scene11\0\depth1-gt-depth.png"
depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

if depth is None:
    print(f"WARNING: GT depth not found, trying sensor depth...")
    depth_path = r"C:\Users\user\Desktop\AUB\Intro2ML\Project\Intro2ML-Project\tanscg-data-2\train\scene11\0\depth1.png"
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    
if depth is None:
    print(f"ERROR: Could not load depth from {depth_path}")
    exit(1)

depth = depth.astype(float) / 1000.0  # Convert to meters
depth_tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).float()

print(f"Loaded mask: {mask.shape}, unique values: {torch.unique(mask_tensor)}")
print(f"Loaded depth: {depth.shape}, range: [{depth.min():.3f}, {depth.max():.3f}]m")

# Camera intrinsics
intrinsics = (927.17, 927.37, 651.32, 349.62)

# Run multi-object version
print("\nRunning build_instance_points_multi...")
points_xyz, uv, point_labels = build_instance_points_multi(
    depth_tensor, 
    mask_tensor, 
    intrinsics, 
    num_samples=4096
)

print("\n" + "="*60)
print("RESULTS:")
print("="*60)
print(f"Points shape: {points_xyz.shape}")  # [4096, 3]
print(f"UV shape: {uv.shape}")              # [4096, 2]
print(f"Labels shape: {point_labels.shape}")  # [4096]
print(f"Unique labels: {torch.unique(point_labels)}")  # Should show [4, 7] or similar

# Check distribution
print("\nPoint distribution by object:")
print("-" * 40)
for obj_id in torch.unique(point_labels):
    count = (point_labels == obj_id).sum()
    print(f"Object {obj_id:3d}: {count:5d} points ({count/4096*100:5.1f}%)")

# Check 3D coordinates
print("\nPoint cloud statistics:")
print("-" * 40)
print(f"X range: [{points_xyz[:, 0].min():.3f}, {points_xyz[:, 0].max():.3f}]")
print(f"Y range: [{points_xyz[:, 1].min():.3f}, {points_xyz[:, 1].max():.3f}]")
print(f"Z range: [{points_xyz[:, 2].min():.3f}, {points_xyz[:, 2].max():.3f}]")

print("\n Test passed! Ready for next step (dense_fusion.py)")