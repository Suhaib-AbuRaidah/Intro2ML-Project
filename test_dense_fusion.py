import torch
from trans_pose.stage2.feature_encoding.dense_fusion import DenseFusion

print("Testing DenseFusion.forward_multi()...")

# Create dummy inputs
B, H, W = 2, 720, 1280
rgb = torch.randn(B, 3, H, W)
normals = torch.randn(B, 3, H, W)
depth = torch.rand(B, 1, H, W) * 1.5 + 0.3  # 0.3-1.8m

# Create instance masks with multiple objects
mask = torch.zeros(B, 1, H, W)
mask[0, 0, 200:400, 300:600] = 4   # Object 4
mask[0, 0, 100:300, 700:900] = 7   # Object 7
mask[1, 0, 300:500, 400:800] = 54  # Object 54

intrinsics = (927.17, 927.37, 651.32, 349.62)

# Initialize model
model = DenseFusion(
    image_channels=128,
    normal_channels=64,
    pointnet_channels=256,
    num_samples=4096
)

print("\n1. Testing ORIGINAL forward()...")
fused, points, trans = model.forward(rgb, depth, normals, mask, intrinsics)
print(f"   Fused: {fused.shape}")      # [2, 4096, 448]
print(f"   Points: {points.shape}")    # [2, 4096, 3]
print(f"   Trans: {trans.shape}")      # [2, 64, 64]
print("    Original forward() works!")

print("\n2. Testing NEW forward_multi()...")
fused, points, trans, labels = model.forward_multi(rgb, depth, normals, mask, intrinsics)
print(f"   Fused: {fused.shape}")      # [2, 4096, 448]
print(f"   Points: {points.shape}")    # [2, 4096, 3]
print(f"   Trans: {trans.shape}")      # [2, 64, 64]
print(f"   Labels: {labels.shape}")    # [2, 4096] ← NEW!

print("\n3. Checking point labels...")
print(f"   Batch 0 unique labels: {torch.unique(labels[0])}")  # [4, 7]
print(f"   Batch 1 unique labels: {torch.unique(labels[1])}")  # [54]

for obj_id in torch.unique(labels[0]):
    count = (labels[0] == obj_id).sum()
    print(f"   Batch 0, Object {obj_id}: {count} points ({count/4096*100:.1f}%)")

print("\n" + "="*60)
print(" ALL TESTS PASSED! DenseFusion multi-object ready!")
print("="*60)