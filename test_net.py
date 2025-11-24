"""
Test the new multi-object network
"""
import torch
from trans_pose.stage2.network_gd import TransPoseNetworkMulti

print("="*60)
print("TESTING MULTI-OBJECT NETWORK")
print("="*60)

# Create model
model = TransPoseNetworkMulti(
    img_outdim=128,
    normals_outdim=64,
    points_outdim=256,
    num_keypoints=10,
    num_classes=61  # 60 objects + background
)

print(f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters")

# Create dummy inputs
B, H, W = 2, 720, 1280
rgb = torch.randn(B, 3, H, W)
normals = torch.randn(B, 3, H, W)
depth = torch.rand(B, 1, H, W) * 2.0 + 0.3

# Create instance masks with multiple objects
mask = torch.zeros(B, 1, H, W)
mask[0, 0, 200:400, 300:600] = 4   # Object 4
mask[0, 0, 100:300, 700:900] = 7   # Object 7
mask[1, 0, 300:500, 400:800] = 54  # Object 54

intrinsics = (927.17, 927.37, 651.32, 349.62)

print("\n1. Testing forward pass...")
seg_logits, offsets, points, trans_feat, point_labels = model(
    rgb, normals, depth, mask, intrinsics
)

print(f"   Segmentation logits: {seg_logits.shape}")  # [2, 4096, 61]
print(f"   Offsets: {offsets.shape}")                 # [2, 4096, 10, 3]
print(f"   Points: {points.shape}")                   # [2, 4096, 3]
print(f"   Trans feat: {trans_feat.shape}")           # [2, 64, 64]
print(f"   Point labels: {point_labels.shape}")       # [2, 4096]

print("\n2. Checking point labels...")
print(f"   Batch 0 unique labels: {torch.unique(point_labels[0])}")
print(f"   Batch 1 unique labels: {torch.unique(point_labels[1])}")

print("\n3. Checking segmentation output...")
print(f"   Logits range: [{seg_logits.min():.3f}, {seg_logits.max():.3f}]")
print(f"   Logits mean: {seg_logits.mean():.3f}")

# Check predicted classes
pred_classes = seg_logits.argmax(dim=-1)  # [B, N]
print(f"   Predicted classes: {torch.unique(pred_classes[0])}")

print("\n4. Checking offset magnitudes...")
offset_norms = torch.norm(offsets, dim=-1)  # [B, N, K]
print(f"   Offset range: [{offset_norms.min():.3f}, {offset_norms.max():.3f}]m")
print(f"   Offset mean: {offset_norms.mean():.3f}m")

print("\n5. Testing backward pass...")
# Dummy loss
seg_loss = seg_logits.sum()
offset_loss = offsets.sum()
total_loss = seg_loss + offset_loss

total_loss.backward()

params_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
total_params = sum(1 for _ in model.parameters())
print(f"   Parameters with gradients: {params_with_grad}/{total_params}")

print("\n" + "="*60)
print(" ALL TESTS PASSED!")
print("="*60)
print("\nKey differences from network.py:")
print("  1. SegHead outputs [B, N, 61] instead of [B, N, 1]")
print("  2. Returns point_labels [B, N] for training")
print("  3. Uses forward_multi() from DenseFusion")