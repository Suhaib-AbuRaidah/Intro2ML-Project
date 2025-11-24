import torch
from trans_pose.stage2.training_gd import (
    compute_segmentation_loss,
    compute_offset_loss_multi
)

print("Testing loss functions...")

# Test 1: Segmentation loss
seg_logits = torch.randn(2, 4096, 61)  # [B, N, num_classes]
point_labels = torch.randint(0, 61, (2, 4096))  # [B, N]

loss_seg = compute_segmentation_loss(seg_logits, point_labels)
print(f" Segmentation loss: {loss_seg.item():.4f}")

# Test 2: Offset loss
offsets = torch.randn(2, 4096, 10, 3)  # [B, N, K, 3]
points = torch.randn(2, 4096, 3)  # [B, N, 3]

# Create dummy keypoints dict
kpts_dict_batch = [
    {'4': torch.randn(10, 3).numpy(), '7': torch.randn(10, 3).numpy()},
    {'54': torch.randn(10, 3).numpy()}
]

# Assign labels
point_labels[0, :2048] = 4
point_labels[0, 2048:] = 7
point_labels[1, :] = 54

loss_off = compute_offset_loss_multi(offsets, points, point_labels, kpts_dict_batch)
print(f" Offset loss: {loss_off.item():.4f}")

print("\n All loss functions working!")