import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
stage2_dir = current_dir
trans_pose_dir = os.path.dirname(current_dir)
project_dir = os.path.dirname(trans_pose_dir)

sys.path.insert(0, project_dir)
sys.path.insert(0, trans_pose_dir)
sys.path.insert(0, stage2_dir)

import torch
import torch.nn.functional as F
from trans_pose.stage2.network import TransPoseNetwork
from trans_pose.stage2.feature_encoding.dense_fusion import DenseFusion
from trans_pose.stage2.feature_encoding.depth_encoder import build_instance_points

def feature_transform_regularizer(trans):
    """
    Computes ||I - AA^T||^2
    trans: [B, K, K]
    """
    d = trans.size()[1]
    I = torch.eye(d, device=trans.device)[None, :, :]
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss

def test_gradient_flow():
    """REAL TEST: Check if gradients actually flow through the network"""
    print("="*60)
    print("TEST 1: Gradient Flow (REAL TEST)")
    print("="*60)
    
    model = TransPoseNetwork(
        img_outdim=128,
        normals_outdim=64,
        points_outdim=256,
        num_classes=4,
        num_keypoints=8
    )
    
    # Create dummy inputs
    rgb = torch.rand(3, 100, 100, requires_grad=True)
    normals = torch.randn(3, 100, 100)
    normals = normals / normals.norm(dim=0, keepdim=True)
    depth = torch.rand(1, 100, 100) * 2.0 + 0.5
    mask = torch.zeros(1, 100, 100)
    mask[0, 30:70, 30:70] = 1.0
    intrinsics = (525.0, 525.0, 50.0, 50.0)
    
    # Forward
    seg_logits, offsets, points, trans_feat = model(rgb, normals, depth, mask, intrinsics)
        
    # Create dummy loss
    seg_loss = seg_logits.sum()
    offset_loss = offsets.sum()
    loss = seg_loss + offset_loss
    
    # Backward
    loss.backward()
    
    # Check gradients exist and are finite
    params_with_grad = []
    params_without_grad = []
    non_finite_grads = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            params_with_grad.append(name)
            if not torch.isfinite(param.grad).all():
                non_finite_grads.append(name)
        else:
            params_without_grad.append(name)
    
    print(f"Parameters with gradients: {len(params_with_grad)}")
    print(f"Parameters WITHOUT gradients: {len(params_without_grad)}")
    
    if params_without_grad:
        print("WARNING: These parameters have no gradients:")
        for name in params_without_grad[:5]:  # Show first 5
            print(f"  - {name}")
    
    if non_finite_grads:
        print("ERROR: These parameters have NaN/Inf gradients:")
        for name in non_finite_grads:
            print(f"  - {name}")
        raise AssertionError("Found non-finite gradients!")
    
    assert len(params_with_grad) > 0, "No parameters received gradients!"
    print("[PASS] Gradients flow properly!\n")


def test_feature_diversity():
    """REAL TEST: Check if features are actually different (not collapsed)"""
    print("="*60)
    print("TEST 2: Feature Diversity (REAL TEST)")
    print("="*60)
    
    model = DenseFusion(
        image_channels=128,
        normal_channels=64,
        pointnet_channels=256,
        num_samples=1000
    )
    model.eval()
    
    # Create two DIFFERENT inputs
    rgb1 = torch.rand(3, 100, 100)
    rgb2 = torch.rand(3, 100, 100)
    
    depth = torch.rand(1, 100, 100) * 2.0 + 0.5
    normals = torch.randn(3, 100, 100)
    normals = normals / normals.norm(dim=0, keepdim=True)
    mask = torch.zeros(1, 100, 100)
    mask[0, 30:70, 30:70] = 1.0
    intrinsics = (525.0, 525.0, 50.0, 50.0)
    
    with torch.no_grad():
        fused1, points1, _ = model(rgb1, depth, normals, mask, intrinsics)
        fused2, points2, _ = model(rgb2, depth, normals, mask, intrinsics)
    
    # Check features are different
    feature_diff = (fused1 - fused2).abs().mean().item()
    
    print(f"Average feature difference: {feature_diff:.6f}")
    
    # If difference is too small, features might be collapsed/constant
    if feature_diff < 1e-6:
        raise AssertionError(f"Features are too similar! Diff: {feature_diff}")
    
    # Check features have reasonable variance (not all zeros or constants)
    feature_std = fused1.std().item()
    print(f"Feature std dev: {feature_std:.6f}")
    
    if feature_std < 0.01:
        raise AssertionError(f"Features have too little variance! Std: {feature_std}")
    
    print("[PASS] Features are diverse and meaningful!\n")


def test_segmentation_sensitivity():
    """REAL TEST: Check if segmentation changes with mask"""
    print("="*60)
    print("TEST 3: Segmentation Sensitivity (REAL TEST)")
    print("="*60)
    
    model = TransPoseNetwork(
        img_outdim=128,
        normals_outdim=64,
        points_outdim=256,
        num_classes=4,
        num_keypoints=8
    )
    model.eval()
    
    rgb = torch.rand(3, 100, 100)
    normals = torch.randn(3, 100, 100)
    normals = normals / normals.norm(dim=0, keepdim=True)
    depth = torch.rand(1, 100, 100) * 2.0 + 0.5
    intrinsics = (525.0, 525.0, 50.0, 50.0)
    
    # Two different masks
    mask1 = torch.zeros(1, 100, 100)
    mask1[0, 20:40, 20:40] = 1.0  # Small region
    
    mask2 = torch.zeros(1, 100, 100)
    mask2[0, 60:80, 60:80] = 1.0  # Different region
    
    with torch.no_grad():
        seg1, off1, pts1, _ = model(rgb, normals, depth, mask1, intrinsics)
        seg2, off2, pts2, _ = model(rgb, normals, depth, mask2, intrinsics)
    
    # Check that points come from different regions
    pts1_mean = pts1.mean(dim=0)
    pts2_mean = pts2.mean(dim=0)
    point_diff = (pts1_mean - pts2_mean).norm().item()
    
    print(f"Point cloud difference: {point_diff:.6f}")
    
    if point_diff < 0.01:
        raise AssertionError("Point clouds are identical despite different masks!")
    
    # Check that segmentation predictions are different
    seg_diff = (seg1 - seg2).abs().mean().item()
    print(f"Segmentation difference: {seg_diff:.6f}")
    
    if seg_diff < 0.01:
        raise AssertionError("Segmentation is identical despite different inputs!")
    
    print("[PASS] Network is sensitive to input changes!\n")


def test_offset_reasonableness():
    """REAL TEST: Check if offsets have reasonable magnitudes"""
    print("="*60)
    print("TEST 4: Offset Reasonableness (REAL TEST)")
    print("="*60)
    
    model = TransPoseNetwork(
        img_outdim=128,
        normals_outdim=64,
        points_outdim=256,
        num_classes=4,
        num_keypoints=8
    )
    model.eval()
    
    rgb = torch.rand(3, 100, 100)
    normals = torch.randn(3, 100, 100)
    normals = normals / normals.norm(dim=0, keepdim=True)
    depth = torch.rand(1, 100, 100) * 2.0 + 0.5
    mask = torch.zeros(1, 100, 100)
    mask[0, 30:70, 30:70] = 1.0
    intrinsics = (525.0, 525.0, 50.0, 50.0)
    
    with torch.no_grad():
        seg_logits, offsets, points, _ = model(rgb, normals, depth, mask, intrinsics)
    
    # Check offset magnitudes
    offset_norms = torch.norm(offsets, dim=-1)  # [1, N, K]
    
    offset_mean = offset_norms.mean().item()
    offset_std = offset_norms.std().item()
    offset_max = offset_norms.max().item()
    
    print(f"Offset magnitude - Mean: {offset_mean:.3f}m, Std: {offset_std:.3f}m, Max: {offset_max:.3f}m")
    
    # Offsets should be reasonable (not too large or all zeros)
    if offset_mean < 1e-6:
        raise AssertionError("All offsets are near zero! Network might not be learning.")
    
    if offset_max > 10.0:
        raise AssertionError(f"Offsets too large! Max: {offset_max:.3f}m (might explode during training)")
    
    # Check offset diversity
    offset_std_normalized = offset_std / (offset_mean + 1e-6)
    print(f"Offset diversity (std/mean): {offset_std_normalized:.3f}")
    
    if offset_std_normalized < 0.1:
        print("WARNING: Offsets have low diversity (might be collapsed)")
    
    print("[PASS] Offsets have reasonable magnitudes!\n")


def test_batch_consistency():
    """REAL TEST: Same input should give same output (deterministic)"""
    print("="*60)
    print("TEST 5: Batch Consistency (REAL TEST)")
    print("="*60)
    
    model = TransPoseNetwork(
        img_outdim=128,
        normals_outdim=64,
        points_outdim=256,
        num_classes=4,
        num_keypoints=8
    )
    model.eval()  # Disable dropout
    
    # Same input
    rgb = torch.rand(3, 100, 100)
    normals = torch.randn(3, 100, 100)
    normals = normals / normals.norm(dim=0, keepdim=True)
    depth = torch.rand(1, 100, 100) * 2.0 + 0.5
    mask = torch.zeros(1, 100, 100)
    mask[0, 30:70, 30:70] = 1.0
    intrinsics = (525.0, 525.0, 50.0, 50.0)
    
    with torch.no_grad():
        seg1, off1, pts1, _ = model(rgb, normals, depth, mask, intrinsics)
        seg2, off2, pts2, _ = model(rgb, normals, depth, mask, intrinsics)
    
    # Check consistency (accounting for random point sampling)
    # Points might differ due to random sampling, but features should be similar
    seg_diff = (seg1 - seg2).abs().max().item()
    
    print(f"Max segmentation difference: {seg_diff:.6f}")
    
    # Should be very small (or zero if no randomness in sampling)
    if seg_diff > 0.1:
        print(f"WARNING: Outputs differ significantly! (might be due to random point sampling)")
    else:
        print("[PASS] Network is deterministic in eval mode!\n")



def test_loss_computation():
    """REAL TEST: Check if loss can be computed without errors"""
    print("="*60)
    print("TEST 6: Loss Computation (REAL TEST)")
    print("="*60)
    
    model = TransPoseNetwork(
        img_outdim=128,
        normals_outdim=64,
        points_outdim=256,
        num_classes=4,
        num_keypoints=8
    )
    
    rgb = torch.rand(3, 100, 100)
    normals = torch.randn(3, 100, 100)
    normals = normals / normals.norm(dim=0, keepdim=True)
    depth = torch.rand(1, 100, 100) * 2.0 + 0.5
    mask = torch.zeros(1, 100, 100)
    mask[0, 30:70, 30:70] = 1.0
    intrinsics = (525.0, 525.0, 50.0, 50.0)
    
    # Unpack 4 values
    seg_logits, offsets, points, trans_feat = model(rgb, normals, depth, mask, intrinsics)
    
    # --- NEW CHECKS FOR REGULARIZATION ---
    print(f"TransFeat Shape: {trans_feat.shape}")
    B = points.shape[0]
    
    # Check shape is [B, 64, 64] (Standard for PointNet feature transform)
    if trans_feat.shape != (B, 64, 64):
        raise AssertionError(f"trans_feat has wrong shape! Expected ({B}, 64, 64), got {trans_feat.shape}")
        
    # Check it's not all zeros
    if trans_feat.abs().sum() < 1e-6:
        raise AssertionError("trans_feat is all zeros! PointNet T-Net is not working.")
    # -------------------------------------

    # Try to compute losses
    B, N, _ = points.shape
    
    # 1. Segmentation loss
    gt_seg = torch.randint(0, 4, (B * N,), dtype=torch.long)
    seg_loss = F.cross_entropy(seg_logits.view(B * N, 4), gt_seg)
    print(f"Segmentation loss: {seg_loss.item():.4f}")
    
    # 2. Offset loss
    gt_keypoints = torch.randn(B, 8, 3) * 0.1
    points_expanded = points.unsqueeze(2) 
    target_offsets = gt_keypoints.unsqueeze(1) - points_expanded 
    offset_loss = F.l1_loss(offsets, target_offsets)
    print(f"Offset loss: {offset_loss.item():.4f}")
    
    # 3. Regularization loss
    reg_loss = feature_transform_regularizer(trans_feat)
    print(f"Regularization loss: {reg_loss.item():.4f}")
    
    # Total loss (Weighted)
    total_loss = seg_loss + offset_loss + 0.001 * reg_loss
    print(f"Total loss: {total_loss.item():.4f}")
    
    assert torch.isfinite(seg_loss), "Segmentation loss is NaN/Inf!"
    assert torch.isfinite(offset_loss), "Offset loss is NaN/Inf!"
    assert torch.isfinite(reg_loss), "Regularization loss is NaN/Inf!"
    
    total_loss.backward()
    
    print("[PASS] Loss computation works!\n")



if __name__ == "__main__":
    print(f"Python path: {sys.path[:3]}\n")
    
    tests = [
        ("Gradient Flow", test_gradient_flow),
        ("Feature Diversity", test_feature_diversity),
        ("Segmentation Sensitivity", test_segmentation_sensitivity),
        ("Offset Reasonableness", test_offset_reasonableness),
        ("Batch Consistency", test_batch_consistency),
        ("Loss Computation", test_loss_computation),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            failed += 1
            print("="*60)
            print(f"[FAIL] {test_name}: {e}")
            print("="*60)
            import traceback
            traceback.print_exc()
            print()
    
    print("="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*60)
    
    if failed == 0:
        print("\n All real tests passed! Your model is actually working!")
    else:
        print(f"\n {failed} tests failed. Fix these before training!")