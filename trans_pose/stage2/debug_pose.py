"""Quick debug script to inspect pose errors."""
import numpy as np
import json
from collections import defaultdict

# Load evaluation results
with open('eval_results/model2/evaluation_results.json') as f:
    results = json.load(f)

print("="*80)
print("POSE ERROR ANALYSIS")
print("="*80)

# Filter by error magnitude
rot_errs = []
trans_errs = []
outlier_samples = []

for sample in results['sample_results']:
    for obj_id_str, obj_info in sample['objects'].items():
        if 'rot_err_deg' not in obj_info:
            continue
        
        rot_err = obj_info['rot_err_deg']
        trans_err = obj_info['trans_err_cm']
        
        rot_errs.append(rot_err)
        trans_errs.append(trans_err)
        
        # Flag outliers
        if rot_err > 90 or trans_err > 100:
            outlier_samples.append({
                'sample_idx': sample['sample_idx'],
                'obj_id': obj_id_str,
                'rot_err': rot_err,
                'trans_err': trans_err,
                'pred_kpts': np.array(obj_info.get('pred_kpts', [])),
                'can_kpts': np.array(obj_info.get('can_kpts', [])),
                'num_points': obj_info['num_points']
            })

rot_errs = np.array(rot_errs)
trans_errs = np.array(trans_errs)

print(f"\n📊 Error Statistics (n={len(rot_errs)}):")
print(f"\nRotation (deg):")
print(f"  mean={np.mean(rot_errs):.2f}°, median={np.median(rot_errs):.2f}°")
print(f"  Q1={np.percentile(rot_errs, 25):.2f}°, Q3={np.percentile(rot_errs, 75):.2f}°")
print(f"  <5°: {(rot_errs < 5).sum()}, 5-15°: {((rot_errs >= 5) & (rot_errs < 15)).sum()}, >90°: {(rot_errs > 90).sum()}")

print(f"\nTranslation (cm):")
print(f"  mean={np.mean(trans_errs):.2f}cm, median={np.median(trans_errs):.2f}cm")
print(f"  Q1={np.percentile(trans_errs, 25):.2f}cm, Q3={np.percentile(trans_errs, 75):.2f}cm")
print(f"  <5cm: {(trans_errs < 5).sum()}, 5-20cm: {((trans_errs >= 5) & (trans_errs < 20)).sum()}, >100cm: {(trans_errs > 100).sum()}")

print(f"\n🚨 OUTLIERS (>90° rotation OR >100cm translation): {len(outlier_samples)}")
print("="*80)

# Analyze first few outliers
for i, outlier in enumerate(outlier_samples[:5]):
    print(f"\nOutlier {i+1}:")
    print(f"  Sample: {outlier['sample_idx']}, Object: {outlier['obj_id']}")
    print(f"  Errors: rot={outlier['rot_err']:.2f}°, trans={outlier['trans_err']:.2f}cm")
    print(f"  Num points: {outlier['num_points']}")
    
    pred = outlier['pred_kpts']
    can = outlier['can_kpts']
    
    if len(pred) > 0 and len(can) > 0:
        print(f"  Canonical kpts range: {can.min():.2f} to {can.max():.2f}")
        print(f"  Predicted kpts range: {pred.min():.2f} to {pred.max():.2f}")
        
        # Check for colinearity
        if len(pred) >= 3:
            p1, p2, p3 = pred[:3]
            v1 = p2 - p1
            v2 = p3 - p1
            cross = np.linalg.norm(np.cross(v1, v2))
            print(f"  Collinearity check (cross product): {cross:.6f} (0=colinear!)")

print("\n" + "="*80)
print("DIAGNOSTIC CHECKS:")
print("="*80)

# Check scale consistency
print("\n1️⃣  SCALE CHECK (canonical vs predicted keypoints):")
for i, outlier in enumerate(outlier_samples[:3]):
    pred = outlier['pred_kpts']
    can = outlier['can_kpts']
    if len(pred) > 1 and len(can) > 1:
        dist_can = np.linalg.norm(can[1] - can[0])
        dist_pred = np.linalg.norm(pred[1] - pred[0])
        scale_ratio = dist_pred / dist_can if dist_can > 0 else 0
        print(f"  Sample {outlier['sample_idx']}: scale_ratio={scale_ratio:.2f}x (should be ~1.0)")

print("\n2️⃣  KEYPOINT ORDER CHECK:")
print("  If keypoints are in wrong order, Kabsch will fail catastrophically.")
print("  → Verify canonical keypoints match predicted keypoints semantically")

print("\n3️⃣  GT POSE FORMAT CHECK:")
print("  Current assumption: T_gt = [R|t] in camera frame")
print("  If GT is in world frame or uses different convention, errors will be huge!")

print("\nDONE")