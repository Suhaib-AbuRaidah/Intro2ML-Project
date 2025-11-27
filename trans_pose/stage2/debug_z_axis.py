"""Diagnose Z-axis prediction failure."""
import json
import numpy as np

with open('eval_results/model2/evaluation_results.json') as f:
    results = json.load(f)

print("="*80)
print("Z-AXIS DIAGNOSTIC")
print("="*80)

z_errors = []
for sample in results['sample_results']:
    for obj_id_str, obj_info in sample['objects'].items():
        pred = np.array(obj_info['pred_kpts'])
        can = np.array(obj_info['can_kpts'])
        
        if len(pred) > 0 and len(can) > 0:
            z_err = np.abs(pred[:, 2].mean() - can[:, 2].mean())
            z_errors.append(z_err)

z_errors = np.array(z_errors)

print(f"\nMean Z prediction error: {np.mean(z_errors):.4f}m")
print(f"Median Z error: {np.median(z_errors):.4f}m")
print(f"Max Z error: {np.max(z_errors):.4f}m")
print(f"\n⚠️  Z errors should be < 0.05m (canonical Z range is ~0.2m)")

if np.mean(z_errors) > 0.1:
    print(f"\n❌ CRITICAL: Z-axis learning has FAILED!")
    print(f"   → Offset head is not predicting correct Z coordinates")
    print(f"   → Solution: Use Z-weighted loss (see training_gd.py)")
else:
    print(f"\n✅ Z-axis looks reasonable")

print("\n" + "="*80)