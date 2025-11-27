"""Inspect predicted vs canonical keypoint distributions."""
import json
import numpy as np
from collections import defaultdict

with open('eval_results/model2/evaluation_results.json') as f:
    results = json.load(f)

print("="*80)
print("KEYPOINT DISTRIBUTION ANALYSIS")
print("="*80)

obj_stats = defaultdict(lambda: {'pred': [], 'can': []})

for sample in results['sample_results']:
    for obj_id_str, obj_info in sample['objects'].items():
        if 'pred_kpts' not in obj_info or 'can_kpts' not in obj_info:
            continue
        
        pred = np.array(obj_info['pred_kpts'])
        can = np.array(obj_info['can_kpts'])
        
        obj_stats[obj_id_str]['pred'].append(pred)
        obj_stats[obj_id_str]['can'].append(can)

print("\nPER-OBJECT ANALYSIS:")
print("-"*80)

for obj_id in sorted(obj_stats.keys())[:10]:  # First 10 objects
    preds = np.array(obj_stats[obj_id]['pred'])  # [N_samples, K, 3]
    cans = np.array(obj_stats[obj_id]['can'])    # [N_samples, K, 3]
    
    if len(preds) == 0:
        continue
    
    # Aggregate across all samples
    pred_all = preds.reshape(-1, 3)  # [N_samples*K, 3]
    can_all = cans.reshape(-1, 3)
    
    print(f"\nObject {obj_id} ({len(preds)} samples):")
    print(f"  Canonical keypoints:")
    print(f"    Range X: [{can_all[:, 0].min():.4f}, {can_all[:, 0].max():.4f}]")
    print(f"    Range Y: [{can_all[:, 1].min():.4f}, {can_all[:, 1].max():.4f}]")
    print(f"    Range Z: [{can_all[:, 2].min():.4f}, {can_all[:, 2].max():.4f}]")
    print(f"    Variance: X={can_all[:, 0].var():.6f}, Y={can_all[:, 1].var():.6f}, Z={can_all[:, 2].var():.6f}")
    
    print(f"  Predicted keypoints:")
    print(f"    Range X: [{pred_all[:, 0].min():.4f}, {pred_all[:, 0].max():.4f}]")
    print(f"    Range Y: [{pred_all[:, 1].min():.4f}, {pred_all[:, 1].max():.4f}]")
    print(f"    Range Z: [{pred_all[:, 2].min():.4f}, {pred_all[:, 2].max():.4f}]")
    print(f"    Variance: X={pred_all[:, 0].var():.6f}, Y={pred_all[:, 1].var():.6f}, Z={pred_all[:, 2].var():.6f}")
    
    # Collinearity check
    if len(pred_all) >= 3:
        v1 = pred_all[1] - pred_all[0]
        v2 = pred_all[2] - pred_all[0]
        cross = np.linalg.norm(np.cross(v1, v2))
        print(f"    Collinearity (cross product): {cross:.6f}")

print("\n" + "="*80)