"""Check what inputs the offset head receives."""
import sys
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trans_pose.stage2.network_gd import TransPoseNetworkMulti

# Initialize model
model = TransPoseNetworkMulti(
    img_outdim=128,
    normals_outdim=64,
    points_outdim=256,
    num_keypoints=10,
    num_classes=61
)

print("="*80)
print("OFFSET HEAD INPUT ANALYSIS")
print("="*80)

# Check what the offset head receives
print(f"\n🔍 Offset Head Input:")
print(f"  Input size to offset head: {model.offset_head.input_size if hasattr(model.offset_head, 'input_size') else 'UNKNOWN'}")

# Inspect the network architecture
print(f"\n🔍 Feature Extraction:")
print(f"  Image features: 128 dims")
print(f"  Normal features: 64 dims")
print(f"  Point features: 256 dims")
print(f"  Total: 128 + 64 + 256 = 448 dims")

print(f"\n❓ QUESTION: Does offset head have access to DEPTH?")
print(f"   Current: Only RGB + Normals + Points (no depth!)")
print(f"   Needed: Need depth feature to predict Z offsets")

print("\n" + "="*80)
print("SOLUTION: Add depth to feature extraction pipeline")
print("="*80)