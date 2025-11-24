import os, glob, json
import numpy as np

root = r"C:\Users\user\Desktop\AUB\Intro2ML\Project\Intro2ML-Project\tanscg-data-2\train"
missing = 0
total = 0
examples = []

for scene in sorted(glob.glob(os.path.join(root, "scene*"))):
    meta_path = os.path.join(scene, "metadata.json")
    if not os.path.exists(meta_path):
        continue
    with open(meta_path,'r') as f:
        meta = json.load(f)
    model_list = meta.get("model_list", [])
    for persp in os.listdir(scene):
        persp_path = os.path.join(scene, persp)
        if not os.path.isdir(persp_path): 
            continue
        pose_path = os.path.join(persp_path, "corrected_pose", "1.npy")
        total += 1
        if not os.path.exists(pose_path):
            missing += 1
            continue
        try:
            data = np.load(pose_path, allow_pickle=True)
            # determine structure
            if isinstance(data, dict):
                keys = list(data.keys())
            elif data.shape == ():
                obj = data.item()
                if isinstance(obj, dict):
                    keys = list(obj.keys())
                else:
                    keys = ["<scalar>"]
            else:
                keys = getattr(data, 'files', list(data.keys())) if hasattr(data,'files') else ["<unknown>"]
            examples.append((pose_path, keys))
        except Exception as e:
            examples.append((pose_path, f"ERR:{e}"))

print("POSE FILE SUMMARY")
print("=================")
print(f"Total perspectives scanned: {total}")
print(f"Missing pose files: {missing}")
print(f"Samples with pose file read: {len(examples)}")
print()
print("First 10 examples:")
for p,k in examples[:10]:
    print(p)
    print("  keys:", k)