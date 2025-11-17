import os
import argparse
from pathlib import Path

def find_first(glob_list, folder: Path):
    for g in glob_list:
        found = list(folder.glob(g))
        if found:
            return str(found[0])
    return None

def find_second(glob_list, folder: Path):
    for g in glob_list:
        found = list(folder.glob(g))
        if found:
            return str(found[1])
    return None
def build_csv(parent_dir, out_csv, recursive=False, rel_paths=False,
              sparse_names=('depth1.png','depth2.png'),
              normals_names=('depth1-gt-sn.png','depth2-gt-sn.png'),
              mask_names=('depth1-gt-mask.png','depth2-gt-mask.png'),
              rgb_names=('rgb1.png','rgb2.png'),
              gt_names=('depth1-gt.png','depth2-gt.png')):
    p = Path(parent_dir)
    lines = []
    missing = []
    for folder in p.iterdir():
        folder = Path(os.path.join(parent_dir, "transcg-data-",folder))
        for sub_folder in folder.iterdir():
            sub_folder=Path(os.path.join(folder,"transcg","scene",sub_folder))
            for scene in sub_folder.iterdir():
                if not scene.is_dir():
                    continue
                scene=Path(os.path.join(sub_folder,scene))    
                for sample in scene.iterdir():
                    sparse = find_first(sparse_names, sample)
                    normals = find_first(normals_names, sample)
                    mask = find_first(mask_names, sample)
                    rgb = find_first(rgb_names, sample)
                    gt = find_first(gt_names, sample)
                    if not (sparse and normals and mask and rgb and gt):
                        missing.append((str(sample), sparse,normals,mask,rgb,gt))
                        continue
                    entries = [sparse, normals, mask, rgb, gt]
                    if rel_paths:
                        entries = [os.path.relpath(x, start=os.getcwd()) for x in entries]
                    lines.append(','.join(entries))
    with open(out_csv, 'w') as f:
        for ln in lines:
            f.write(ln + '\n')
    print(f"Wrote {len(lines)} samples to {out_csv}")
    if missing:
        print(f"Skipped {len(missing)} folders (missing files). Example missing entries:")
        for x in missing[:5]:
            print("  folder:", x[0], "sparse,normals,mask,rgb,gt:", x[1:])
    return len(lines), missing

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="train_list.txt", help="Output CSV/TSV file")
    parser.add_argument("--recursive", action="store_true", help="Scan subfolders recursively")
    parser.add_argument("--rel", action="store_true", help="Write relative paths instead of absolute")
    args = parser.parse_args()
    parent_dir="F:\ML-Dataset"
    build_csv(parent_dir, args.out, recursive=args.recursive, rel_paths=args.rel)