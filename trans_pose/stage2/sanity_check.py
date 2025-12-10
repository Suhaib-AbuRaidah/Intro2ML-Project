import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys
import os
import tqdm  # <--- Added for progress bar

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trans_pose.stage2.network import TransPoseNetwork
from trans_pose.stage2.dataset_stage2 import collate_fn

# --- 1. FAKE DATASET GENERATOR ---
class SyntheticDataset(Dataset):
    def __init__(self, length=32):
        self.length = length
        self.camera_intrisics = np.array([
            [525.0, 0.0, 319.5],
            [0.0, 525.0, 239.5],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Random RGB [3, 120, 160]
        rgb = torch.rand(3, 120, 160)
        
        # Random Depth [1, 120, 160] (Meters 0.5 to 2.0)
        depth = torch.rand(1, 120, 160) * 1.5 + 0.5
        
        # Random Normals [3, 120, 160] (Normalized)
        sn = torch.randn(3, 120, 160)
        sn = sn / sn.norm(dim=0, keepdim=True)
        
        # Random Mask [1, 120, 160] (Object ID 1 in center)
        mask = torch.zeros(1, 120, 160)
        mask[0, 40:80, 60:100] = 1.0 
        
        # Fake Keypoints for Object 1
        kpts = np.random.rand(10, 3).astype(np.float32)
        keypoints = {'1': kpts.tolist()}

        return {
            'rgb': rgb,
            'depth': depth,
            'mask': mask,
            'sn': sn,
            'keypoints': keypoints,
            'intrinsics': self.camera_intrisics
        }

# --- 2. TRAINING LOOP SIMULATION ---
def test_synthetic_training():
    print("=== STARTING SYNTHETIC TRAINING TEST ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Setup Fake Data
    dataset = SyntheticDataset(length=64) # 64 fake images
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    
    # Setup Model
    model = TransPoseNetwork(
        img_outdim=128,
        normals_outdim=64,
        points_outdim=256,
        num_classes=4,
        num_keypoints=10
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    intrinsics = dataset.camera_intrisics
    intrinsics_tuple = (intrinsics[0,0], intrinsics[1,1], intrinsics[0,2], intrinsics[1,2])

    model.train()
    
    print("\n--- Running 5 Epochs on Fake Data ---")
    
    for epoch in range(5):
        epoch_loss = 0.0
        
        # WRAP DATALOADER WITH TQDM
        pbar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/5")
        
        for step, data in enumerate(pbar):
            rgb = data['rgb'].to(device)
            sn = data['sn'].to(device)
            depth = data['depth'].to(device)
            o_mask = data['mask'].to(device)
            kpts_list = data['keypoints']
            
            # Forward
            seg_logits, offsets, points, trans_feat = model(rgb, sn, depth, o_mask, intrinsics_tuple)
            
            # --- Target Generation ---
            B, N, _ = points.shape
            fx, fy, cx, cy = intrinsics_tuple
            
            x, y, z = points[:, :, 0], points[:, :, 1], points[:, :, 2]
            z = torch.clamp(z, min=1e-8)
            u = (x * fx / z) + cx
            v = (y * fy / z) + cy
            
            H, W = o_mask.shape[2], o_mask.shape[3]
            grid = torch.zeros(B, N, 1, 2, device=device)
            grid[:, :, 0, 0] = 2.0 * u / (W - 1) - 1.0
            grid[:, :, 0, 1] = 2.0 * v / (H - 1) - 1.0
            
            sampled_mask = F.grid_sample(o_mask.float(), grid, mode='nearest', align_corners=True)
            point_obj_ids = sampled_mask.squeeze(-1).squeeze(1).long()
            
            gt_seg = torch.zeros((B, N), dtype=torch.long, device=device)
            target_offsets = torch.zeros((B, N, 10, 3), device=device)
            offset_mask = torch.zeros((B, N), dtype=torch.bool, device=device)
            
            for b in range(B):
                for obj_id_str, kpts in kpts_list[b].items():
                    obj_id = int(obj_id_str)
                    mask_b = (point_obj_ids[b] == obj_id)
                    if mask_b.sum() == 0: continue
                    
                    class_idx = obj_id - 1 
                    if class_idx >= 0:
                        gt_seg[b, mask_b] = class_idx
                        offset_mask[b, mask_b] = True
                        kpts_tensor = torch.tensor(kpts, device=device).float()
                        target_offsets[b, mask_b] = kpts_tensor.unsqueeze(0) - points[b, mask_b].unsqueeze(1)

            # Losses
            loss_seg = F.cross_entropy(seg_logits.view(-1, 4), gt_seg.view(-1))
            
            if offset_mask.sum() > 0:
                loss_off = F.l1_loss(offsets[offset_mask], target_offsets[offset_mask])
            else:
                loss_off = torch.tensor(0.0, device=device)
                
            d = trans_feat.size()[1]
            I = torch.eye(d, device=device)[None, :, :]
            loss_reg = torch.mean(torch.norm(torch.bmm(trans_feat, trans_feat.transpose(2, 1)) - I, dim=(1, 2)))
            
            loss_total = loss_seg + loss_off + 0.001 * loss_reg
            
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            
            epoch_loss += loss_total.item()
            
            # UPDATE PROGRESS BAR
            pbar.set_postfix({
                "Loss": f"{loss_total.item():.4f}",
                "Seg": f"{loss_seg.item():.4f}",
                "Off": f"{loss_off.item():.4f}"
            })
        
        avg_loss = epoch_loss / len(dataloader)
        # print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")

    print("\n=== TEST PASSED ===")
    print("The training loop is valid. When you get real data, just update the path in training.py!")

if __name__ == "__main__":
    test_synthetic_training()