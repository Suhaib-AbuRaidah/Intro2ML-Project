import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torchvision import transforms as T
import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trans_pose.stage2.dataset_stage2 import Stage2Dataset, collate_fn
from trans_pose.stage2.network import TransPoseNetwork

def feature_transform_regularizer(trans):
    """Computes ||I - AA^T||^2"""
    d = trans.size()[1]
    I = torch.eye(d, device=trans.device)[None, :, :]
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss

def train_epoch(model, dataloader, intrinsics, device, optimizer, epochs=10):
    # Unpack intrinsics
    fx, fy, cx, cy = intrinsics[0,0], intrinsics[1,1], intrinsics[0,2], intrinsics[1,2]
    intrinsics_tuple = (fx, fy, cx, cy)
    
    model.train()
    
    for epoch in range(epochs):
        pbar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        epoch_loss = 0.0
        
        for step, data in enumerate(pbar):
            # Move data to device
            rgb = data['rgb'].to(device)              # (B, 3, H, W)
            sn = data['sn'].to(device)                # (B, 3, H, W)
            depth = data['depth'].to(device)          # (B, 1, H, W)
            o_mask = data['mask'].to(device)          # (B, 1, H, W) - Contains Object IDs
            
            kpts_list = data['keypoints']             # List of dicts
            
            # Forward pass (Expect 4 outputs now)
            seg_logits, offsets, points, trans_feat = model(rgb, sn, depth, o_mask, intrinsics_tuple)
            
            B, N, _ = points.shape
            
            # --- GENERATE TARGETS ON THE FLY ---
            # We need to know which object each point belongs to.
            # Since DenseFusion samples randomly, we project points back to 2D to sample the mask.
            
            # 1. Project points to 2D (u, v)
            # u = x * fx / z + cx
            # v = y * fy / z + cy
            x, y, z = points[:, :, 0], points[:, :, 1], points[:, :, 2]
            u = (x * fx / z) + cx
            v = (y * fy / z) + cy
            
            # Normalize to [-1, 1] for grid_sample
            H, W = o_mask.shape[2], o_mask.shape[3]
            grid = torch.zeros(B, N, 1, 2, device=device)
            grid[:, :, 0, 0] = 2.0 * u / (W - 1) - 1.0
            grid[:, :, 0, 1] = 2.0 * v / (H - 1) - 1.0
            
            # Sample the mask to get Object IDs for each point
            # mode='nearest' is crucial for integer labels!
            sampled_mask = F.grid_sample(o_mask.float(), grid, mode='nearest', align_corners=True)
            point_obj_ids = sampled_mask.squeeze(-1).squeeze(1).long() # (B, N)
            
            # 2. Prepare Segmentation Targets & Offset Targets
            gt_seg = torch.zeros((B, N), dtype=torch.long, device=device)
            target_offsets = torch.zeros((B, N, model.offset_head.num_keypoints, 3), device=device)
            offset_mask = torch.zeros((B, N), dtype=torch.bool, device=device) # Only supervise valid objects
            
            for b in range(B):
                # Get available objects in this batch item
                available_objs = kpts_list[b] 
                
                for obj_id_str, kpts in available_objs.items():
                    obj_id = int(obj_id_str)
                    
                    # Find points belonging to this object
                    mask_b = (point_obj_ids[b] == obj_id)
                    
                    if mask_b.sum() == 0: continue
                    
                    # Assign Segmentation Target (assuming obj_id 1 -> class 0)
                    # If your classes are 0-indexed in mask, remove the -1
                    class_idx = obj_id - 1 
                    if class_idx >= 0 and class_idx < model.seg_head.mlp[-1].out_features:
                        gt_seg[b, mask_b] = class_idx
                        offset_mask[b, mask_b] = True
                        
                        # Assign Offset Target
                        # kpts: (K, 3)
                        kpts_tensor = torch.tensor(kpts, device=device).float()
                        
                        # points[b, mask_b]: (M, 3)
                        # kpts_tensor: (K, 3) -> (1, K, 3)
                        # diff: (M, K, 3)
                        current_points = points[b, mask_b].unsqueeze(1) # (M, 1, 3)
                        diff = kpts_tensor.unsqueeze(0) - current_points
                        
                        target_offsets[b, mask_b] = diff

            # --- COMPUTE LOSSES ---
            
            # 1. Segmentation Loss
            # Flatten for CrossEntropy: (B*N, C) vs (B*N)
            loss_seg = F.cross_entropy(seg_logits.view(-1, seg_logits.shape[-1]), gt_seg.view(-1))
            
            # 2. Offset Loss (Only for valid object points)
            # offsets: (B, N, K, 3)
            if offset_mask.sum() > 0:
                loss_off = F.l1_loss(offsets[offset_mask], target_offsets[offset_mask])
            else:
                loss_off = torch.tensor(0.0, device=device)
            
            # 3. Regularization Loss
            loss_reg = feature_transform_regularizer(trans_feat)
            
            # Total Loss
            # Weights: Seg=1.0, Off=1.0, Reg=0.001 (Standard PointNet weight)
            loss_total = loss_seg + loss_off + 0.001 * loss_reg
            
            optimizer.zero_grad()
            loss_total.backward()
            
            # Gradient Clipping (Recommended)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            
            optimizer.step()
            
            epoch_loss += loss_total.item()
            pbar.set_postfix({"Loss": loss_total.item(), "Seg": loss_seg.item(), "Off": loss_off.item()})
            
        print(f"Epoch {epoch+1} Average Loss: {epoch_loss / len(dataloader):.4f}")

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Ensure dataset path is correct
    # Update this path to your actual data location
    dataset = Stage2Dataset(
        root_dir="/home/suhaib/ML_Project/data/transcg-data-1/transcg",
        transforms=T.ToTensor()
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=2, drop_last=True, shuffle=True, collate_fn=collate_fn
    )

    intrinsics = dataset.camera_intrisics

    params = {
         "img_outdim": 128,
         "normals_outdim": 64, 
         "points_outdim": 256,
         "num_classes": 4,     # Ensure this matches your dataset object count
         "num_keypoints": 10   # Ensure this matches your dataset keypoint count
    }

    model = TransPoseNetwork(**params).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    train_epoch(model, train_dataloader, intrinsics, device=device, optimizer=optimizer, epochs=50)