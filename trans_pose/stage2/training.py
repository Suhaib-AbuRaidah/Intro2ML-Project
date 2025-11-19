import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torchvision import transforms as T
import sys
import os
import glob

# Adjust path to point to project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trans_pose.stage2.dataset_stage2 import Stage2Dataset, collate_fn
from trans_pose.stage2.network import TransPoseNetwork
from read_intrinsics import scale_intrinsics

# --- CONFIGURATION ---
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
EPOCHS = 20
NUM_WORKERS = 0

# Define target output size for images
TARGET_H = 360
TARGET_W = 640

# --- CRITICAL: YOU MUST SET THESE TO YOUR ACTUAL IMAGE RESOLUTION ---
ORIGINAL_H = 720  # Example: If your images are 720p (1280x720)
ORIGINAL_W = 1280 # Example: If your images are 720p (1280x720)
# ------------------------------------------------------------------

WEIGHT_DECAY = 1e-4
LR_DECAY_STEP = 10
LR_DECAY_GAMMA = 0.5
SAVE_DIR = "checkpoints"
RESUME = True
# ---------------------

def feature_transform_regularizer(trans):
    d = trans.size()[1]
    I = torch.eye(d, device=trans.device)[None, :, :]
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss

def run_epoch(model, dataloader, intrinsics_tuple_scaled, device, optimizer=None, is_train=True, epoch_idx=0):
    if is_train:
        model.train()
        desc = f"Train Ep {epoch_idx+1}"
    else:
        model.eval()
        desc = f"Valid Ep {epoch_idx+1}"

    total_loss = 0.0
    total_seg = 0.0
    total_off = 0.0
    
    pbar = tqdm.tqdm(dataloader, desc=desc, leave=True, dynamic_ncols=True)

    for step, data in enumerate(pbar):
        rgb = data['rgb'].to(device)
        sn = data['sn'].to(device)
        depth = data['depth'].to(device)
        o_mask = data['mask'].to(device)
        kpts_list = data['keypoints']
        # binary_mask is no longer needed as input to the model, o_mask is used for point sampling
        
        # --- Forward pass with o_mask for point sampling ---
        with torch.set_grad_enabled(is_train):
            seg_logits, offsets, points, trans_feat = model(rgb, sn, depth, o_mask, intrinsics_tuple_scaled)
            
            # --- TARGET GENERATION ---
            B, N, _ = points.shape
            
            # Use the scaled intrinsics for 3D-to-2D projection
            fx, fy, cx, cy = intrinsics_tuple_scaled
            
            x, y, z = points[:, :, 0], points[:, :, 1], points[:, :, 2]
            z = torch.clamp(z, min=1e-8)
            
            # 3D-to-2D Projection using SCALED intrinsics
            u = (x * fx / z) + cx
            v = (y * fy / z) + cy
            
            # Use the fixed, scaled dimensions for H and W
            H, W = TARGET_H, TARGET_W 
            
            grid = torch.zeros(B, N, 1, 2, device=device)
            grid[:, :, 0, 0] = 2.0 * u / (W - 1) - 1.0
            grid[:, :, 0, 1] = 2.0 * v / (H - 1) - 1.0
            
            # The sampled mask is now at the TARGET_H/W resolution
            sampled_mask = F.grid_sample(o_mask.float(), grid, mode='nearest', align_corners=False) # Use align_corners=False
            point_obj_ids = sampled_mask.squeeze(-1).squeeze(1).long()
            
            gt_seg = torch.zeros((B, N), dtype=torch.long, device=device)
            target_offsets = torch.zeros((B, N, model.offset_head.num_keypoints, 3), device=device)
            offset_mask = torch.zeros((B, N), dtype=torch.bool, device=device)
            
            for b in range(B):
                available_objs = kpts_list[b]
                for obj_id_str, kpts in available_objs.items():
                    obj_id = int(obj_id_str)
                    mask_b = (point_obj_ids[b] == obj_id)
                    if mask_b.sum() == 0: continue
                    
                    # The check for class_idx is removed as it was incorrect for binary segmentation.
                    # We now calculate offsets for ANY valid object that has keypoints.
                    offset_mask[b, mask_b] = True
                    kpts_tensor = torch.tensor(kpts, device=device).float()
                    current_points = points[b, mask_b].unsqueeze(1)
                    target_offsets[b, mask_b] = kpts_tensor.unsqueeze(0) - current_points

            # --- LOSSES ---
            # MODIFIED LOSS CALCULATION
            # 1. Segmentation Loss: Binary classification (Object vs Background)
            # Use BCEWithLogitsLoss for binary (1-channel) output. It's more stable.
            gt_seg_binary = (point_obj_ids > 0).float() # Target should be float for BCE
            # seg_logits is [B, N, 1], so we squeeze it. gt_seg_binary is [B, N].
            loss_seg = F.binary_cross_entropy_with_logits(seg_logits.squeeze(-1), gt_seg_binary)
            
            # 2. Offset Loss: Calculated ONLY on points that are ground-truth objects.
            # This forces the offset head to learn even if the seg head is wrong.
            if offset_mask.sum() > 0:
                loss_off = F.l1_loss(offsets[offset_mask], target_offsets[offset_mask])
            else:
                loss_off = torch.tensor(0.0, device=device)
                
            loss_reg = feature_transform_regularizer(trans_feat)

            # Re-weighted loss to give importance to the offset task
            loss_total = 1.0 * loss_seg + 1.0 * loss_off + 0.001 * loss_reg

            if step == 0:                
                # Check what percentage of points are on screen (within image bounds)
                valid_u = torch.logical_and(u >= 0, u < W)
                valid_v = torch.logical_and(v >= 0, v < H)
                valid_points = torch.logical_and(valid_u, valid_v).sum().item()
                            
            if is_train:
                optimizer.zero_grad()
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
        
        # Update stats
        total_loss += loss_total.item()
        total_seg += loss_seg.item()
        total_off += loss_off.item()
        
        # Real-time print in progress bar
        pbar.set_postfix({
            "Loss": f"{loss_total.item():.4f}", 
            "Seg": f"{loss_seg.item():.4f}", 
            "Off": f"{loss_off.item():.4f}"
        })

    return total_loss / len(dataloader)

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # 1. DATASETS
    keypoints_dir ="F:/ML-Dataset/keypoints"
    train_dir = "F:/ML-Dataset/train" 
    valid_dir = "F:/ML-Dataset/valid"

    data_transforms = T.ToTensor() 

    train_dataset = Stage2Dataset(
        root_dir=train_dir,
        keypoints_dir=keypoints_dir,
        transforms=data_transforms,
        # --- PASS TARGET SIZE TO DATASET ---
        target_size=(TARGET_W, TARGET_H)
    )
    
    original_intrinsics_matrix = train_dataset.camera_intrisics
    
    # Extract fx, fy, cx, cy from the original intrinsics matrix
    original_fx = original_intrinsics_matrix[0,0]
    original_fy = original_intrinsics_matrix[1,1]
    original_cx = original_intrinsics_matrix[0,2]
    original_cy = original_intrinsics_matrix[1,2]

    if not os.path.exists(valid_dir):
        valid_dataset = train_dataset
    else:
        valid_dataset = Stage2Dataset(
            root_dir=valid_dir, 
            keypoints_dir=keypoints_dir, 
            transforms=data_transforms,
            # --- PASS TARGET SIZE TO DATASET ---
            target_size=(TARGET_W, TARGET_H) 
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        collate_fn=collate_fn, num_workers=NUM_WORKERS, drop_last=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        collate_fn=collate_fn, num_workers=NUM_WORKERS, drop_last=False
    )

    # The intrinsics loaded from Stage2Dataset are already scaled to TARGET_W, TARGET_H (640x360).
    # Therefore, we use them directly without further scaling.
    # If Stage2Dataset were to provide ORIGINAL (1280x720) intrinsics,
    # then the scale_intrinsics function would be correctly applied here.
    intrinsics_tuple_scaled = (original_fx, original_fy, original_cx, original_cy)
 
    # 2. MODEL
    params = {
         "img_outdim": 128,
         "normals_outdim": 64, 
         "points_outdim": 256,
         "num_keypoints": 10   
    }
    model = TransPoseNetwork(**params).to(device)

    # 3. OPTIMIZER & SCHEDULER
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_DECAY_STEP, gamma=LR_DECAY_GAMMA)

    # 4. RESUME LOGIC
    start_epoch = 0
    best_val_loss = float('inf')
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    latest_ckpt = os.path.join(SAVE_DIR, "latest_model.pth")
    if RESUME and os.path.exists(latest_ckpt):
        print(f"Resuming from {latest_ckpt}...")
        checkpoint = torch.load(latest_ckpt)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resumed at Epoch {start_epoch}")

    # 5. TRAINING LOOP
    for epoch in range(start_epoch, EPOCHS):
        current_lr = scheduler.get_last_lr()[0]
        print(f"\n=== Epoch {epoch+1}/{EPOCHS} | LR: {current_lr:.6f} ===")
        
        # Pass the scaled intrinsics tuple
        train_loss = run_epoch(model, train_loader, intrinsics_tuple_scaled, device, optimizer, is_train=True, epoch_idx=epoch)
        val_loss = run_epoch(model, valid_loader, intrinsics_tuple_scaled, device, is_train=False, epoch_idx=epoch)
        
        scheduler.step()
        
        print(f"Summary Ep {epoch+1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Checkpoint Dictionary
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'loss': train_loss
        }
        
        # Save Latest (Overwrites)
        torch.save(checkpoint, latest_ckpt)
        
        # Save Per-Epoch (Cache History)
        epoch_ckpt = os.path.join(SAVE_DIR, f"model_epoch_{epoch+1}.pth")
        torch.save(checkpoint, epoch_ckpt)
        
        # Save Best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, os.path.join(SAVE_DIR, "best_model.pth"))
            print(f">>> New Best Model! (Val Loss: {val_loss:.4f})")