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

# --- CONFIGURATION ---
BATCH_SIZE = 8          # Decrease to 4 if you run out of GPU memory
LEARNING_RATE = 1e-3
EPOCHS = 50
NUM_WORKERS = 4         # Keeps GPU fed with data
WEIGHT_DECAY = 1e-4
LR_DECAY_STEP = 10
LR_DECAY_GAMMA = 0.5
SAVE_DIR = "checkpoints"
RESUME = True           # Set to True to auto-resume from latest checkpoint
# ---------------------

def feature_transform_regularizer(trans):
    d = trans.size()[1]
    I = torch.eye(d, device=trans.device)[None, :, :]
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss

def run_epoch(model, dataloader, intrinsics_tuple, device, optimizer=None, is_train=True, epoch_idx=0):
    if is_train:
        model.train()
        desc = f"Train Ep {epoch_idx+1}"
    else:
        model.eval()
        desc = f"Valid Ep {epoch_idx+1}"

    total_loss = 0.0
    total_seg = 0.0
    total_off = 0.0
    
    # Create progress bar with real-time updates
    pbar = tqdm.tqdm(dataloader, desc=desc, leave=True, dynamic_ncols=True)
    
    for step, data in enumerate(pbar):
        rgb = data['rgb'].to(device)
        sn = data['sn'].to(device)
        depth = data['depth'].to(device)
        o_mask = data['mask'].to(device)
        kpts_list = data['keypoints']
        
        with torch.set_grad_enabled(is_train):
            seg_logits, offsets, points, trans_feat = model(rgb, sn, depth, o_mask, intrinsics_tuple)
            
            # --- TARGET GENERATION ---
            B, N, _ = points.shape
            fx, fy, cx, cy = intrinsics_tuple
            
            x, y, z = points[:, :, 0], points[:, :, 1], points[:, :, 2]
            # Avoid div by zero
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
            target_offsets = torch.zeros((B, N, model.offset_head.num_keypoints, 3), device=device)
            offset_mask = torch.zeros((B, N), dtype=torch.bool, device=device)
            
            for b in range(B):
                available_objs = kpts_list[b]
                for obj_id_str, kpts in available_objs.items():
                    obj_id = int(obj_id_str)
                    mask_b = (point_obj_ids[b] == obj_id)
                    if mask_b.sum() == 0: continue
                    
                    # MAPPING: Adjust if your IDs differ (e.g. 1->0)
                    class_idx = obj_id - 1 
                    
                    if class_idx >= 0 and class_idx < model.seg_head.mlp[-1].out_features:
                        gt_seg[b, mask_b] = class_idx
                        offset_mask[b, mask_b] = True
                        kpts_tensor = torch.tensor(kpts, device=device).float()
                        current_points = points[b, mask_b].unsqueeze(1)
                        target_offsets[b, mask_b] = kpts_tensor.unsqueeze(0) - current_points

            # --- LOSSES ---
            loss_seg = F.cross_entropy(seg_logits.view(-1, seg_logits.shape[-1]), gt_seg.view(-1))
            
            if offset_mask.sum() > 0:
                loss_off = F.l1_loss(offsets[offset_mask], target_offsets[offset_mask])
            else:
                loss_off = torch.tensor(0.0, device=device)
                
            loss_reg = feature_transform_regularizer(trans_feat)
            
            loss_total = loss_seg + loss_off + 0.001 * loss_reg
            
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
    # UPDATE THESE PATHS TO YOUR ACTUAL WINDOWS PATHS
    train_dir = "c:/Users/user/Desktop/AUB/Intro2ML/Project/data/train" 
    valid_dir = "c:/Users/user/Desktop/AUB/Intro2ML/Project/data/valid"

    # Check if paths exist
    if not os.path.exists(train_dir):
        print(f"WARNING: Train dir {train_dir} not found. Please edit the path in training.py")
    
    train_dataset = Stage2Dataset(root_dir=train_dir, transforms=T.ToTensor())
    # If valid dir doesn't exist, use train for valid (just for testing code)
    if not os.path.exists(valid_dir):
        print("Validation dir not found, using split of train or same dataset...")
        valid_dataset = train_dataset 
    else:
        valid_dataset = Stage2Dataset(root_dir=valid_dir, transforms=T.ToTensor())

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        collate_fn=collate_fn, num_workers=NUM_WORKERS, drop_last=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        collate_fn=collate_fn, num_workers=NUM_WORKERS, drop_last=False
    )

    # 2. MODEL
    intrinsics = train_dataset.camera_intrisics
    fx, fy, cx, cy = intrinsics[0,0], intrinsics[1,1], intrinsics[0,2], intrinsics[1,2]
    intrinsics_tuple = (fx, fy, cx, cy)

    params = {
         "img_outdim": 128,
         "normals_outdim": 64, 
         "points_outdim": 256,
         "num_classes": 4,     
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
        
        train_loss = run_epoch(model, train_loader, intrinsics_tuple, device, optimizer, is_train=True, epoch_idx=epoch)
        val_loss = run_epoch(model, valid_loader, intrinsics_tuple, device, is_train=False, epoch_idx=epoch)
        
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