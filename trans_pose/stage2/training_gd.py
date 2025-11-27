"""
Multi-Object 6D Pose Estimation Training Script

Key Differences from training.py:
1. Uses TransPoseNetworkMulti (multi-class segmentation)
2. CrossEntropy loss for segmentation
3. Object-aware offset loss (per-object keypoint matching)
4. Skips scenes containing object ID 0
5. Loads keypoints for ALL objects in scene
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torchvision import transforms as T
import sys
import os
import json
from pathlib import Path

# Adjust path to point to project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trans_pose.stage2.dataset_stage2 import Stage2Dataset, collate_fn
from trans_pose.stage2.network_gd import TransPoseNetworkMulti
from read_intrinsics import scale_intrinsics

# --- CONFIGURATION ---
BATCH_SIZE = 2
LEARNING_RATE = 0.0005
EPOCHS = 10
NUM_WORKERS = 0

# Define target output size for images
TARGET_H = 360
TARGET_W = 640

# Original image resolution
ORIGINAL_H = 720
ORIGINAL_W = 1280

WEIGHT_DECAY = 1e-4
LR_DECAY_STEP = 10
LR_DECAY_GAMMA = 0.5
SAVE_DIR = "checkpoints_multi"
RESUME = False
PATIENCE = 5  # Early stopping patience

# Multi-object specific settings
NUM_CLASSES = 61  # 60 objects + 1 background
SKIP_OBJECT_ZERO = True  # Skip scenes with object ID 0
# ---------------------


def feature_transform_regularizer(trans):
    """Regularization for PointNet feature transformation matrix."""
    d = trans.size()[1]
    I = torch.eye(d, device=trans.device)[None, :, :]
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


def compute_segmentation_loss(seg_logits, point_labels, num_classes=NUM_CLASSES):
    """
    Multi-class segmentation loss using CrossEntropy.
    
    Args:
        seg_logits: [B, N, num_classes] raw class scores
        point_labels: [B, N] ground truth object IDs
    
    Returns:
        loss: scalar
    """
    B, N, C = seg_logits.shape
    
    # Flatten for CrossEntropy
    B, N, C = seg_logits.shape
    
    # Flatten for CrossEntropy (use reshape to handle non-contiguous tensors)
    seg_logits_flat = seg_logits.reshape(B * N, C)  # [B*N, num_classes]
    point_labels_flat = point_labels.reshape(B * N).long()  # [B*N]
    
    # CrossEntropy expects long tensor targets
    loss = F.cross_entropy(seg_logits_flat, point_labels_flat, ignore_index=-1)
    
    return loss



def compute_offset_loss_multi(offsets, points, point_labels, kpts_dict_batch, num_keypoints=10):
    """
    Object-aware offset loss - uses point Z as additional feature context.
    """
    B, N, K, _ = offsets.shape
    device = offsets.device
    
    total_loss = 0.0
    num_objects_processed = 0
    
    for b in range(B):
        kpts_dict = kpts_dict_batch[b]
        unique_objs = torch.unique(point_labels[b])
        unique_objs = unique_objs[unique_objs > 0]
        
        if len(unique_objs) == 0:
            continue
        
        for obj_id in unique_objs:
            obj_id_int = obj_id.item()
            
            if str(obj_id_int) not in kpts_dict:
                continue
            
            mask_obj = (point_labels[b] == obj_id_int)
            
            if mask_obj.sum() == 0:
                continue
            
            offsets_obj = offsets[b, mask_obj]       # [M, K, 3]
            points_obj = points[b, mask_obj]         # [M, 3]
            
            kpts_gt = torch.tensor(kpts_dict[str(obj_id_int)], device=device, dtype=torch.float32)
            
            if kpts_gt.shape[0] != K:
                continue
            
            target_offsets = kpts_gt.unsqueeze(0) - points_obj.unsqueeze(1)  # [M, K, 3]
            
            # ✅ ONLY X,Y offsets matter (points already have correct Z from depth)
            # Z can be near-zero, penalize it lightly
            loss_xy = F.l1_loss(offsets_obj[..., :2], target_offsets[..., :2], reduction='mean')
            loss_z = F.l1_loss(offsets_obj[..., 2], target_offsets[..., 2], reduction='mean') * 0.1
            
            loss_obj = loss_xy + loss_z
            total_loss += loss_obj
            num_objects_processed += 1
    
    if num_objects_processed > 0:
        return total_loss / num_objects_processed
    else:
        return torch.tensor(0.0, device=device)



def run_epoch(model, dataloader, intrinsics_tuple_scaled, device, optimizer=None, is_train=True, epoch_idx=0):
    """Run one training/validation epoch."""
    if is_train:
        model.train()
        desc = f"Train Ep {epoch_idx+1}"
    else:
        model.eval()
        desc = f"Valid Ep {epoch_idx+1}"

    total_loss = 0.0
    total_seg = 0.0
    total_off = 0.0
    total_reg = 0.0
    num_batches = 0
    
    pbar = tqdm.tqdm(dataloader, desc=desc, leave=False, dynamic_ncols=True)

    for step, data in enumerate(pbar):
        # ⚡ GPU Memory Check (every 10 steps)
        if step % 10 == 0 and torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
            mem_reserved = torch.cuda.memory_reserved(device) / 1024**3    # GB
            if step == 0:
                pbar.write(f"GPU Memory: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")

        # --- DEBUG WRAPPER: surface dataset / forward errors with context ---
        try:
            rgb = data['rgb'].to(device)
            sn = data['sn'].to(device)
            depth = data['depth'].to(device)
            o_mask = data['mask'].to(device)
            kpts_list = data['keypoints']
        except Exception as e:
            print(f"\nERROR: failed preparing batch at step={step} in {desc}")
            print(f"  data keys: {list(data.keys())}")
            try:
                # try to print basic info for each key
                for k, v in data.items():
                    if torch.is_tensor(v):
                        print(f"   {k}: tensor shape={v.shape}, dtype={v.dtype}")
                    else:
                        print(f"   {k}: type={type(v)}")
            except Exception:
                pass
            # save raw batch for inspection
            try:
                torch.save({'data': data}, "/tmp/failing_batch_preparation.pt")
                print("  Saved failing batch to /tmp/failing_batch_preparation.pt")
            except Exception:
                pass
            raise

        # SKIP SCENES WITH OBJECT ID 0 (if enabled)
        if SKIP_OBJECT_ZERO:
            skip_batch = False
            for kpts_dict in kpts_list:
                if '0' in kpts_dict:
                    skip_batch = True
                    break
            
            if skip_batch:
                    if step == 0:
                        pbar.write(f"Skipping batch with object ID 0")
                    continue
        
        # --- Forward + loss inside try to capture runtime errors ---
        try:
            with torch.set_grad_enabled(is_train):
                seg_logits, offsets, points, trans_feat, point_labels = model(
                    rgb, sn, depth, o_mask, intrinsics_tuple_scaled
                )
                
                # --- LOSSES ---
                loss_seg = compute_segmentation_loss(seg_logits, point_labels, num_classes=NUM_CLASSES)
                # loss_off = compute_offset_loss_multi(
                #     offsets, points, point_labels, kpts_list, 
                #     num_keypoints=model.offset_head.num_keypoints
                # )
                loss_off = compute_offset_loss_multi(
                    offsets, points, point_labels, kpts_list,
                    num_keypoints=model.offset_head.num_keypoints
                )

                pred_kpts_from_offsets = points.unsqueeze(2) + offsets  # [B, N, 1, 3] + [B, N, K, 3] = [B, N, K, 3]
                loss_var = torch.tensor(0.0, device=device)
                
                loss_reg = feature_transform_regularizer(trans_feat)
                
                # Total loss (SIMPLE - NO TRICKS)
                loss_total = 1.0 * loss_seg + 1.0 * loss_off + 0.001 * loss_reg

                if is_train:
                    optimizer.zero_grad()
                    loss_total.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                    optimizer.step()
        except Exception as e:
            pbar.write(f"ERROR: exception during forward/loss at step={step} in {desc}: {e}")
            # print shapes to help debug
            try:
                pbar.write(f" seg_logits: {getattr(seg_logits,'shape',None)}")
            except Exception:
                pass
            try:
                print(f" offsets: {getattr(offsets,'shape',None)}")
            except Exception:
                pass
            try:
                print(f" points: {getattr(points,'shape',None)}")
            except Exception:
                pass
            try:
                print(f" point_labels: {getattr(point_labels,'shape',None)}")
            except Exception:
                pass
            # save problematic batch (CPU)
            try:
                save_dict = {}
                for k, v in data.items():
                    try:
                        save_dict[k] = v.cpu() if torch.is_tensor(v) else v
                    except Exception:
                        save_dict[k] = str(type(v))
                torch.save({'data': save_dict}, "/tmp/failing_batch_forward.pt")
                print("  Saved failing batch to /tmp/failing_batch_forward.pt")
            except Exception:
                pass
            raise

        # ⚡ Clear cache after each batch (helps with memory)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Update stats
        total_loss += loss_total.item()
        total_seg += loss_seg.item()
        total_off += loss_off.item()
        total_reg += loss_reg.item()
        num_batches += 1
        
        # Real-time print in progress bar
        pbar.set_postfix({
            "Loss": f"{loss_total.item():.4f}",
            "Seg": f"{loss_seg.item():.4f}",
            "Off": f"{loss_off.item():.4f}"
        })
    
    if num_batches == 0:
        return float('inf')
    
    return total_loss / num_batches

def compute_offset_loss_multi_weighted(offsets, points, point_labels, kpts_dict_batch, num_keypoints=10):
    """
    Object-aware offset loss WITH Z-AXIS WEIGHTING.
    
    Penalizes Z errors 5x more than X,Y to force proper depth learning.
    
    Args:
        offsets: [B, N, K, 3] predicted offsets
        points: [B, N, 3] 3D point cloud
        point_labels: [B, N] ground truth object ID for each point
        kpts_dict_batch: List of dicts (length B), each dict maps obj_id -> keypoints [K, 3]
    
    Returns:
        loss: scalar
    """
    B, N, K, _ = offsets.shape
    device = offsets.device
    
    total_loss = 0.0
    num_objects_processed = 0
    
    for b in range(B):
        kpts_dict = kpts_dict_batch[b]
        unique_objs = torch.unique(point_labels[b])
        unique_objs = unique_objs[unique_objs > 0]
        
        if len(unique_objs) == 0:
            continue
        
        for obj_id in unique_objs:
            obj_id_int = obj_id.item()
            
            if str(obj_id_int) not in kpts_dict:
                continue
            
            mask_obj = (point_labels[b] == obj_id_int)
            
            if mask_obj.sum() == 0:
                continue
            
            offsets_obj = offsets[b, mask_obj]  # [M, K, 3]
            points_obj = points[b, mask_obj]    # [M, 3]
            
            kpts_gt = torch.tensor(kpts_dict[str(obj_id_int)], device=device, dtype=torch.float32)
            
            if kpts_gt.shape[0] != K:
                continue
            
            target_offsets = kpts_gt.unsqueeze(0) - points_obj.unsqueeze(1)  # [M, K, 3]
            
            # ✅ NEW: Compute per-axis errors with Z weighting
            error_x = F.l1_loss(offsets_obj[..., 0], target_offsets[..., 0], reduction='mean')
            error_y = F.l1_loss(offsets_obj[..., 1], target_offsets[..., 1], reduction='mean')
            error_z = F.l1_loss(offsets_obj[..., 2], target_offsets[..., 2], reduction='mean')
            
            # Weight Z 5x more to force proper depth learning
            loss_obj = 1.0 * error_x + 1.0 * error_y + 5.0 * error_z
            total_loss += loss_obj
            num_objects_processed += 1
    
    if num_objects_processed > 0:
        return total_loss / num_objects_processed
    else:
        return torch.tensor(0.0, device=device)


def compute_keypoint_variance_loss(pred_kpts):
    """
    Penalize colinear keypoints by encouraging variance in each dimension.
    
    Args:
        pred_kpts: [B, N, K, 3] predicted keypoints from offsets
    
    Returns:
        loss: scalar (lower = better diversity)
    """
    B, N, K, _ = pred_kpts.shape
    
    if K < 2:
        return torch.tensor(0.0, device=pred_kpts.device)
    
    # Compute variance per dimension per batch
    var_x = pred_kpts[..., 0].var(dim=2, keepdim=True)  # [B, N, 1]
    var_y = pred_kpts[..., 1].var(dim=2, keepdim=True)
    var_z = pred_kpts[..., 2].var(dim=2, keepdim=True)
    
    # Total variance across all dimensions
    total_var = var_x + var_y + var_z  # [B, N, 1]
    
    # ✅ FIXED: Penalize LOW variance (not negate!)
    # We want to maximize variance, so loss = 1 / (1 + variance)
    # Lower variance → higher loss
    loss = 1.0 / (1.0 + total_var.mean())
    
    return loss

class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float('inf')
        self.counter = 0

    def step(self, value):
        if value < self.best - self.min_delta:
            self.best = value
            self.counter = 0
            return False  # not stopping
        else:
            self.counter += 1
            return self.counter >= self.patience


if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    print(f"Multi-object training with {NUM_CLASSES} classes")
    print(f"Skip object ID 0: {SKIP_OBJECT_ZERO}")

    # 1. DATASETS
    keypoints_dir = "/media/ahmad/New Volume/ML-GD/TransCG/transcg-data-2/keypoints"
    train_dir = "/media/ahmad/New Volume/ML-GD/TransCG/transcg-data-2/train"
    valid_dir = "/media/ahmad/New Volume/ML-GD/TransCG/transcg-data-2/valid"

    data_transforms = T.ToTensor()

    train_dataset = Stage2Dataset(
        root_dir=train_dir,
        keypoints_dir=keypoints_dir,
        transforms=data_transforms,
        target_size=(TARGET_W, TARGET_H),
        augment=True   
    )
    print(f"\n CANONICAL KEYPOINTS LOADED:")
    loaded_kpts = train_dataset.canonical_kpts
    print(f"   Total objects: {len(loaded_kpts)}")
    
    if len(loaded_kpts) > 0:
        print(f"   All IDs: {sorted(loaded_kpts.keys())}")
        
        # Check if your problem objects exist
        problem_ids = ['4', '7', '11', '36', '50', '54']
        for obj_id in problem_ids:
            if obj_id in loaded_kpts:
                print(f"    Object {obj_id}: shape {loaded_kpts[obj_id].shape}")
            else:
                print(f"    Object {obj_id}: NOT FOUND")
    else:
        print(f"    ERROR: No keypoints loaded!")
        print(f"   Keypoints dir: {keypoints_dir}")
        print(f"   Files in dir:")
        import glob
        files = glob.glob(os.path.join(keypoints_dir, "*.npz"))
        for f in sorted(files)[:10]:
            print(f"      {os.path.basename(f)}")
    
    #  DEBUG: Test loading one sample
    print(f"\n TESTING FIRST SAMPLE:")
    sample = train_dataset[0]
    print(f"   RGB shape: {sample['rgb'].shape}")
    print(f"   Keypoints in sample: {list(sample['keypoints'].keys())}")
    
    if len(sample['keypoints']) > 0:
        for obj_id, kpts in sample['keypoints'].items():
            print(f"      Object {obj_id}: {len(kpts)} keypoints")
    else:
        print(f"    No keypoints in first sample!")
    

    original_intrinsics_matrix = train_dataset.camera_intrisics
    
    # Extract fx, fy, cx, cy from the original intrinsics matrix
    original_fx = original_intrinsics_matrix[0, 0]
    original_fy = original_intrinsics_matrix[1, 1]
    original_cx = original_intrinsics_matrix[0, 2]
    original_cy = original_intrinsics_matrix[1, 2]

    if not os.path.exists(valid_dir):
        valid_dataset = train_dataset
    else:
        valid_dataset = Stage2Dataset(
            root_dir=valid_dir,
            keypoints_dir=keypoints_dir,
            transforms=data_transforms,
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

    intrinsics_tuple_scaled = (original_fx, original_fy, original_cx, original_cy)

    # 2. MODEL
    params = {
        "img_outdim": 128,
        "normals_outdim": 64,
        "points_outdim": 256,
        "num_keypoints": 10,
        "num_classes": NUM_CLASSES
    }
    model = TransPoseNetworkMulti(**params).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

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
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resumed at Epoch {start_epoch}")

    earlystop = EarlyStopping(patience=PATIENCE)
    train_losses = []
    val_losses = []

    # 5. TRAINING LOOP
    for epoch in range(start_epoch, EPOCHS):
        current_lr = scheduler.get_last_lr()[0]
        print(f"\n=== Epoch {epoch+1}/{EPOCHS} | LR: {current_lr:.6f} ===")
        
        train_loss = run_epoch(
            model, train_loader, intrinsics_tuple_scaled, device,
            optimizer, is_train=True, epoch_idx=epoch
        )
        val_loss = run_epoch(
            model, valid_loader, intrinsics_tuple_scaled, device,
            optimizer=None, is_train=False, epoch_idx=epoch
        )
        
        scheduler.step()
        
        print(f"Summary Ep {epoch+1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        #  append histories 
        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))
        
        # checkpointing same as before...
        if earlystop.step(val_loss):
            print(f"Early stopping triggered (no improvement for {PATIENCE} epochs).")
            break

        # Checkpoint Dictionary
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'loss': train_loss,
            'num_classes': NUM_CLASSES,
            'train_losses': train_losses, 
            'val_losses': val_losses,
            'scheduler_state_dict': scheduler.state_dict()
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

        try:
            import numpy as _np, json as _json
            _np.savez_compressed(os.path.join(SAVE_DIR, "loss_history.npz"),
                                train=_np.array(train_losses), val=_np.array(val_losses))
            with open(os.path.join(SAVE_DIR, "loss_history.json"), "w") as _f:
                _json.dump({"train": train_losses, "val": val_losses}, _f)
        except Exception as e:
            print(f"Warning: saving loss history failed: {e}")

    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved in: {SAVE_DIR}")
    print("="*60)