import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torchvision import transforms as T
import sys
import os
import glob
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import open3d as o3d
import cv2
from torch.utils.data import random_split
import datetime

# Adjust path to point to project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from trans_pose.stage2.utilis import votes_from_offsets, mean_shift_clustering, rigid_transform_3D
from trans_pose.stage2.dataset2_stage2 import Stage2Dataset, collate_fn
from trans_pose.stage2.network import TransPoseNetwork
from read_intrinsics import scale_intrinsics

# --- CONFIGURATION ---
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EPOCHS = 20
NUM_WORKERS = 16



WEIGHT_DECAY = 1e-4
LR_DECAY_STEP = 10
LR_DECAY_GAMMA = 0.5
SAVE_DIR = "checkpoints"
RESUME = False

# ---------------- loss weights ----------------
W_POSE     = 1.0
W_SEG      = 0.5
W_OFFSETS  = 0.25

def run_epoch(model, dataloader, intrinsics_tuple_scaled, device, 
              optimizer=None, is_train=True, epoch_idx=0):

    if is_train:
        model.train()
        desc = f"Train Ep {epoch_idx+1}"
    else:
        model.eval()
        desc = f"Valid Ep {epoch_idx+1}"

    total_loss = 0.0
    pose_loss = 0.0
    seg_loss = 0.0
    offset_loss = 0.0

    pbar = tqdm.tqdm(dataloader, desc=desc, leave=True, dynamic_ncols=True)

    for step, data in enumerate(pbar):

        rgb   = data['rgb'].to(device)
        sn    = data['sn'].to(device)
        depth = data['depth'].to(device)
        o_mask = data['mask'].to(device)

        poses_list        = data['poses']
        zero_kpts_list    = data['zero_keypoints']
        target_kpts_list  = data['target_keypoints']   # REQUIRED for Fix #2

        with torch.set_grad_enabled(is_train):

            seg_logits, offsets, points, trans_feat = model(
                rgb, sn, depth, o_mask, intrinsics_tuple_scaled)

            B, N, K, _ = votes_from_offsets(points, offsets).shape
            seg_labels = seg_logits.argmax(dim=-1)
            votes = votes_from_offsets(points, offsets)

            loss_total = 0.0
            loss_pose_batch    = 0.0
            loss_seg_batch     = 0.0
            loss_offsets_batch = 0.0

            # ---------------------------------------------------------
            # Process each sample independently
            # ---------------------------------------------------------
            for b in range(B):

                poses_gt       = poses_list[b]
                zero_kp_dict   = zero_kpts_list[b]
                target_kp_dict = target_kpts_list[b]

                object_ids = sorted([int(obj_id) for obj_id in poses_gt.keys()])
                O = len(object_ids)

                # Local class IDs: real_id → 0..O-1
                class_map = {real_id: idx for idx, real_id in enumerate(object_ids)}

                seg_b   = seg_labels[b]     # (N,)
                votes_b = votes[b]          # (N,K,3)
                pts_b   = points[b]         # (N,3)

                # ---------------------- FIX #2 ----------------------
                # Create seg_gt by assigning each point to the nearest object's keypoints
                seg_gt = torch.zeros(N, dtype=torch.long, device=device)
                offsets_gt = torch.zeros_like(offsets[b])  # (N,K,3)

                # Collect keypoints of all objects into one tensor (O,K,3)
                all_kpts = []
                for real_obj in object_ids:
                    k = torch.tensor(target_kp_dict[str(real_obj)], device=device).float()  # (K,3)
                    all_kpts.append(k)
                all_kpts = torch.stack(all_kpts, dim=0)   # (O,K,3)

                # pts_b : (N, 3)
                # all_kpts : (O, K, 3)

                N = pts_b.shape[0]
                O, K, _ = all_kpts.shape

                all_kpts_flat = all_kpts.reshape(O*K, 3)     # (O*K, 3)

                # Pairwise distances
                d_flat = torch.cdist(pts_b, all_kpts_flat)   # (N, O*K)

                # Reshape back
                dists = d_flat.reshape(N, O, K)              # (N, O, K)

                # For each point, reduce distances to (N,O) by min over keypoints
                d_obj = dists.min(dim=2).values  # (N,O)

                # Finally assign the nearest object for each point
                seg_gt = d_obj.argmin(dim=1)     # (N,), integer class id ∈ [0, O-1]
                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(pts_b.cpu().numpy())
                # pcd.paint_uniform_color([1, 0, 0])
                # o3d.visualization.draw_geometries([pcd])

                # ---------------------- offsets supervision ----------------------
                # For each point → offsets to all K keypoints of its assigned object
                for real_obj in object_ids:
                    cls = class_map[real_obj]

                    obj_mask = (seg_gt == cls)

                    # pcd_ob = o3d.geometry.PointCloud()
                    # pcd_ob.points = o3d.utility.Vector3dVector(pts_b[obj_mask].cpu().numpy())
                    # pcd_ob.paint_uniform_color([0, 1, 0])
                    # o3d.visualization.draw_geometries([pcd_ob])

                    if obj_mask.sum() == 0:
                        continue

                    kpts = torch.tensor(target_kp_dict[str(real_obj)],
                                        device=device).float()  # (K,3)

                    pts_obj = pts_b[obj_mask]       # (M,3)
                    # broadcast: pts → (M,1,3), kpts → (1,K,3)
                    offsets_gt[obj_mask] = kpts.unsqueeze(0) - pts_obj.unsqueeze(1)

                # ---------------- segmentation loss ----------------
                loss_seg = F.cross_entropy(seg_logits[b], seg_gt)

                # ---------------- offset regression loss ----------------
                loss_offset = F.l1_loss(offsets[b], offsets_gt)

                # ---------------- pose loss ----------------
                loss_pose = 0.0
                centers_b = torch.zeros((O, K, 3), device=device)

                # mean-shift clustering per object
                for real_obj in object_ids:
                    cls = class_map[real_obj]
                    mask = (seg_b == cls)
                    votes_obj = votes_b[mask]

                    if votes_obj.shape[0] == 0:
                        centers_b[cls] = votes_b.mean(dim=0)
                    else:
                        inp = votes_obj.unsqueeze(0)
                        ctr = mean_shift_clustering(inp, None, 0.05, 15)
                        centers_b[cls] = ctr[0]

                # pose loss over objects
                for real_obj in object_ids:
                    local_idx = class_map[real_obj]

                    pred_kp = centers_b[local_idx]   # (K,3)
                    zero_kp = torch.tensor(zero_kp_dict[str(real_obj)],
                                           device=device).float()  # (K,3)

                    pred_pose = rigid_transform_3D(zero_kp, pred_kp)

                    target_pose = torch.tensor(
                        poses_gt[str(real_obj)], device=device).float()  # (4,4)

                    loss_pose += F.mse_loss(pred_pose, target_pose)

                loss_pose_batch    += loss_pose
                loss_seg_batch     += loss_seg
                loss_offsets_batch += loss_offset

            # Average across batch
            loss_pose_batch    /= B
            loss_seg_batch     /= B
            loss_offsets_batch /= B

            # Final weighted loss
            loss_total = (W_POSE * loss_pose_batch +
                          W_SEG  * loss_seg_batch +
                          W_OFFSETS * loss_offsets_batch)

            if is_train:
                optimizer.zero_grad()
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()

        pose_loss += loss_pose_batch.item()
        seg_loss += loss_seg_batch.item()
        offset_loss += loss_offsets_batch.item()
        total_loss += loss_total.item()

        pbar.set_postfix({
            "Loss": f"{loss_total.item():.4f}",
            "Seg": f"{loss_seg_batch.item():.4f}",
            "Off": f"{loss_offsets_batch.item():.4f}",
            "Pose": f"{loss_pose_batch.item():.4f}",
        })

    return total_loss / len(dataloader), pose_loss / len(dataloader), seg_loss / len(dataloader), offset_loss / len(dataloader)


if __name__ == "__main__":


    torch.seed(42)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    start_training_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    writer = SummaryWriter(f'runs/{start_training_time}')
    print(f"TensorBoard logs will be saved to: './runs/{start_training_time}'")
    dataset = Stage2Dataset(
        root_dir="/home/suhaib/ML_Project/data/transcg-data-1/transcg",transforms=T.ToTensor())
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=8, drop_last=True, shuffle=True, collate_fn=collate_fn)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, val_size])

    original_intrinsics_matrix = dataset.camera_intrisics
    
    # Extract fx, fy, cx, cy from the original intrinsics matrix
    original_fx = original_intrinsics_matrix[0,0]
    original_fy = original_intrinsics_matrix[1,1]
    original_cx = original_intrinsics_matrix[0,2]
    original_cy = original_intrinsics_matrix[1,2]



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
         "normals_outdim": 128, 
         "points_outdim": 256,
         "num_classes": 4,     
         "num_keypoints": 10   
    }
    model = TransPoseNetwork(**params).float().to(device)


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
    patience = 5
    no_improve = 0
    epoch = start_epoch

    while True:
        current_lr = scheduler.get_last_lr()[0]
        print(f"\n=== Epoch {epoch+1} | LR: {current_lr:.6f} ===")

        train_loss, pose_loss, seg_loss, offset_loss = run_epoch(
            model, train_loader, intrinsics_tuple_scaled, device,
            optimizer, is_train=True, epoch_idx=epoch
        )

        val_loss, val_pose_loss, val_seg_loss, val_offset_loss = run_epoch(
            model, valid_loader, intrinsics_tuple_scaled, device,
            is_train=False, epoch_idx=epoch
        )

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/pose_loss', pose_loss, epoch)
        writer.add_scalar('Loss/seg_loss', seg_loss, epoch)
        writer.add_scalar('Loss/offset_loss', offset_loss, epoch)
        writer.add_scalar('Loss/val_loss', val_loss, epoch)
        writer.add_scalar('Loss/val_pose_loss', val_pose_loss, epoch)
        writer.add_scalar('Loss/val_seg_loss', val_seg_loss, epoch)
        writer.add_scalar('Loss/val_offset_loss', val_offset_loss, epoch)

        scheduler.step()

        print(f"Summary Ep {epoch+1}:\n"
            f"Train Loss: {train_loss:.4f} | Pose: {pose_loss:.4f} | "
            f"Seg: {seg_loss:.4f} | Offset: {offset_loss:.4f}\n"
            f"Val Loss: {val_loss:.4f}")

        # Save latest
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'loss': train_loss
        }
        torch.save(checkpoint, latest_ckpt)

        # Best model logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, os.path.join(SAVE_DIR, "best_model.pth"))
            print(f">>> New Best Model! (Val Loss: {val_loss:.4f})")
            no_improve = 0
        else:
            no_improve += 1
            print(f"No improvement for {no_improve}/{patience} epochs")

        # Stop criterion
        if no_improve >= patience:
            print(f"Stopping because no improvement for {patience} epochs.")
            break

        epoch += 1
