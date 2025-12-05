import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torchvision import transforms as T
import sys
import os
import glob
import numpy as np
from torch.utils.data import random_split
# Adjust path to point to project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from trans_pose.stage2.utilis import votes_from_offsets, mean_shift_clustering, rigid_transform_3D
import cv2
from trans_pose.stage2.dataset2_stage2 import Stage2Dataset, collate_fn
from trans_pose.stage2.network import TransPoseNetwork
from read_intrinsics import scale_intrinsics
import open3d as o3d
# --- CONFIGURATION ---
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
EPOCHS = 20
NUM_WORKERS = 0



WEIGHT_DECAY = 1e-4
LR_DECAY_STEP = 10
LR_DECAY_GAMMA = 0.5
SAVE_DIR = "checkpoints"
RESUME = True
# ---------------------
def summarize(x):
    return {
        "shape": tuple(x.shape),
        "dtype": x.dtype,
        "device": x.device,
        "min": x.min().item(),
        "max": x.max().item(),
        "mean": x.float().mean().item(),
        "std": x.float().std().item(),
        "nan_count": torch.isnan(x).sum().item(),
        "inf_count": torch.isinf(x).sum().item(),
    }
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
    pbar = tqdm.tqdm(dataloader, desc=desc, leave=True, dynamic_ncols=True)

    for step, data in enumerate(pbar):
        rgb = data['rgb'].to(device)
        sn = data['sn'].to(device)
        depth = data['depth'].to(device)
        o_mask = data['mask'].to(device)

        poses_list        = data['poses']
        zero_kpts_list    = data['zero_keypoints']    # dict[obj_id] → (K,3)
        target_kpts_list  = data['target_keypoints']  # dict[obj_id] → (K,3)

        with torch.set_grad_enabled(is_train):

            seg_logits, offsets, points, trans_feat = model(
                rgb, sn, depth, o_mask, intrinsics_tuple_scaled
            )

            seg_labels = seg_logits.argmax(dim=-1)      # (B,N)
            votes = votes_from_offsets(points, offsets) # (B,N,K,3)

            B, N, K, _ = votes.shape
            loss_total = 0.0

            # =====================================================================
            #                        PROCESS EACH SAMPLE
            # =====================================================================
            for b in range(B):
                # img = depth[b]
                # # img = cv2.cvtColor(img.permute(1,2,0).cpu().numpy(), cv2.COLOR_RGB2BGR)
                # img=img.permute(1,2,0).cpu().numpy()
                # print(f"img shape: {img.shape},min: {img.min()}, max: {img.max()}")
                # cv2.imshow("img", img)
                # if cv2.waitKey(0)==ord('q'):
                #     cv2.destroyAllWindows()
                # points_obj = points[b]
                # pcd_obj = o3d.geometry.PointCloud()
                # pcd_obj.points = o3d.utility.Vector3dVector(points_obj.cpu().numpy())
                # o3d.visualization.draw_geometries([pcd_obj])
                poses_gt       = poses_list[b]            # dict obj_id → (4,4)
                zero_kpts_gt   = zero_kpts_list[b]        # dict obj_id → (K,3)
                target_kpts_gt = target_kpts_list[b]      # dict obj_id → (K,3)

                object_ids = sorted([int(obj_id) for obj_id in poses_gt.keys()])
                O = len(object_ids)

                # consistent local class mapping 0..O-1
                class_map = { real_id: idx for idx, real_id in enumerate(object_ids) }

                seg_b   = seg_labels[b]      # (N,)
                votes_b = votes[b]           # (N,K,3)
                centers_b = torch.zeros((O,K,3), device=device)

                # print(f"seg_b min: \n{seg_b.min()}")
                # print(f"seg_b max: \n{seg_b.max()}")
                # print(points[b].shape)
                # =================================================================
                #               MEAN SHIFT CLUSTERING (voted keypoints)
                # =================================================================
                for local_idx, real_obj_id in enumerate(object_ids):
                    # print(f"real_obj_id: {real_obj_id}")
                    # print(f"class_map: \n{class_map[real_obj_id]}")
                    # print("\n\n")
                    # print(f"seg_b: \n{seg_b}")
                    cls = class_map[real_obj_id]
                    mask = (seg_b == cls)
                    votes_obj = votes_b[mask]   # (M,K,3)

                    if epoch_idx>=0:
                        points_obj = points[b][mask]
                        # print(points_obj.shape)
                        pcd_obj = o3d.geometry.PointCloud()
                        pcd_obj.points = o3d.utility.Vector3dVector(points_obj.cpu().numpy())
                        pcd_obj.paint_uniform_color([1, 0, 0])
                        o3d.visualization.draw_geometries([pcd_obj])



                    if votes_obj.shape[0] == 0:
                        centers_b[local_idx] = votes_b.mean(dim=0)
                    else:
                        inp = votes_obj.unsqueeze(0)   # (1,M,K,3)
                        ctr = mean_shift_clustering(inp, None, 0.05, 15)
                        centers_b[local_idx] = ctr[0]
                # print(f"centers_b: \n{centers_b.shape}")    
                # =================================================================
                #      SUPERVISE VOTED KEYPOINTS → target pose keypoints
                #                AND compute transform to zero pose
                # =================================================================
                for local_idx, real_obj_id in enumerate(object_ids):

                    pred_kp = centers_b[local_idx]  # (K,3)

                    target_kp = torch.tensor(
                        target_kpts_gt[str(real_obj_id)], device=device
                    ).float()                       # (K,3)

                    zero_kp = torch.tensor(
                        zero_kpts_gt[str(real_obj_id)], device=device
                    ).float()

                    # print(f"pred_kp: \n{pred_kp}")
                    # print(f"target_kp: \n{target_kp}")
                    # print(f"zero_kp: \n{zero_kp}")
   
                    # 1. supervise voted keypoints to match the target pose keypoints
                    kpt_loss = F.mse_loss(pred_kp, target_kp)
                    loss_total += kpt_loss

                    # 2. compute rigid transform between predicted and zero pose
                    # if epoch_idx==1:
                    #     pred_T_zero = rigid_transform_3D(zero_kp, target_kp)
                    #     print(f"Predicted Transformation: \n{pred_T_zero}")
                    #     print(f"Target Transformation: \n{poses_gt[str(real_obj_id)]}")
                    #     print("\n\n")
                    # (NOT ADDED TO LOSS unless requested)

            loss_total /= B

            if is_train:
                optimizer.zero_grad()
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()

        total_loss += loss_total.item()
        pbar.set_postfix({"Loss": f"{loss_total.item():.6f}"})

    return total_loss / len(dataloader)


if __name__ == "__main__":

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

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

    # The intrinsics loaded from Stage2Dataset are already scaled to TARGET_W, TARGET_H (640x360).
    # Therefore, we use them directly without further scaling.
    # If Stage2Dataset were to provide ORIGINAL (1280x720) intrinsics,
    # then the scale_intrinsics function would be correctly applied here.
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