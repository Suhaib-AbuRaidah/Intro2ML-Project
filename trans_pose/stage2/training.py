import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torchvision import transforms as T
import sys
sys.path.append("/home/suhaib/ML_Project")

from trans_pose.stage2.dataset_stage2 import Stage2Dataset, collate_fn
from trans_pose.stage2.network import TransPoseNetwork
from trans_pose.stage2.utilis import votes_from_offsets, mean_shift_clustering, rigid_transform_3D

def train_epoch(model, dataloader,intrinsics, device, optimizer, epochs=10):
    intrinsics_param = (intrinsics[0,0], intrinsics[1,1], intrinsics[0,2], intrinsics[1,2])  # fx, fy, cx, cy
    for epoch in range(epochs):

        for step, data in enumerate(dataloader):

            poses_list = data['poses']        # list of length B, each is dict: obj_id → (4,4)
            kpts_list  = data['keypoints']    # list of length B, each is dict: obj_id → (K,3)
            rgb = data['rgb'].to(device)              # (B, 3, H, W)
            sn = data['sn'].to(device)                # (B, 3, H, W)
            depth = data['depth'].to(device)          # (B, 1, H, W)
            o_mask = data['mask'].to(device)            # (B, 1, H, W)
            seg_logits, offsets,points = model(rgb,depth,sn,o_mask,intrinsics_param)  # (B, N, C), (B, N, K, 3)
            seg_labels = seg_logits.argmax(dim=-1)  # (B,N)

            votes = votes_from_offsets(points, offsets)  # (B,N,K,3)

            B, N, K, _ = votes.shape

            loss_total = 0.0

            # ---------------- process each sample independently ----------------
            for b in range(B):

                poses_gt = poses_list[b]      # dict: obj_id → (4,4)
                kpts_gt  = kpts_list[b]       # dict: obj_id → (K,3)

                # object ids for THIS sample only
                object_ids = sorted([int(obj_id) for obj_id in poses_gt.keys()])
                O = len(object_ids)

                # assign local classes: 0..O-1
                class_map = { real_id: idx for idx, real_id in enumerate(object_ids) }

                # predicted segmentation for this sample
                seg_b = seg_labels[b]     # (N,)
                votes_b = votes[b]        # (N,K,3)

                centers_b = torch.zeros((O, K, 3), device=device)

                # ---------------- mean shift per object ----------------
                for local_idx, real_obj_id in enumerate(object_ids):

                    cls = class_map[real_obj_id]
                    mask = (seg_b == cls)        # (N,)
                    votes_obj = votes_b[mask]    # (M,K,3)

                    if votes_obj.shape[0] == 0:
                        centers_b[local_idx] = votes_b.mean(dim=0)
                    else:
                        inp = votes_obj.unsqueeze(0)     # (1,M,K,3)
                        ctr = mean_shift_clustering(inp, None, 0.05, 15)
                        centers_b[local_idx] = ctr[0]

                # ---------------- pose loss per object ----------------
                for local_idx, real_obj_id in enumerate(object_ids):

                    pred_kp = centers_b[local_idx]   # (K,3)

                    target_kp = torch.tensor(
                        kpts_gt[str(real_obj_id)], device=device
                    ).float()                         # (K,3)

                    pred_pose = rigid_transform_3D(pred_kp, target_kp)

                    target_pose = torch.tensor(
                        poses_gt[str(real_obj_id)], device=device
                    ).float()                         # (4,4)

                    loss_total += F.mse_loss(pred_pose, target_pose)

            # normalize by batch size
            loss_total /= B

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss={loss_total.item():.4f}")





if __name__ == "__main__":

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    dataset = Stage2Dataset(
        root_dir="/home/suhaib/ML_Project/data/transcg-data-1/transcg",transforms=T.ToTensor())

    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=8, drop_last=True, shuffle=True, collate_fn=collate_fn)

    intrinsics = dataset.camera_intrisics

    params = {
         "img_outdim":128,
         "normals_outdim":128,
         "points_outdim":256,
        "num_classes": 4,
        "num_keypoints": 10
    }

    model = TransPoseNetwork(**params).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    train_epoch(model, train_dataloader,intrinsics, device=device, optimizer=optimizer, epochs=50)
