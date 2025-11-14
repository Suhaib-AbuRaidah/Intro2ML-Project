import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from stage2.dataset_stage2 import Stage2_dataset
from stage2.network import TransPoseNetwork
from stage2.utilis import votes_from_offsets


def train_epoch(model, dataloader, num_objects, device):
    for epoch in tqdm.tqdm(range(50)):
        for step, data in enumerate(dataloader, start=1):
            seg_logits, offsets = model(data)
            seg_probs = F.softmax(seg_logits, dim=-1)
            seg_labels = seg_probs.argmax(dim=-1)  # (B,N) semantic labels

            votes = votes_from_offsets(data, offsets)  # (B,N,K,3)

            # For each object class in the scene, we will cluster votes for points predicted as that class.
            # For this example, we assume single foreground class id = 1 (adapt as needed).
            B, N, K, _ = votes.shape
            centers = torch.zeros((B, K, 3), device=device)
            for b in range(B):
                # mask points that belong to object(s) of interest (semantic label >0)
                mask = (seg_labels[b] > 0)  # boolean (N,)
                if mask.sum() == 0:
                    # fallback: use all points
                    mask = torch.ones(N, dtype=torch.bool, device=device)
                votes_b = votes[b]  # (N,K,3)
                mask_f = mask.to(device)
                votes_masked = votes_b[mask_f]  # (M,K,3)
                if votes_masked.shape[0] == 0:
                    centers[b] = votes_b.mean(dim=0)
                else:
                    # run mean-shift on votes_masked (convert to shape (1, M, K, 3))
                    v = votes_masked.unsqueeze(0)  # (1,M,K,3)
                    centers_b = mean_shift_clustering(v, mask=None, bandwidth=0.05, num_iters=15)
                    centers[b] = centers_b[0]


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

dataset = Stage2_dataset(
    root_dir="/home/suhaib/ML_Project/data/transcg-data-2/transcg")

train_dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=8, drop_last=True, shuffle=True)

params = {
    "in_dim": 3,
    "feature_outdim": 1024,
    "num_classes": 4,
    "num_keypoints": 10
}

model = TransPoseNetwork(**params).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
model.train()


# return seg_logits, offsets, votes, centers