import os
import time
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Import model from depth_completion_network
from depth_completion_network import DepthCompletionNet, load_and_preprocess_input_fast, extract_boundaries_from_rgb, INPUT_H, INPUT_W

class DepthCompletionDataset(Dataset):
    """CSV: sparse_path,normals_path,mask_path,rgb_path,gt_dense_path
    Extracts boundaries from RGB on-the-fly using Canny edge detection."""
    def __init__(self, samples_file):
        with open(samples_file, 'r') as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        self.items = [ln.split(',') for ln in lines]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        sparse_path, normals_path, mask_path, rgb_path, gt_path = self.items[idx]
        
        d = load_and_preprocess_input_fast(sparse_path, 'Sparse Depth')
        n = load_and_preprocess_input_fast(normals_path, 'Normals')
        m = load_and_preprocess_input_fast(mask_path, 'Mask')
        b = extract_boundaries_from_rgb(rgb_path) 
        gt = load_and_preprocess_input_fast(gt_path, 'Sparse Depth')
        
        inp = torch.cat([d, n, m, b], dim=0)
        return {'input': inp.float(), 'gt': gt.float()}
    

def masked_l1_loss(pred, gt):
    """Compute L1 loss only on valid pixels."""
    valid = (gt > 0).float()
    num_valid = valid.sum()
    if num_valid < 1:
        return F.l1_loss(pred, gt)
    loss = (torch.abs(pred - gt) * valid).sum() / (num_valid + 1e-6)
    return loss

def smoothness_loss(pred):
    """Compute total variation smoothness loss."""
    dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1]).mean()
    dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :]).mean()
    return dx + dy

def train_epoch(model, loader, optim, device, lambda_smooth=1e-3, scaler=None, use_amp=False):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    num_samples = 0
    
    for batch_idx, batch in enumerate(loader):
        inp = batch['input'].to(device, non_blocking=True)
        gt = batch['gt'].to(device, non_blocking=True)
        
        optim.zero_grad()
        
        try:
            if use_amp and scaler:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    pred = model(inp)
                    loss_data = masked_l1_loss(pred, gt)
                    loss_smooth = smoothness_loss(pred)
                    loss = loss_data + lambda_smooth * loss_smooth
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                pred = model(inp)
                loss_data = masked_l1_loss(pred, gt)
                loss_smooth = smoothness_loss(pred)
                loss = loss_data + lambda_smooth * loss_smooth
                loss.backward()
                optim.step()
            
            running_loss += loss.item() * inp.size(0)
            num_samples += inp.size(0)
            
            if (batch_idx + 1) % 5 == 0:
                print(f"  Batch {batch_idx + 1}/{len(loader)}: loss={loss.item():.6f}")
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
            else:
                raise
    
    return running_loss / max(1, num_samples)

def validate(model, loader, device):
    """Validate the model."""
    if loader is None:
        return None
    
    model.eval()
    tot_l1 = 0.0
    count = 0
    
    with torch.no_grad():
        for batch in loader:
            inp = batch['input'].to(device, non_blocking=True)
            gt = batch['gt'].to(device, non_blocking=True)
            pred = model(inp)
            l1 = masked_l1_loss(pred, gt)
            tot_l1 += l1.item() * inp.size(0)
            count += inp.size(0)
    
    return tot_l1 / max(1, count)

def main():
    parser = argparse.ArgumentParser(description="Train Depth Completion Network with Canny Boundaries")
    parser.add_argument("--train_list", default="train_list.txt", help="Training list (sparse,normals,mask,rgb,gt)")
    parser.add_argument("--val_list", default="val_list.txt", help="Validation list (same format)")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--out_dir", default="checkpoints", help="Output dir")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="Use AMP")
    parser.add_argument("--lambda_smooth", type=float, default=1e-3, help="Smoothness weight")
    parser.add_argument("--val_freq", type=int, default=1, help="Validate every N epochs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    # --- File paths for logging losses ---
    train_loss_file = os.path.join(args.out_dir, "train_losses.txt")
    val_loss_file = os.path.join(args.out_dir, "val_losses.txt")

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True

    print(f"\n{'='*60}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] AMP: {args.amp}")
    print(f"[INFO] Batch size: {args.batch}")
    print(f"[INFO] Validate every {args.val_freq} epochs")
    print(f"[INFO] Boundary extraction: Canny (on-the-fly)")
    print(f"{'='*60}\n")
    
    if not os.path.exists(args.train_list):
        print(f"[ERROR] Train list not found: {args.train_list}")
        return
    
    train_ds = DepthCompletionDataset(args.train_list)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=False)
    print(f"[INFO] Training samples: {len(train_ds)}")

    val_loader = None
    if args.val_list and os.path.exists(args.val_list):
        val_ds = DepthCompletionDataset(args.val_list)
        val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=False)
        print(f"[INFO] Validation samples: {len(val_ds)}")

    print("\n[INFO] Creating model...")
    model = DepthCompletionNet(input_channels=6).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model parameters: {total_params:,}\n")
    
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.5)
    scaler = torch.amp.GradScaler('cuda') if args.amp else None

    best_val = 1e9
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        print(f"[EP {ep}/{args.epochs}] Training...")
        train_loss = train_epoch(model, train_loader, optim, device, args.lambda_smooth, scaler, args.amp)
        scheduler.step()

        t1 = time.time()
        print(f"[EP {ep}/{args.epochs}] train_loss={train_loss:.6f} time={t1-t0:.1f}s\n")

        # Save training loss
        with open(train_loss_file, 'a') as f:
            f.write(f"{train_loss}\n")

        # Validate every N epochs
        if val_loader and ep % args.val_freq == 0:
            print(f"[EP {ep}/{args.epochs}] Validating...")
            val_loss = validate(model, val_loader, device)
            print(f"[EP {ep}/{args.epochs}] val_loss={val_loss:.6f}\n")

            # Save validation loss
            with open(val_loss_file, 'a') as f:
                f.write(f"{val_loss}\n")

            if val_loss < best_val:
                best_val = val_loss
                best_ckpt_path = os.path.join(args.out_dir, "best.pth")
                torch.save({'epoch': ep, 'model_state': model.state_dict()}, best_ckpt_path)
                print(f"[BEST] val_loss={best_val:.6f}\n")

        # Save checkpoint every epoch
        ckpt_path = os.path.join(args.out_dir, f"ckpt_ep{ep:03d}.pth")
        torch.save({'epoch': ep, 'model_state': model.state_dict()}, ckpt_path)
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("[INFO] Training complete!")
    print(f"[INFO] Checkpoints: {args.out_dir}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()