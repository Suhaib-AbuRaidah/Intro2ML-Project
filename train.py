import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from datasets import TransCG_D435_Dataset
import sys
sys.path.append('/home/suhaib/6DPOSE')
import numpy as np
from model import PoseNet
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import datetime
import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/suhaib/6DPOSE/data/transcg-data-1/transcg")
    parser.add_argument("--use_depth", default=True, action="store_true")
    parser.add_argument("--num_epochs", default=50, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = args.data
    dataset = TransCG_D435_Dataset(root)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)


    start_training_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    checkpoint_path = f"./pre_trained_models/{start_training_time}"
    os.makedirs(checkpoint_path, exist_ok=True)

    writer = SummaryWriter(f'runs/{start_training_time}')
    model = PoseNet(pretrained=False).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    best_loss_train = float('inf')
    best_loss_val = float('inf')

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        for i, batch in enumerate(train_loader):
            rgb = batch["rgb"].to(device).float()
            if args.use_depth and batch["depth"] is not None:
                depth = batch["depth"].to(device).float()
            else:
                depth = None

            quat_gt = batch["quat"].to(device).float()
            trans_gt = batch["trans"].to(device).float()
            if epoch ==0 and i==0:
                writer.add_graph(model, (rgb,depth))
                traced_model = torch.jit.trace(model, (rgb,depth))
                traced_model.save("pose_net.pt")
            
            quat_pred, trans_pred = model(rgb, depth)
            loss = loss_fn(quat_pred, quat_gt) + loss_fn(trans_pred, trans_gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                rgb = batch["rgb"].to(device).float()
                depth = batch["depth"].to(device).float() if batch["depth"] is not None else None
                quat_gt = batch["quat"].to(device).float()
                trans_gt = batch["trans"].to(device).float()

                quat_pred, trans_pred = model(batch["rgb"].to(device).float(), batch["depth"].to(device).float())

                val_loss = loss_fn(quat_pred, quat_gt) + loss_fn(trans_pred, trans_gt)
                val_loss_total += val_loss.item()

        avg_val_loss = val_loss_total / len(val_loader)

        writer.add_scalar('Val/Loss', avg_val_loss, epoch)

        print(f"Epoch {epoch}: train loss= {avg_train_loss:.4f}, validation loss= {avg_val_loss:.4f}\n")

        if avg_train_loss < best_loss_train:
            best_loss_train = avg_train_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_path, f"chkpt_best_model_train.pth"))
        
        if avg_val_loss < best_loss_val:
            best_loss_val = avg_val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_path, f"chkpt_best_model_val.pth"))

        # if (epoch + 1) % 5 == 0:
        #     torch.save(model.state_dict(), os.path.join(checkpoint_path, f"chkpt_{epoch}.pth"))

if __name__ == "__main__":
    main()
