import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import argparse

# Import constants from your other script for consistency
from depth_completion_network import INPUT_H, INPUT_W
from torchvision.transforms import functional as F

# --- 1. DATASET CLASS ---

class TransparentObjectDataset(Dataset):
    """
    Dataset for loading RGB images and their corresponding ground truth masks for instance segmentation.
    The CSV file should have the format: sparse,normals,mask,rgb,gt
    """
    def __init__(self, samples_file, transforms=None):
        with open(samples_file, 'r') as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        self.items = [ln.split(',') for ln in lines]
        self.transforms = transforms

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item_paths = self.items[idx]
        if len(item_paths) != 5:
            raise ValueError(f"Expected 5 paths per line in samples file, but got {len(item_paths)} for line {idx}")
        
        _, _, mask_path, rgb_path, _ = item_paths

        # Load RGB image
        img = cv2.imread(rgb_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_h, original_w, _ = img_rgb.shape

        # --- FIX: Resize image to a consistent size ---
        img_rgb = cv2.resize(img_rgb, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)

        # Load grayscale mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found at {mask_path}")

        # Find contours to identify each separate object instance
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        num_objs = len(contours)
        boxes = []
        masks = []

        for i in range(num_objs):
            # Bounding box
            x_orig, y_orig, w_orig, h_orig = cv2.boundingRect(contours[i])

            # --- FIX: Scale bounding boxes to match the resized image ---
            x_new = int(x_orig * (INPUT_W / original_w))
            y_new = int(y_orig * (INPUT_H / original_h))
            w_new = int(w_orig * (INPUT_W / original_w))
            h_new = int(h_orig * (INPUT_H / original_h))
            boxes.append([x_new, y_new, x_new + w_new, y_new + h_new])

            # Instance mask
            instance_mask = np.zeros_like(mask)
            cv2.drawContours(instance_mask, contours, i, (255), thickness=cv2.FILLED)
            masks.append(instance_mask)

        if num_objs == 0:
            # If no objects, return empty tensors in the expected format
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64),
                "masks": torch.zeros((0, INPUT_H, INPUT_W), dtype=torch.uint8),
                "image_id": torch.tensor([idx])
            }
        else:
            # Convert to tensors
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # All objects are "transparent object", so label is 1
            labels = torch.ones((num_objs,), dtype=torch.int64)

            # --- FIX: Resize masks to the consistent size ---
            resized_masks = [cv2.resize(m, (INPUT_W, INPUT_H), interpolation=cv2.INTER_NEAREST) for m in masks]
            # Convert to a single tensor
            masks = torch.as_tensor(np.array(resized_masks), dtype=torch.uint8)

            target = {
                "boxes": boxes,
                "labels": labels,
                "masks": masks,
                "image_id": torch.tensor([idx])
            }

        # Apply transformations
        img_tensor = F.to_tensor(img_rgb)

        return img_tensor, target

# --- 2. MODEL DEFINITION ---

def get_model_instance_segmentation(num_classes):
    """
    Loads a pre-trained Mask R-CNN model and modifies the heads for our custom number of classes.
    """
    # Load a model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT)

    # --- Replace the box predictor ---
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)

    # --- Replace the mask predictor ---
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model

# --- 3. UTILITY & TRAINING LOOP ---

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    parser = argparse.ArgumentParser(description="Train Mask R-CNN for transparent object segmentation.")
    parser.add_argument("--train_list", required=True, help="Training list CSV from load_data_to_txt.py (sparse,normals,mask,rgb,gt)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (reduce if OOM error)")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--out_dir", default="mask_rcnn_checkpoints", help="Directory to save checkpoints")
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    # Our dataset has one class (transparent object) + background
    num_classes = 2

    # --- DATASET & DATALOADER ---
    dataset = TransparentObjectDataset(samples_file=args.train_list)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=0, # Set to 0 for stability on Windows
        collate_fn=collate_fn
    )

    # --- MODEL, OPTIMIZER, SCHEDULER ---
    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # --- TRAINING LOOP ---
    print("\nStarting training...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        for i, (images, targets) in enumerate(data_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # The model returns a dict of losses when in training mode
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            epoch_loss += loss_value

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print(f"  Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(data_loader)}], Loss: {loss_value:.4f}")

        # Update the learning rate
        lr_scheduler.step()

        avg_epoch_loss = epoch_loss / len(data_loader)
        print(f"\n--- Epoch {epoch+1} Summary ---")
        print(f"Average Loss: {avg_epoch_loss:.4f}")
        
        # Save checkpoint
        ckpt_path = os.path.join(args.out_dir, f"mask_rcnn_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}\n")

    print("Training complete!")
    print(f"Final model saved in: {args.out_dir}")

if __name__ == "__main__":
    main()

"""
HOW TO RUN:

1. Ensure you have a 'train_list.txt' file generated by `load_data_to_txt.py`.
   Each line should contain 5 paths, separated by commas.

   Example line:
   .../depth1.png,.../depth1-gt-sn.png,.../depth1-gt-mask.png,.../rgb1.png,.../depth1-gt.png

2. Run the script from your terminal:
   python train_mask_rcnn_segmentation.py --train_list train_list.txt --batch 2 --epochs 10

   - Adjust --batch size based on your GPU memory. Start with a small number like 1 or 2.
   - The trained models (.pth files) will be saved in the 'mask_rcnn_checkpoints' directory.
"""