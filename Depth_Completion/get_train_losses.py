import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import tqdm

# Import necessary components from your existing scripts
from train_depth_completion import DepthCompletionDataset, masked_l1_loss, smoothness_loss
from depth_completion_network import DepthCompletionNet

def get_losses(model, loader, device, lambda_smooth=1e-3):
    """
    Iterate through a dataset and calculate the loss for each batch without training.
    """
    model.eval()  # Set the model to evaluation mode
    batch_losses = []
    
    print("Calculating losses for each batch...")
    pbar = tqdm.tqdm(loader, desc="Evaluating Batches")

    with torch.no_grad():  # Disable gradient calculations
        for batch in pbar:
            inp = batch['input'].to(device, non_blocking=True)
            gt = batch['gt'].to(device, non_blocking=True)
            
            # Forward pass
            pred = model(inp)
            
            # Calculate loss (same as in training)
            loss_data = masked_l1_loss(pred, gt)
            loss_smooth = smoothness_loss(pred)
            loss = loss_data + lambda_smooth * loss_smooth
            
            batch_losses.append(loss.item())
            
            pbar.set_postfix({"Last Batch Loss": f"{loss.item():.6f}"})

    return batch_losses

def save_losses_to_txt(losses, filename):
    """Saves a list of losses to a text file, one per line."""
    try:
        with open(filename, 'w') as f:
            for loss in losses:
                f.write(f"{loss:.6f}\n")
        print(f"\nSuccessfully saved {len(losses)} batch losses to {filename}")
    except Exception as e:
        print(f"Error saving losses to {filename}. Reason: {e}")

def main():
    parser = argparse.ArgumentParser(description="Calculate and save training losses using a trained Depth Completion model.")
    parser.add_argument("--checkpoint", default="checkpoints_3/best.pth", help="Path to the trained model checkpoint.")
    parser.add_argument("--train_list", default="train_list.txt", help="Path to the list of training samples.")
    parser.add_argument("--output_file", default="train_batch_losses.txt", help="File to save the per-batch training losses.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size to use for evaluation (should match training for consistency).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on.")
    parser.add_argument("--lambda_smooth", type=float, default=1e-3, help="Smoothness weight, must match the value used during training.")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"\n{'='*60}")
    print(f"Using device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")
    print(f"Using training data from: {args.train_list}")
    print(f"{'='*60}\n")

    # 1. Load Model
    if not os.path.exists(args.checkpoint):
        print(f"[ERROR] Checkpoint file not found: {args.checkpoint}")
        return

    model = DepthCompletionNet().to(device)
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        # Handle different checkpoint formats
        state_dict = None
        if 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load model weights: {e}")
        print("Please ensure the model architecture in the script matches the checkpoint.")
        return

    # 2. Load Dataset
    if not os.path.exists(args.train_list):
        print(f"[ERROR] Training list file not found: {args.train_list}")
        return

    train_dataset = DepthCompletionDataset(args.train_list)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,  # No need to shuffle for evaluation
        num_workers=0, 
        pin_memory=True
    )
    print(f"Found {len(train_dataset)} samples in the training set.")

    # 3. Calculate Losses
    batch_losses = get_losses(model, train_loader, device, args.lambda_smooth)

    # 4. Save Losses
    if batch_losses:
        save_losses_to_txt(batch_losses, args.output_file)
        avg_loss = np.mean(batch_losses)
        print(f"Average training loss across all batches: {avg_loss:.6f}")
    else:
        print("No losses were calculated. Please check your data loader and files.")

    print(f"\n{'='*60}")
    print("Evaluation complete.")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()