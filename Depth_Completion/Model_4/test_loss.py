import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import cv2 
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION ---
CHECKPOINT_PATH = "checkpoints/best.pth" 
TEST_INDEX_FILE = "../test_list.txt" # The file containing your data paths (comma-separated)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"INFO: Using device: {DEVICE}")

# Input parameters from your model
IN_CHANNELS = 6
BASE_CHANNELS = 64
INPUT_H, INPUT_W = 360, 640 
BATCH_SIZE = 16 

# --- 2. MODEL DEFINITION (UNMODIFIED) ---

class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class DepthCompletionNet(nn.Module):
    """
    Simple model: 1 encoder + 1 decoder (FAST)
    Includes Dropout regularization to combat overfitting.
    """
    def __init__(self, input_channels=6, base_channels=64, dropout_rate=0.3):
        super(DepthCompletionNet, self).__init__()
        
        # Encoder (compress)
        self.enc = nn.Sequential(
            ConvBlock(input_channels, base_channels, 3, 1, 1),
            ConvBlock(base_channels, 32, 3, 2, 1),  # /2
        )

        # REGULARIZATION: Dropout applied to the feature maps from the encoder
        self.encoder_dropout = nn.Dropout2d(dropout_rate)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(32, 16, 3, 1, 1),
            ConvBlock(16, 32, 3, 1, 1),
        )
        
        # Decoder (decompress)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(32, base_channels, 4, 2, 1),  # x2
            ConvBlock(base_channels, base_channels, 3, 1, 1),
            nn.Conv2d(base_channels, 1, 1),
            nn.ReLU()
        )

    def forward(self, x):
        enc = self.enc(x)
        enc = self.encoder_dropout(enc) # Apply dropout here
        bottleneck = self.bottleneck(enc)
        out = self.dec(bottleneck)
        return out

# --- 3. DATA LOADING AND PREPROCESSING HELPERS (UNMODIFIED) ---

def load_and_preprocess_input_fast(path: str, input_type: str) -> torch.Tensor:
    """Load image, resize, and preprocess (Sparse Depth, Normals, or Mask)."""
    if input_type == 'Sparse Depth' or input_type == 'GT Depth':
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        interpolation = cv2.INTER_NEAREST if input_type == 'GT Depth' else cv2.INTER_LINEAR
        img = cv2.resize(img, (INPUT_W, INPUT_H), interpolation=interpolation)
        img = img / 1000.0  # mm to meters
        img = np.clip(img, 0, 10)
        return torch.from_numpy(img).unsqueeze(0)
    
    elif input_type == 'Normals':
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
             raise FileNotFoundError(f"Normals image not found: {path}")
        img = cv2.resize(img, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = 2.0 * img - 1.0 # Normalize to [-1, 1]
        return torch.from_numpy(img.transpose(2, 0, 1))
    
    elif input_type == 'Mask':
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        if img is None:
             raise FileNotFoundError(f"Mask image not found: {path}")
        img = cv2.resize(img, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
        img = img / 255.0
        return torch.from_numpy(img).unsqueeze(0)
    
    else:
        raise ValueError(f"Unknown input type: {input_type}")

def extract_boundaries_from_rgb(rgb_path: str) -> torch.Tensor:
    """Extract boundaries from RGB using Canny edge detection."""
    img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"RGB image not found: {rgb_path}")
    
    img = cv2.resize(img, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    edges = edges.astype(np.float32) / 255.0
    return torch.from_numpy(edges).unsqueeze(0)

def load_input_and_target(paths: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Loads all 5 files, stacks the 4 inputs, and separates the target GT depth."""
    sparse_depth_path, normals_path, mask_path, rgb_path, gt_depth_path = paths
    
    # 4 Input Channels: D, N, M, B
    d = load_and_preprocess_input_fast(sparse_depth_path, 'Sparse Depth') # 1 channel
    n = load_and_preprocess_input_fast(normals_path, 'Normals')          # 3 channels
    m = load_and_preprocess_input_fast(mask_path, 'Mask')                # 1 channel
    b = extract_boundaries_from_rgb(rgb_path)                            # 1 channel
    
    # Stack inputs (1 + 3 + 1 + 1 = 6 channels)
    stacked_input = torch.cat([d, n, m, b], dim=0) # Shape: (6, H, W)

    # Ground Truth Target (Y)
    gt_depth = load_and_preprocess_input_fast(gt_depth_path, 'GT Depth') # Shape: (1, H, W)

    return stacked_input, gt_depth

def load_sample_paths_from_file(index_file: str) -> List[List[str]]:
    """Reads the index file and returns a list of path lists, using comma as delimiter."""
    print(f"INFO: Reading sample paths from {index_file}...")
    paths_list = []
    try:
        with open(index_file, 'r') as f:
            for line in f:
                # Use comma (,) as the delimiter for splitting paths
                paths = [p.strip() for p in line.strip().split(',')]
                paths = [p for p in paths if p] 

                if len(paths) == 5:
                    paths_list.append(paths)
                else:
                    print(f"WARNING: Skipping line with incorrect number of paths ({len(paths)}) or invalid format: {line.strip()}")
        
        if not paths_list:
            raise ValueError(f"Index file '{index_file}' is empty or improperly formatted.")
        return paths_list
    except FileNotFoundError:
        print(f"FATAL: The file '{index_file}' was not found. Please ensure it exists.")
        exit()

# --- 4. DATASET DEFINITION (UNMODIFIED) ---
class PathTestDataset(Dataset):
    """Dataset that loads data on the fly based on file paths."""
    def __init__(self, path_list: List[List[str]]):
        self.path_list = path_list
        
    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        # paths is a list of 5 strings: [sparse_depth, normals, mask, rgb, gt_depth]
        paths = self.path_list[idx]
        
        try:
            X, Y = load_input_and_target(paths)
            return X, Y
        except Exception as e:
            print(f"ERROR: Failed to load sample at index {idx} with paths: {paths}. Error: {e}")
            return torch.zeros(IN_CHANNELS, INPUT_H, INPUT_W), torch.zeros(1, INPUT_H, INPUT_W)

# --- 5. UTILITY FUNCTIONS ---

def calculate_threshold_metrics(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float, float, int]:
    """
    Calculates the 1.25^t threshold accuracy metrics on valid ground truth pixels.
    
    Returns: (acc1, acc2, acc3, N_valid)
        acc1/2/3: Number of pixels satisfying the threshold.
        N_valid: Total number of valid pixels in the batch.
    """
    # Create mask for valid ground truth pixels (D_GT > 0)
    mask = target > 0
    
    # Apply mask to predictions and targets
    target_valid = target[mask]
    pred_valid = pred[mask]
    
    if target_valid.numel() == 0:
        return 0.0, 0.0, 0.0, 0 # Return zero if no valid pixels

    # Calculate the ratio: max(D_hat / D_GT, D_GT / D_hat)
    # The minimum value this can take is 1.0 (perfect match).
    ratio = torch.max(pred_valid / target_valid, target_valid / pred_valid)
    
    # 1.25^1 accuracy (delta < 1.25)
    acc1_count = torch.sum(ratio < 1.25).item()
    
    # 1.25^2 accuracy (delta < 1.5625)
    acc2_count = torch.sum(ratio < 1.25**2).item()
    
    # 1.25^3 accuracy (delta < 1.953125)
    acc3_count = torch.sum(ratio < 1.25**3).item()
    
    N_valid = target_valid.numel()
    
    return acc1_count, acc2_count, acc3_count, N_valid

def plot_batch_loss(batch_losses: List[float], total_avg_loss: float):
    """Plots the loss for each batch evaluated."""
    plt.figure(figsize=(12, 6))
    plt.plot(batch_losses, marker='.', linestyle='-', color='#3b82f6', label='Batch Mean Loss')
    
    # Add the overall average loss as a horizontal line
    plt.axhline(total_avg_loss, color='red', linestyle='--', label=f'Overall Average Loss ({total_avg_loss:.4f})')
    
    plt.title('Test Set Loss per Batch (Mean MSE)', fontsize=16)
    plt.xlabel(f'Batch Index (Batch Size = {BATCH_SIZE})', fontsize=12)
    plt.ylabel('Loss (Mean MSE)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()
    print("\nINFO: Test Loss Plot displayed.")


# --- 6. MAIN EVALUATION FUNCTION (MODIFIED) ---

MetricResults = Dict[str, float]

@torch.no_grad()
def calculate_test_metrics(model: nn.Module, test_loader: DataLoader, criterion: nn.Module) -> Tuple[MetricResults, List[float]]:
    """
    Runs the evaluation loop, calculating MSE loss and Threshold Accuracy metrics.
    """
    model.eval() 
    total_sum_squared_error = 0.0 # Tracks the *sum* of errors, not the mean
    total_valid_pixels = 0
    total_acc1, total_acc2, total_acc3 = 0, 0, 0
    batch_losses_mean = [] # Stores the MEAN MSE for each batch
    
    print("\nINFO: Starting comprehensive evaluation loop...")

    for i, (inputs, targets) in enumerate(test_loader):
        print(f"  > Processing Batch {i+1}/{len(test_loader)}...", end='\r')
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)

        # Forward pass (prediction)
        outputs = model(inputs)
        
        # --- 1. Calculate Loss (MSE) ---
        mask = targets > 0
        
        # Loss contains individual squared errors for each valid pixel
        pixel_losses = criterion(outputs[mask], targets[mask]) 
        
        # Get the number of valid pixels in this batch
        N_valid_batch = mask.sum().item()
        
        if N_valid_batch == 0:
            continue
            
        # Calculate the Mean MSE for the current batch
        batch_mean_loss = pixel_losses.mean().item()
        batch_losses_mean.append(batch_mean_loss)

        # Accumulate the *sum* of all squared errors for the overall average calculation
        total_sum_squared_error += pixel_losses.sum().item() 
        total_valid_pixels += N_valid_batch
        
        # --- 2. Calculate Threshold Metrics (Accuracy/Precision) ---
        acc1, acc2, acc3, _ = calculate_threshold_metrics(outputs, targets)
        
        total_acc1 += acc1
        total_acc2 += acc2
        total_acc3 += acc3

    # Calculate final averages/percentages
    
    # The final average MSE is the total sum of errors divided by the total number of pixels
    final_avg_mse = total_sum_squared_error / total_valid_pixels if total_valid_pixels > 0 else 0.0

    # Calculate percentages for threshold metrics
    if total_valid_pixels > 0:
        final_acc1 = (total_acc1 / total_valid_pixels) * 100.0
        final_acc2 = (total_acc2 / total_valid_pixels) * 100.0
        final_acc3 = (total_acc3 / total_valid_pixels) * 100.0
    else:
        final_acc1, final_acc2, final_acc3 = 0.0, 0.0, 0.0
        
    results: MetricResults = {
        'avg_mse': final_avg_mse,
        'acc_delta_1.25': final_acc1,
        'acc_delta_1.25^2': final_acc2,
        'acc_delta_1.25^3': final_acc3,
    }
    
    return results, batch_losses_mean

if __name__ == "__main__":
    
    # --- A. Setup and Data Loading ---
    path_list = load_sample_paths_from_file(TEST_INDEX_FILE)
    test_dataset = PathTestDataset(path_list)
    # NOTE: Set shuffle=False for reproducible metric evaluation
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # --- B. Initialize Model and Load Checkpoint ---
    model = DepthCompletionNet(input_channels=IN_CHANNELS, base_channels=BASE_CHANNELS).to(DEVICE)
        
    print(f"INFO: Loading model checkpoint from {CHECKPOINT_PATH}...")
    
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state']) 
        print("SUCCESS: Model weights loaded successfully!")
    except KeyError:
        print("\nERROR: Found the file, but could not find the 'model_state' key inside it. Please check the checkpoint file contents.")
        exit()
    except RuntimeError as e:
        print(f"\nFATAL RUNTIME ERROR: Mismatch occurred during weight loading: {e}")
        print("ACTION: Double-check the model architecture (DepthCompletionNet) and its parameters (e.g., `base_channels`).")
        exit()
        
    # Define the loss function (MSE is used for the loss tracking)
    criterion = nn.MSELoss(reduction='none') # Use reduction='none' for manual masking

    # --- C. Calculate Metrics ---
    metrics, batch_losses = calculate_test_metrics(model, test_dataloader, criterion)

    # --- D. Final Output and Plotting ---
    print("\n---------------------------------------------------------")
    print(f"Total samples evaluated: {len(test_dataset)}")
    print("--- DEPTH COMPLETION METRICS (Calculated on Valid Pixels) ---")
    
    # Print Loss (MSE)
    print(f"Average Test Loss (MSE): {metrics['avg_mse']:.6f}")
    print(f"Average Test Loss (RMSE): {np.sqrt(metrics['avg_mse']):.6f} (Root Mean Squared Error)")
    
    # Print Threshold Metrics
    print("\nThreshold Accuracy (Often referred to as Accuracy/Precision in papers):")
    print(f"  Delta < 1.25   (Acc1): {metrics['acc_delta_1.25']:.2f}%")
    print(f"  Delta < 1.25^2 (Acc2): {metrics['acc_delta_1.25^2']:.2f}%")
    print(f"  Delta < 1.25^3 (Acc3): {metrics['acc_delta_1.25^3']:.2f}%")
    print("---------------------------------------------------------")
    
    # Plot the results
    plot_batch_loss(batch_losses, metrics['avg_mse'])