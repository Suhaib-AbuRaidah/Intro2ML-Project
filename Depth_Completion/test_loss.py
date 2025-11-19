import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os

# Define the expected input size constants from your model file
INPUT_H, INPUT_W = 360, 640
NUM_INPUT_CHANNELS = 6 # Sparse Depth (1), Normals (3), Mask (1), Boundaries (1)

# --- 1. MODEL DEFINITION (DepthCompletionNet) ---
# Your model needs to be defined here so the script can instantiate and load weights.

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
    """Depth Completion model for 6-channel input -> 1-channel depth output"""
    def __init__(self, input_channels=NUM_INPUT_CHANNELS, base_channels=32):
        super(DepthCompletionNet, self).__init__()
        
        # Encoder (compress)
        self.enc = nn.Sequential(
            ConvBlock(input_channels, base_channels, 3, 1, 1),
            ConvBlock(base_channels, base_channels * 2, 3, 2, 1),  # /2
            ConvBlock(base_channels * 2, base_channels * 4, 3, 2, 1),  # /4
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(base_channels * 4, base_channels * 8, 3, 1, 1),
            ConvBlock(base_channels * 8, base_channels * 4, 3, 1, 1),
        )
        
        # Decoder (decompress)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1),  # x2
            ConvBlock(base_channels * 2, base_channels * 2, 3, 1, 1),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1),  # x2
            ConvBlock(base_channels, base_channels, 3, 1, 1),
            nn.Conv2d(base_channels, 1, 1),
            nn.ReLU() # Output depth must be non-negative
        )

    def forward(self, x):
        enc = self.enc(x)
        bottleneck = self.bottleneck(enc)
        out = self.dec(bottleneck)
        return out

# --- 2. PLACEHOLDER DATASET AND LOADER ---

class MockDepthDataset(Dataset):
    """
    Generates mock data matching the expected format for your Depth Completion model:
    6-channel input and 1-channel Ground Truth Depth.
    """
    def __init__(self, num_samples=100):
        self.num_samples = num_samples
        # Simulated 6-channel input: (N, 6, H, W)
        self.inputs = torch.randn(num_samples, NUM_INPUT_CHANNELS, INPUT_H, INPUT_W, dtype=torch.float32)
        # Simulated Ground Truth Full Depth: (N, 1, H, W). Values between 0.1 and 10 meters.
        self.gt_depth = torch.rand(num_samples, 1, INPUT_H, INPUT_W, dtype=torch.float32) * 9.9 + 0.1
        # Simulated Mask: Pixels where GT depth is valid (e.g., > 0).
        self.mask = (self.gt_depth > 0.05).float() # Simple non-zero mask

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Returns: Stacked Input, Ground Truth Depth, Validity Mask
        return self.inputs[idx], self.gt_depth[idx], self.mask[idx]

# --- 3. REGRESSION METRICS FOR DEPTH COMPLETION ---

def compute_depth_errors(y_true, y_pred, mask):
    """
    Calculates standard depth completion metrics over valid pixels (mask=1).

    Args:
        y_true (np.array): Flattened Ground truth depth map (valid pixels).
        y_pred (np.array): Flattened Predicted depth map (valid pixels).
        mask (np.array): Flattened Validity mask (Boolean array for valid pixels).
    
    Returns:
        dict: A dictionary containing 'rmse', 'mae', 'delta1', 'delta2', 'delta3'.
    """
    # Filter arrays to include only valid pixels defined by the mask
    y_true_valid = y_true[mask]
    y_pred_valid = y_pred[mask]
    
    if y_true_valid.size == 0:
        print("Warning: No valid pixels found for evaluation. Returning NaN metrics.")
        return {'rmse': np.nan, 'mae': np.nan, 'delta1': np.nan, 'delta2': np.nan, 'delta3': np.nan}

    # 1. Error calculation: Difference and Squared Difference
    diff = np.abs(y_true_valid - y_pred_valid)
    
    # 2. MAE (Mean Absolute Error)
    mae = np.mean(diff)
    
    # 3. RMSE (Root Mean Squared Error)
    rmse = np.sqrt(np.mean(diff ** 2))
    
    # 4. Accuracy Thresholds (Delta N)
    # The percentage of pixels where max(pred/true, true/pred) < threshold
    # Note: Use a small epsilon to avoid division by zero if mask is imperfect
    epsilon = 1e-6
    ratio = np.maximum(
        y_pred_valid / (y_true_valid + epsilon), 
        y_true_valid / (y_pred_valid + epsilon)
    )
    
    # Thresholds: 1.25^1, 1.25^2, 1.25^3
    delta1 = np.mean(ratio < 1.25) * 100
    delta2 = np.mean(ratio < 1.25**2) * 100
    delta3 = np.mean(ratio < 1.25**3) * 100
    
    return {
        'rmse': rmse, 
        'mae': mae, 
        'delta1': delta1, 
        'delta2': delta2, 
        'delta3': delta3
    }

# --- 4. MAIN EVALUATION FUNCTION ---
def evaluate_test_set_regression(model_path, test_loader, device):
    """
    Loads the Depth Completion model, performs inference, and calculates
    standard regression metrics (RMSE, MAE, Delta Accuracies).
    """
    
    # Initialize the model and load weights
    model = DepthCompletionNet(input_channels=NUM_INPUT_CHANNELS, base_channels=32).to(device)
    
    # Load weights
    try:
        # Attempt to load the state dict from the standard 'model_state' key
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
        else:
            # Assume the file only contains the state_dict directly
            model.load_state_dict(checkpoint)
        print(f"Successfully loaded model weights from {model_path}.")
    except Exception as e:
        print(f"ERROR: Could not load model weights from {model_path}. Please check file path and content.")
        print(f"Details: {e}")
        return

    model.eval()
    
    all_preds = []
    all_labels = []
    all_masks = []

    print(f"Starting evaluation on device: {device}. Evaluating {len(test_loader.dataset)} samples...")
    
    with torch.no_grad():
        for inputs, gt_depth, mask in test_loader:
            inputs = inputs.to(device)
            gt_depth = gt_depth.to(device)
            mask = mask.to(device)
            
            # Forward pass: Output is the completed depth map
            predicted_depth = model(inputs)
            
            # Flatten and store results for metric calculation
            all_preds.extend(predicted_depth.cpu().numpy().flatten())
            all_labels.extend(gt_depth.cpu().numpy().flatten())
            all_masks.extend(mask.cpu().numpy().flatten())

    # Convert lists to NumPy arrays
    y_true_flat = np.array(all_labels)
    y_pred_flat = np.array(all_preds)
    mask_flat = np.array(all_masks).astype(bool) # Convert mask to boolean for indexing

    # --- METRICS CALCULATION ---
    metrics = compute_depth_errors(y_true_flat, y_pred_flat, mask_flat)
    
    print("\n" + "="*70)
    print("      Depth Completion Regression Evaluation Results")
    print("="*70)
    print(f"Total Images Evaluated: {len(test_loader.dataset)}")
    print(f"Total Valid Pixels Evaluated: {np.sum(mask_flat):,}")
    print("-" * 70)
    print(f"1. RMSE (Root Mean Squared Error): {metrics['rmse']:.4f} m")
    print(f"2. MAE (Mean Absolute Error): {metrics['mae']:.4f} m")
    print("-" * 70)
    print("3. Threshold Accuracy ($\delta$): Percentage of pixels where max(pred/true, true/pred) < threshold")
    print(f"   $\delta < 1.25^1$ (Delta 1): {metrics['delta1']:.2f}%")
    print(f"   $\delta < 1.25^2$ (Delta 2): {metrics['delta2']:.2f}%")
    print(f"   $\delta < 1.25^3$ (Delta 3): {metrics['delta3']:.2f}%")
    print("="*70)


if __name__ == "__main__":
    # --- Configuration ---
    MODEL_PATH = 'best.pth' 
    # Adjust BATCH_SIZE based on your GPU memory
    BATCH_SIZE = 4 
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # --- Data Setup (Replace with your actual data loading) ---
    # NOTE: You MUST replace MockDepthDataset and DataLoader with your real dataset and loader
    test_dataset = MockDepthDataset(num_samples=200)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    try:
        # Simple check to allow the script to run even without a real model file
        if not os.path.exists(MODEL_PATH):
            dummy_model = DepthCompletionNet(input_channels=NUM_INPUT_CHANNELS)
            torch.save({'model_state': dummy_model.state_dict()}, MODEL_PATH)
            print(f"NOTE: Created a dummy model file '{MODEL_PATH}'. REPLACE THIS with your actual checkpoint!")
            
        evaluate_test_set_regression(MODEL_PATH, test_loader, DEVICE)
        
    except FileNotFoundError:
        print(f"\nERROR: Model file '{MODEL_PATH}' not found. Please ensure it exists.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during evaluation: {e}")
        
# To run this script, you need: pip install torch numpy