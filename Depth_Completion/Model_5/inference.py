import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os
import time

from depth_completion_network import ConvBlock, DepthCompletionNet, load_and_preprocess_input_fast, extract_boundaries_from_rgb, INPUT_H, INPUT_W

# --- Configuration (Must match training config) ---
NUM_INPUT_CHANNELS = 6 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Preprocessing Functions (Must match training setup) ---

def load_and_preprocess_input_fast(path, input_type):
    """Load, resize, and normalize a single input channel/image."""
    if input_type == 'Sparse Depth':
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        # Ensure correct resizing interpolation for depth
        img = cv2.resize(img, (INPUT_W, INPUT_H), interpolation=cv2.INTER_NEAREST) 
        img = img / 1000.0  # mm to meters (Standard conversion for many datasets)
        img = np.clip(img, 0, 10) # Clip to maximum meaningful depth (e.g., 10m)
        return torch.from_numpy(img).unsqueeze(0)
    
    elif input_type == 'Normals':
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        # Ensure correct resizing interpolation for color/normals
        img = cv2.resize(img, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # Normalize to [-1, 1]
        img = 2.0 * img - 1.0
        return torch.from_numpy(img.transpose(2, 0, 1)) # (3, H, W)
    
    elif input_type == 'Mask':
        # Mask is often derived from sparse depth (non-zero regions) but loaded here
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        img = cv2.resize(img, (INPUT_W, INPUT_H), interpolation=cv2.INTER_NEAREST)
        img = img / 255.0
        return torch.from_numpy(img).unsqueeze(0) # (1, H, W)
    
    elif input_type == 'RGB':
        # Load and resize RGB image for edge detection purposes
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"RGB image not found: {path}")
        img = cv2.resize(img, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
        return img
    
    else:
        raise ValueError(f"Unknown input type: {input_type}")

def extract_boundaries_from_rgb(rgb_path):
    """Extract boundaries (Canny edges) from RGB image."""
    # Note: We load the RGB here just to get the edges, but don't include it 
    # in the 6-channel stack. If you decide to include RGB, you'll need 9 channels.
    img = load_and_preprocess_input_fast(rgb_path, 'RGB')
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Canny Edge Detection
    edges = cv2.Canny(gray, 50, 150)
    edges = edges.astype(np.float32) / 255.0
    return torch.from_numpy(edges).unsqueeze(0) # (1, H, W)

def load_and_stack_inputs(depth_path, normals_path, mask_path, rgb_path):
    """Load all inputs, resize, extract boundaries, and stack into 6-channel tensor."""
    d = load_and_preprocess_input_fast(depth_path, 'Sparse Depth') # (1, H, W)
    n = load_and_preprocess_input_fast(normals_path, 'Normals')     # (3, H, W)
    m = load_and_preprocess_input_fast(mask_path, 'Mask')         # (1, H, W)
    b = extract_boundaries_from_rgb(rgb_path)                     # (1, H, W)
    
    # Stack: Sparse Depth (1) + Normals (3) + Mask (1) + Boundary (1) = 6 channels
    stacked = torch.cat([d, n, m, b], dim=0)
    return stacked.unsqueeze(0) # Add batch dimension: (1, 6, H, W)


# --- Core Inference Logic ---

def run_inference(checkpoint_path, depth_path, normals_path, mask_path, rgb_path, output_dir="output"):
    """
    Loads model, runs inference on a single scene, and saves the output.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Load Model
    print(f"Loading model on device: {DEVICE}")
    model = DepthCompletionNet().to(DEVICE)
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint file not found at {checkpoint_path}")
        return

    # Load the checkpoint file (which is a dictionary containing state_dict)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    # Fix for the state dict error: look for nested model state
    STATE_DICT_KEYS = ['state_dict', 'model_state', 'model']
    model_state = None
    
    if isinstance(checkpoint, dict):
        for key in STATE_DICT_KEYS:
            if key in checkpoint:
                model_state = checkpoint[key]
                print(f"INFO: Successfully extracted model state from checkpoint key: '{key}'")
                break
        
        if model_state is None:
            print("WARNING: Assuming the loaded dictionary is the model state_dict itself.")
            model_state = checkpoint
    else:
        model_state = checkpoint
        print("INFO: Loaded checkpoint directly as model state_dict.")

    # Load the extracted state dictionary
    try:
        model.load_state_dict(model_state)
    except Exception as e:
        print(f"FATAL ERROR during state_dict loading: {e}")
        print("This usually means your model architecture and the saved weights do not match.")
        return

    model.eval() # Set model to evaluation mode (important for BatchNorm and Dropout)

    # 2. Prepare Input
    print("Preparing input data...")
    try:
        input_tensor = load_and_stack_inputs(depth_path, normals_path, mask_path, rgb_path)
    except FileNotFoundError as e:
        print(f"Input file error: {e}")
        return
        
    input_tensor = input_tensor.to(DEVICE)

    # 3. Forward Pass
    print("Running forward pass...")
    start_time = time.time()
    with torch.no_grad():
        predicted_depth_tensor = model(input_tensor)
    end_time = time.time()
    
    # 4. Process Output
    # The output is (1, 1, H, W)
    predicted_depth_np = predicted_depth_tensor.squeeze().cpu().numpy()
    
    # --- Visualization and Raw Data Saving ---
    MAX_DEPTH = 10.0 # Match the clip in load_and_preprocess_input_fast
    
    # 1. Clip the predicted depth (in meters, float32)
    predicted_depth_np = np.clip(predicted_depth_np, 0, MAX_DEPTH)
    
    # 2. Save the RAW depth data (uint16 in millimeters)
    # Convert from meters (float) to millimeters (uint16)
    predicted_depth_mm = (predicted_depth_np * 1000.0).astype(np.uint16)
    raw_output_filename = os.path.join(output_dir, "completed_depth_raw.png")
    cv2.imwrite(raw_output_filename, predicted_depth_mm)
    
    # 5. Report Results
    print("-" * 30)
    print(f"Inference Time: {(end_time - start_time):.4f} seconds")
    print(f"RAW 16-bit depth (mm) saved to: {raw_output_filename}")
    print(f"Shape: {predicted_depth_mm.shape}")
    print("-" * 30)


if __name__ == '__main__':
    # --- Configuration for testing ---
    
    # 1. CHECKPOINT PATH: Replace this with the actual path to your trained model weights (.pth or .pt file)
    CHECKPOINT_PATH = "checkpoints/best.pth" 
    
    # 2. INPUT PATHS: Replace these with the actual paths to your test images
    # Example data (replace with your file names)
    SPARSE_DEPTH_PATH = "F:/ML-Dataset/transcg-data-9/transcg/scene81/9/depth1.png"
    NORMALS_PATH = "F:/ML-Dataset/transcg-data-9/transcg/scene81/9/depth1-gt-sn.png"
    MASK_PATH = "F:/ML-Dataset/transcg-data-9/transcg/scene81/9/depth1-gt-mask.png"
    RGB_PATH = "F:/ML-Dataset/transcg-data-9/transcg/scene81/9/rgb1.png"

    # Run the inference function
    run_inference(
        checkpoint_path=CHECKPOINT_PATH,
        depth_path=SPARSE_DEPTH_PATH,
        normals_path=NORMALS_PATH,
        mask_path=MASK_PATH,
        rgb_path=RGB_PATH
    )