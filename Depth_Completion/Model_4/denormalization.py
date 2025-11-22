import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration ---
# Use the same dimensions and path rules as your evaluation script
INPUT_W, INPUT_H = 640, 360 
MAX_DISPLAY_DEPTH_M = 3.0 # Set a max depth for visualization (e.g., 10 meters)

def load_depth_map_for_visualization(path: str) -> np.ndarray:
    """
    Loads a 16-bit depth PNG, converts it to meters, and prepares it for display.
    """
    if not os.path.exists(path):
        print(f"Error: Depth file not found at {path}")
        return None
    
    # Use IMREAD_UNCHANGED to read the 16-bit data
    depth_mm = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    
    if depth_mm is None:
        print(f"Error: Failed to load image from {path}")
        return None
        
    # Resize and convert to float32 meters
    depth_m = depth_mm.astype(np.float32) / 1000.0
    depth_m = cv2.resize(depth_m, (INPUT_W, INPUT_H), interpolation=cv2.INTER_NEAREST)
    
    # Clip values for cleaner visualization (e.g., max 10m)
    depth_m = np.clip(depth_m, 0, MAX_DISPLAY_DEPTH_M)
    
    return depth_m

def visualize_depth_map(depth_data: np.ndarray, title: str):
    """
    Displays the depth data using Matplotlib's 'jet' colormap for clarity.
    """
    if depth_data is None:
        return
        
    # Pixels with 0 depth (no reading) should ideally be masked or set to a specific color.
    # We will use a mask here.
    mask = depth_data > 0
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use 'viridis' or 'jet' colormap. We normalize based on the MAX_DISPLAY_DEPTH_M.
    # The 'None' values (masked) will appear white/gray depending on the colormap setup.
    c = ax.imshow(depth_data, cmap='viridis', vmax=MAX_DISPLAY_DEPTH_M)
    ax.set_title(title, fontsize=14)
    ax.axis('off') # Hide axes for a clean image display
    
    # Add a color bar to show the depth scale (in meters)
    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label('Depth (Meters)', rotation=270, labelpad=15)
    
    plt.show()
    print(f"\nINFO: Displayed {title} with max depth {MAX_DISPLAY_DEPTH_M}m.")

# --- Example Usage ---
if __name__ == "__main__":
    # 1. Get the path for a GT Depth map (You need to update this line)
    # This path must point to one of your 16-bit GT depth PNG files.
    # Example structure from your test.txt line:
    # sparse_depth_path, normals_path, mask_path, rgb_path, gt_depth_path

    example_gt_path = 'output/completed_depth_raw.png'  # <-- UPDATE THIS PATH
    
    # Load and display the depth map
    depth_data = load_depth_map_for_visualization(example_gt_path)
    
    if depth_data is not None:
        visualize_depth_map(depth_data, "Correctly Visualized Ground Truth Depth Map")

    else:
        print("\nACTION REQUIRED: Please update the 'example_gt_path' variable in the script to a real GT depth file path from your dataset.")