import numpy as np

def read_intrinsics(filepath):
    """
    Reads camera intrinsic parameters from a NumPy binary file (.npy).

    Args:
        filepath (str): The path to the .npy intrinsics file,
                        expected to contain a 3x3 intrinsic matrix.

    Returns:
        tuple: A tuple containing:
            - fx (float): Focal length in x-direction.
            - fy (float): Focal length in y-direction.
            - cx (float): Principal point x-coordinate.
            - cy (float): Principal point y-coordinate.
    """
    intrinsics_matrix = np.load(filepath)
    
    # Ensure it's a 3x3 matrix
    if intrinsics_matrix.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 intrinsic matrix, but got shape {intrinsics_matrix.shape}")

    fx = intrinsics_matrix[0, 0]
    fy = intrinsics_matrix[1, 1]
    cx = intrinsics_matrix[0, 2]
    cy = intrinsics_matrix[1, 2]
    return fx, fy, cx, cy

def scale_intrinsics(fx, fy, cx, cy, original_size, new_size):
    """
    Scales camera intrinsics to a new image size.

    Args:
        fx, fy, cx, cy (float): Original intrinsic parameters.
        original_size (tuple): The original (width, height).
        new_size (tuple): The new (width, height).

    Returns:
        tuple: The new (fx, fy, cx, cy).
    """
    orig_w, orig_h = original_size
    new_w, new_h = new_size
    
    scale_w = new_w / orig_w
    scale_h = new_h / orig_h
    
    return fx * scale_w, fy * scale_h, cx * scale_w, cy * scale_h

if __name__ == "__main__":
    intrinsics_filepath = r"C:\\Users\\user\\Desktop\\AUB\\Intro2ML\\Project\\TransCG\\camera_intrinsics\\1-camIntrinsics-D435.npy"
    fx, fy, cx, cy = read_intrinsics(intrinsics_filepath)

    print("Original Camera Intrinsics (for 1280x720):")
    print(f"  fx: {fx:.2f}, fy: {fy:.2f}, cx: {cx:.2f}, cy: {cy:.2f}")

    new_fx, new_fy, new_cx, new_cy = scale_intrinsics(fx, fy, cx, cy, original_size=(1280, 720), new_size=(640, 360))

    print("\nScaled Camera Intrinsics (for 640x360):")
    print(f"  fx: {new_fx:.2f}, fy: {new_fy:.2f}, cx: {new_cx:.2f}, cy: {new_cy:.2f}")
