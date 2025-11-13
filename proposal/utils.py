import torch
import numpy as np
from scipy.spatial.transform import Rotation as Rscipy
from torch.nn import functional as F

def quat_to_rotmat(q):
    # q: (B,4) in xyzw (scipy default is xyzw)
    q = q.detach().cpu().numpy()
    R = Rscipy.from_quat(q).as_matrix()
    return torch.from_numpy(R).float()

def pose_loss(pred_q, pred_t, gt_q, gt_t, w_rot=1.0, w_trans=1.0):
    # rotation error: geodesic or quaternion distance
    # compute angular difference via quaternion inner product
    # pred_q and gt_q are normalized
    dot = (pred_q * gt_q).sum(dim=1).abs().clamp(-1+1e-7,1-1e-7)
    ang = 2 * torch.acos(dot)  # radians
    rot_loss = ang.mean()
    trans_loss = F.l1_loss(pred_t, gt_t)
    return w_rot * rot_loss + w_trans * trans_loss, rot_loss.item(), trans_loss.item()

import cv2
import numpy as np

def draw_pose_on_rgb(rgb, poses, K, axis_length=0.05, colors=None):
    """
    Draw 3D coordinate axes for each object on the RGB image.

    Args:
        rgb: numpy array HxWx3 (RGB image, range 0-1 or 0-255)
        poses: list of 4x4 numpy arrays, one per object
        K: 3x3 intrinsic matrix (fx, fy, cx, cy)
        axis_length: length of axes in meters
        colors: list of BGR tuples per object. Default: [(0,0,255),(0,255,0),(255,0,0),(0,255,255)]
    Returns:
        img_out: RGB image with axes drawn (uint8)
    """
    img = (rgb*255).astype(np.uint8) if rgb.max() <= 1.0 else rgb.copy()
    if colors is None:
        colors = [(0,0,255),(0,255,0),(255,0,0),(0,255,255)]

    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    for i, T in enumerate(poses):
        R = T[:3,:3]
        t = T[:3,3]

        # Define axes in object frame
        axes = np.array([
            [axis_length,0,0],
            [0,axis_length,0],
            [0,0,axis_length]
        ])  # X,Y,Z

        # Transform to camera frame
        pts_cam = (R @ axes.T).T + t  # shape (3,3)

        # Project to 2D
        pts_2d = np.zeros((3,2), dtype=np.int32)
        for j in range(3):
            X, Y, Z = pts_cam[j]
            u = int(fx*X/Z + cx)
            v = int(fy*Y/Z + cy)
            pts_2d[j] = [u,v]

        # Origin
        u0 = int(fx*t[0]/t[2] + cx)
        v0 = int(fy*t[1]/t[2] + cy)

        color = colors[i % len(colors)]
        # Draw lines for axes
        cv2.line(img, (u0,v0), tuple(pts_2d[0]), color=(0,0,255), thickness=2)  # X: red
        cv2.line(img, (u0,v0), tuple(pts_2d[1]), color=(0,255,0), thickness=2)  # Y: green
        cv2.line(img, (u0,v0), tuple(pts_2d[2]), color=(255,0,0), thickness=2)  # Z: blue

    return img
import matplotlib.pyplot as plt
import numpy as np

def plot_pose_errors(rotation_errors, translation_errors):
    """
    Plot per-instance rotation and translation errors.

    Args:
        rotation_errors (list[float]): Rotation errors in degrees.
        translation_errors (list[float]): Translation errors in meters.
        save_path (str, optional): If given, saves the plot to this path.
    """
    idx = np.arange(len(rotation_errors))

    plt.figure(figsize=(16, 10))

    # Rotation error plot
    plt.subplot(1, 2, 1)
    plt.plot(idx, rotation_errors, 'r-', label='Rotation Error (deg)')
    plt.xlabel('Instance Index')
    plt.ylabel('Rotation Error (°)')
    plt.title('Per-instance Rotation Error')
    plt.legend()
    plt.grid(True)

    # Translation error plot
    plt.subplot(1, 2, 2)
    plt.plot(idx, translation_errors, 'b-', label='Translation Error (m)')
    plt.xlabel('Instance Index')
    plt.ylabel('Translation Error (m)')
    plt.title('Per-instance Translation Error')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()


    plt.savefig("error.svg", dpi=300)
    plt.show()


def plot_histogram(rot_errs, trans_errs, acc, rot_threshold):
    # === Visualization ===
    plt.figure(figsize=(16, 10))

    # Histogram of rotation error
    plt.subplot(1, 3, 1)
    plt.hist(rot_errs, bins=50, color='steelblue', alpha=0.8)
    plt.title('Rotation Error (°)')
    plt.xlabel('Error (°)')
    plt.ylabel('Count')
    plt.grid(True, linestyle='--', alpha=0.4)

    # Histogram of translation error
    plt.subplot(1, 3, 2)
    plt.hist(trans_errs * 100, bins=50, color='darkorange', alpha=0.8)
    plt.title('Translation Error (cm)')
    plt.xlabel('Error (cm)')
    plt.ylabel('Count')
    plt.grid(True, linestyle='--', alpha=0.4)

    # Cumulative accuracy curve
    plt.subplot(1, 3, 3)
    rot_threshs = np.linspace(0, 30, 100)
    acc_curve = [np.mean(rot_errs < t) * 100 for t in rot_threshs]
    plt.plot(rot_threshs, acc_curve, label='Rotation Accuracy', color='green')
    plt.axvline(rot_threshold, color='gray', linestyle='--', label='Threshold')
    plt.title('Cumulative Rotation Accuracy')
    plt.xlabel('Rotation Threshold (°)')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)

    plt.suptitle(f"Pose Evaluation\nAccuracy={acc:.2f}%, "
                 f"Median Rot Error={np.median(rot_errs):.2f}°, "
                 f"Median Trans Error={np.median(trans_errs)*100:.2f}cm")

    plt.tight_layout()
    plt.subplots_adjust(top=0.8)
    plt.savefig("pose_evaluation.svg", dpi=300)
    plt.show()
