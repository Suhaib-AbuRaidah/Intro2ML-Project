import torch
import numpy as np
import open3d as o3d
import cv2
from torchvision import transforms as T
from matplotlib import cm
import sys
sys.path.append("/home/suhaib/ML_Project")

from trans_pose.stage2.dataset2_stage2 import Stage2Dataset
from trans_pose.stage2.network import TransPoseNetwork
from trans_pose.stage2.utilis import rigid_transform_3D

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT = "checkpoints/best_model.pth"


def to_numpy_img(tensor):
    """
    Convert CHW torch image in [0,1] or [0,255] to HWC uint8 RGB.
    """
    img = tensor.detach().cpu().numpy()
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    # If values are in [0,1], scale
    if img.max() <= 1.0:
        img = (img * 255.0).astype(np.uint8)
    else:
        img = img.astype(np.uint8)
    return img


def depth_to_color(depth_np, colormap=cm.viridis):
    """
    Normalize depth and map to color using matplotlib colormap.
    depth_np: HxW (float)
    Returns HxW x 3 uint8
    """
    d = depth_np.copy()
    # handle constant depth
    if d.max() - d.min() < 1e-6:
        d_norm = np.zeros_like(d)
    else:
        d_norm = (d - d.min()) / (d.max() - d.min())
    d_col = colormap(d_norm)[..., :3]
    return (d_col * 255).astype(np.uint8)


def create_colored_pointcloud(points_np, seg=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    if seg is None:
        pcd.paint_uniform_color([0.75, 0.75, 0.75])
    else:
        seg = np.asarray(seg).astype(int)
        num_cls = int(seg.max()) + 1
        rng = np.random.RandomState(0)
        palette = rng.rand(num_cls, 3)
        colors = palette[seg]
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def visualize_all(points_np,
                  seg=None,
                  pred_kpts=None,
                  gt_kpts=None,
                  zero_kp=None,
                  pred_pose_dict=None,
                  gt_pose_dict=None,
                  rgb_img=None,
                  depth_img=None):
    """
    Visualize point cloud, segmentation, predicted keypoints, ground-truth keypoints,
    zero-pose keypoints (transformed), coordinate frames (zero and predicted),
    and show RGB + depth windows using Open3D + OpenCV.
    - points_np: (N,3)
    - seg: (N,) int labels or None
    - pred_kpts: (K,3) or None
    - gt_kpts: (K,3) or None (target keypoints in world)
    - zero_kp: (K,3) or None (canonical keypoints in object canonical frame)
    - pred_pose: (4,4) numpy transform mapping zero_kp -> world (same frame as points_np)
    - rgb_img: HxWx3 uint8 (RGB)
    - depth_img: HxWx3 uint8 colorized
    """
    vis_objs = []

    # main point cloud (colored by seg if provided)
    pcd = create_colored_pointcloud(points_np, seg=seg)
    vis_objs.append(pcd)

    # predicted keypoints (red)
    if pred_kpts is not None:
        kp_pc = o3d.geometry.PointCloud()
        kp_pc.points = o3d.utility.Vector3dVector(np.asarray(pred_kpts))
        kp_pc.paint_uniform_color([1.0, 0.0, 0.0])
        vis_objs.append(kp_pc)

        # small spheres for predicted keypoints
        for kp in pred_kpts:
            s = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            s.translate(kp)
            s.paint_uniform_color([1.0, 0.6, 0.6])
            vis_objs.append(s)

    # ground-truth keypoints in world (green)
    if gt_kpts is not None:
        kp2 = o3d.geometry.PointCloud()
        kp2.points = o3d.utility.Vector3dVector(np.asarray(gt_kpts))
        kp2.paint_uniform_color([0.0, 1.0, 0.0])
        vis_objs.append(kp2)

        for kp in gt_kpts:
            s = o3d.geometry.TriangleMesh.create_sphere(radius=0.0075)
            s.translate(kp)
            s.paint_uniform_color([0.6, 1.0, 0.6])
            vis_objs.append(s)

    # zero-pose keypoints transformed to world using pred_pose (blue)
    if zero_kp is not None and pred_pose_dict is not None:
        # zero_kp: (K,3) canonical; pred_pose maps zero -> world
        zero_kp_h = np.concatenate([np.asarray(zero_kp), np.ones((len(zero_kp), 1))], axis=1)  # (K,4)
        transformed = (pred_pose_dict @ zero_kp_h.T).T[:, :3]
        kpt_z = o3d.geometry.PointCloud()
        kpt_z.points = o3d.utility.Vector3dVector(transformed)
        kpt_z.paint_uniform_color([0.0, 0.0, 1.0])
        vis_objs.append(kpt_z)

        for kp in transformed:
            s = o3d.geometry.TriangleMesh.create_sphere(radius=0.006)
            s.translate(kp)
            s.paint_uniform_color([0.6, 0.6, 1.0])
            vis_objs.append(s)

    # coordinate frames: zero (identity at origin) and predicted (transformed)
    frame_zero = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    vis_objs.append(frame_zero)

    if pred_pose_dict is not None:
        for key in pred_pose_dict.keys():
            frame_pred = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            frame_pred.transform(pred_pose_dict[key])
            vis_objs.append(frame_pred)

    # if gt_pose_dict is not None:
    #     for key in gt_pose_dict.keys():
    #         frame_pred = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    #         frame_pred.transform(gt_pose_dict[key])
    #         frame_pred.paint_uniform_color([0.0, 0.0, 0.0])  # cyan for GT
    #         vis_objs.append(frame_pred)

    # # Offscreen renderer
    # width, height = 3400, 1440  # adjust resolution for high-DPI
    # renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)

    # scene = renderer.scene
    # for i, obj in enumerate(vis_objs):
    #     scene.add_geometry(f"obj_{i}", obj, o3d.visualization.rendering.MaterialRecord())

    # # Optional: set camera
    # center = frame_zero.get_center()
    # renderer.setup_camera(40, center, center +[0,0,1], [0,1,1])

    # # Capture image
    # img = renderer.render_to_image()
    # o3d.io.write_image("high_res_capture.png", img)

    # show point cloud + kps + frames in Open3D visualizer
    o3d.visualization.draw_geometries(vis_objs)

    # show RGB and depth using OpenCV windows (if provided)
    if rgb_img is not None:
        # Expect RGB uint8 HxWx3
        cv2.imshow("RGB", cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
    if depth_img is not None:
        # depth_img already colorized HxWx3
        cv2.imshow("Depth (colorized)", cv2.cvtColor(depth_img, cv2.COLOR_RGB2BGR))

    if (rgb_img is not None) or (depth_img is not None):
        # wait key; close on any key
        cv2.waitKey(0)
        cv2.destroyAllWindows()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_coord_frame(ax, T, length=0.02, lw=3,linestyle='-'):
    """
    Draw a coordinate frame (XYZ axes) at transform T (4x4).
    """
    origin = T[:3, 3]

    x_axis = origin + T[:3, 0] * length
    y_axis = origin + T[:3, 1] * length
    z_axis = origin + T[:3, 2] * length

    ax.plot([origin[0], x_axis[0]], [origin[1], x_axis[1]], [origin[2], x_axis[2]],
            color='red', linewidth=lw,linestyle=linestyle)
    ax.plot([origin[0], y_axis[0]], [origin[1], y_axis[1]], [origin[2], y_axis[2]],
            color='green', linewidth=lw,linestyle=linestyle)
    ax.plot([origin[0], z_axis[0]], [origin[1], z_axis[1]], [origin[2], z_axis[2]],
            color='blue', linewidth=lw,linestyle=linestyle)

from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D

def plot_scene_matplotlib(points_np, seg, pred_pose_dict,gt_pose_dict,
                          rgb_img, depth_img, sn, out_path="scene.svg"):

    fig = plt.figure(figsize=(16, 10))

    # ------------------------------
    # GRID: 3 rows × 2 columns
    #  - Left column: images
    #  - Right column: big 3D scene
    # ------------------------------
    gs = gridspec.GridSpec(
        3, 2,
        width_ratios=[1, 3],     # <<< right side is 3× larger
        height_ratios=[1, 1, 1]
    )
    plt.subplots_adjust(wspace=0.0, hspace=0.0)  # remove white space

    # 3D SCENE: span rows 0:3, col 1
    ax1 = fig.add_subplot(gs[:, 1], projection='3d')

    # main point cloud
    points = np.asarray(points_np)
    if seg is None:
        ax1.scatter(points[:,0], points[:,1], points[:,2], s=1, c="gray")
    else:
        seg = np.asarray(seg)
        ax1.scatter(points[:,0], points[:,1], points[:,2], s=2, c=seg, cmap='tab20')

    # plot predicted frames
    for key, T in pred_pose_dict.items():
        plot_coord_frame(ax1, np.asarray(T), length=0.07, lw=4)

    for key, T in gt_pose_dict.items():
        plot_coord_frame(ax1, np.asarray(T), length=0.07, lw=2,linestyle='--')
    # formatting
    ax1.set_xlabel("X", fontsize=12)
    ax1.set_ylabel("Y", fontsize=12)
    ax1.set_zlabel("Z", fontsize=12)
    ax1.set_title("3D Scene with Predicted Poses", fontsize=24)
    ax1.view_init(elev=-90, azim=-90)
    ax1.grid(False)
    ax1.axis('off')

    # ---------------------------------
    # LEFT COLUMN: RGB, Depth, SN
    # ---------------------------------

    ax2 = fig.add_subplot(gs[0, 0])
    ax2.imshow(rgb_img)
    ax2.axis("off")
    ax2.set_title("Input", fontsize=24)
    ax3 = fig.add_subplot(gs[1, 0])
    if depth_img.ndim == 2:
        ax3.imshow(depth_img, cmap='gray')
    else:
        ax3.imshow(depth_img)
    ax3.axis("off")

    ax4 = fig.add_subplot(gs[2, 0])
    if sn.ndim == 2:
        ax4.imshow(sn, cmap='gray')
    else:
        ax4.imshow(sn)
    ax4.axis("off")

    # ---------------------------------
    # SAVE AS SVG
    # ---------------------------------
    plt.savefig(out_path, format='png', dpi=300)
    plt.close(fig)
    print(f"Saved SVG to {out_path}")


# Example usage
# plot_scene_matplotlib(points_np, seg, pred_pose_dict, "test_scene.svg")


def run_inference(index=0, show_all=True):
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)

    params = {
        "img_outdim": 128,
        "normals_outdim": 128,
        "points_outdim": 256,
        "num_classes": 4,
        "num_keypoints": 10,
    }
    model = TransPoseNetwork(**params).float().to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dataset = Stage2Dataset(
        root_dir="/home/suhaib/ML_Project/data/transcg-data-1/transcg", transforms=T.ToTensor())
    sample = dataset[index]

    # prepare inputs
    rgb   = sample["rgb"].unsqueeze(0).to(DEVICE)    # (1,3,H,W) or (1,3,H,W) torch
    sn    = sample["sn"].unsqueeze(0).to(DEVICE)
    depth = sample["depth"].unsqueeze(0).to(DEVICE)  # assume (1,H,W) or (1,1,H,W) ; handle below
    o_mask = sample["mask"].unsqueeze(0).to(DEVICE)

    intr = dataset.camera_intrisics
    intrinsics_tuple = (intr[0,0], intr[1,1], intr[0,2], intr[1,2])

    with torch.no_grad():
        seg_logits, offsets, points, _ = model(rgb, sn, depth, o_mask, intrinsics_tuple)

    # convert point cloud and segmentation to numpy
    points_np = points[0].cpu().numpy()         # (N,3)
    seg = seg_logits[0].argmax(dim=-1).cpu().numpy()  # (N,)

    # compute votes and predicted keypoints (global mean across points)
    votes = (offsets + points.unsqueeze(2))[0].cpu().numpy()  # (N,K,3)
    pred_kpts = votes.mean(axis=0)  # (K,3) -- global average of votes

    # pick first object from sample's GT for visualization (same as before)
    obj_ids = list(sample["target_keypoints"].keys())
    if len(obj_ids) == 0:
        raise RuntimeError("No objects in sample target_keypoints")

    pred_poses = {}  # store per-object predicted transforms

    for o_i, oid in enumerate(obj_ids):
        o_mask= (seg == (o_i))  # object masks start from 1
        gt_kpts = np.array(sample["target_keypoints"][oid])      # (K,3)
        zero_kp = np.array(sample["zero_keypoints"][oid])        # (K,3)
        pred_kpts_obj= votes[o_mask].mean(axis=0)  # (K,3) -- mean votes for this object
        A = torch.tensor(zero_kp, device=DEVICE).float()
        B = torch.tensor(pred_kpts_obj,    device=DEVICE).float()

        pred_pose_T = rigid_transform_3D(A, B).cpu().numpy()
        pred_poses[oid] = pred_pose_T

    # Prepare RGB & depth images for display
    # rgb: tensor CHW; depth: likely HxW or 1xHxW
    rgb_img = to_numpy_img(sample["rgb"])  # uses original sample (not preprocessed) -> HWC uint8
    sn = to_numpy_img(sample["sn"])  # (H,W) single-channel; convert to uint8
    # depth may be single-channel; convert to numpy 2D
    depth_t = sample["depth"]
    if isinstance(depth_t, torch.Tensor):
        depth_np = depth_t.detach().cpu().numpy()
        # if shape (H,W) or (1,H,W)
        if depth_np.ndim == 3 and depth_np.shape[0] == 1:
            depth_np = depth_np[0]
        if depth_np.ndim == 3 and depth_np.shape[2] == 1:
            depth_np = depth_np[..., 0]
    else:
        depth_np = np.array(depth_t)

    depth_color = depth_to_color(depth_np)

    for key in pred_poses.keys():
        # show text info in console
        print(f"Object ID: {key}")
        print("Predicted transform (zero_kp -> pred_kp):\n", pred_poses[key])
        # if pose GT available
        sample_poses = sample.get("poses", None)
        if sample_poses is not None and key in sample_poses:
            print("Target/GT pose for object:\n", sample_poses[key])

    plot_scene_matplotlib(points_np, seg, pred_poses,sample.get("poses", None),rgb_img, depth_color, sn, out_path="inference_scene.png")
    print("Saved scene plot to inference_scene.svg")
    # visualize everything
    if show_all:
        visualize_all(points_np,
                      seg=seg,
                      pred_pose_dict=pred_poses,
                      gt_pose_dict=sample.get("poses", None),
                      rgb_img=rgb_img,
                      depth_img=depth_color)
    else:
        # minimal: show point cloud + pred keypoints and frames
        pcd = create_colored_pointcloud(points_np, seg=seg)
        pred_kp_pc = o3d.geometry.PointCloud()
        pred_kp_pc.points = o3d.utility.Vector3dVector(pred_kpts)
        pred_kp_pc.paint_uniform_color([1, 0, 0])

        zero_kp_h = np.concatenate([zero_kp, np.ones((len(zero_kp), 1))], axis=1)
        zero_transformed = (pred_pose_T @ zero_kp_h.T).T[:, :3]
        zero_kp_pc = o3d.geometry.PointCloud()
        zero_kp_pc.points = o3d.utility.Vector3dVector(zero_transformed)
        zero_kp_pc.paint_uniform_color([0, 0, 1])

        frame_zero = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        frame_pred = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        frame_pred.transform(pred_pose_T)

        o3d.visualization.draw_geometries([pcd, pred_kp_pc, zero_kp_pc, frame_zero, frame_pred])

        # show rgb/depth quickly
        cv2.imshow("RGB", cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
        cv2.imshow("Depth (colorized)", cv2.cvtColor(depth_color, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # run inference on index 0 by default
    run_inference(index=1800, show_all=True)
