
import os
import argparse
import fnmatch
import json
import numpy as np
import torch
from src.utils.utils import focal2fov, getWorld2View2, getProjectionMatrix
from src.data.databundle import GaussianSplattingImageBundle
import matplotlib
from matplotlib import pyplot as plt
import cv2
matplotlib.use('Agg')
import open3d as o3d
import tqdm

VISUALIZE = False


def visualize_sparsedepth(points2D, disparities, colors, reference_disparity, fname):
    plt.scatter(points2D[:, 0].numpy().tolist(), (-points2D[:, 1].numpy()).tolist(), c=colors.numpy().tolist(), s=1)
    plt.xlim(0, 470)
    plt.ylim(-260, 0)
    plt.savefig('colmap_%s.png' % fname)

    viz_disparity = torch.clip((disparities - disparities.mean()) / disparities.std(), 0, 1)
    plt.scatter(points2D[:, 0].numpy().tolist(), (-points2D[:, 1].numpy()).tolist(), c=viz_disparity, s=1)
    plt.xlim(0, 470)
    plt.ylim(-260, 0)
    plt.savefig('colmap_disparity_%s.png' % fname)
    plt.close('all')

    # get matplotlib cmap for viridis
    viz_disparity = torch.clip((reference_disparity - reference_disparity.mean()) / reference_disparity.std(), 0, 1)
    cmap = plt.get_cmap('viridis')
    # get the rgba values for the colormap
    rgba = cmap(viz_disparity.numpy())
    cv2.imwrite('predicted_disparity_%s.png' % fname, rgba * 255)


def load_colmap_pc(pc_path, cam_opencv):
    colmap_pc = np.load(pc_path)  # this is in colmap coordindate frame!
    # world coordinate transform: map colmap gravity guess (-y) to nerfstudio convention (+z)

    # first, move back from opengl (dycheck) to opencv (colmap) coordinate frame
    colmap_pc[:, 1:3] *= -1

    # then, move PC into our nerfstudio world frame (x-right, y-forward, z-up)
    colmap_pc[:, [0, 1, 2]] = colmap_pc[:, [0, 2, 1]]
    colmap_pc[:, 2] *= -1

    # remove duplicates
    pc_rounded = np.round(colmap_pc, decimals=1)
    _, u_idxs = np.unique(pc_rounded, return_index=True, axis=0)
    colmap_pc = colmap_pc[u_idxs, :]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(colmap_pc[:, :3]))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colmap_pc[:, 3:6]) / 255.0)
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=32, std_ratio=0.5)
    colmap_pc = torch.from_numpy(np.array(pcd.points)).float()
    colors = torch.from_numpy(np.array(pcd.colors))

    # project colmap pc into 2D
    colmap_2D, colmap_depths = project_pc(colmap_pc, cam_opencv)
    colmap_disparities = 1 / colmap_depths
    return colmap_2D, colmap_depths, colmap_disparities, colors


def project_pc(points, camera_opencv):
    means3D_affine = torch.cat([points, torch.ones(points.size(0), 1).to(points)], dim=-1)  # N x 4
    projected_unhomog = means3D_affine.double() @ camera_opencv.full_proj_transform.double()
    reg = 1.0 / (projected_unhomog[:, 3:4].double() + 0.0001)
    depths = projected_unhomog[:, 3].clone()
    projected_homogenous = projected_unhomog.double() * reg.double()
    h, w = camera_opencv.image_height, camera_opencv.image_width
    projected_homogenous[:, 0] = ((projected_homogenous[:, 0] + 1.0) * w - 1.0) * 0.5  # convert from NDC coords to pixel space
    projected_homogenous[:, 1] = ((projected_homogenous[:, 1] + 1.0) * h - 1.0) * 0.5
    means2D = projected_homogenous[:, :2]
    return means2D, depths


def load_camera(camera_path):
    with open(camera_path) as f:
        cam = json.load(f)
    c2w_nerfstudio = load_camera_into_nerfstudio(cam)
    fx = cam["focal_length"]
    fy = cam["focal_length"] * cam["pixel_aspect_ratio"]
    cx = cam["principal_point"][0]
    cy = cam["principal_point"][1]
    width = cam["image_size"][0]
    height = cam["image_size"][1]
    znear, zfar = 0.01, 100.0
    cam_opencv = camera_nerfstudio_to_opencv(c2w_nerfstudio, fx, fy, cx, cy, width, height, znear, zfar)
    return cam_opencv


def load_camera_into_nerfstudio(cam_json):
    c2w = torch.as_tensor(cam_json["orientation"]).T
    position = torch.as_tensor(cam_json["position"])
    pose = torch.zeros([3, 4])
    pose[:3, :3] = c2w
    pose[:3, 3] = position
    # move into our weird hybrid coord system (which is unnessary, because we'll undo this later)
    pose[0:3, 1:3] *= -1
    pose[2, :] *= -1  # invert world z
    pose = pose[[0, 2, 1], :]  # switch y and z
    return pose


def camera_nerfstudio_to_opencv(extrinsics, fx, fy, cx, cy, width, height, znear, zfar):
        fovx = focal2fov(fx, width)
        fovy = focal2fov(fy, height)

        # Convert from Blender Coordinate Frame to (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w = torch.cat([extrinsics, torch.tensor([[0, 0, 0, 1]]).to(extrinsics)], dim=0)
        c2w[:3, 1:3] *= -1

        # Convert from camera2world to world2camera
        w2c = torch.linalg.inv(c2w)
        R = w2c[:3, :3]
        T = w2c[:3, 3]

        # R is stored transposed due to 'glm' in CUDA code
        R = R.T

        # convert Rt into final transforms
        trans = torch.tensor([0., 0., 0.])
        scale = 1.0
        world_view_transform = getWorld2View2(R, T, trans, scale).transpose(0, 1)
        cx_gsplat = (cx - width / 2) / (2 * fx)
        cy_gsplat = (cy - height / 2) / (2 * fy)
        projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy, cx=cx_gsplat, cy=cy_gsplat).transpose(0, 1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)

        raster_info = GaussianSplattingImageBundle(
            image_width=width,
            image_height=height,
            FoVx=fovx,
            FoVy=fovy,
            znear=znear,
            zfar=zfar,
            world_view_transform=world_view_transform,
            full_proj_transform=full_proj_transform,
            time=0  # dummy time index
        )
        return raster_info


def estimate_scale_and_shift(pred_disparity, gt_disparity, num_iterations, pct_keep=0.3):
    errs, scales, shifts, subset_errs = [], [], [], []
    for i in range(num_iterations):
        randidxs = torch.randperm(pred_disparity.size(0))[:int(pct_keep * pred_disparity.size(0))]
        this_pred_disparity = pred_disparity[randidxs]
        this_gt_disparity = gt_disparity[randidxs]
        A = torch.stack([this_pred_disparity, torch.ones(this_pred_disparity.size(0)).to(pred_disparity)], dim=1)
        out = torch.linalg.lstsq(A.float(), this_gt_disparity[:, None].float())
        scale, shift = out[0].flatten()
        err = torch.mean(torch.abs(gt_disparity - (scale * pred_disparity + shift)))
        this_err = torch.mean(torch.abs(this_gt_disparity - (scale * this_pred_disparity + shift)))
        err = this_err  # replace with local error - see if we can do better!
        errs.append(err)
        scales.append(scale)
        shifts.append(shift)
        subset_errs.append(this_err)

    best_idx = torch.argmin(torch.stack(errs))
    return scales[best_idx].item(), shifts[best_idx].item(), errs[best_idx].item(), subset_errs[best_idx].item()


def estimate_scales_and_shifts(args, depth_fnames):
    scales, shifts = [], []
    for fname in tqdm.tqdm(depth_fnames):
        # load depth map
        pred_depthmap = np.load(os.path.join(args.input_depth_dir, fname))
        pred_disparity = torch.from_numpy(1 / pred_depthmap)

        # load camera intrinsics and extrinsics
        cam_opencv = load_camera(os.path.join(args.input_camera_dir, fname.replace('npy', 'json')))
        width, height = cam_opencv.image_width, cam_opencv.image_height

        # load colmap point cloud into our world coordinate frame
        colmap_2D, colmap_depths, colmap_disparities, colmap_clrs = load_colmap_pc(args.input_colmap_pc, cam_opencv)

        # only keep valid projected points
        image_bounds_mask = (colmap_2D[:, 0] < (width-1)) & (colmap_2D[:, 0] > 0) & (colmap_2D[:, 1] < (height-1)) & (colmap_2D[:, 1] > 0)
        image_bounds_mask = image_bounds_mask & ~torch.isnan(colmap_disparities) & ~torch.isinf(colmap_disparities)
        colmap_2D = colmap_2D[image_bounds_mask]
        colmap_2D = colmap_2D.long()
        colmap_clrs = colmap_clrs[image_bounds_mask]
        colmap_disparities = colmap_disparities[image_bounds_mask]

        # visualize colmap_2D and colmap_disparities
        if VISUALIZE:
            visualize_sparsedepth(colmap_2D, colmap_disparities, colmap_clrs, pred_disparity, fname)

        # apply segmentation mask
        segmentation = torch.from_numpy(np.load(os.path.join(args.input_segmentation_dir, fname)))
        segmentation_mask = (segmentation == 0)[colmap_2D[:, 1], colmap_2D[:, 0]]
        colmap_2D = colmap_2D[segmentation_mask]
        colmap_disparities = colmap_disparities[segmentation_mask]

        # estimate scale and shift
        pred_disparity = pred_disparity[colmap_2D[:, 1], colmap_2D[:, 0]]
        scale, shift, err, subset_err = estimate_scale_and_shift(pred_disparity.cuda(), colmap_disparities.cuda(), num_iterations=250, pct_keep=0.15)

        print("Output:")
        print("Best scale %s and shift %s" % (scale, shift))
        print("Error is: %s" % err)
        scales.append(scale)
        shifts.append(shift)

    return scales, shifts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_colmap_pc",
        help="npy file containing metric point cloud from colmap",
    )
    parser.add_argument(
        "--input_depth_dir",
        default="./out.mp4",
        help="path to folder containing relative depth maps",
    )
    parser.add_argument(
        "--input_camera_dir",
        default="",
        help="path to folder containing camera parameters",
    )
    parser.add_argument(
        "--input_segmentation_dir",
        default="",
        help="path to folder containing SAM segmentations (background is 0)",
    )
    parser.add_argument(
        "--output_depth_dir",
        default="",
        help="path to folder for putting output scale+shifted depth maps",
    )
    parser.add_argument(
        "--path_fnmatch",
        default="",
        help="glob pattern to match valid filenames",
    )
    parser.add_argument(
        "--dataset_source",
        default="dycheck",
        help="One of ['dycheck', 'colmap']",
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_depth_dir):
        os.makedirs(args.output_depth_dir)

    # load rgb names and format them
    depth_fnames = os.listdir(args.input_depth_dir)
    depth_fnames = [name for name in depth_fnames if fnmatch.fnmatch(name, args.path_fnmatch)]
    depth_fnames = sorted(depth_fnames)

    # get all scales and shifts
    scales, shifts = estimate_scales_and_shifts(args, depth_fnames)

    # print("Subset error is: %s" % subset_err)
    scale = np.mean(scales)
    shift = np.mean(shifts)
    print(">>>>>> Average scale %s and shift %s" % (scale, shift))
    updated_depth_medians = []
    for fname in tqdm.tqdm(depth_fnames):
        # load depth map
        pred_depthmap = np.load(os.path.join(args.input_depth_dir, fname))
        pred_disparity = torch.from_numpy(1 / pred_depthmap)  # we've already accounted for nan when first converting disparity

        # apply scale and shift
        updated_depthmap = 1 / (scale * (1 / pred_depthmap) + shift)
        # updated_depthmap = np.expand_dims(updated_depthmap, axis=-1)  # h, w, 1
        np.save(os.path.join(args.output_depth_dir, fname), updated_depthmap)
        updated_depth_medians.append(np.median(updated_depthmap))

    # update scene.json file with correct scale info
    avg_median = np.mean(updated_depth_medians)
    scaling = 4.0 / avg_median  # put median at 4 units away
    far_plane = 20 / scaling
    near_plane = 0.05 / scaling
    with open(os.path.join(args.input_camera_dir, "..", "scene.json"), "w") as f:
        scene = { "far": far_plane, "near": near_plane, "scale": scaling}
        json.dump(scene, f)




