
import os.path
from pathlib import Path
from typing import List, Tuple, Union, Dict
import math

import cv2
import numpy as np
import torch

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.data_utils import get_depth_image_from_path
import open3d as o3d


class SceneFlowDataset(InputDataset):
    """Dataset that returns images and depths.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs.
    """

    def __init__(self, dataparser_outputs: DataparserOutputs,
                 scale_factor: float = 1.0,
                 depth_remove_outliers: bool = False
                 ):
        super().__init__(dataparser_outputs, scale_factor)
        assert (
            "gt_tracks" in dataparser_outputs.metadata.keys()
            and dataparser_outputs.metadata["gt_tracks"] is not None
        )
        assert (
                "depth_filenames" in dataparser_outputs.metadata.keys()
                and dataparser_outputs.metadata["depth_filenames"] is not None
        )
        self.depth_filenames = self.metadata["depth_filenames"]
        self.depth_unit_scale_factor = self.metadata["depth_unit_scale_factor"]
        self.tracks_filenames = self.metadata["tracks_filenames"]
        self.gt_tracks_3D, self.gt_tracks_2D = self.metadata["gt_tracks"]
        self.mask_filenames = self.metadata["mask_filenames"]
        self.eval_mask_filenames = self.metadata["evalmask_filenames"] if 'evalmask_filenames' in self.metadata else []
        self.segmentation_filenames = self.metadata["segmentation_filenames"]
        self.nframes = dataparser_outputs.metadata["nframes"]
        self.far_plane = dataparser_outputs.metadata["far"]
        self.alpha_color = dataparser_outputs.alpha_color
        self.dataset_source = dataparser_outputs.metadata['source']
        self.split = dataparser_outputs.metadata['split']

        # options for filtering input depth maps up-front (used for Dy-Check data)
        self.depth_remove_outliers = depth_remove_outliers

    def get_data(self, image_idx: int) -> Dict:
        """Returns the ImageDataset data as a dictionary.

        Args:
            image_idx: The image index in the dataset.
        """
        image = self.get_image(image_idx)
        data = {"image_idx": image_idx, "image": image}
        if self.mask_filenames is not None:
            mask_filepath = self.mask_filenames[image_idx]
            foreground_mask = self.get_foreground_mask(mask_filepath)
            foreground_mask = foreground_mask.astype(bool)
            data["valid_mask"] = foreground_mask
            # image[~foreground_mask] = self.alpha_color

        metadata = self.get_metadata(data)
        data.update(metadata)
        return data

    def get_metadata(self, data: Dict) -> Dict:
        # get depth image
        depth_image = self.get_depth(data)

        # get ground truth tracks
        gt_tracks_2d = torch.from_numpy(self.gt_tracks_2D)

        # get segmentation labels
        if os.path.exists(str(self.segmentation_filenames[data['image_idx']])):
            segmentation = np.load(str(self.segmentation_filenames[data['image_idx']]))
        else:
            segmentation = np.zeros(depth_image.shape[:2]).astype(np.uint8)

        # get predicted tracks and foreground mask
        tracks, track_mask, track_segs = self.get_pred_trajectories(data)

        w, h = depth_image.size(1), depth_image.size(0)
        eval_mask = self.get_evaluation_mask(data, w, h)

        return {"depth_image": depth_image, "tracks": tracks, "track_mask": track_mask, "track_segs": track_segs,
                "gt_tracks_2D": gt_tracks_2d, "eval_mask": eval_mask, "segmentation": segmentation}

    def get_depth(self, data: Dict) -> np.array:
        if self.split not in ['train', 'virtual-train']:
            return torch.zeros_like(data['image'][:, :, 0:1])

        filepath = self.depth_filenames[data["image_idx"]]
        height = int(self._dataparser_outputs.cameras.height[data["image_idx"]])
        width = int(self._dataparser_outputs.cameras.width[data["image_idx"]])
        if self.dataset_source == 'point-odyssey':
            depth_img = self.get_depth_image_from_png(filepath, height, width)
            depth_img *= self.depth_unit_scale_factor
        elif self.dataset_source == 'dycheck':
            # Scale depth images to meter units and also by scaling applied to cameras
            depth_img = get_depth_image_from_path(
                filepath=filepath, height=height, width=width, scale_factor=self.depth_unit_scale_factor
            )
        else:
            raise RuntimeError("Dataset source must be point odyssey or dycheck")

        # clip depth to far plane
        depth_img[depth_img.isnan()] = 0
        depth_img[depth_img > self.far_plane] = self.far_plane
        return depth_img

    def get_evaluation_mask(self, data: Dict, w: int, h: int) -> torch.Tensor:
        if len(self.eval_mask_filenames) > 0:
            if os.path.exists(self.eval_mask_filenames[data['image_idx']]):
                eval_mask = self.get_foreground_mask(self.eval_mask_filenames[data['image_idx']])
            else:  # there is no mask
                eval_mask = np.ones((h, w), dtype=bool)
        else:  # there is no mask
            eval_mask = np.ones((h, w), dtype=bool)

        # if eval_mask.shape != (h, w):
        #     eval_mask = cv2.resize(eval_mask.astype(float), (w, h), interpolation=cv2.INTER_NEAREST) > 0

        return torch.from_numpy(eval_mask)

    def get_pred_trajectories(self, data: Dict):
        if self.split != 'train':
            return torch.empty(1), torch.empty(1), torch.empty(1)

        filepath = self.tracks_filenames[data["image_idx"]]
        if os.path.exists(filepath):
            tracks = np.load(filepath)  # shape (T, num_tracks, 3), with each (x, y, visible) tracked across T frames
        else:
            tracks = np.zeros((self.nframes, 1, 3))

        if self.segmentation_filenames is not None:
            segmentation = np.load(str(self.segmentation_filenames[data["image_idx"]]))
            foreground_mask = segmentation > 0
            xs, ys = tracks[data["image_idx"]].astype(int)[:, 0], tracks[data["image_idx"]].astype(int)[:, 1]
            trajectories_in_foreground = foreground_mask[ys, xs]
            track_segmentation_classes = segmentation[ys, xs]
            if data["image_idx"] < 15:
                temp_idx = 15  # 0-15 before, 16 cur, 17-32 inclusive after
            elif data["image_idx"] >= (self.nframes - 15):  # if nframe=100, 84 or higher inclusive
                temp_idx = self.nframes - 16  # set it to 83  --> 83:100 is 17 frames (
            else:
                temp_idx = data["image_idx"]
            tracks = tracks[temp_idx - 15: temp_idx + 16, :, :]
        else:
            trajectories_in_foreground = np.ones(tracks.shape[1], dtype=bool)
            track_segmentation_classes = np.zeros(tracks.shape[1], dtype=int)
        # tracks = tracks[:, trajectories_in_foreground, :]
        return tracks, trajectories_in_foreground, track_segmentation_classes

    def get_full_depth_pc_sequence(self):
        """
        :return:
        """
        pcs = [None for k in range(self.nframes)]
        for i, depth_fname in enumerate(self.depth_filenames):
            print("Processed point cloud %s" % i)
            # get depth image
            height = int(self._dataparser_outputs.cameras.height[i])
            width = int(self._dataparser_outputs.cameras.width[i])
            image = self.get_image(i)
            if self.dataset_source == 'point-odyssey':
                depth_img = self.get_depth_image_from_png(depth_fname, height, width)
                depth_img *= self.depth_unit_scale_factor
                depth_img = depth_img.squeeze(-1)
                foreground_mask = self.get_foreground_mask(self.mask_filenames[i])
            else:  # self.dataset_source == 'dycheck':
                # Scale depth images to meter units and also by scaling applied to cameras
                depth_img = get_depth_image_from_path(
                    filepath=depth_fname, height=height, width=width, scale_factor=self.depth_unit_scale_factor
                )
                depth_img = depth_img.squeeze(-1)
                foreground_mask = depth_img > 0
                if self.mask_filenames is not None:
                    additional_mask = self.get_foreground_mask(self.mask_filenames[i])
                    foreground_mask &= additional_mask

            # get segmentation mask
            segmentation = torch.from_numpy(np.load(str(self.segmentation_filenames[i])))

            # get camera info for this timestep
            fx, fy = self.cameras.fx[i].item(), self.cameras.fy[i].item()
            cx, cy = self.cameras.cx[i].item(), self.cameras.cy[i].item()
            extrinsics = self.cameras.camera_to_worlds[i]
            depth_img[depth_img > self.far_plane] = self.far_plane
            pc, pix, segmentation, rgb, depth = self.depth2pc(depth_img, height, width, fx, fy, cx, cy, extrinsics, foreground_mask, self.far_plane, segmentation, image)

            # remove outliers
            if self.depth_remove_outliers:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pc.cpu().numpy())
                pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
                segmentation = segmentation[ind]
                rgb = rgb[ind]
                depth = depth[ind]
                # mesh_pcd.paint_uniform_color([0, 0.651, 0.929])
                pc = torch.from_numpy(np.array(pcd.points))

            # apply foreground/background segmentation
            pc = torch.cat((pc, segmentation.unsqueeze(-1), rgb, depth.view(-1, 1)), dim=-1)

            # get frame ID
            frameid = int(self.cameras.times[i])
            pcs[frameid] = pc

        return pcs

    @staticmethod
    def depth2pc(depth_img: np.array, height: int, width: int, fx: float, fy: float,
                 cx: float, cy:float, extrinsics: np.array, foreground_mask: np.array, far: float,
                 segmentation: np.array, image: np.array) -> np.array:

        # start with coordinate frame x-right, y-down
        xs, ys = np.meshgrid(np.arange(width), np.arange(height))
        xs, ys = torch.from_numpy(xs), torch.from_numpy(ys)

        # compute depth mask and filter
        distance_mask = (depth_img <= far).numpy()
        if foreground_mask is not None:
            mask = foreground_mask & distance_mask
        else:
            mask = distance_mask
        depth_map = -depth_img[mask]  # z is reflected in blender / nerfstudio!
        xs = xs[mask]
        ys = ys[mask]
        segmentation = segmentation[mask]
        image = image[mask]
        xs_orig, ys_orig = xs.clone(), ys.clone()
        xys_orig = torch.stack((xs_orig, ys_orig), dim=-1)

        # opengl uses x-right, y-down, and z-backward for its camera coordinate frame
        xs = depth_map * (xs - cx) / fx
        ys = - depth_map * (ys - cy) / fy  # TODO: this is a left-handed coordinate system! But we immediately undo this in the couple lines
        xyz_cam = torch.stack((xs, ys, depth_map), dim=-1)

        # -------------------------------------------------------------------------------------
        # change basis of camera coords to match that of our extrinsics C2W (this is 180 degree rotation about z-axis)
        # our extrinsics matrix actually takes in OpenCV coordinates! All of OUR properties are in opencv; but nerfstudio needs opengl!
        xyz_cam[:, [0, 1]] *= -1  # now that's interesting... this is x and y, not y and z!!!
        # -------------------------------------------------------------------------------------

        # convert from camera to world coordinates
        xyz_world = xyz_cam.float() @ extrinsics[:3, :3].T + extrinsics[:3, 3:4].reshape((1, 3))
        return xyz_world, xys_orig, segmentation, image, depth_img[mask]

    @staticmethod
    def get_foreground_mask(mask_fname):
        if str(mask_fname).endswith('png'):
            mask = cv2.imread(str(mask_fname.absolute()))
            foreground_mask = np.sum(mask, axis=-1) > 0
        elif str(mask_fname).endswith('npy'):
            foreground_mask = np.load(str(mask_fname.absolute()))
            foreground_mask = foreground_mask.astype(bool)
        else:
            raise RuntimeError("Mask must be png or npy.")
        return foreground_mask

    @staticmethod
    def get_depth_image_from_tiff(
        filepath: Path,
        height: int,
        width: int,
        interpolation: int = cv2.INTER_NEAREST,
        ) -> torch.Tensor:
        """Loads, rescales and resizes depth images.
        Filepath points to a 16-bit or 32-bit depth image, or a numpy array `*.npy`.

        Args:
            filepath: Path to depth image.
            height: Target depth image height.
            width: Target depth image width.
            interpolation: Depth value interpolation for resizing.

        Returns:
            Depth image torch tensor with shape [width, height, 1].
        """
        depth = cv2.imread(str(filepath.absolute()), cv2.IMREAD_UNCHANGED).astype(np.float64)
        depth = cv2.resize(depth, (width, height), interpolation=interpolation)
        return torch.from_numpy(depth[:, :, np.newaxis])

    @staticmethod
    def get_depth_image_from_png(
        filepath: Path,
        height: int,
        width: int,
        interpolation: int = cv2.INTER_NEAREST,
        ) -> torch.Tensor:
        depth_16bit = cv2.imread(str(filepath.absolute()), cv2.IMREAD_ANYDEPTH)
        depth = depth_16bit.astype(np.float32) / 65535.0 * 1000.0
        depth = cv2.resize(depth, (width, height), interpolation=interpolation)
        return torch.from_numpy(depth[:, :, np.newaxis])
