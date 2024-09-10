from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Type

import cv2
import numpy as np
import torch

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.io import load_from_json
from nerfstudio.utils.rich_utils import CONSOLE


def downscale(img, scale: int) -> np.ndarray:
    """Function from DyCheck's repo. Downscale an image.

    Args:
        img: Input image
        scale: Factor of the scale

    Returns:
        New image
    """
    if scale == 1:
        return img
    height, width = img.shape[:2]
    if height % scale > 0 or width % scale > 0:
        raise ValueError(f"Image shape ({height},{width}) must be divisible by the" f" scale ({scale}).")
    out_height, out_width = height // scale, width // scale
    resized = cv2.resize(img, (out_width, out_height), cv2.INTER_AREA)  # type: ignore
    return resized


def upscale(img, scale: int) -> np.ndarray:
    """Function from DyCheck's repo. Upscale an image.

    Args:
        img: Input image
        scale: Factor of the scale

    Returns:
        New image
    """
    if scale == 1:
        return img
    height, width = img.shape[:2]
    out_height, out_width = height * scale, width * scale
    resized = cv2.resize(img, (out_width, out_height), cv2.INTER_AREA)  # type: ignore
    return resized


def rescale(img, scale_factor: float, interpolation: int = cv2.INTER_AREA) -> np.ndarray:
    """Function from DyCheck's repo. Rescale an image.

    Args:
        img: Input image
        scale: Factor of the scale
        interpolation: Interpolation method in opencv

    Returns:
        New image
    """
    scale_factor = float(scale_factor)
    if scale_factor <= 0.0:
        raise ValueError("scale_factor must be a non-negative number.")
    if scale_factor == 1.0:
        return img

    height, width = img.shape[:2]
    if scale_factor.is_integer():
        return upscale(img, int(scale_factor))

    inv_scale = 1.0 / scale_factor
    if inv_scale.is_integer() and (scale_factor * height).is_integer() and (scale_factor * width).is_integer():
        return downscale(img, int(inv_scale))

    print(f"Resizing image by non-integer factor {scale_factor}, this may lead to artifacts.")
    height, width = img.shape[:2]
    out_height = math.ceil(height * scale_factor)
    out_height -= out_height % 2
    out_width = math.ceil(width * scale_factor)
    out_width -= out_width % 2

    return cv2.resize(img, (out_width, out_height), interpolation)  # type: ignore


def _load_scene_info(data_dir: Path) -> Tuple[np.ndarray, float, float, float]:
    """Function from DyCheck's repo. Load scene info from json.

    Args:
        data_dir: data path

    Returns:
        A tuple of scene info: center, scale, near, far
    """
    scene_dict = load_from_json(data_dir / "scene.json")
    center = np.array(scene_dict["center"], dtype=np.float32)
    scale = scene_dict["scale"]
    near = scene_dict["near"]
    far = scene_dict["far"]
    return center, scale, near, far


def _rescale_depth(depth_raw: np.ndarray, cam: Dict) -> np.ndarray:
    """Depth rescale function from DyCheck.

    Args:
        depth: A numpy ndarray of the raw depth
        cam: Dict of the camera

    Returns:
        A numpy ndarray of the processed depth
    """
    xx, yy = np.meshgrid(np.arange(cam["width"], dtype=np.float32), np.arange(cam["height"], dtype=np.float32))
    pixels = np.stack([xx, yy], axis=-1)
    batch_shape = pixels.shape[:-1]
    pixels = np.reshape(pixels, (-1, 2))
    y = (pixels[..., 1] - cam["cy"]) / cam["fy"]
    x = (pixels[..., 0] - cam["cx"]) / cam["fx"]
    # x = (pixels[..., 0] - self.principal_point_x - y * self.skew) / self.scale_factor_x
    # assume skew = 0
    viewdirs = np.stack([x, y, np.ones_like(x)], axis=-1)
    local_viewdirs = viewdirs / np.linalg.norm(viewdirs, axis=-1, keepdims=True)
    viewdirs = (cam["camera_to_worlds"][:3, :3] @ local_viewdirs[..., None])[..., 0]
    viewdirs /= np.linalg.norm(viewdirs, axis=-1, keepdims=True)
    viewdirs = viewdirs.reshape((*batch_shape, 3))
    cosa = viewdirs @ (cam["camera_to_worlds"][:, 2])
    if depth_raw.ndim == cosa.ndim:
        depth = depth_raw[..., None] / cosa[..., None]
    else:
        depth = depth_raw / cosa[..., None]
    return depth


@dataclass
class DycheckDataParserConfig(DataParserConfig):
    """Dycheck (https://arxiv.org/abs/2210.13445) dataset parser config"""

    _target: Type = field(default_factory=lambda: Dycheck)
    """target class to instantiate"""
    data: Path = Path("data/iphone/mochi-high-five")
    """Directory specifying location of data."""
    scale_factor: float = 4.0
    """How much to scale the camera origins by."""
    alpha_color: str = "white"
    """alpha color of background"""
    downscale_factor: int = 1
    """How much to downscale images."""
    scene_box_bound: float = 1.5
    """Boundary of scene box."""
    dataset_source: str = "dycheck"


@dataclass
class Dycheck(DataParser):
    """Dycheck (https://arxiv.org/abs/2210.13445) Dataset `iphone` subset"""

    config: DycheckDataParserConfig
    includes_time: bool = True

    def __init__(self, config: DycheckDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data
        self.alpha_color = config.alpha_color

        # get all scale factors
        self.scale_factor: float = config.scale_factor
        self.dycheck_scale_factor = None

        # load extra info from "extra.json"
        data_info = load_from_json(self.data / "dataset.json")
        self._num_frames = data_info['num_exemplars']
        self._center, self._scale, self._near, self._far = _load_scene_info(self.data)

    def _generate_dataparser_outputs(self, split="train"):
        if self.alpha_color is not None:
            alpha_color_tensor = get_color(self.alpha_color)
        else:
            alpha_color_tensor = None
        data_dir = self.data if split in ["train", "val"] else self.data / split  # covers virtual viewpoints
        splits_dir = data_dir / "splits"

        # scale the scene to fill the aabb bbox
        sf = self.config.scene_box_bound / 4 / (self._scale * self._far)
        self.dycheck_scale_factor = sf
        CONSOLE.print(f"Dycheck scale factor is {self.dycheck_scale_factor}")

        if not (splits_dir / f"{split}.json").exists():
            CONSOLE.print(f"split {split} not found, using split train")
            split = "train"
        split_dict = load_from_json(splits_dir / f"{split}.json")
        frame_names = np.array(split_dict["frame_names"])
        time_ids = np.array(split_dict["time_ids"])

        nframes = self._num_frames  # len(frame_names.tolist())
        image_filenames, depth_filenames, trajectory_filenames, gt_trajectories, cams, evalmask_filenames, mask_filenames, segmentation_filenames = self.process_frames(frame_names.tolist(), time_ids, split)

        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-self.config.scene_box_bound] * 3, [self.config.scene_box_bound] * 3], dtype=torch.float32
            )
        )
        cam_dict = {}
        for k in cams[0].keys():
            cam_dict[k] = torch.stack([torch.as_tensor(c[k]) for c in cams], dim=0)
        cam_dict['metadata'] = {'nframes': torch.tensor([nframes for i in range(50)])}
        cameras = Cameras(camera_type=CameraType.PERSPECTIVE, **cam_dict)

        scale = self._scale * self.dycheck_scale_factor * self.scale_factor
        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            alpha_color=alpha_color_tensor,
            scene_box=scene_box,
            metadata={
                "depth_filenames": depth_filenames,
                "depth_unit_scale_factor": scale,
                "camera_translation": self._center,
                "near": self._near * scale,
                "far": self._far * scale * 10,  # we artificially scale by 10x to fit all gaussians

                # add my additional inputs to metadata field
                "segmentation_filenames": segmentation_filenames,
                "gt_tracks": gt_trajectories,
                "mask_filenames": None if len(mask_filenames) == 0 else mask_filenames,
                "tracks_filenames": trajectory_filenames,
                "nframes": nframes,
                "source": 'dycheck',
                "split": split,
                "evalmask_filenames": evalmask_filenames
            },
        )

        return dataparser_outputs

    def process_frames(self, frame_names: List[str], time_ids: np.ndarray, split: str) -> Tuple[List, List, List, List, List, List]:
        """Read cameras and filenames from the name list.

        Args:
            frame_names: list of file names.
            time_ids: time id of each frame.

        Returns:
            A list of camera, each entry is a dict of the camera.
        """
        data_dir = self.data if split in ["train", "val"] else self.data / split  # covers virtual viewpoints
        image_filenames, depth_filenames, trajectory_filenames, evalmask_filenames, mask_filenames, segmentation_filenames = [], [], [], [], [], []
        cams = []
        for idx, frame in enumerate(frame_names):
            image_filenames.append(data_dir / f"rgb/{self.config.downscale_factor}x/{frame}.png")
            depth_filenames.append(data_dir / f"depth/{self.config.downscale_factor}x/{frame}.npy")
            if frame.startswith("0_"):  # covers case of dycheck vs in-the-wild data
                trajectory_filenames.append(data_dir / f"tracks/{self.config.downscale_factor}x/track_{frame[2:]}.npy")
            else:
                trajectory_filenames.append(data_dir / f"tracks/{self.config.downscale_factor}x/track_{frame}.npy")
            segmentation_filenames.append(data_dir / f"segmentation/{self.config.downscale_factor}x/{frame}.npy")
            if os.path.exists(str(data_dir / f"mask/1x/{frame}.npy")):
                mask_filenames.append(data_dir / f"mask/{self.config.downscale_factor}x/{frame}.npy")
            if 'train' not in split:
                evalmask_filenames.append(data_dir / f"covisible/2x/val/{frame}.png")

            cam_json = load_from_json(data_dir / f"camera/{frame}.json")
            w2c = torch.as_tensor(cam_json["orientation"])  # w2c really
            position = torch.as_tensor(cam_json["position"])  # location of camera in world space
            position -= self._center  # some scenes look weird (wheel)
            position *= self._scale * self.dycheck_scale_factor * self.scale_factor
            pose = torch.zeros([3, 4])

            # IF IN DYCHECK, FORMAT AS FOLLOWS:
            if self.config.dataset_source == "dycheck":
                pose[:3, :3] = w2c.T  # dycheck stores rotation in world-to-camera
                pose[:3, 3] = position  # position is already in c2w, as it specifies camera's location in world space

                # First, everything nerfstudio passes us is in OpenGL camera coordinates. This transforms them
                #  into OpenCV camera coordinates. Because' it's applied before the rotation, it's column operations
                pose[0:3, 1:3] *= -1  # todo: this is unnecessary, but deeply buried in my code somewhere!

                # Also, this suggests that DyCheck is in OpenGL actually
                # After applying the rotation and translation, we'll be in an opencv world space,
                # with (x-right, y-down, and z-forward). For visualization, we want (x-right, y-forward, z-up)
                pose[2, :] *= -1  # invert world z
                pose = pose[[0, 2, 1], :]  # switch y and z

            else:  # FOR COLMAP, FORMAT THIS WAY: We've now organized nvidia as dycheck, so nothing uses this
                # pose[:3, :3] = c2w.T  # from w2c to c2w
                # pose[:3, 3] = -position  # position is already in c2w AND opengl format
                # pose[:3, 3] = - c2w.T @ position  # go from world to camera space

                # our input is in opencv coord system, which is x-right, y-down, and z-forward
                # for visualization purposes, we want x-right, y-forward, and z-up
                # pose = pose[[0, 2, 1], :]  # flip y and z
                # pose[2, :] *= -1  # invert z to become up

                pose[:3, :3] = w2c
                pose[:3, 3] = position
                pose = pose[[0, 2, 1], :]
                pose[2, :] *= -1

            cams.append(
                {
                    "camera_to_worlds": pose,
                    "fx": cam_json["focal_length"] / self.config.downscale_factor,
                    "fy": cam_json["focal_length"] * cam_json["pixel_aspect_ratio"] / self.config.downscale_factor,
                    "cx": cam_json["principal_point"][0] / self.config.downscale_factor,
                    "cy": cam_json["principal_point"][1] / self.config.downscale_factor,
                    "height": cam_json["image_size"][1] // self.config.downscale_factor,
                    "width": cam_json["image_size"][0] // self.config.downscale_factor,
                    "times": torch.as_tensor(time_ids[idx]).float(),
                }
            )

        d = self.config.downscale_factor
        if not image_filenames[0].exists():
            CONSOLE.print(f"downscale factor {d}x not exist, converting")
            ori_h, ori_w = cv2.imread(str(data_dir / f"rgb/1x/{frame_names[0]}.png")).shape[:2]
            (data_dir / f"rgb/{d}x").mkdir(exist_ok=True)
            h, w = ori_h // d, ori_w // d
            for frame in frame_names:
                cv2.imwrite(
                    str(data_dir / f"rgb/{d}x/{frame}.png"),
                    cv2.resize(cv2.imread(str(data_dir / f"rgb/1x/{frame}.png")), (w, h)),
                )
            CONSOLE.print("finished")

        if not depth_filenames[0].exists() and 'train' in split:  # val set has no depth
            CONSOLE.print(f"processed depth downscale factor {d}x not exist, converting")
            (data_dir / f"depth/{d}x").mkdir(exist_ok=True, parents=True)
            for idx, frame in enumerate(frame_names):
                depth = np.load(data_dir / f"depth/1x/{frame}.npy")
                ori_h, ori_w = depth.shape[:2]
                h, w = ori_h // d, ori_w // d
                depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
                np.save(str(data_dir / f"depth/{d}x/{frame}.npy"), depth)
            CONSOLE.print("finished")

        if not segmentation_filenames[0].exists() and 'train' in split:  # val set has no segmentation
            CONSOLE.print(f"processed segmentation downscale factor {d}x not exist, converting")
            (data_dir / f"segmentation/{d}x").mkdir(exist_ok=True, parents=True)
            for idx, frame in enumerate(frame_names):
                segmentation = np.load(str(data_dir / f"segmentation/1x/{frame}.npy"))
                ori_h, ori_w = segmentation.shape[:2]
                h, w = ori_h // d, ori_w // d
                segmentation = cv2.resize(segmentation, (w, h), interpolation=cv2.INTER_NEAREST)
                np.save(str(data_dir / f"segmentation/{d}x/{frame}.npy"), segmentation)
            CONSOLE.print("finished")

        if not trajectory_filenames[0].exists() and split == 'train':  # all other splits have no segmentation
            CONSOLE.print(f"processed tracks downscale factor {d}x not exist, converting")
            (data_dir / f"tracks/{d}x").mkdir(exist_ok=True, parents=True)
            for idx, frame in enumerate(frame_names):
                if frame.startswith("0_"):  # covers case of dycheck vs in-the-wild data
                    tracks = np.load(str(data_dir / f"tracks/1x/track_{frame[2:]}.npy"))
                else:
                    tracks = np.load(str(data_dir / f"tracks/1x/track_{frame}.npy"))
                tracks[:, :, :2] = tracks[:, :, :2] / d
                if frame.startswith("0_"):  # covers case of dycheck vs in-the-wild data
                    np.save(str(data_dir / f"tracks/{d}x/track_{frame[2:]}.npy"), tracks)
                else:
                    np.save(str(data_dir / f"tracks/{d}x/track_{frame}.npy"), tracks)
            CONSOLE.print("finished")

        if len(mask_filenames) > 0 and not mask_filenames[0].exists():  # we have masks at 1x, but not Dx
            CONSOLE.print(f"processed masks downscale factor {d}x not exist, converting")
            (data_dir / f"mask/{d}x").mkdir(exist_ok=True, parents=True)
            for idx, frame in enumerate(frame_names):
                mask = np.load(str(data_dir / f"mask/1x/{frame}.npy"))
                ori_h, ori_w = mask.shape[:2]
                h, w = ori_h // d, ori_w // d
                invalid_mask = cv2.resize((~mask).astype(float), (w, h), interpolation=cv2.INTER_AREA)
                invalid_mask = invalid_mask > 0
                np.save(str(data_dir / f"mask/{d}x/{frame}.npy"), ~invalid_mask)

        # load ground truth keypoint tracks
        if os.path.exists(str(data_dir / f"validation-tracks/{frame_names[0]}.json")):
            with open(str(data_dir / f"validation-tracks/{frame_names[0]}.json"), "r") as f:
                template = json.load(f)
            n_tracks = len(template)
        else:
            print("WARNING: no validation tracks found for split %s. WE WILL SKIP TRACKING EVALUATION" % split)
            n_tracks = 0

        # set trajectories 3D to zeros, because we don't have this for dycheck
        trajectories_3d = np.zeros((len(frame_names), n_tracks, 3))

        # assign trajectories 2D for all cameras
        trajectories_2d = np.zeros((len(frame_names), n_tracks, 3))  # 3rd entry is whether valid or not
        for idx, frame in enumerate(sorted(frame_names)):
            if not os.path.exists(str(data_dir / f"validation-tracks/{frame}.json")):
                continue
            with open(str(data_dir / f"validation-tracks/{frame}.json"), "r") as f:
                tracked_locs = json.load(f)
            time_idx = int(frame[2:7])
            trajectories_2d[idx, :, :] = np.array(tracked_locs)
            trajectories_2d[idx, :, :2] = trajectories_2d[idx, :, :2] / self.config.downscale_factor
        gt_trajectories = (trajectories_3d, trajectories_2d)

        return image_filenames, depth_filenames, trajectory_filenames, gt_trajectories, cams, evalmask_filenames, mask_filenames, segmentation_filenames