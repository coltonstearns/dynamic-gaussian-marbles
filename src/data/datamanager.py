
from dataclasses import dataclass, field
from pathlib import Path
import os
from typing import (
    Any,
    Callable,
    Dict,
    Literal,
    Tuple,
    Type,
    Union,
    cast,
)

import torch

from nerfstudio.configs.dataparser_configs import AnnotatedDataParserUnion
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.dataloaders import (
    CacheDataloader,
)
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.data.datamanagers.base_datamanager import DataManagerConfig, DataManager, variable_res_collate
from src.data.dataloader import \
    GaussianSplattingImageSampler, GaussianSplattingFixedIndicesEvalDataloader
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from src.data.databundle import GaussianSplattingImageBundle
from src.data.sceneflow_dataset import SceneFlowDataset


@dataclass
class GaussianSplattingDataManagerConfig(DataManagerConfig):
    """A basic data manager"""

    _target: Type = field(default_factory=lambda: GaussianSplattingDataManager)
    """Target class to instantiate."""
    dataparser: AnnotatedDataParserUnion = BlenderDataParserConfig()
    """Specifies the dataparser used to unpack the data."""
    camera_optimizer: CameraOptimizerConfig = CameraOptimizerConfig()
    """Specifies the camera pose optimizer used during training. Helpful if poses are noisy, such as for data from
    Record3D."""
    collate_fn: Callable[[Any], Any] = cast(Any, staticmethod(nerfstudio_collate))
    """Specifies the collate function to use for the train and eval dataloaders."""
    camera_res_scale_factor: float = 1.0
    """The scale factor for scaling spatial data such as images, mask, semantics
    along with relevant information about camera intrinsics
    """

    # our parameters
    depth_remove_outliers: bool = False
    znear: float = 0.01
    zfar: float = 100.0


class GaussianSplattingDataManager(DataManager):
    """Gaussian splatting stored data manager implementation.
    """

    config: GaussianSplattingDataManagerConfig
    train_dataset: SceneFlowDataset
    eval_dataset: SceneFlowDataset
    train_dataparser_outputs: DataparserOutputs

    def __init__(
        self,
        config: GaussianSplattingDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        # set internal variables
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.sampler = None

        # set up test split
        self.test_mode = test_mode
        self.test_split = "test" if test_mode in ["test", "inference"] else "val"

        # configure dataparser - load training dataparser outputs
        self.dataparser_config = self.config.dataparser
        if self.config.data is not None:
            self.config.dataparser.data = Path(self.config.data)
        else:
            self.config.data = self.config.dataparser.data
        self.dataparser = self.dataparser_config.setup()
        if test_mode == "inference":
            self.dataparser.downscale_factor = 1  # Avoid opening images
        self.includes_time = self.dataparser.includes_time
        self.train_dataparser_outputs: DataparserOutputs = self.dataparser.get_dataparser_outputs(split="train")

        # build train and eval datasets
        self.train_dataset = self.load_dataset(self.train_dataparser_outputs, self.config.camera_res_scale_factor)
        self.eval_dataset = self.load_dataset(self.dataparser.get_dataparser_outputs(split=self.test_split), self.config.camera_res_scale_factor)

        # choose what to cache on GPU cache batch info on GPU
        self.exclude_batch_keys_from_device = self.train_dataset.exclude_batch_keys_from_device
        # self.exclude_batch_keys_from_device += ["tracks", "track_mask", "gt_tracks_2D", "eval_mask", "segmentation", "depth_image"]

        if self.config.masks_on_gpu is True:
            self.exclude_batch_keys_from_device.remove("mask")
        if self.config.images_on_gpu is True:
            self.exclude_batch_keys_from_device.remove("image")

        # if variable resolution, apply variable-resolution collate
        if self.train_dataparser_outputs is not None:
            cameras = self.train_dataparser_outputs.cameras
            if len(cameras) > 1:
                for i in range(1, len(cameras)):
                    if cameras[0].width != cameras[i].width or cameras[0].height != cameras[i].height:
                        CONSOLE.print("Variable resolution, using variable_res_collate")
                        self.config.collate_fn = variable_res_collate
                        break
        super().__init__()

    def load_dataset(self, dataparser_outputs, camera_res_scale_factor):
        return SceneFlowDataset(
            dataparser_outputs=dataparser_outputs,
            scale_factor=camera_res_scale_factor,
            depth_remove_outliers=self.config.depth_remove_outliers
        )

    def setup_train(self):
        """Sets up the data loaders for training"""
        assert self.train_dataset is not None
        CONSOLE.print("Setting up training dataset...")
        self.train_image_dataloader = CacheDataloader(
            self.train_dataset,
            num_images_to_sample_from=-1,
            num_times_to_repeat_images=-1,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
            exclude_batch_keys_from_device=self.exclude_batch_keys_from_device,
        )
        self.iter_train_image_dataloader = iter(self.train_image_dataloader)
        self.train_image_sampler = GaussianSplattingImageSampler(self.train_dataset.cameras, znear=self.config.znear, zfar=self.config.zfar)
        self.train_camera_optimizer = self.config.camera_optimizer.setup(
            num_cameras=self.train_dataset.cameras.size, device=self.device
        )

    def setup_eval(self):
        """Sets up the data loader for evaluation"""
        assert self.eval_dataset is not None
        CONSOLE.print("Setting up evaluation dataset...")
        self.eval_image_dataloader = CacheDataloader(
            self.eval_dataset,
            num_images_to_sample_from=-1,
            num_times_to_repeat_images=-1,
            device='cpu',  # for eval dataloader, we want to save GPU memory
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
            exclude_batch_keys_from_device=self.exclude_batch_keys_from_device,
        )
        self.iter_eval_image_dataloader = iter(self.eval_image_dataloader)
        self.eval_image_sampler = GaussianSplattingImageSampler(self.eval_dataset.cameras, znear=self.config.znear, zfar=self.config.zfar)

        # for inference-time --> called directly in Pipeline class
        self.fixed_indices_eval_dataloader = GaussianSplattingFixedIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            device='cpu',
            num_workers=self.world_size * 4,
        )

        self.fixed_indices_train_dataloader = GaussianSplattingFixedIndicesEvalDataloader(
            input_dataset=self.train_dataset,
            device='cpu',
            num_workers=self.world_size * 4,
        )

    def next_train(self, step: int, valid_idxs: tuple = None) -> Tuple[GaussianSplattingImageBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        assert self.train_image_sampler is not None
        assert isinstance(image_batch, dict)
        ray_bundle, batch, _ = self.train_image_sampler.sample(image_batch, valid_idxs)
        ray_bundle = ray_bundle.to(self.device)
        return ray_bundle, batch

    def next_eval(self, step: int) -> Tuple[GaussianSplattingImageBundle, Dict]:
        """Returns the next batch of data from the eval dataloader."""
        image_idx, ray_bundle, batch = self.next_eval_image(step)
        return ray_bundle, batch

    def next_eval_image(self, step: int) -> Tuple[int, GaussianSplattingImageBundle, Dict]:
        """
        This will return the same as next_eval for Gaussian Splatting (because we work on a per-image basis)
        """
        self.eval_count += 1
        image_batch = next(self.iter_eval_image_dataloader)
        assert self.eval_image_sampler is not None
        assert isinstance(image_batch, dict)
        ray_bundle, batch, image_idx = self.eval_image_sampler.sample(image_batch)
        ray_bundle = ray_bundle.to(self.device)
        return image_idx, ray_bundle, batch

    # =========== Below is for backwards compatability with Nerfstudio -- Not Really Meaningful ================
    def get_train_rays_per_batch(self) -> int:
        rays = int(torch.mean(self.train_dataset.cameras.width.float()).item() * torch.mean(self.train_dataset.cameras.height.float()))
        return rays

    def get_eval_rays_per_batch(self) -> int:
        return self.get_eval_rays_per_image()

    def get_eval_rays_per_image(self) -> int:
        rays = int(torch.mean(self.eval_dataset.cameras.width.float()).item() * torch.mean(self.eval_dataset.cameras.height.float()))
        return rays

