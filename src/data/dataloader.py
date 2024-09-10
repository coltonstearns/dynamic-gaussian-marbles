
from typing import Tuple

from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.utils.misc import get_dict_to_torch
from nerfstudio.data.utils.dataloaders import EvalDataloader

from typing import Dict, Optional, Union

import torch
from torch import nn
from nerfstudio.cameras.cameras import Cameras
from src.data.databundle import GaussianSplattingImageBundle
from src.utils.utils import focal2fov, getWorld2View2, getProjectionMatrix


class GaussianSplattingImageSampler(nn.Module):
    """
    Given all input image data, this samples and processes the cached data, yielding a rapid
    training output.
    """
    def __init__(self, cameras: Cameras, znear=0.01, zfar=100, **kwargs) -> None:
        super().__init__()
        self.kwargs = kwargs
        self.cameras = cameras
        self.zfar = zfar
        self.znear = znear

        # for updating the extrinsics matrix
        self.scale = 1.0
        self.trans = torch.tensor([0, 0, 0])

    def sample(self, image_batch: Dict, valid_idxs: tuple = None, time_idxs: tuple = None) -> tuple[GaussianSplattingImageBundle, dict, int]:
        """
        image_batch (dict): contains the following {"image_idx": int, "image": tensor, "__dataloader_metatada__"}
        """
        if isinstance(image_batch["image"], list):
            N = len(image_batch["image"])
        elif isinstance(image_batch["image"], torch.Tensor):
            N = image_batch["image"].size(0)
        else:
            raise ValueError("image_batch['image'] must be a list or torch.Tensor")

        # select camera index and generate rendering info
        if valid_idxs is None and time_idxs is None:
            image_idx = torch.randint(high=N, size=(1,)).item()
            time_idx = None
        elif valid_idxs is not None:
            image_idx = torch.randint(high=len(valid_idxs), size=(1,)).item()
            image_idx = valid_idxs[image_idx]
            time_idx = None
        elif time_idxs is not None:
            time_idx = torch.randint(high=len(time_idxs), size=(1,)).item()
            time_idx = time_idxs[time_idx]
            image_idx = None

        raster_info, batch = self.sample_idx(image_batch, image_idx, time_idx)
        return raster_info, batch, image_idx

    def sample_idx(self, image_batch: Dict, camidx: int, timeidx: int = None) -> tuple[GaussianSplattingImageBundle, dict]:
        assert not (timeidx is not None and camidx is not None), "Cannot specify both timeidx and camidx"
        if timeidx is not None:
            valid_camidxs = torch.where(self.cameras.times == timeidx)[0]
            if len(valid_camidxs) == 0:
                return None, None
            camidx = torch.randint(high=len(valid_camidxs), size=(1,)).item()
            camidx = valid_camidxs[camidx].item()


        camera = self.cameras[camidx]
        raster_info = self.get_raster_info(camera)

        # load train image supervision from passed in batch
        batch = {}
        # note that image_batch randomly shuffles the camera indices
        random_shuffle_cam_idx = torch.where(image_batch['image_idx'] == camidx)[0].item()
        gt_image = image_batch["image"][random_shuffle_cam_idx].clone()

        # load tracks strategically - context window max of 16
        tracks = image_batch["tracks"][random_shuffle_cam_idx].clone()
        # if tracks.size(0) > 1:  # otherwise we're in validation mode
        #     tracks = tracks[:, image_batch["track_mask"][random_shuffle_cam_idx]]

        batch["image"] = gt_image
        batch["tracks"] = tracks.float()
        batch["tracks_foreground"] = image_batch["track_mask"][random_shuffle_cam_idx].clone()
        batch["tracks_segmentations"] = image_batch["track_segs"][random_shuffle_cam_idx].clone()
        batch["gt_tracks_2D"] = image_batch["gt_tracks_2D"].clone()
        batch["eval_mask"] = image_batch["eval_mask"][random_shuffle_cam_idx].clone()
        batch["segmentation"] = image_batch["segmentation"][random_shuffle_cam_idx].clone()
        batch["depth_image"] = image_batch["depth_image"][random_shuffle_cam_idx].clone()
        if "valid_mask" in image_batch:
            batch["valid_mask"] = image_batch["valid_mask"][random_shuffle_cam_idx].clone()
        return raster_info, batch

    def get_raster_info(self, camera: Cameras):
        # load rendering info
        fovx = focal2fov(camera.fx.item(), camera.width.item())
        fovy = focal2fov(camera.fy.item(), camera.height.item())
        time_idx = int(camera.times.item())
        world_view_transform, full_proj_transform = self.load_gaussian_splat_transforms(camera, fovx, fovy, self.znear, self.zfar, self.trans, self.scale)
        raster_info = GaussianSplattingImageBundle(
            image_width=camera.width.item(),
            image_height=camera.height.item(),
            FoVx=fovx,
            FoVy=fovy,
            znear=self.znear,
            zfar=self.zfar,
            world_view_transform=world_view_transform,
            full_proj_transform=full_proj_transform,
            time=time_idx,
            fx=camera.fx.item(),
            fy=camera.fy.item(),
            cx=camera.cx.item(),
            cy=camera.cy.item(),
            nstudio_c2w=camera.camera_to_worlds.squeeze(0)
        )
        return raster_info

    @staticmethod
    def load_gaussian_splat_transforms(camera, fovx, fovy, znear, zfar, trans=torch.tensor([0., 0., 0.]), scale=1.0):
        # Convert from Blender Coordinate Frame to (Y up, Z back) to COLMAP (Y down, Z forward)
        matrix = camera.camera_to_worlds
        c2w = torch.cat([matrix, torch.tensor([[0, 0, 0, 1]]).to(matrix)], dim=0)
        c2w[:3, 1:3] *= -1

        # Convert from camera2world to world2camera
        w2c = torch.linalg.inv(c2w)
        R = w2c[:3, :3]
        T = w2c[:3, 3]

        # R is stored transposed due to 'glm' in CUDA code
        R = R.T

        # convert Rt into final transforms
        world_view_transform = getWorld2View2(R, T, trans, scale).transpose(0, 1)
        cx_gsplat = (camera.cx.item() - camera.width.item()/2) / (2*camera.fx.item())
        cy_gsplat = (camera.cy.item() - camera.height.item()/2) / (2*camera.fy.item())
        projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy, cx=cx_gsplat, cy=cy_gsplat).transpose(0,1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        return world_view_transform, full_proj_transform


class GaussianSplattingFixedIndicesEvalDataloader(EvalDataloader):
    """Dataloader that iterates through all images"""
    def __init__(
        self,
        input_dataset: InputDataset,
        image_indices: Optional[Tuple[int]] = None,
        device: Union[torch.device, str] = "cpu",
        **kwargs,
    ):
        super().__init__(input_dataset, device, **kwargs)
        if image_indices is None:
            self.image_indices = list(range(len(input_dataset)))
        else:
            self.image_indices = image_indices
        self.count = 0
        self.image_sampler = GaussianSplattingImageSampler(self.cameras)

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count < len(self.image_indices):
            image_idx = self.image_indices[self.count]
            batch = self.input_dataset[image_idx]
            batch = get_dict_to_torch(batch, device='cuda:0', exclude=["image"])
            ray_bundle = self.image_sampler.get_raster_info(camera=self.cameras[image_idx])
            ray_bundle = ray_bundle.to('cuda:0')
            self.count += 1
            return ray_bundle, batch
        raise StopIteration

    def get_data_from_image_idx(self, image_idx: int):
        """Returns the data for a specific image index.

        Args:
            image_idx: Camera image index
        """
        batch = self.input_dataset[image_idx]
        batch = get_dict_to_torch(batch, device='cuda:0', exclude=["image"])
        ray_bundle = self.image_sampler.get_raster_info(camera=self.cameras[image_idx])
        ray_bundle = ray_bundle.to('cuda:0')
        return ray_bundle, batch

    def get_data_from_time_idx(self, time_idx: int):
        """Returns the data for a specific time index.

        Args:
            time_idx: Time index
        """
        image_idx = torch.where(self.cameras.times == time_idx)[0]
        assert len(image_idx) == 1
        return self.get_data_from_image_idx(image_idx.item())

