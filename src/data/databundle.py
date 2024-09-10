
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, Literal, Optional, Tuple, Union, overload

import torch
from jaxtyping import Float, Int, Shaped
from torch import Tensor
from nerfstudio.utils.tensor_dataclass import TensorDataclass
from nerfstudio.cameras.rays import RaySamples


@dataclass
class GaussianSplattingImageBundle(TensorDataclass):
    """A bundle of ray parameters."""

    image_width: float
    """Image width to render"""
    image_height: float
    """Image height to render"""
    FoVy: float
    """Image y-axis field of view to render"""
    FoVx: float
    """Image x-axis field of view to render"""
    znear: float
    """Image near plane"""
    zfar: float
    """Image far plane"""
    time: int
    """Image far plane"""
    world_view_transform: [Float[Tensor, "*batch 1"]]
    """World view transform"""
    full_proj_transform: [Float[Tensor, "*batch 1"]]
    """World view transform compose with projection matrix"""
    camera_center: [Float[Tensor, "*batch 1"]]
    """Camera xyz position in world space"""
    fx: float
    """Focal length in x"""
    fy: float
    """Focal length in y"""
    cx: float
    """Principal point in x"""
    cy: float
    """Principal point in y"""
    nstudio_c2w: [Float[Tensor, "*batch 4 4"]]
    """NerfStudio camera to world matrix"""

    def __init__(self, image_width, image_height, FoVy, FoVx, znear, zfar, world_view_transform, full_proj_transform, time=None, camera_center=None,
                 fx=None, fy=None, cx=None, cy=None, nstudio_c2w=None):
        self.image_width = image_width
        self.image_height = image_height
        self.FoVy = FoVy
        self.FoVx = FoVx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        self.camera_center = world_view_transform.inverse()[3, :3]
        self.time = time
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.nstudio_c2w = nstudio_c2w

    def __len__(self) -> int:
        return 1

    def set_camera_indices(self, camera_index: int) -> None:
        raise RuntimeError("Gaussians splatting is a rasterization framework, and does not sample rays.")

    def sample(self, num_rays: int):
        raise RuntimeError("Gaussians splatting is a rasterization framework, and does not sample rays.")

    def get_row_major_sliced_ray_bundle(self, start_idx: int, end_idx: int):
        raise RuntimeError("Gaussians splatting is a rasterization framework, and does not sample rays.")

    def get_ray_samples(
        self,
        bin_starts: Float[Tensor, "*bs num_samples 1"],
        bin_ends: Float[Tensor, "*bs num_samples 1"],
        spacing_starts: Optional[Float[Tensor, "*bs num_samples 1"]] = None,
        spacing_ends: Optional[Float[Tensor, "*bs num_samples 1"]] = None,
        spacing_to_euclidean_fn: Optional[Callable] = None,
    ) -> RaySamples:
        raise RuntimeError("Gaussians splatting is a rasterization framework, and does not sample rays.")
