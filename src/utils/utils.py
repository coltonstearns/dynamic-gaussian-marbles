#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
from typing import NamedTuple
import lpips
import torch.nn.functional as F
from typing import Optional

PHASE_NAMES = {0: 'bundle-adjust', 1: 'motion-estimation', 2: 'motion-expansion'}
PHASE_IDS = {'bundle-adjust': 0, 'motion-estimation': 1, 'motion-expansion': 2}


def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t, device='cpu'):
    Rt = torch.zeros((4, 4)).to(device)
    Rt[:3, :3] = R.transpose(0, 1)
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return Rt.float()

def getWorld2View2(R, t, translate=torch.tensor([.0, .0, .0]), scale=1.0, device='cpu'):
    Rt = torch.zeros((4, 4)).to(device)
    Rt[:3, :3] = R.transpose(0, 1)
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = torch.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate.to(device)) * scale
    C2W[:3, 3] = cam_center
    Rt = torch.linalg.inv(C2W)
    return Rt.float()

def getProjectionMatrix(znear, zfar, fovX, fovY, cx=None, cy=None):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)
    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    if cx is None:
        P[0, 2] = (right + left) / (right - left)
    else:
        P[0, 2] = cx
    if cy is None:
        P[1, 2] = (top + bottom) / (top - bottom)
    else:
        P[1, 2] = cy
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)  # 100/99, so scales everything by 1.01
    P[2, 3] = -(zfar * znear) / (zfar - znear)  # then they subtract 100/99, so that near-plane becomes 0, and far-plane is far-plane
    # this actually is all just an accomodation to make near-plane be at 0-depth
    # the fourth is the unnormalized depth!! The third is actually a near-far-plane normalized depth!!
    # however, BOTH are simply z-buffer values, ie not the depth along the ray!
    # when you divide by actual depth, everything between the near and far planes becomes between [0,1] --> truly NDC space!
    # note to self: using such a large far plane might be bad for numerical precision
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))








#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import sys
from datetime import datetime
import numpy as np
import random

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))


# My utils

def to_batch_padded(tensor, val2batchidx, pad_size=None, batch_size=None):
    # preliminaries
    ptr = batch2ptr(val2batchidx, with_ends=True)
    counts = ptr[1:] - ptr[:-1]
    if batch_size is None:
        batch_size = ptr.size(0) - 1
    if pad_size is None:
        pad_size = torch.max(counts).item()

    # in trivial case of batch_size=1, simply return the same tensor
    if batch_size == 1:
        mask = torch.ones((1, tensor.size(0)), dtype=torch.bool, device=tensor.device)
        lengths = torch.tensor([tensor.size(0)]).to(tensor.device)
        offsets = torch.zeros(1).to(tensor.device)
        return tensor.unsqueeze(0), mask, lengths, offsets

    # create indices for sparse tensor
    val2ptr = torch.index_select(ptr, dim=0, index=val2batchidx)  # gets ptr offset for each value in batch
    local_idx = torch.arange(val2batchidx.size(0)).to(val2ptr) - val2ptr
    sparse_idx = torch.stack([val2batchidx, local_idx], dim=0)  # 2 x N set of sparse indices

    # convert to sparse tensor, then rapidly convert to dense
    tensor_coo = torch.sparse_coo_tensor(sparse_idx, tensor, dtype=tensor.dtype, device=tensor.device)
    mask = torch.zeros((batch_size, pad_size), dtype=torch.bool, device=tensor.device)
    mask[sparse_idx[0], sparse_idx[1]] = True
    out = tensor_coo.to_dense()
    lengths = mask.sum(dim=-1)
    offsets = torch.cumsum(lengths, dim=0)[:-1]
    return out, mask, lengths, offsets


def batch2ptr(batch, with_ends=False):
    device, end_idx = batch.device, batch.size(0)+1
    assert torch.all(((batch[1:] - batch[:-1]) == 0) | ((batch[1:] - batch[:-1]) == 1))  # make sure is sorted
    ptr = torch.where(batch[1:] - batch[:-1] > 0)[0]
    ptr += 1
    if with_ends:
        ptr = torch.cat([torch.zeros(1).to(ptr), ptr, torch.tensor([batch.size(0)]).to(ptr)])
    return ptr


def compute_masked_psnr(
    img0: torch.tensor, img1: torch.tensor, mask: torch.tensor
) -> torch.tensor:
    """Compute PSNR between two images.

    Args:
        img0 (jnp.ndarray): An image of shape (H, W, 3) in float32.
        img1 (jnp.ndarray): An image of shape (H, W, 3) in float32.
        mask (Optional[jnp.ndarray]): An optional forground mask of shape (H,
            W, 1) in float32 {0, 1}. The metric is computed only on the pixels
            with mask == 1.

    Returns:
        jnp.ndarray: PSNR in dB of shape ().
    """
    mse = (img0 - img1) ** 2
    return -10.0 / np.log(10.0) * torch.log(torch.mean(mse[mask]))


def get_compute_lpips():
    """Get the LPIPS metric function.
    """
    model = lpips.LPIPS(net="alex", spatial=True)

    @torch.inference_mode()
    def compute_lpips(img0: torch.tensor, img1: torch.tensor, mask: torch.tensor) -> torch.tensor:
        """Compute LPIPS between two images.

        This function computes mean LPIPS over masked regions. The input images
        are also masked. The following previous works leverage this metric:

        [1] Neural Scene Flow Fields for Space-Time View Synthesis of Dynamic
        Scenes.
            Li et al., CVPR 2021.
            https://arxiv.org/abs/2011.13084

        [2] Transforming and Projecting Images into Class-conditional
        Generative Networks.
            Huh et al., CVPR 2020.
            https://arxiv.org/abs/2005.01703

        [3] Controlling Perceptual Factors in Neural Style Transfer.
            Gatys et al., CVPR 2017.
            https://arxiv.org/abs/1611.07865

        Args:
            img0 (jnp.ndarray): An image of shape (H, W, 3) in float32.
            img1 (jnp.ndarray): An image of shape (H, W, 3) in float32.
            mask (Optional[jnp.ndarray]): An optional forground mask of shape
                (H, W, 1) in float32 {0, 1}. The metric is computed only on the
                pixels with mask == 1.

        Returns:
            np.ndarray: LPIPS in range [0, 1] in shape ().
        """
        img0 = lpips.im2tensor((img0 * mask).cpu().numpy(), factor=1 / 2)
        img1 = lpips.im2tensor((img1 * mask).cpu().numpy(), factor=1 / 2)
        out = model(img0, img1)
        return out[0, 0, :, :].cpu()[mask.squeeze(-1).cpu()].mean()

    return compute_lpips


def get_compute_lpips_diff():
    """Get the LPIPS metric function.
    """
    model = lpips.LPIPS(net="alex", spatial=True)
    model = model.cuda()

    @torch.inference_mode()
    def compute_lpips(img0: torch.tensor, img1: torch.tensor, mask: torch.tensor) -> torch.tensor:
        img0 = ((img0 * mask) / 0.5 - 1.0).unsqueeze(-1).permute(3, 2, 0, 1).float()
        img1 = ((img1 * mask) / 0.5 - 1.0).unsqueeze(-1).permute(3, 2, 0, 1).float()
        out = model(img0, img1)
        return out[0, 0, :, :][mask.squeeze(-1)].mean()

    return compute_lpips



import torch
import torch.nn.functional as F
from typing import Optional

def compute_ssim(
    img0: torch.Tensor,
    img1: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    max_val: float = 1.0,
    filter_size: int = 11,
    filter_sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> torch.Tensor:
    if mask is None:
        mask = torch.ones_like(img0[..., :1])

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((torch.arange(filter_size) - hw + shift) / filter_sigma) ** 2
    filt = torch.exp(-0.5 * f_i)
    filt /= torch.sum(filt)
    filt = filt.to(img0.device)

    def convolve2d(z, m, f):
        rs = F.conv1d(z[:, :, 0:1].permute(0, 2, 1) * m.permute(0, 2, 1), f.view(1, 1, -1), padding=hw)
        gs = F.conv1d(z[:, :, 1:2].permute(0, 2, 1) * m.permute(0, 2, 1), f.view(1, 1, -1), padding=hw)
        bs = F.conv1d(z[:, :, 2:3].permute(0, 2, 1) * m.permute(0, 2, 1), f.view(1, 1, -1), padding=hw)
        m_ = F.conv1d(m.permute(0, 2, 1).float(), torch.ones_like(f).view(1, 1, -1), padding=hw).permute(0, 2, 1)
        z_ = torch.cat([rs, gs, bs], dim=1).permute(0, 2, 1)
        return torch.where(m_ != 0, z_ * torch.sum(f) / m_, 0), (m_ != 0).to(z.dtype)

    filt_fn1 = lambda z, m: convolve2d(z, m, filt)
    filt_fn2 = lambda z, m: convolve2d(z, m, filt.view(1, -1))

    # Vmap the blurs to the tensor size, and then compose them.
    def filt_fn(z, m):
        kernel_2d = filt.unsqueeze(0) * filt.unsqueeze(1)  # (K, K)
        # format image to (minibatch, in-channels, h, w), i.e. (3, 1, h, w)
        # format kernel to (out_channels, groups, k, k), i.e. (1, 1, k, k)
        img = F.conv2d(z.permute(2, 0, 1).unsqueeze(1), kernel_2d[None, None, :, :], padding=hw)
        mask = F.conv2d(m.permute(2, 0, 1).unsqueeze(1).float(), torch.ones_like(kernel_2d)[None, None, :, :], padding=hw)
        img = img * torch.sum(torch.ones_like(kernel_2d).sum()) / mask
        img[0, mask[0] == 0] = 0
        img[1, mask[0] == 0] = 0
        img[2, mask[0] == 0] = 0
        img = img.squeeze(1).permute(1, 2, 0)
        return img
        # return filt_fn1(*filt_fn2(z, m))

    mu0 = filt_fn(img0, mask)
    mu1 = filt_fn(img1, mask)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0**2, mask) - mu00
    sigma11 = filt_fn(img1**2, mask) - mu11
    sigma01 = filt_fn(img0 * img1, mask) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = torch.clip(sigma00, min=0)
    sigma11 = torch.clip(sigma11, min=0)
    sigma01 = torch.sign(sigma01) * torch.minimum(torch.sqrt(sigma00 * sigma11), torch.abs(sigma01))

    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = ssim_map.mean()

    return ssim
