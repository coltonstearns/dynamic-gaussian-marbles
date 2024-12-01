

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import torchvision.transforms
from torchvision.transforms import functional as F
import torch.nn.functional as FT
from torch import Tensor
import wandb
import time
from copy import deepcopy

import numpy as np
import torch
from torch.nn import Parameter

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.model_components.losses import (
    L1Loss
)
from nerfstudio.models.base_model import Model, ModelConfig
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# from src.models.field import GaussianField
from src.models.field import GaussianField
from src.data.databundle import GaussianSplattingImageBundle
from src.utils.utils import fov2focal, getWorld2View2, getProjectionMatrix, compute_masked_psnr, get_compute_lpips, compute_ssim
# from src.models.gaussian_splatting.gsplatting_field.gsplatting_fieldheaders import GaussianSplattingFieldHeadNames
from nerfstudio.cameras import camera_utils
from src.visualization.render_trajectories import visualize_trajectories
from pytorch3d.ops import knn_points

WANDB_LOG_STEP = 0



@dataclass
class GaussianSplattingModelConfig(ModelConfig):
    """Gaussian Splatting Model Config"""

    _target: Type = field(default_factory=lambda: GaussianSplattingModel)
    """Spherical harmonics degree"""
    point_cloud_sequence = []  # : list[TensorType, ...]
    """ Sequence of initial point clouds, obtained from depth sensor or depth estimation."""

    # number of Guassians
    number_of_gaussians: int = 120000

    # adjustable learning rates
    delta_position_lr: float = 0.002
    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.002

    # parameters controlling our background
    no_background: bool = False
    static_background: bool = False
    render_background_probability: float = 0.25
    pretrained_background: str = ''

    # loss terms
    isometry_loss_weight: float = 10.0
    chamfer_loss_weight: float = 2.0
    photometric_loss_weight: float = 0.7
    tracking_loss_weight: float = 0.4
    segmentation_loss_weight: float = 0.1
    depthmap_loss_weight: float = 0.2
    lpips_loss_weight: float = 0.0
    instance_isometry_loss_weight: float = 2.0
    consistency_loss_weight: float = 0.0
    tracking_depth_loss_weight: float = 2.0

    # loss terms - not used in final model
    velocity_smoothing_loss_weight: float = 0.0
    scaling_loss_weight: float = 0.1

    # loss term hyperparameters
    isometry_knn: int = 32
    isometry_knn_radius: float = 0.3  # change me for synthetic?
    isometry_per_segment: bool = True
    isometry_use_l2: bool = False
    isometry_weight_background_factor: float = 1.0
    chamfer_agg_group_ratio: float = 0.125  # when dividing frames into 2 groups, what percentage of frames do we use?
    tracking_window: int = 12
    tracking_knn: int = 32
    tracking_radius: int = 8
    tracking_loss_per_segment: bool = True
    instance_isometry_numpairs: int = 4096

    # other training parameters
    supervision_downscale: int = 1  # supervise rendering losses on image downresoluted by 1x, 2x, 3x, etc.
    frame_transport_dropout: float = 0.5  # chance of dropping out frames
    freeze_previous_in_motion_estimation: bool = True
    freeze_frames_of_origin: bool = True

    # merge downsampling parameters
    prune_points: bool = True
    downsample_reducescale: float = 0.85

    # background color
    background: Tensor = torch.tensor([1.0, 1.0, 1.0])


class GaussianSplattingModel(Model):
    """Gaussian Splatting model

    Args:
        config: Gaussian Splatting configuration to instantiate model
    """

    config: GaussianSplattingModelConfig

    def __init__(
        self,
        config: GaussianSplattingModelConfig,
        scene_box,
        num_train_data,
        datamanager,
        **kwargs,
    ) -> None:
        self.scene_scale = datamanager.train_dataset.depth_unit_scale_factor
        super().__init__(config, scene_box, num_train_data, **kwargs)
        self.datamanager = datamanager
        self.corresponding_idxs = None

    def randselect_active_training_frames(self):
        return self.field.randselect_active_training_frames(motion_training_window=1)

    def step_motion_scope(self):
        self.field.step_motion_scope()

    def step_stage(self, stage, fgbg, scope_increase=None):
        self.field.step_stage(stage, fgbg, scope_increase)

    def prepare_load_weights(self, weights):
        self.field.prepare_load_weights(weights)

    def prune_points(self):
        self.field.prune_points()

    def upsample_gaussians(self, factor, fgbg):
        self.field.upsample_gaussians(factor, fgbg)

    def populate_modules(self):
        """Set the fields and modules."""
        # Set up Gaussian Field
        self.field = GaussianField(
            num_images=self.num_train_data,
            config=self.config,
            point_clouds=self.config.point_cloud_sequence,
            background=self.config.background,
            background_cls=torch.tensor([0]),  # hard code for now
            pretrained_background=self.config.pretrained_background,
            scene_scale=self.scene_scale
        )
        self.nframes = len(self.config.point_cloud_sequence)

        # losses
        self.l1_loss = L1Loss()
        self.lpips = get_compute_lpips()
        self.lpips_diff = LearnedPerceptualImagePatchSimilarity()

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        return param_groups

    def get_outputs(self, ray_bundle: GaussianSplattingImageBundle):
        outputs = self.field(ray_bundle)
        return outputs

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: [GaussianSplattingImageBundle, RayBundle], t=None) -> Dict[str, torch.Tensor]:
        # Note: the viewer will pass in a standard NeRF camera ray bundle, so we must convert it!
        correspondences = None
        if isinstance(camera_ray_bundle, RayBundle):
            tracking_tstep = camera_ray_bundle.metadata.get('viz_tracking', None)
            t = t+1 if t is not None else self.nframes
            camera_ray_bundle = nerfstudio_bundle_to_gsplatting_camera(camera_ray_bundle, t)
        else:
            tracking_tstep = None

        # render the ray bundle
        outputs = self.field(camera_ray_bundle)

        if tracking_tstep is not None:
            initial_gaussian_idx = int(tracking_tstep.flatten()[0].item())
            correspondences = self.render_correspondences(camera_ray_bundle, initial_gaussian_idx)

        # superimpose if we're visualizing correspondences
        if correspondences is not None:
            mask = torch.mean(correspondences.float(), dim=2) < 190  # where it's clearly not white
            correspondences = correspondences.float() / 255
            # make rgb output grayscale
            # outputs['rgb'] = torch.mean(outputs['rgb'], dim=2, keepdim=True).repeat(1, 1, 3)
            outputs['rgb'][mask] = correspondences[mask]

        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None, dataloader=None, data_sampler=None, ray_bundle=None):
        # compute l1 loss
        gt_image = batch["image"].to(self.device).squeeze(0).clone()
        h, w = gt_image.size(0), gt_image.size(1)

        # remove background if we didn't render it:
        if not self.field._render_background or self.config.no_background:
            background_color = self.field._tmp_background_clr
            background_mask = batch['segmentation'].clone() == 0
            gt_image[background_mask.to(self.device)] = background_color
            gt_image = gt_image.detach().clone()  # NOTE: VERY IMPORTANT TO DETACH AND CLONE, OR ELSE CUDA ERROR!!
            batch["depth_image"][background_mask] = 0.0
            batch["depth_image"] = batch["depth_image"].detach().clone()

        # if appropriate, downscale image before computing loss
        if self.config.supervision_downscale > 1:
            h, w = h // self.config.supervision_downscale, w // self.config.supervision_downscale
            outputs["rgb"] = F.resize(outputs["rgb"].permute(2, 0, 1), size=[h, w]).permute(1, 2, 0)
            outputs['depthmap'] = F.resize(outputs['depthmap'].unsqueeze(0), size=[h, w]).squeeze(0)
            outputs['segmentation'] = F.resize(outputs['segmentation'].permute(2, 0, 1), size=[h, w]).permute(1, 2, 0)

            gt_image = F.resize(gt_image.permute(2, 0, 1), size=[h, w]).permute(1, 2, 0)
            batch['depth_image'] = F.resize(batch['depth_image'].unsqueeze(0), size=[h, w], interpolation=F.InterpolationMode.NEAREST).squeeze(0)
            batch['segmentation'] = F.resize(batch['segmentation'].unsqueeze(0), size=[h, w], interpolation=F.InterpolationMode.NEAREST).squeeze(0)

        # compute L1 Photometric loss
        photometric_loss = self.l1_loss(outputs["rgb"], gt_image)
        photometric_loss = self.config.photometric_loss_weight * photometric_loss

        # compute LPIPs loss
        if self.config.lpips_loss_weight > 0:
            img0 = (outputs["rgb"].float() * 2 - 1.0).unsqueeze(-1).permute(3, 2, 0, 1).float()
            img1 = (gt_image.float() * 2 - 1.0).unsqueeze(-1).permute(3, 2, 0, 1).float()
            img0, img1 = torch.clip(img0, -1.0, 1.0), torch.clip(img1, -1.0, 1.0)
            lpips_loss = self.lpips_diff(img0, img1) * self.config.lpips_loss_weight
        else:
            lpips_loss = torch.zeros(1).to(self.device)

        # compute segmentation loss
        seg_loss = torch.zeros(1).to(self.device)
        if self.config.segmentation_loss_weight > 0:
            pred_seg = outputs['segmentation']  # h, w, K
            num_classes = max(batch['segmentation'].unique().numel(), pred_seg.size(2))
            gt_seg = torch.nn.functional.one_hot(batch['segmentation'].to(self.device).long(), num_classes=num_classes).squeeze(0)
            if num_classes > pred_seg.size(2):
                pred_seg = torch.cat([pred_seg, torch.zeros(h, w, num_classes - pred_seg.size(2)).to(self.device)], dim=2)
            if not self.field._render_background or self.config.no_background:
                gt_seg[gt_seg[..., 0] == 1] = 0  # make background "black"
            gt_seg = F.resize(gt_seg.permute(2, 0, 1), size=[h, w]).permute(1, 2, 0)
            seg_loss = self.config.segmentation_loss_weight * self.l1_loss(pred_seg, gt_seg)
            # l1_viz = torch.abs(pred_seg - gt_seg).mean(dim=-1)
            # l1_viz = l1_viz / l1_viz.max()
            # cv2.imwrite("l1_seg_viz_%s.png" % np.random.randint(0, 10), (l1_viz * 255).cpu().detach().numpy().astype(np.uint8))

        # compute in isometry loss
        isometry_loss = torch.zeros(1).to(photometric_loss)
        if self.config.isometry_loss_weight > 0:
            isometry_loss = self.field.compute_isometry_loss(self.config.isometry_knn, self.config.isometry_knn_radius,
                                                             self.config.isometry_per_segment, self.config.isometry_use_l2,
                                                             self.config.isometry_weight_background_factor)
            isometry_loss = self.config.isometry_loss_weight * isometry_loss

        # add in chamfer loss
        chamfer_loss = torch.zeros(1).to(photometric_loss)
        if self.config.chamfer_loss_weight > 0:
            chamfer_loss = self.field.compute_chamfer_loss(self.config.chamfer_agg_group_ratio)
            chamfer_loss = self.config.chamfer_loss_weight * chamfer_loss

        # add in tracking loss
        tracking_loss = torch.zeros(1).to(chamfer_loss)
        tracking_depth_loss = torch.zeros(1).to(chamfer_loss)
        if self.config.tracking_loss_weight > 0 or self.config.tracking_depth_loss_weight > 0:
            assert dataloader is not None and data_sampler is not None
            gt_depthmap = batch['depth_image'].squeeze(-1).to(self.device)  # note: if anything is 0, ignore it
            tracking_loss, tracking_depth_loss = self.field.compute_tracking_loss(dataloader, data_sampler, self.config.tracking_window, gt_depthmap)
            tracking_loss = self.config.tracking_loss_weight * tracking_loss
            tracking_depth_loss = self.config.tracking_depth_loss_weight * tracking_depth_loss

        # add in global rigidity loss
        instance_iso_loss = torch.zeros(1).to(photometric_loss)
        if self.config.instance_isometry_loss_weight > 0:
            instance_iso_loss = self.field.compute_instance_isometry_loss(self.config.instance_isometry_numpairs)
            instance_iso_loss = self.config.instance_isometry_loss_weight * instance_iso_loss

        # compute depth map loss
        depth_loss = torch.zeros(1).to(photometric_loss)
        if self.config.depthmap_loss_weight > 0:
            pred_depthmap = outputs['depthmap']  # note: if anything is 0, ignore it (indicates transparent)
            gt_depthmap = batch['depth_image'].squeeze(-1).to(self.device)  # note: if anything is 0, ignore it
            ignore_mask = (pred_depthmap == 0) | (gt_depthmap == 0)
            pred_disp, gt_disp = torch.clip(1.0 / pred_depthmap[~ignore_mask], 0, 1/1e-3), torch.clip(1.0 / gt_depthmap[~ignore_mask], 0, 1/1e-3)
            depth_loss = self.config.depthmap_loss_weight * self.l1_loss(pred_disp, gt_disp)
            # viz_gt_depthmap = gt_depthmap.clone().detach()
            # viz_pred_depthmap = pred_depthmap.clone().detach()
            # viz_gt_depthmap[ignore_mask] = 0.0
            # viz_pred_depthmap[ignore_mask] = 0.0
            # l1_viz = torch.abs(viz_pred_depthmap - viz_gt_depthmap)
            # l1_viz = l1_viz / l1_viz.max()
            # randval = np.random.randint(0, 30)
            # cv2.imwrite("%s_l1_viz_depth.png" % randval, (l1_viz * 255).cpu().detach().numpy().astype(np.uint8))
            # cv2.imwrite("%s_gt_viz_depth.png" % randval, (torch.clip(gt_depthmap / 10, 0, 1) * 255).cpu().detach().numpy().astype(np.uint8))
            # cv2.imwrite("%s_pred_viz_depth.png" % randval, (torch.clip(pred_depthmap / 10, 0, 1) * 255).cpu().detach().numpy().astype(np.uint8))
            # viz_gt_disp = torch.clip(1.0 / gt_depthmap, 0, 1/1e-3).clone().detach()
            # viz_pred_disp = torch.clip(1.0 / pred_depthmap, 0, 1/1e-3).clone().detach()
            # l1_viz_disp = torch.abs(viz_gt_disp - viz_pred_disp)
            # viz_gt_disp[ignore_mask] = 0.0
            # viz_pred_disp[ignore_mask] = 0.0
            # l1_viz_disp = l1_viz_disp / l1_viz_disp.max()
            # cv2.imwrite("%s_l1_viz_disp.png" % randval, (l1_viz_disp * 255).cpu().detach().numpy().astype(np.uint8))
            # cv2.imwrite("%s_gt_viz_disp.png" % randval, (torch.clip(viz_gt_disp / 10, 0, 1) * 255).cpu().detach().numpy().astype(np.uint8))
            # cv2.imwrite("%s_pred_viz_disp.png" % randval, (torch.clip(viz_pred_disp / 10, 0, 1) * 255).cpu().detach().numpy().astype(np.uint8))

        # compute velocity smoothing loss
        velocity_smoothing_loss = torch.zeros(1).to(chamfer_loss)
        if self.config.velocity_smoothing_loss_weight > 0:
            velocity_smoothing_loss = self.field.compute_velocity_smoothing_loss()
            velocity_smoothing_loss = self.config.velocity_smoothing_loss_weight * velocity_smoothing_loss

        # compute Gaussian scale loss
        scaling_loss = torch.zeros(1).to(chamfer_loss)
        if self.config.scaling_loss_weight > 0:
            scaling_loss = self.field.compute_scaling_loss()
            scaling_loss = self.config.scaling_loss_weight * scaling_loss


        consistency_loss = torch.zeros(1).to(chamfer_loss)
        if self.config.consistency_loss_weight > 0 and ray_bundle is not None:
            random_t = (torch.randn(3) * 0.002).to(chamfer_loss)
            dx, dy = self.compute_pixel_translation(ray_bundle, random_t)
            translated_ray_bundle = deepcopy(ray_bundle)
            translated_ray_bundle.camera_center += torch.tensor(random_t, dtype=torch.float32).to(translated_ray_bundle.camera_center.device)
            t_tensor = torch.tensor(random_t, dtype=torch.float32).to(translated_ray_bundle.nstudio_c2w.device).unsqueeze(-1)  # Shape (3, 1)
            translated_ray_bundle.nstudio_c2w[..., :3, 3] += t_tensor.squeeze(-1)
            translated_ray_bundle.world_view_transform[..., :3, 3] += t_tensor.squeeze(-1)

            translated_bundle_out = self.field(translated_ray_bundle)
            
            translated_output = self.translate_image(outputs["rgb"], dx, dy)
            # translated mask marking out the pixels that should not be counted for loss because they are outside the image
            translated_mask = self.translate_image(torch.ones_like(outputs["rgb"]), dx, dy)

            consistency_loss = torch.mean(torch.abs(translated_output - translated_bundle_out["rgb"]) * translated_mask)

            consistency_loss = self.config.consistency_loss_weight * consistency_loss

        # perform backward pass
        loss = photometric_loss + isometry_loss + chamfer_loss + tracking_loss + depth_loss + velocity_smoothing_loss + lpips_loss + instance_iso_loss + scaling_loss + tracking_depth_loss + consistency_loss
        loss.backward(retain_graph=False)

        # record losses
        loss_dict = {}
        loss_dict["total_loss"] = loss
        loss_dict["photometric"] = photometric_loss
        loss_dict["isometry"] = isometry_loss
        loss_dict["chamfer"] = chamfer_loss
        loss_dict["tracking"] = tracking_loss
        loss_dict["segmentation"] = seg_loss
        loss_dict["depth"] = depth_loss
        loss_dict["velocity_smoothing"] = velocity_smoothing_loss
        loss_dict["global_rigidity_loss"] = instance_iso_loss
        loss_dict["lpips_loss"] = lpips_loss
        loss_dict["scaling_loss"] = scaling_loss
        loss_dict["tracking_depth_loss"] = tracking_depth_loss
        loss_dict["consistency_loss"] = consistency_loss

        # log losses into wandb every 10 steps (because there are so many steps)
        global WANDB_LOG_STEP
        if WANDB_LOG_STEP % 10 == 0:
            wandb.log(loss_dict)
        WANDB_LOG_STEP += 1

        loss_dict = {'rgb_dummy': torch.zeros(1, requires_grad=True).to(photometric_loss)}
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:

        # Compute L1 RGB loss
        image = batch["image"].to(self.device).clone()
        eval_mask = batch["eval_mask"].to(self.device)
        rgb = outputs["rgb"].clone()
        images_dict = {"img": rgb}

        # resize image and prediction to match that of our evaluation mask
        h, w = eval_mask.size()
        image = torchvision.transforms.Resize((h, w))(torch.moveaxis(image, -1, 0))
        image = torch.moveaxis(image, 0, -1)
        rgb = torchvision.transforms.Resize((h, w))(torch.moveaxis(rgb, -1, 0))
        rgb = torch.moveaxis(rgb, 0, -1)

        # compute masked psnr, lpips, and ssim
        psnr = compute_masked_psnr(image, torch.clip(rgb, 0.0, 1.0), eval_mask.clone())
        lpips = self.lpips(image, torch.clip(rgb, 0.0, 1.0), eval_mask.unsqueeze(-1).clone())
        ssim = compute_ssim(image, torch.clip(rgb, 0.0, 1.0), eval_mask.unsqueeze(-1).clone())

        # compute non-masked psnr, lpips, and ssim
        nomask_psnr = compute_masked_psnr(image, torch.clip(rgb, 0.0, 1.0), torch.ones_like(eval_mask))
        nomask_lpips = self.lpips(image, torch.clip(rgb, 0.0, 1.0), torch.ones_like(eval_mask.unsqueeze(-1)))
        nomask_ssim = compute_ssim(image, torch.clip(rgb, 0.0, 1.0), torch.ones_like(eval_mask.unsqueeze(-1)))
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim), "lpips": float(lpips),
                        "nomask_psnr": float(nomask_psnr.item()), "nomask_ssim": float(nomask_ssim),
                        "nomask_lpips": float(nomask_lpips)}  # type: ignore

        return metrics_dict, images_dict

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []

        # callback that applies flow update to point positions and velocities
        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=self.field.optimizer_step,
            )
        )

        # callback that applies flow update to point positions and velocities
        # callbacks.append(
        #     TrainingCallback(
        #         where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
        #         update_every_num_iters=self.config.stage_transition_every,
        #         func=self.field.step_stage,
        #     )
        # )

        return callbacks

    def eval(self):
        r"""Sets the module in evaluation mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        This is equivalent with :meth:`self.train(False) <torch.nn.Module.train>`.

        See :ref:`locally-disable-grad-doc` for a comparison between
        `.eval()` and several similar mechanisms that may be confused with it.

        Returns:
            Module: self
        """
        self.field.eval()
        return self.train(False)

    def train(self, mode: bool = True):
        r"""Sets the module in training mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                         mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.field.train(mode)
        return self

    def render_correspondences(self, camera_ray_bundle, initial_gaussian_idx):
        PREVIOUS_WINDOW = 5
        NPOINTS = 200

        # get sequence of corresponding 3D points
        interval = self.field.foreground_field.gaussians[initial_gaussian_idx].get_xyz.size(0) // NPOINTS
        tracked_idxs = torch.arange(NPOINTS) * interval
        # corresponding_idxs = self.glue_correspondences()

        means3d_sequence = []
        for i in range(PREVIOUS_WINDOW):
            frameidx = max(0, camera_ray_bundle.time - i)
            gaussian_idx = self.field.foreground_field.frameidx2gaussianidx[frameidx].item()
            g = self.field.foreground_field.gaussians[gaussian_idx]
            means3d = g.get_properties(time_idx=frameidx)[1]
            if self.corresponding_idxs is not None:
                corresponding_pnts = means3d[self.corresponding_idxs[gaussian_idx][tracked_idxs]]
            else:
                corresponding_pnts = means3d[tracked_idxs]
            means3d_sequence.append(corresponding_pnts)

        # get image rendering
        h, w = int(camera_ray_bundle.image_height / self.config.supervision_downscale), int(camera_ray_bundle.image_width / self.config.supervision_downscale)
        fx, fy = fov2focal(camera_ray_bundle.FoVx, w), fov2focal(camera_ray_bundle.FoVy, h)
        K = np.array([[fx, 0, w / 2], [0, fy, h / 2], [0, 0, 1]])
        w2c = camera_ray_bundle.world_view_transform.cpu().numpy()
        img = visualize_trajectories(means3d_sequence, w2c, K, h, w)

        return img

    def compute_correspondences(self):
        corresponding_idxs = self.glue_correspondences()
        self.corresponding_idxs = corresponding_idxs

    def translate_image(self, image, dx, dy):
        """
        Translate a 2D RGB image by [dx, dy].

        Args:
            image: A torch tensor of shape (H, W, 3) representing the RGB image.
            dx: The translation in the x-direction (pixels).
            dy: The translation in the y-direction (pixels).

        Returns:
            Translated image tensor of shape (H, W, 3).
        """
        # Ensure the image has a batch and channel dimension
        if image.dim() == 3:
            image = image.permute(2, 0, 1).unsqueeze(0)  # Shape: (1, 3, H, W)

        # Get image dimensions
        _, _, H, W = image.shape

        # Create normalized grid
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=image.device),
            torch.linspace(-1, 1, W, device=image.device),
            indexing='ij'
        )
        grid = torch.stack((x, y), dim=-1)  # Shape: (H, W, 2)

        # Compute normalized translation
        norm_dx = dx / (W / 2)
        norm_dy = dy / (H / 2)

        # Apply the translation to the grid
        translated_grid = grid.clone()
        translated_grid[..., 0] -= norm_dx  # x-coordinate
        translated_grid[..., 1] -= norm_dy  # y-coordinate

        # Apply the grid transformation using grid_sample
        translated_image = FT.grid_sample(
            image,
            translated_grid.unsqueeze(0),  # Add batch dimension
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )

        # Remove batch and channel dimensions if necessary
        translated_image = translated_image.squeeze(0).permute(1, 2, 0)  # Shape: (H, W, 3)

        return translated_image
    
    def compute_pixel_translation(self, ray_bundle, t):
        """
        Compute the 2D pixel translation [dx, dy] caused by a 3D translation t=[x, y, z].
        
        Args:
            ray_bundle: An instance of RayBundle with the required attributes.
            t: A 3D translation vector [x, y, z] (Tensor of shape (3,)).
        
        Returns:
            dx, dy: The 2D pixel translation caused by the 3D translation.
        """
        # Extract attributes
        fx, fy = ray_bundle.fx, ray_bundle.fy
        cx, cy = ray_bundle.cx, ray_bundle.cy
        c2w = ray_bundle.nstudio_c2w  # Camera-to-world matrix (batch x 4 x 4)
        camera_center = ray_bundle.camera_center  # Camera center in world space (batch x 3)
        
        # Convert translation t to a tensor if not already
        t = torch.tensor(t, dtype=torch.float32).to(camera_center.device)

        # Transform the camera center by the translation t
        translated_camera_center = camera_center + t

        # Project original camera center to 2D
        camera_center_h = torch.cat([camera_center, torch.ones_like(camera_center[..., :1])], dim=-1)  # Homogeneous coords
        camera_center_proj = c2w @ camera_center_h.unsqueeze(-1)  # Shape: (batch x 4 x 1)
        camera_center_proj = camera_center_proj.squeeze(-1)  # Shape: (batch x 4)

        # Apply projection (normalized device coordinates -> pixel coordinates)
        x_ndc = camera_center_proj[..., 0] / camera_center_proj[..., 2]
        y_ndc = camera_center_proj[..., 1] / camera_center_proj[..., 2]
        x_pixel = fx * x_ndc + cx
        y_pixel = fy * y_ndc + cy
        original_pixel = torch.stack([x_pixel, y_pixel], dim=-1)
    
        # Project translated camera center to 2D
        translated_center_h = torch.cat([translated_camera_center, torch.ones_like(translated_camera_center[..., :1])], dim=-1)
        translated_center_proj = c2w @ translated_center_h.unsqueeze(-1)  # Shape: (batch x 4 x 1)
        translated_center_proj = translated_center_proj.squeeze(-1)  # Shape: (batch x 4)

        x_ndc_t = translated_center_proj[..., 0] / translated_center_proj[..., 2]
        y_ndc_t = translated_center_proj[..., 1] / translated_center_proj[..., 2]
        x_pixel_t = fx * x_ndc_t + cx
        y_pixel_t = fy * y_ndc_t + cy
        translated_pixel = torch.stack([x_pixel_t, y_pixel_t], dim=-1)

        # Compute the 2D pixel translation
        pixel_translation = translated_pixel - original_pixel
        dx, dy = pixel_translation[..., 0], pixel_translation[..., 1]

        return dx, dy


    def glue_correspondences(self, context_frames=4, start_frame_idx=0):
        # find gaussian idx to start from
        start_gaussian_idx = self.field.foreground_field.frameidx2gaussianidx[start_frame_idx].cpu()
        rolling_correspondences = torch.arange(self.field.foreground_field.gaussians[start_gaussian_idx].get_xyz.shape[0])
        corresponding_idxs = [rolling_correspondences.clone()]
        for i in range(start_gaussian_idx, len(self.field.foreground_field.gaussians)-1):
            print("Gluing together tracks from gaussian cluster %s and %s" % (i, i+1))
            this_g, next_g = self.field.foreground_field.gaussians[i], self.field.foreground_field.gaussians[i+1]

            # active_frames_orig = this_g.learned_motions.sum(dim=0).nonzero().flatten()
            active_frames_orig = torch.unique(this_g._batch).flatten()

            this_boundary, next_boundary = active_frames_orig.max().item(), active_frames_orig.max().item() + 1
            # this_boundary, next_boundary = torch.max(this_g.active_frames.clone()).item(), torch.min(next_g.active_frames.clone()).item()
            this_means_3d, this_means_2d, this_frame_idxs = [], [], []
            context_frames_valid = 0
            for k in range(context_frames):
                # get backtracked time index
                time_idx = this_boundary - k
                if time_idx not in active_frames_orig:
                    continue
                this_frame_idxs.append(time_idx)

                # get 3D means
                m3d = this_g.get_properties(time_idx=time_idx)[1].clone().detach().cpu()
                this_means_3d.append(m3d)

                # get 2D means as well
                camera_ray_bundle, batch = self.datamanager.fixed_indices_train_dataloader.get_data_from_image_idx(time_idx)
                frame_means_2d = self.field.foreground_field.get_means_2D(time_idx, camera_ray_bundle).detach().clone().cpu()
                this_means_2d.append(frame_means_2d)
                context_frames_valid += 1
            context_frames = context_frames_valid

            # merge all info
            this_means_3d = torch.stack(this_means_3d)  # [context_frames, num_gaussians, 3]
            this_means_2d = torch.stack(this_means_2d)  # [context_frames, num_gaussians, 2]
            this_seg = this_g.get_segmentation.unsqueeze(-1).float().detach().cpu()  # [context_frames, num_gaussians]
            this_frame_idxs = torch.tensor(this_frame_idxs).to('cpu')  # [context_frames]
            frame_min, frame_range = this_frame_idxs.min(), max(1, this_frame_idxs.max() - this_frame_idxs.min())
            this_frame_idxs = (this_frame_idxs - frame_min) / frame_range

            # compute std normalization for
            means3D_reg = this_means_3d.std(dim=1).mean() * 3  # make weaker!
            means2D_reg = this_means_2d.std(dim=1).mean()
            seg_reg = 1 / 50.0  # make segmentation matching score really high!

            # forecast this gaussian collection forward in time
            # A = torch.stack([torch.ones_like(this_frame_idxs), this_frame_idxs, this_frame_idxs ** 2], dim=1).to(self.device)  # context_frames x 3
            A = torch.stack([torch.ones_like(this_frame_idxs), this_frame_idxs], dim=1).to('cpu')  # context_frames x 3
            if torch.linalg.matrix_rank(A) == 2:
                A = A.unsqueeze(0).float()  # 1 x context_frames x 3

                # set up b for Ax=b
                all_means = torch.cat([this_means_3d, this_means_2d], dim=2)  # context_frames x num_gaussians x 5
                all_means = all_means.permute(1, 2, 0).reshape(-1, context_frames)  # num_gaussians*5 x context_frames
                all_means = all_means.unsqueeze(-1)  # num_gaussians*5 x context_frames x 1
                b = all_means

                # solve Ax=b
                x, residuals, _, _ = torch.linalg.lstsq(A, b)  # num_gaussians*5 x 3 x 1

                # forecast everything
                next_boundary_reg = (next_boundary - frame_min) / frame_range
                # A_forecast = torch.tensor([[1, next_boundary_reg, next_boundary_reg**2]]).to(self.device)
                A_forecast = torch.tensor([[1, next_boundary_reg]]).to('cpu')
                A_forecast = A_forecast.unsqueeze(0).float()
                forecasted_means = A_forecast @ x  # num_gaussians*5 x 1 x 1
                forecasted_means = forecasted_means.view(-1, 5)  # num_gaussians x 5
            else:
                forecasted_means = torch.cat([this_means_3d[0], this_means_2d[0]], dim=1)  # num_gaussians x 5

            this_features = torch.cat([forecasted_means[:, :3] / means3D_reg,
                                       forecasted_means[:, 3:] / means2D_reg,
                                       this_seg / seg_reg], dim=1)  # num_gaussians x 6

            # get next gaussian means and seg
            next_means_3d = next_g.get_properties(time_idx=next_boundary)[1].clone().detach().cpu()
            next_camera_ray_bundle, next_batch = self.datamanager.fixed_indices_train_dataloader.get_data_from_image_idx(next_boundary)
            next_means_2d = self.field.foreground_field.get_means_2D(next_boundary, next_camera_ray_bundle).detach().clone().cpu()
            next_seg = next_g.get_segmentation.unsqueeze(-1).float().clone().detach().cpu()
            next_features = torch.cat([next_means_3d / means3D_reg,
                                       next_means_2d / means2D_reg,
                                       next_seg / seg_reg], dim=1)  # num_gaussians x 6

            # compute matching based on features (KNN=1)
            dists2, cur2next_idxs, _ = knn_points(this_features[None].clone().cuda(), next_features[None].clone().cuda(), K=1)
            cur2next_idxs = cur2next_idxs.flatten()  # size (num_gaussians,), idxs of the nearest neighbor in next_features
            rolling_correspondences = cur2next_idxs[rolling_correspondences].clone().detach().cpu()
            corresponding_idxs.append(rolling_correspondences)

        return corresponding_idxs


def nerfstudio_bundle_to_gsplatting_camera(camera_ray_bundle, nframes=0):
    ''' Converts nerfstudio camera ray bundle into Gaussian Splatting format '''
    # step 1: recover image width, height, cx, cy from ray bundle info
    image_height, image_width = camera_ray_bundle.origins.shape[:2]
    cx = image_width / 2
    cy = image_height / 2

    # step 1.5: recover the time parameters if applicable
    if camera_ray_bundle.times is None:
        time = 0
    else:
        time = camera_ray_bundle.times[0, 0].item()
        time = min(round(time * nframes), nframes-1)

    # step 2: recover fov_x, fov_y from ray bundle info
    ray1_x, ray2_x = camera_ray_bundle.directions[int(cy), 0, [0, 1, 2]], camera_ray_bundle.directions[int(cy), -1, [0, 1, 2]]
    assert np.isclose(torch.linalg.norm(ray1_x).item(), 1.0)
    assert np.isclose(torch.linalg.norm(ray2_x).item(), 1.0)
    fov_x = torch.arccos(torch.dot(ray1_x, ray2_x)).item()
    ray1_y, ray2_y = camera_ray_bundle.directions[0, int(cx), [0, 1, 2]], camera_ray_bundle.directions[-1, int(cx), [0, 1, 2]]
    fov_y = torch.arccos(torch.dot(ray1_y, ray2_y)).item()
    # NOTE: USING CX AND CY IS AN APPROXIMATION --> THIS IS NOT COMPLETELY PRECISE!

    # step 3: recover fx, fy from fov info
    fx = fov2focal(fov_x, image_width)
    fy = fx

    # step 4: recover global translation
    translation = camera_ray_bundle.origins[0, 0]  # size (3, ), goes from camera to world
    # znear, zfar = camera_ray_bundle.nears[0, 0].item(), camera_ray_bundle.fars[0, 0].item()f
    znear = 0.01
    zfar = 100.0

    # step 4: get camera-space (i.e. image-origin centered) xyz directions from focal length
    pixel_offset = 0.5
    image_coords = torch.meshgrid(torch.arange(image_height), torch.arange(image_width), indexing="ij")
    image_coords = torch.stack(image_coords, dim=-1).to(camera_ray_bundle.origins.device) + pixel_offset  # stored as (y, x) coordinates  (h, w, 2)
    y = image_coords[..., 0]  # (num_rays,) get rid of the last dimension
    x = image_coords[..., 1]  # (num_rays,) get rid of the last dimension
    coord = torch.stack([(x - cx) / fx, -(y - cy) / fy, -torch.ones(x.size()).to(x)], -1)  # (num_rays, 2)  # xzy coords in camera ref frame
    directions, directions_norm = camera_utils.normalize_with_norm(coord, -1)

    # step 5: get world-space normalized directions
    ray_directions_world = camera_ray_bundle.directions

    # step 6: solve lstsq for rotation
    lt, rt, lb, rb = 0, image_width-1, image_height*image_width - image_width - 1, image_height*image_width-1
    A, B = directions.view(-1, 3)[[lt, rt, lb, rb], :], ray_directions_world.view(-1, 3)[[lt, rt, lb, rb], :]  # solve for A in Ax = b
    rotation = torch.linalg.lstsq(A, B)[0].T

    # step 7: put together into nerfstudio c2w matrix
    c2w = torch.eye(4, 4)
    c2w[:3, :3] = rotation
    c2w[:3, 3] = translation

    # step 8: convert into Guassian Splatting format
    c2w[:3, 1:3] *= -1  # opengl --> COLMAP
    w2c = torch.linalg.inv(c2w)
    R = w2c[:3, :3]
    T = w2c[:3, 3]
    R = R.T  # R is stored transposed due to 'glm' in CUDA code

    # convert Rt into final transforms
    trans, scale = torch.tensor([0., 0., 0.]), 1.0
    world_view_transform = getWorld2View2(R, T, trans, scale).transpose(0, 1)
    projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fov_x, fovY=fov_y).transpose(0, 1)
    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
    raster_info = GaussianSplattingImageBundle(
        image_width=image_width,
        image_height=image_height,
        FoVx=fov_x,
        FoVy=fov_y,
        znear=znear,
        zfar=zfar,
        world_view_transform=world_view_transform,
        full_proj_transform=full_proj_transform,
        time=time,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy
    )

    return raster_info.to(camera_ray_bundle.origins.device)