from typing import Dict
import numpy as np

import torch
from torch import Tensor
from torchtyping import TensorType
from torch import nn
import torch.nn.functional as F

from src.models.sequence import GaussianSequence
from src.models.static_gaussians import GaussianStaticScene
from gsplat.rendering import rasterization
from src.utils.shutils import SH2RGB
from src.data.databundle import GaussianSplattingImageBundle
from src.utils.tracking_utils import TrackingLoss, get_local_cotracker_index


class GaussianField(nn.Module):

    def __init__(
            self,
            num_images: int,
            config,
            point_clouds: list[TensorType],
            background: Tensor = torch.tensor([1.0, 1.0, 1.0]),
            background_cls: Tensor = torch.tensor([0])
    ) -> None:
        super().__init__()

        if not config.no_background:
            foreground_cls = torch.tensor(list({i for i in range(16)} - set(background_cls.cpu().numpy().tolist())))
        else:
            foreground_cls = torch.tensor(list({i for i in range(16)}))
        self.foreground_field = GaussianSequence(num_images, config, point_clouds, config.number_of_gaussians, foreground_cls)
        if config.no_background:
            self.background_field = None
        elif config.static_background:
            num_background_pts = config.number_of_gaussians - self.foreground_field.maxpoints
            self.background_field = GaussianStaticScene(config, point_clouds, num_background_pts, background_cls)
        else:
            self.background_field = GaussianSequence(num_images, config, point_clouds, config.number_of_gaussians, background_cls)

        self.register_buffer("background_clr", background)  # registering as buffer will send it to("cuda") when called
        self.num_images = num_images
        self.config = config
        self.register_buffer("phase", torch.tensor(0, dtype=torch.long))
        self.register_buffer("learning_foreground", torch.tensor(True, dtype=torch.bool))
        self.register_buffer("learning_background", torch.tensor(True, dtype=torch.bool))

        # set up gaussians
        self.nframes = len(point_clouds)

        # temporary state -- keeps track of training params at each forward iteration
        self._render_background = True
        self._tmp_background_clr = torch.tensor([1.0, 1.0, 1.0])
        self._time = 0
        self._viewpoint_camera = None

        # initialize tracking loss module
        self.tracking_loss = TrackingLoss(self.config.tracking_knn, self.config.tracking_radius,
                                          self.config.tracking_loss_per_segment)

    def prepare_load_weights(self, weights):
        self.foreground_field.prepare_load_weights(weights, 'foreground_field')
        if not self.config.no_background:
            self.background_field.prepare_load_weights(weights, 'background_field')

    def forward(self, viewpoint_cam: GaussianSplattingImageBundle) -> Dict[str, Tensor]:
        # cache this iteration info
        self._time = viewpoint_cam.time
        self._viewpoint_camera = viewpoint_cam
        device = self.background_clr.device

        # get foreground gaussians to render (always happens)
        foreground_gproperties = self.foreground_field.nextstep_get_gaussians(viewpoint_cam)

        # get background properties
        self._render_background = (not self.training) or np.random.random() < self.config.render_background_probability\
            or (not self.config.static_background) or (self.learning_background and not self.learning_foreground)
        self._render_background = self._render_background or foreground_gproperties[0].size(0) == 0  # always render background if no foreground
        self._render_background = self._render_background and not self.config.no_background
        background_gproperties = None
        if self._render_background:
            background_gproperties = self.background_field.nextstep_get_gaussians(viewpoint_cam)  # , self.foreground_field._gaussian_idxs

        # merge foreground and background gaussians
        gproperties = self._merge_fg_gaussians(foreground_gproperties, background_gproperties)

        # if we're not rendering background gaussians, set scene background to either black or white
        rand_clr = torch.ones(3).to(device) if np.random.random() < 0.5 else torch.ones(3).to(device)
        self._tmp_background_clr = self.background_clr if self._render_background or self.config.no_background else rand_clr

        # forward pass through rendering module
        render_pkg = self.render(viewpoint_cam, gproperties, background_clr=self._tmp_background_clr)

        # format outputs
        image = render_pkg["render"]
        outputs = {}
        outputs.update({'rgb': image.permute(1, 2, 0)})
        outputs.update({'depthmap': render_pkg["depthmap"]})

        # process segmentation for vis
        if render_pkg["segmentation"].size(-1) < 3:
            (h, w, nseg), device = render_pkg["segmentation"].shape, render_pkg["segmentation"].device
            render_pkg["segmentation"] = torch.cat([render_pkg["segmentation"], torch.zeros(h, w, 3 - nseg).to(device)], dim=2)
        outputs.update({'segmentation': render_pkg["segmentation"]})
        return outputs

    def _merge_fg_gaussians(self, gaussians1, gaussians2):
        if gaussians2 is None:
            return gaussians1

        means3D = torch.cat([gaussians1[0], gaussians2[0]], dim=0)
        rotations = torch.cat([gaussians1[1], gaussians2[1]], dim=0)
        opacity = torch.cat([gaussians1[2], gaussians2[2]], dim=0)
        shs = torch.cat([gaussians1[3], gaussians2[3]], dim=0)
        scales = torch.cat([gaussians1[4], gaussians2[4]], dim=0)
        segmentation = torch.cat([gaussians1[5], gaussians2[5]], dim=0)
        return means3D, rotations, opacity, shs, scales, segmentation

    def render(self, viewpoint_camera, gaussians, background_clr=None):
        # get gaussian properties
        means3D, rotations, opacity, shs, scales, segmentation = gaussians

        # convert segmentations into 1-hot
        num_classes = int(segmentation.max().item()) + 1
        seg_feat = F.one_hot(segmentation.long().flatten(), num_classes=num_classes).float().view(-1, num_classes)
        rgb = SH2RGB(shs).view(-1, 3)
        feat = torch.cat([rgb, seg_feat], dim=1)

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        viewmat = viewpoint_camera.world_view_transform.clone().detach().cuda().T  # or nstudio_c2w
        fx, fy, cx, cy = viewpoint_camera.fx, viewpoint_camera.fy, viewpoint_camera.cx, viewpoint_camera.cy
        Ks = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).float().cuda()
        backgrounds = self.background_clr if background_clr is None else background_clr
        backgrounds = torch.cat([backgrounds, torch.zeros(num_classes).to(backgrounds)], dim=0).unsqueeze(0)
        rendered_image, render_alphas, info = rasterization(
            means=means3D,
            quats=rotations,
            scales=scales,
            opacities=opacity.flatten(),
            colors=feat,
            viewmats=viewmat[None, :, :],
            Ks=Ks[None, :, :],
            width=int(viewpoint_camera.image_width),
            height=int(viewpoint_camera.image_height),
            packed=True,
            absgrad=False,
            sparse_grad=False,
            sh_degree=None,
            render_mode="RGB+D",
            rasterize_mode='classic',
            backgrounds=backgrounds
        )

        rgb_render = rendered_image[0, :, :, :3].permute(2, 0, 1).contiguous()
        seg_render = rendered_image[0, :, :, 3: 3+num_classes]  # .permute(2, 0, 1).contiguous()
        depth_render = rendered_image[0, :, :, 3+num_classes]
        return {"render": rgb_render, "depthmap": depth_render, "segmentation": seg_render}

    def optimizer_step(self, step):
        if self.learning_foreground:
            self.foreground_field.optimizer_step(step)
        else:
            self.foreground_field.optimizer_zero_grad()

        if self._render_background and self.learning_background:
            self.background_field.optimizer_step(step)
        elif self._render_background and not self.learning_background:  # if we didn't render, no need to reset!
            self.background_field.optimizer_zero_grad()

    def randselect_active_training_frames(self, motion_training_window):
        if self.learning_foreground:
            return self.foreground_field.randselect_active_training_frames(motion_training_window)
        else:
            return self.background_field.randselect_active_training_frames(motion_training_window)

    def step_stage(self, new_phase, fgbg, scope_increase=None):
        if fgbg == 'foreground' or self.config.no_background:
            self.foreground_field.step_stage(new_phase, scope_increase)
            self.phase.data = self.foreground_field.phase.data
            self.learning_foreground |= True
            self.learning_background &= False
        elif fgbg == 'background':
            self.background_field.step_stage(new_phase, scope_increase)
            self.phase.data = self.background_field.phase.data
            self.learning_foreground &= False
            self.learning_background |= True
        elif fgbg == 'foreground-and-background':
            self.foreground_field.step_stage(new_phase, scope_increase)
            self.background_field.step_stage(new_phase, scope_increase)
            self.phase.data = self.foreground_field.phase.data
            self.learning_foreground |= True
            self.learning_background |= True
        else:
            raise RuntimeError("Must specify foreground and/or background for stage. Got %s" % fgbg)

    def step_motion_scope(self):
        if self.learning_foreground:
            self.foreground_field.step_motion_scope()

        if not self.config.static_background and not self.config.no_background and self.learning_background:
            self.background_field.step_motion_scope()

    # ========================================= LOSSES ============================================
    def compute_isometry_loss(self, knn, knn_radius, per_segment):
        errs = torch.zeros(1).to(self.background_clr.device)
        if self.learning_foreground:
            foreground_errs = self.foreground_field.compute_isometry_loss(knn, knn_radius, per_segment)
            errs = torch.cat([errs, foreground_errs], dim=0)
        if self.learning_background and not self.config.static_background and not self.config.no_background:
            bkgrnd_errs = self.background_field.compute_isometry_loss(knn, knn_radius, per_segment)
            errs = torch.cat([errs, bkgrnd_errs], dim=0)
        return errs.mean()

    def compute_chamfer_loss(self, agg_group_ratio):
        loss = torch.zeros(1).to(self.background_clr.device)
        if self.learning_foreground:
            loss += self.foreground_field.compute_chamfer_loss(agg_group_ratio)
        if self.learning_background and not self.config.static_background and not self.config.no_background:
            loss += self.background_field.compute_chamfer_loss(agg_group_ratio)
        return loss

    def compute_scaling_loss(self):
        loss = torch.zeros(1).to(self.background_clr.device)
        if self.learning_foreground:
            loss += self.foreground_field.compute_scaling_loss()
        if self._render_background and self.learning_background:
            loss += self.background_field.compute_scaling_loss()
        return loss

    def compute_velocity_smoothing_loss(self):
        loss = torch.zeros(1).to(self.background_clr.device)
        if self.learning_foreground:
            loss += self.foreground_field.compute_velocity_smoothing_loss()
        if self.learning_background and not self.config.static_background and not self.config.no_background:
            loss += self.background_field.compute_velocity_smoothing_loss()
        return loss

    def compute_instance_isometry_loss(self, num_pairs):
        loss = torch.zeros(1).to(self.background_clr.device)
        if self.learning_foreground:
            loss += self.foreground_field.compute_instance_isometry_loss(num_pairs)
        if self.learning_background and not self.config.static_background and not self.config.no_background:
            loss += self.background_field.compute_instance_isometry_loss(num_pairs)
        return loss

    def compute_tracking_loss(self, dataloader, datasampler, window=16):
        # load foreground info
        target_fidx = self._time
        target_camera = self._viewpoint_camera

        if self.learning_foreground:  # if learning foreground, always sample source and target from foreground
            _, (src_fidx, src_xyz, target_xyz, opacity, scales, segmentation) = \
                self.foreground_field.get_source_and_target_gaussians(window, target_fidx)

            # load and append background gaussians (only if background is not static)
            if self.learning_background and not self.config.static_background and not self.config.no_background:
                success, bkrnd = self.background_field.get_source_and_target_gaussians(window, target_fidx, src_fidx)
                if success:
                    _, b_src_xyz, b_target_xyz, b_opacity, b_scales, b_segmentation = bkrnd
                    src_xyz = torch.cat([src_xyz, b_src_xyz], dim=0)
                    target_xyz = torch.cat([target_xyz, b_target_xyz], dim=0)
                    opacity = torch.cat([opacity, b_opacity], dim=0)
                    scales = torch.cat([scales, b_scales], dim=0)
                    segmentation = torch.cat([segmentation, b_segmentation], dim=0)

        else:  # only learning background
            success, (src_fidx, src_xyz, target_xyz, opacity, scales, segmentation) = \
                self.background_field.get_source_and_target_gaussians(window, target_fidx)

        # load source camera and source tracks
        device = src_xyz.device
        src_camera, src_batch, _ = datasampler.sample(next(dataloader), valid_idxs=[src_fidx])
        src_camera = src_camera.to(device)
        src_tracks = src_batch['tracks'].float().to(device)
        src_track_seg = src_batch['tracks_segmentations'].float().to(device)
        if self.config.static_background and not self.config.no_background:  # remove background tracks
            src_tracks = src_tracks[:, src_track_seg != 0]
            src_track_seg = src_track_seg[src_track_seg != 0]

        # select source and target frames from tracks
        source_relative, target_relative = get_local_cotracker_index(self.nframes, src_fidx, target_fidx)
        src_particles = src_tracks[source_relative, :, :2]  # should all be visible
        target_particles = src_tracks[target_relative, :, :2]

        if src_xyz.size(0) == 0:
            return torch.zeros(1).to(device)

        tracking_loss = self.tracking_loss(src_camera, target_camera, src_xyz, target_xyz, opacity, scales,
                                           segmentation, src_particles, target_particles, src_track_seg)
        return tracking_loss

    # ========================================= OTHER FUNCTIONS ============================================
    def upsample_gaussians(self, factor, fgbg):
        if 'foreground' in fgbg:
            self.foreground_field.upsample_gaussians(factor)
        if 'background' in fgbg and not self.config.no_background:
            self.background_field.upsample_gaussians(factor)

    def get_means_2D(self, frameidx, camera):
        means2D = self.foreground_field.get_means_2D(frameidx, camera)
        if not self.config.static_background and not self.config.no_background:
            means2D_background = self.background_field.get_means_2D(frameidx, camera)
            means2D = torch.cat([means2D, means2D_background], dim=0)
        return means2D
