import gc
import torch
from torch import Tensor
import torch.nn as nn
from torchtyping import TensorType
from src.utils.utils import PHASE_NAMES
from src.models.gaussians import GaussianModel


class GaussianStaticScene(nn.Module):

    def __init__(
        self,
        config,
        point_clouds: list[TensorType],
        maxpoints: int,
        seg_classes: Tensor = torch.tensor([0, 1, 2, 3]),
    ) -> None:
        super().__init__()

        self.config = config
        self.register_buffer("maxpoints",  torch.tensor(maxpoints))
        self.register_buffer("phase", torch.tensor(0, dtype=torch.long))
        self.gaussian_set = self.initialize_gaussian_set(point_clouds, seg_classes, maxpoints)

    def initialize_gaussian_set(self, point_clouds, seg_classes, maxpoints):
        aggregate_pc = []
        for i, pc in enumerate(point_clouds):
            valid = torch.isin(pc[:, 3], seg_classes)
            aggregate_pc.append(pc[valid].clone())

        # initialize one set of static Gaussians
        aggregate_pc = torch.cat(aggregate_pc, dim=0)
        gaussians = self._initialize_gaussian_set(aggregate_pc, maxpoints)
        return gaussians

    def _initialize_gaussian_set(self, properties, maxpnts):
        """
        properties: [N, 8] tensor of properties [x, y, z, segmentation, r, g, b, depth]
        frameidx: frame timestep that this Gaussian set is initialized from
        """
        # perform random downsampling
        idxs = torch.randperm(properties.size(0))[:maxpnts]
        properties = properties[idxs]

        # reorder points by segmentation
        segmentation_reordering = torch.argsort(properties[:, 7])
        properties = properties[segmentation_reordering]

        # create gaussian splat of 1 static frame
        g = GaussianModel(num_frames=1, freeze_frames_of_origin=self.config.freeze_frames_of_origin)
        g.create_from_input_pcd(properties[:, :3], rgb=properties[:, 4:7], frameidx=0, segmentation=properties[:, 3])
        g.training_setup(self.config)
        g.optimize_all_except_motion()
        return g

    def prepare_load_weights(self, weights, field_name='background_field'):
        static_bkgrnd_key = '_model.field.{0}.gaussian_set'.format(field_name)
        gaussian_set = self._prepare_gaussian_set(static_bkgrnd_key, weights, nframes=1)
        self.gaussian_set = gaussian_set
        self.gaussian_set.optimize_all_except_motion()

    def _prepare_gaussian_set(self, key, weights, nframes):
        if key + "._xyz" not in weights:
            return None
        xyz = weights[key + "._xyz"]
        active_frames = weights[key + '.active_frames']
        g = GaussianModel(nframes, freeze_frames_of_origin=self.config.freeze_frames_of_origin)
        g.create_empty(num_gaussians=xyz.size(0), active_frames=active_frames)
        g.training_setup(self.config)
        return g

    def nextstep_get_gaussians(self, viewpoint_camera):
        batch, means3D, rotations, opacity, shs, scales = self.gaussian_set.get_properties(0)
        segmentation = self.gaussian_set.get_segmentation
        return means3D, rotations, opacity, shs, scales, segmentation

    def step_stage(self, new_phase, reduce_scale=True):
        phase = PHASE_NAMES[self.phase.item()]
        if (phase == 'motion-estimation' or phase == 'motion-expansion') and new_phase == 'bundle-adjust':
            if reduce_scale:
                self.gaussian_set.reduce_scale(factor=self.config.downsample_background_reducescale)
            self.phase *= 0

        if new_phase == 'motion-estimation':
            if self.config.prune_points:
                self.gaussian_set.prune_points(min_opacity=0.02, min_size=0.002)
            self.phase *= 0
            self.phase += 1

        if new_phase == 'motion-expansion':
            self.phase *= 0
            self.phase += 2

    def optimizer_step(self, step):
        self.gaussian_set.optimizer.step()
        self.gaussian_set.optimizer.zero_grad(set_to_none=True)

    def optimizer_zero_grad(self):
        self.gaussian_set.optimizer.zero_grad(set_to_none=True)

    def compute_scaling_loss(self):
        loss = self.gaussian_set.compute_scaling_loss(idxs=torch.ones_like(self.gaussian_set.get_segmentation).long())
        return loss

    def upsample_gaussians(self, factor):
        g_new = GaussianModel(num_frames=1, freeze_frames_of_origin=self.config.freeze_frames_of_origin)
        g_new.create_upsampled_gaussians(self.gaussian_set, int(factor))
        g_new.training_setup(self.config)
        g_new.optimize_all_except_motion()
        self.gaussian_set = g_new
        torch.cuda.empty_cache()
        gc.collect()

