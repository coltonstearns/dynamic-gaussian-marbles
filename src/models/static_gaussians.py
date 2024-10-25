import gc
import torch
from torch import Tensor
from nerfstudio.fields.base_field import Field
import torch.nn as nn
from torchtyping import TensorType
from src.utils.utils import PHASE_NAMES
from src.models.gaussians import GaussianModel
from typing import Optional, Tuple
import torch.nn.functional as F
import math

class GaussianStaticScene(nn.Module):

    def __init__(
        self,
        config,
        point_clouds: list[TensorType],
        maxpoints: int,
        seg_classes: Tensor = torch.tensor([0, 1, 2, 3]),
        pretrained_ckpt: Optional[str] = None,
        scene_scale: float = 1.0
    ) -> None:
        super().__init__()

        self.config = config
        self.register_buffer("maxpoints",  torch.tensor(maxpoints))
        self.register_buffer("phase", torch.tensor(0, dtype=torch.long))
        if pretrained_ckpt is not None and pretrained_ckpt != '':
            self.gaussian_set = load_pretrained_gsplat(pretrained_ckpt, scene_scale)
            self.pretrained = True
        else:
            self.gaussian_set = self.initialize_gaussian_set(point_clouds, seg_classes, maxpoints)
            self.pretrained = False

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
        if self.pretrained:
            return
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
        if self.pretrained:
            means3D, rotations, opacity, shs, scales = self.gaussian_set.get_properties()
            segmentation = torch.zeros_like(means3D)[:, 0]
            return means3D, rotations, opacity, shs, scales, segmentation
        else:
            batch, means3D, rotations, opacity, shs, scales = self.gaussian_set.get_properties(0)
            segmentation = self.gaussian_set.get_segmentation
            return means3D, rotations, opacity, shs, scales, segmentation

    def step_stage(self, new_phase, reduce_scale=True):
        phase = PHASE_NAMES[self.phase.item()]
        if (phase == 'motion-estimation' or phase == 'motion-expansion') and new_phase == 'bundle-adjust':
            if reduce_scale and not self.pretrained:
                self.gaussian_set.reduce_scale(factor=self.config.downsample_background_reducescale)
            self.phase *= 0

        if new_phase == 'motion-estimation':
            if self.config.prune_points and not self.pretrained:
                self.gaussian_set.prune_points(min_opacity=0.02, min_size=0.002)
            self.phase *= 0
            self.phase += 1

        if new_phase == 'motion-expansion':
            self.phase *= 0
            self.phase += 2

    def optimizer_step(self, step):
        if not self.pretrained:
            self.gaussian_set.optimizer.step()
            self.gaussian_set.optimizer.zero_grad(set_to_none=True)

    def optimizer_zero_grad(self):
        if not self.pretrained:
            self.gaussian_set.optimizer.zero_grad(set_to_none=True)

    def compute_scaling_loss(self):
        if self.pretrained:
            return torch.zeros(1).to(self.gaussian_set.means.device)
        else:
            loss = self.gaussian_set.compute_scaling_loss(idxs=torch.ones_like(self.gaussian_set.get_segmentation).long())
            return loss

    def upsample_gaussians(self, factor):
        if not self.pretrained:
            g_new = GaussianModel(num_frames=1, freeze_frames_of_origin=self.config.freeze_frames_of_origin)
            g_new.create_upsampled_gaussians(self.gaussian_set, int(factor))
            g_new.training_setup(self.config)
            g_new.optimize_all_except_motion()
            self.gaussian_set = g_new
            torch.cuda.empty_cache()
            gc.collect()



class PretrainedGaussianSplatting:

    def __init__(self, means, quats, scales, opacities, colors, sh_degree=3):

        # gaussian params
        self.means = means
        self.quats = quats
        self.scales = scales
        self.opacities = opacities
        self.colors = colors
        self.sh_degree = sh_degree

    def get_properties(self):
        return self.means, self.quats, self.opacities, self.colors, self.scales

def load_pretrained_gsplat(ckpt_path, scene_scale):
    device = torch.device("cuda")
    ckpt = torch.load(ckpt_path, map_location=device)["splats"]

    # load checkpoint properties
    means = ckpt["means"]
    quats = F.normalize(ckpt["quats"], p=2, dim=-1)
    scales = torch.exp(ckpt["scales"])
    opacities = torch.sigmoid(ckpt["opacities"])
    sh0 = ckpt["sh0"]
    shN = ckpt["shN"]
    colors = sh0
    # colors = torch.cat([sh0, shN], dim=-2)
    sh_degree = int(math.sqrt(colors.shape[-2]) - 1)

    # convert from colmap to our coordinate frame
    # NOTE: this is actually a 90 degree rotation!
    means[:, 1:] *= -1  # make y and z coordinates negative
    means[:, 2] *= -1  # invert world z
    means = means[:, [0, 2, 1]]  # switch y and z
    means *= scene_scale
    scales *= scene_scale

    # rotate 90 degrees
    rotx90_quat = torch.tensor([[-0.7071067818211393, 0.7071067805519557, 0.0, 0.0]]).to(quats)
    quats = quaternion_multiply(rotx90_quat, quats)
    quats = F.normalize(quats, p=2, dim=-1)

    return PretrainedGaussianSplatting(means, quats, scales, opacities.view(-1, 1), colors, sh_degree)


def quaternion_multiply(q1, q2):
    a, b, c, d = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    e, f, g, h = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    x1 = a*e - b*f - c*g - d*h
    x2 = b*e + a*f + c*h - d*g
    x3 = a*g - b*h + c*e + d*f
    x4 = a*h + b*g - c*f + d*e
    out = torch.stack([x1, x2, x3, x4], dim=-1)
    return out


