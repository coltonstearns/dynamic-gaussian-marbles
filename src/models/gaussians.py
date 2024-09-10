

import torch
import numpy as np
import torch_scatter

from src.utils.utils import inverse_sigmoid, strip_symmetric, build_scaling_rotation, to_batch_padded
from src.utils.shutils import RGB2SH, SH2RGB
from torch import nn
from simple_knn._C import distCUDA2
from pytorch3d.ops import knn_points
from pytorch3d.loss import chamfer_distance
import frnn


class GaussianModel(nn.Module):

    def __init__(self, num_frames: int = 1, freeze_frames_of_origin=True):
        self.freeze_frames_of_origin = freeze_frames_of_origin

        # gaussian params
        self._xyz = torch.empty(0)
        self._segmentation = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._scaling = torch.empty(0)
        self._opacity = torch.empty(0)
        self._delta_xyz = torch.empty(0)
        self._batch = torch.empty(0)

        # dynamics state
        self.num_frames = num_frames
        self.active_frames = torch.empty(0)
        self.learned_motions = torch.empty(0)  # todo: refactor into something simpler! Just make this 4 numbers; min/max/min/max
        self.boundary = torch.empty(0)

        # cache state for learning algorithm
        self._delta_xyz_frozen = torch.empty(0)
        self.cache_left_neighbors = None
        self.cache_right_neighbors = None

        # optimizer params
        self.optimizer = None
        self.learning_rates = None

        # initialize
        super().__init__()
        self.setup_functions()

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

    def training_setup(self, training_args, factor=1.0):
        self.learning_rates = {'f_dc': training_args.feature_lr * factor,
                               'opacity': training_args.opacity_lr * factor,
                               'scaling': training_args.scaling_lr * factor,
                               'delta_xyz': training_args.delta_position_lr * factor}

        l = [
            {'params': [self._features_dc], 'lr': training_args.feature_lr * factor, "name": "f_dc"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr * factor, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr * factor, "name": "scaling"},
        ]
        for i in range(len(self._delta_xyz)):
            l.append({'params': [self._delta_xyz[i]], 'lr': training_args.delta_position_lr * factor, "name": "delta_xyz_%s" % i})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    @property
    def get_scaling(self):
        scaling = self._scaling.repeat(1, 3)
        return self.scaling_activation(scaling)

    @property
    def get_rotation(self):
        rotation = torch.zeros((self._xyz.shape[0], 4), device="cuda")
        rotation[:, 0] = 1
        return rotation

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_delta_xyz(self):
        return self._delta_xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        return features_dc

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        rotation = torch.zeros((self._xyz.shape[0], 4), device="cuda")
        rotation[:, 0] = 1
        return self.covariance_activation(self.get_scaling, scaling_modifier, rotation)

    @property
    def get_segmentation(self):
        return self._segmentation.clone()

    def get_properties(self, time_idx):
        means3D = self.get_mapped_means3D(time_idx)
        rotations = self.get_rotation.clone()
        opacity = self.get_opacity.clone()
        scales = self.get_scaling.clone()
        shs = self.get_features.clone()
        batch = self._batch.clone()
        return batch, means3D, rotations, opacity, shs, scales

    def get_mapped_means3D(self, frameidx, frozen_offset=False):
        xyz = self.get_xyz.clone()
        if not frozen_offset:
            delta_xyz = self.get_delta_xyz[frameidx].clone()
        else:
            delta_xyz = self._delta_xyz_frozen[frameidx].clone()
        if self.freeze_frames_of_origin:
            delta_xyz[self._batch == frameidx] = torch.zeros_like(delta_xyz[self._batch == frameidx])
        xyz = xyz + delta_xyz
        return xyz

    def current_motion_estimation_frames(self, window=1):
        learned_motions = self.learned_motions[self.active_frames]
        fully_learned = torch.prod(learned_motions, dim=0).to(bool)
        assert torch.any(fully_learned)
        fully_learned = torch.where(fully_learned)[0]
        min_learned, max_learned = torch.min(fully_learned).item(), torch.max(fully_learned).item()
        active_frames = (fully_learned < (min_learned + window)) | (fully_learned > (max_learned - window))
        active_frames = fully_learned[active_frames].cpu().numpy().tolist()
        return active_frames

    # ====================================================================================================
    # ================================== INITIALIZING GAUSSIAN POINT CLOUD ===============================
    def create_from_input_pcd(self, pc, rgb, frameidx, segmentation=None):
        print(f"Loading Depth Point Cloud ({pc.shape[0]})...")

        # get gaussian static properties
        pc = pc.reshape(-1, 3).float().cuda()
        features = RGB2SH(rgb.reshape(-1, 3).float().cuda()).reshape(-1, 1, 3)
        if pc.size(0) > 0:
            dist2 = torch.clamp_min(distCUDA2(pc), 0.0000001)
        else:
            dist2 = torch.zeros((0,), device="cuda")
        scales = torch.log(torch.sqrt(dist2))[..., None]
        opacities = inverse_sigmoid(0.1 * torch.ones((pc.shape[0], 1), dtype=torch.float, device="cuda"))
        segmentation = segmentation.float().cuda() if segmentation is not None else torch.zeros_like(pc[:, 0])

        # initialize motion properties and bookeeping
        active_frames = torch.tensor([frameidx], device="cuda", dtype=torch.long)
        delta_xyz = [torch.zeros((0, 3), device="cuda").requires_grad_(True) for i in range(self.num_frames)]
        delta_xyz[frameidx] = torch.zeros((pc.shape[0], 3), device="cuda").requires_grad_(True)
        delta_xyz_frozen = [delta_xyz[i].clone().detach().cuda().requires_grad_(False) for i in range(len(delta_xyz))]
        boundary = torch.tensor(-1, device="cuda", dtype=torch.long)
        learned_motions = torch.zeros((self.num_frames, self.num_frames), device="cuda", dtype=torch.bool)
        learned_motions[frameidx, frameidx] = True
        batch = torch.ones(pc.shape[0], device="cuda", dtype=torch.long) * frameidx

        # initialize all fields
        self._initialize_parameters(active_frames, learned_motions, boundary, batch, pc, segmentation,
                                    features, scales, opacities, delta_xyz, delta_xyz_frozen)

    def create_empty(self, num_gaussians: int, active_frames: torch.tensor):
        # initialize parameters with empty values -- prepares tensor sizes for loading weights
        empty, empty_seg = torch.zeros(num_gaussians, 3), torch.zeros(num_gaussians)
        self.create_from_input_pcd(pc=empty, rgb=empty, frameidx=0, segmentation=empty_seg)

        # manually override the active_frames parameters
        del self.active_frames
        self.register_buffer('active_frames', active_frames.clone().cuda().long())

        # manually override sizes for delta_xyz's
        delta_xyz = [torch.zeros((0, 3), device="cuda").requires_grad_(True) for i in range(self.num_frames)]
        for i in active_frames:
            delta_xyz[i] = torch.zeros((num_gaussians, 3), device="cuda").requires_grad_(True)
        self._delta_xyz = nn.ParameterList(delta_xyz)
        frozen_delta_xyz = [delta_xyz[i].clone().detach().cuda().requires_grad_(False) for i in range(len(delta_xyz))]
        self._delta_xyz_frozen = frozen_delta_xyz

    def merge_two_gaussian_sets(self, g1, g2):
        # set valid delta_xyz parameters into each other's frame spans
        self._extend_active_frames_for_merge(g1, g2)

        # Prepare to learn delta-xyz across the motion boundary
        boundary = self._prepare_gaussians_at_boundary(g1, g2)

        # get merged parameters of the two gaussians
        props = {}
        for prop in ['_xyz', '_features_dc', '_scaling', '_opacity', '_segmentation', '_batch', 'active_frames']:
            props[prop] = torch.cat([getattr(g1, prop).data, getattr(g2, prop).data], dim=0)

        # custom merging of properties - note that frozen delta xyz is only valid for half the gaussian/frame pairs
        delta_xyz, frozen_delta_xyz = [], []
        for i in range(self.num_frames):
            merged_delta_xyz = torch.cat([g1._delta_xyz[i].data, g2._delta_xyz[i].data], dim=0)
            delta_xyz.append(merged_delta_xyz.requires_grad_(True))
            frozen_delta_xyz.append(merged_delta_xyz.clone().detach().requires_grad_(False))
        learned_motions = g1.learned_motions | g2.learned_motions

        # initialize all fields
        self._initialize_parameters(props['active_frames'], learned_motions, boundary, props['_batch'], props['_xyz'],
                                    props['_segmentation'], props['_features_dc'], props['_scaling'], props['_opacity'],
                                    delta_xyz, frozen_delta_xyz)

    def _extend_active_frames_for_merge(self, g1, g2):
        combined_active_frames = torch.unique(torch.cat([g1.active_frames, g2.active_frames]))
        g1_newly_active_frames = [f for f in combined_active_frames if f not in g1.active_frames]
        g2_newly_active_frames = [f for f in combined_active_frames if f not in g2.active_frames]
        for (g, newly_active_frames) in zip([g1, g2], [g1_newly_active_frames, g2_newly_active_frames]):
            for frameidx in newly_active_frames:
                g._delta_xyz[frameidx].data = torch.zeros((g.get_xyz.shape[0], 3), device="cuda")

    def _prepare_gaussians_at_boundary(self, g1, g2):
        assert torch.max(g1.active_frames) == (torch.min(g2.active_frames) - 1)
        boundary_lower, boundary_upper = torch.max(g1.active_frames), torch.min(g2.active_frames)
        g1._delta_xyz[boundary_lower+1].data = g1._delta_xyz[boundary_lower].data
        g2._delta_xyz[boundary_upper-1].data = g2._delta_xyz[boundary_upper].data
        g1.learned_motions[g1.active_frames, torch.ones_like(g1.active_frames) * (boundary_lower+1)] = True
        g2.learned_motions[g2.active_frames, torch.ones_like(g2.active_frames) * (boundary_upper-1)] = True
        return boundary_upper.item()

    def downsample(self, max_gaussians, min_gaussians=4):
        """
        Note: because this re-initializes all parameters, it must be followed by training_setup() call!
        """
        if self._xyz.size(0) == 0:  # in case where empty
            self._initialize_parameters(self.active_frames.clone(), self.learned_motions.clone(), self.boundary.clone(),
                                        self._batch, self._xyz, self._segmentation, self._features_dc, self._scaling,
                                        self._opacity, self._delta_xyz, self._delta_xyz_frozen)
            return

        N = self._xyz.size(0)
        if max_gaussians >= N:
            return

        all_indices = []
        for local_frame_idx in range(self.active_frames.size(0)):
            frame_idx = self.active_frames[local_frame_idx].item()
            print("Running FPS frames %s" % frame_idx)
            frame_mask = self._batch == frame_idx
            unique_segments = torch.unique(self._segmentation)
            for seg in unique_segments:
                seg_mask = self._segmentation == seg

                idxs = torch.nonzero(seg_mask & frame_mask).squeeze()
                if idxs.size(0) > min_gaussians:
                    # get number of points to keep
                    percent = idxs.size(0) / N
                    num_points = max(int(percent * max_gaussians), min_gaussians)

                    # get the entire point trajectory - we're running FPS on this higher-dimensional spline
                    splines = []
                    for i in self.active_frames:
                        splines.append(self._xyz[idxs] + self._delta_xyz[i][idxs])
                    splines = torch.cat(splines, dim=-1)  # N x 3T

                    # use pytorch3D sample_farthest_points to get the indices
                    fps_idxs = torch.randperm(splines.size(0))[:num_points]
                    idxs = idxs[fps_idxs]
                    all_indices.append(idxs)

                else:
                    all_indices.append(idxs)

        # combine indices across all frames and segments
        indices = torch.sort(torch.cat(all_indices, dim=0))[0]
        num_frames = self.active_frames.size(0)
        assert torch.unique(self._batch[indices]).size(0) == num_frames

        # downsample fields
        batch, xyz, segmentation, features_dc, scaling, opacity, delta_xyz, delta_xyz_frozen \
            = self._subset_parameters(indices)

        # update all fields
        self._initialize_parameters(self.active_frames.clone(), self.learned_motions.clone(), self.boundary.clone(),
                                    batch, xyz, segmentation, features_dc, scaling, opacity, delta_xyz, delta_xyz_frozen)

    def create_upsampled_gaussians(self, g, upsample_factor=8):
        """
        Note: because this re-initializes all parameters, it must be followed by training_setup() call!
        """
        # get upsampled parameters
        batch, xyz, segmentation, features_dc, scaling, opacity, delta_xyz, delta_xyz_frozen \
            = self._upsample_repeat_parameters(g, upsample_factor)

        # add noise to upsampled positions
        noise = torch.randn_like(xyz) * self.scaling_activation(scaling) * 0.66
        xyz = xyz + noise

        # shrink scale of each gaussian
        scaling = self.scaling_activation(scaling) / upsample_factor
        scaling = self.scaling_inverse_activation(scaling)

        # update all fields
        self._initialize_parameters(g.active_frames, g.learned_motions, g.boundary, batch, xyz, segmentation,
                                    features_dc, scaling, opacity, delta_xyz, delta_xyz_frozen)

    def prune_points(self, min_opacity, min_size):
        """
        Note: because this re-initializes all parameters, it must be followed by training_setup() call!
        """
        # check empty
        if self._xyz.size(0) == 0:  # in case where empty
            self._initialize_parameters(self.active_frames.clone(), self.learned_motions.clone(), self.boundary.clone(),
                                        self._batch, self._xyz, self._segmentation, self._features_dc, self._scaling,
                                        self._opacity, self._delta_xyz, self._delta_xyz_frozen)
            return

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        prune_mask |= (self.get_scaling.mean(dim=-1) < min_size)

        # make sure we don't prune too many --> challenging or ill posed cases
        if prune_mask.sum() > (prune_mask.size(0) / 2):
            print("Abnormal pruning! More than half our gaussians will be pruned! Randomly reducing pruning.")
            keep_idxs = torch.randperm(prune_mask.size(0))[:int(prune_mask.size(0) / 2)]
            prune_mask[keep_idxs] = False

        # get per-frame-of-origin points, and don't filter all points from a frame
        num_segments = torch.max(self._segmentation) + 1
        batch_wsegment = (self._batch.clone() * num_segments + self._segmentation.clone()).long()
        numpnts_remaining = torch_scatter.scatter_sum((~prune_mask).view(-1, 1).float(), batch_wsegment.view(-1, 1), dim=0).flatten()
        fewpnts_remaining = numpnts_remaining < 4
        dont_prune = torch.nonzero(fewpnts_remaining.flatten())  # indices we are forcing to keep around
        dont_prune = dont_prune.flatten()
        points_canprune_mask = ~torch.isin(batch_wsegment, dont_prune)
        prune_mask = torch.logical_and(prune_mask, points_canprune_mask)
        valid_points_mask = ~prune_mask

        # downsample fields
        batch, xyz, segmentation, features_dc, scaling, opacity, delta_xyz, delta_xyz_frozen \
            = self._subset_parameters(valid_points_mask)

        # update all fields
        self._initialize_parameters(self.active_frames, self.learned_motions, self.boundary, batch, xyz, segmentation,
                                    features_dc, scaling, opacity, delta_xyz, delta_xyz_frozen)
        torch.cuda.empty_cache()

    def _subset_parameters(self, indices):
        # downsample all fields
        props = {}
        for prop in ['_xyz', '_segmentation', '_features_dc', '_scaling', '_opacity', '_batch']:
            props[prop] = getattr(self, prop).data[indices]

        # custom downsample of delta xyz mappings
        delta_xyz, delta_xyz_frozen = [], []
        for i in range(self.num_frames):
            valid_map = i in self.active_frames
            new_delta_xyz = self._delta_xyz[i].data[indices] if valid_map else self._delta_xyz[i].data
            new_fr_delta_xyz = self._delta_xyz_frozen[i].data[indices] if valid_map else self._delta_xyz_frozen[i].data
            delta_xyz.append(new_delta_xyz.requires_grad_(True))
            delta_xyz_frozen.append(new_fr_delta_xyz.requires_grad_(False))

        return props['_batch'], props['_xyz'], props['_segmentation'], props['_features_dc'], props['_scaling'],\
                  props['_opacity'], delta_xyz, delta_xyz_frozen

    @staticmethod
    def _upsample_repeat_parameters(g, upsamples):
        # upsample regular fields
        props = {}
        for prop in ['_xyz', '_segmentation', '_features_dc', '_scaling', '_opacity', '_batch']:
            data = getattr(g, prop).data
            extra_dims = [1 for i in range(len(data.size()) - 1)]
            props[prop] = data.unsqueeze(1).repeat([1, upsamples] + extra_dims)
            trailing_dim = data.size(-1) if len(data.size()) > 1 else None
            props[prop] = props[prop].view(-1, trailing_dim) if trailing_dim is not None else props[prop].flatten()
        props['_features_dc'] = props['_features_dc'].unsqueeze(1)  # correct for unique SH formatted tensor

        # custom upsample of delta xyz mappings
        delta_xyz, delta_xyz_frozen = [], []
        for i in range(g.num_frames):
            valid_map = i in g.active_frames
            if valid_map:
                new_delta_xyz = g._delta_xyz[i].data.unsqueeze(1).repeat(1, upsamples, 1).view(-1, 3)
                new_fr_delta_xyz = g._delta_xyz[i].data.unsqueeze(1).repeat(1, upsamples, 1).view(-1, 3)
            else:
                new_delta_xyz = g._delta_xyz[i].data
                new_fr_delta_xyz = g._delta_xyz[i].data
            delta_xyz.append(new_delta_xyz.requires_grad_(True))
            delta_xyz_frozen.append(new_fr_delta_xyz.clone().detach().requires_grad_(False))

        return props['_batch'], props['_xyz'], props['_segmentation'], props['_features_dc'], props['_scaling'],\
                  props['_opacity'], delta_xyz, delta_xyz_frozen

    def _initialize_parameters(self, active_frames, learned_motions, boundary, batch, xyz, segmentation, features, scaling,
                           opacity, delta_xyz, frozen_delta_xyz):
        # delete and reinitialize buffers
        del self._batch
        self.register_buffer("_batch", batch)
        del self.active_frames
        self.register_buffer('active_frames', active_frames)
        del self.learned_motions
        self.register_buffer("learned_motions", learned_motions)
        del self.boundary
        self.register_buffer('boundary', torch.tensor(boundary, device="cuda", dtype=torch.long))

        # update parameters
        self._xyz = nn.Parameter(xyz.requires_grad_(True))
        self._segmentation = nn.Parameter(segmentation.requires_grad_(False))
        self._features_dc = nn.Parameter(features.contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scaling.requires_grad_(True))
        self._opacity = nn.Parameter(opacity.requires_grad_(True))
        self._delta_xyz = nn.ParameterList(delta_xyz)
        self._delta_xyz_frozen = frozen_delta_xyz

    # ====================================================================================================
    # =================================== UPDATE CALLS DURING OPTIMIZATION ===============================
    def only_optimize_motion(self):
        for param_group in self.optimizer.param_groups:
            if param_group["name"].startswith("delta_xyz"):
                param_group['lr'] = self.learning_rates['delta_xyz']
            else:
                param_group['lr'] = 0.0

    def optimize_all_except_motion(self):
        for param_group in self.optimizer.param_groups:
            if param_group["name"].startswith("delta_xyz"):
                param_group['lr'] = self.learning_rates['delta_xyz'] / 5
            else:
                param_group['lr'] = self.learning_rates[param_group['name']]

    def optimize_all(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rates[param_group['name']]

    def clone_motion(self, frameidx, source_motion, target_motion):
        """
        For all points originally belonging to frameidx, this clones the delta xyz motion from the source frame
        into the target frame.
        """
        motion = self._delta_xyz[source_motion].data[self._batch == frameidx]
        self._delta_xyz[target_motion].data[self._batch == frameidx] = motion.clone()

    def increase_scope_of_learned_motion(self):
        """
        If lower and upper idx are given, we manually set these frames to being 'learned'. Otherwise,
        we increment our scope by 1 in each direction.
        """
        # filter to N'xN' sub-array of frames that we are actually covering
        learned_motions = self.learned_motions[self.active_frames]
        learned_motions = learned_motions[:, self.active_frames]  # active frames x active frames
        source_idxs, target_idxs = torch.where(learned_motions)  # which frame to learn where

        # go through active frames and update their "learned ranges"
        for idx in range(len(self.active_frames)):
            min_target = torch.min(target_idxs[source_idxs == idx]).item()
            max_target = torch.max(target_idxs[source_idxs == idx]).item()
            if min_target > 0:  # we're learning transform from points in min_learned to (min_learned - 1)
                global_source, global_min_target = self.active_frames[idx], self.active_frames[min_target]
                self.learned_motions[global_source, global_min_target-1] = True
                self.clone_motion(global_source, global_min_target, global_min_target-1)
            if max_target < (len(self.active_frames) - 1):
                global_source, global_max_target = self.active_frames[idx], self.active_frames[max_target]
                self.learned_motions[global_source, global_max_target+1] = True
                self.clone_motion(global_source, global_max_target, global_max_target+1)

    def increase_active_frame_scope(self, scope_increase=4):
        orig_min, orig_max = torch.min(self.active_frames).item(), torch.max(self.active_frames).item()
        new_min = max(orig_min - scope_increase, 0)
        new_max = min(orig_max + scope_increase, self.num_frames - 1)
        self.active_frames = torch.arange(new_min, new_max+1, device="cuda")

        # Update newly added frames (which we won't have any points for), to have mappings into all previously
        # learned frames
        assert self.learned_motions[:, :new_min].sum() == 0 and self.learned_motions[:, new_max+1:].sum() == 0
        assert self.learned_motions[:new_min, :].sum() == 0 and self.learned_motions[new_max+1:, :].sum() == 0
        self.learned_motions[new_min: orig_min, orig_min: orig_max+1] = True
        self.learned_motions[orig_max: new_max+1, orig_min: orig_max+1] = True

        # initialize delta_xyz and delta_rotations
        num_gaussians = self._xyz.size(0)
        for frameidx in range(new_min, orig_min):
            self._delta_xyz[frameidx] = torch.zeros((num_gaussians, 3), device="cuda").requires_grad_(True)
            new_params = {'params': [self._delta_xyz[frameidx]], 'lr': self.learning_rates['delta_xyz'], "name": "delta_xyz_%s" % frameidx}
            self.optimizer.add_param_group(new_params)
        for frameidx in range(orig_max+1, new_max+1):
            self._delta_xyz[frameidx] = torch.zeros((num_gaussians, 3), device="cuda").requires_grad_(True)
            new_params = {'params': [self._delta_xyz[frameidx]], 'lr': self.learning_rates['delta_xyz'], "name": "delta_xyz_%s" % frameidx}
            self.optimizer.add_param_group(new_params)

        return orig_min, orig_max, new_min, new_max

    def reset_previously_learned_motions(self, frameidx):
        """
        This is used during motion estimation. During motion estimation, we want to freeze gaussian mappings
        that we've already learned in previous phases, and instead only learn NEW unexplored mappings.
        """
        if self.boundary == -1:  # indicates no boundary yet!
            return
        previous_mappings = self._delta_xyz_frozen[frameidx]
        already_learned = self._batch < self.boundary if frameidx < self.boundary else self._batch > self.boundary
        self._delta_xyz[frameidx].data[already_learned] = previous_mappings.data[already_learned].detach().clone()

    def reduce_scale(self, factor=0.5):
        self._scaling.data = self.scaling_inverse_activation(self.scaling_activation(self._scaling.data) * factor)

    # =================================================================================================
    # =============================== Geometric Regularization Losses =================================
    # =================================================================================================
    def compute_scaling_loss(self, idxs):
        if self._scaling.size(0) == 0:
            return torch.zeros(1).to(self._batch.device)
        return torch.mean(torch.square(self.scaling_activation(self._scaling[idxs])))

    def compute_adjacent_isometry_loss(self, frameidx, KNN=32, knn_radius=0.1, phase='motion-estimation',
                                      per_segment=True, dropout_idxs=None):
        device = self._batch.device
        if self.boundary == -1 or self._xyz.size(0) == 0:
            return torch.zeros(1).to(device)

        # compute and cache the KNN from the previous learning stage (if needed)
        if self.cache_right_neighbors is None or self.cache_left_neighbors is None:
            print("Caching Isometry! Will be faster after this is cached...")
            self._cache_isometry_knn(KNN, knn_radius, per_segment)

        # map gaussian positions to the current frame
        xyz = self.get_mapped_means3D(frameidx)

        errs = torch.tensor([]).to(device)
        if phase == 'motion-estimation' or phase == "motion-expansion":  # only update the side we're learning into
            # get neighborhood
            cache = self.cache_right_neighbors if frameidx < self.boundary else self.cache_left_neighbors
            if phase == 'motion-estimation':
                mask = self._batch >= self.boundary if frameidx < self.boundary else self._batch < self.boundary
            else:  # motion-expansion
                mask = torch.ones_like(self._batch).bool()
            cache_data = cache[0] if frameidx < self.boundary else cache[-1]
            if cache_data is None:
                return torch.zeros(1).to(device)  # no points!
            point_idxs, knn_idxs = cache_data
            point_idxs, knn_idxs = point_idxs.long(), knn_idxs.long()

            # get source xyz
            src_idx = frameidx + 1 if frameidx < self.boundary else frameidx - 1
            this_errs = self._compute_adjacent_isometry_helper(xyz, src_idx, point_idxs, knn_idxs, mask)
            errs = torch.cat((errs, this_errs))

        else:  # phase is bundle adjust - all active frames have learned mappings
            # we randomly drop out gaussians at each iteration - don't flow gradients to dropped out gaussians
            if dropout_idxs is not None:
                valid_mask = torch.zeros_like(self._batch).bool()
                valid_mask[dropout_idxs] = True
            else:
                valid_mask = None

            # compute isometry on left-half Gaussians and right-half Gaussians
            right_mask, left_mask = self._batch >= self.boundary, self._batch < self.boundary
            right_cache = self.cache_right_neighbors[np.random.randint(0, len(self.cache_right_neighbors))]
            left_cache = self.cache_left_neighbors[np.random.randint(0, len(self.cache_left_neighbors))]
            src_options = [frameidx + 1, frameidx - 1]
            src_options = [opt for opt in src_options if opt in self.active_frames]
            src_idx = src_options[np.random.randint(len(src_options))]
            for mask, cache in zip([right_mask, left_mask], [right_cache, left_cache]):
                if cache is None:
                    continue
                point_idxs, knn_idxs = cache
                point_idxs, knn_idxs = point_idxs.long(), knn_idxs.long()
                this_errs = self._compute_adjacent_isometry_helper(xyz, src_idx, point_idxs, knn_idxs, mask, valid_mask=valid_mask)
                errs = torch.cat((errs, this_errs))

        return errs

    def _compute_adjacent_isometry_helper(self, xyz, src_idx, point_idxs, knn_idxs, mask, valid_mask=None):
        # get source distances
        src_xyz = self.get_mapped_means3D(src_idx)  # don't use frozen, due to adjacent isometry
        src_xyz_subset = src_xyz[mask].clone()
        all_idxs = torch.cat([point_idxs, knn_idxs])
        src_gathered = torch.gather(src_xyz_subset, 0, torch.stack([all_idxs, all_idxs, all_idxs], dim=-1)).reshape(2, -1, 3)
        src_neighbor_dists = torch.norm(src_gathered[0] - src_gathered[1], dim=-1).detach().clone()

        # get current distances
        xyz_subset = xyz[mask].clone()
        xyz_gathered = torch.gather(xyz_subset, 0, torch.stack([all_idxs, all_idxs, all_idxs], dim=-1)).reshape(2, -1, 3)
        neighbor_dists = torch.norm(xyz_gathered[0] - xyz_gathered[1], dim=-1)

        # compute difference
        diffs = (neighbor_dists - src_neighbor_dists)
        if valid_mask is not None:
            dropout_mask = torch.gather(valid_mask[mask], 0, point_idxs).reshape(-1)
            diffs = diffs[dropout_mask]

        # get l1 and l2 errors
        # l2_err = diffs ** 2
        l1_err = torch.abs(diffs)
        return l1_err

    def _cache_isometry_knn(self, KNN=32, knn_radius=0.1, per_segment=True):
        device = self._xyz.device
        min_active, max_active = torch.min(self.active_frames).item(), torch.max(self.active_frames).item()

        # adjust for motion-expansion (i.e. the min-max boundaries will NOT have cached delta-xyzs)
        while self._delta_xyz_frozen[min_active].size(0) == 0:
            min_active += 1
        while self._delta_xyz_frozen[max_active].size(0) == 0:
            max_active -= 1

        # go through active frames and estimate previous-stage KNN and distances
        left_templates, right_templates = [], []
        for frameidx in range(min_active, max_active + 1):
            # map xyz into this frame index
            xyz_template = self.get_mapped_means3D(frameidx, frozen_offset=True)
            valid_gidxs = self._batch >= self.boundary if frameidx >= self.boundary else self._batch < self.boundary
            xyz_template = xyz_template[valid_gidxs]

            # check for if foreground Gaussians have 0 points (in one half or the other)
            if xyz_template.size(0) == 0:
                left_templates.append(None) if frameidx < self.boundary else right_templates.append(None)

            # append segmentation as additional dim
            seg_template = self._segmentation[valid_gidxs] * knn_radius * 2
            if per_segment:
                xyz_template = torch.cat([xyz_template, seg_template.view(-1, 1)], dim=-1)

            # compute knn
            lengths = torch.tensor([xyz_template.size(0)], device=xyz_template.device)
            dists2, idxs, nn, grid = frnn.frnn_grid_points(xyz_template[None].clone(),
                                                           xyz_template[None].clone(), lengths.clone(),
                                                           lengths.clone(), KNN, knn_radius, grid=None,
                                                           return_nn=False, return_sorted=True)
            mask = idxs != -1  # B x N x K=16

            # convert to sparse tensor format
            N = idxs.size(1)
            point_idxs = torch.arange(N, device='cuda').repeat_interleave(KNN).to(device)
            point_idxs = point_idxs[mask.flatten()].detach().clone()
            knn_idxs = idxs[mask].detach().clone()

            # cache this info
            if frameidx < self.boundary:
                left_templates.append((point_idxs.int(), knn_idxs.int()))
            else:
                right_templates.append((point_idxs.int(), knn_idxs.int()))

        self.cache_left_neighbors = left_templates
        self.cache_right_neighbors = right_templates

    def compute_chamfer_loss(self, frame_idx, agg_group_ratio=1.0, dropout_idxs=None):
        if len(self.active_frames) == 1 or self._xyz.size(0) == 0:
            return torch.zeros(1).to(self._xyz.device)
        chamfer_loss = self._compute_chamfer_aggregate(frame_idx, agg_group_ratio, dropout_idxs)
        return chamfer_loss

    def _compute_chamfer_aggregate(self, frame_idx, group_ratio=0.25, dropout_idxs=None):
        # step 1: get all frames that validly map into frame_idx
        valid_mask = self.learned_motions[self.active_frames, frame_idx]
        valid_idxs_local = torch.nonzero(valid_mask).flatten()
        valid_idxs = self.active_frames[valid_idxs_local]
        valid_idxs, valid_idxs_local = valid_idxs.tolist(), valid_idxs_local.tolist()

        # step 1.5 - check for >2 frames
        if self.boundary == -1 or len(valid_idxs) < 2:
            return torch.zeros(1).to(self._xyz.device)

        # randomly subdivide frames into 2 sets
        randidxs = torch.randperm(len(valid_idxs))
        num_select = max(1, int(len(valid_idxs) / 2 * group_ratio))
        subset1_idxs = torch.tensor(valid_idxs)[randidxs[:num_select]].cuda()
        subset2_idxs = torch.tensor(valid_idxs)[randidxs[num_select:2*num_select]].cuda()

        # reformat dropout mask
        dropout_mask = torch.zeros_like(self._batch).bool()
        dropout_mask[dropout_idxs] = True

        # get left and right boundary pcs
        xyz_mapped = self.get_mapped_means3D(frame_idx)
        subset1_mask, subset2_mask = torch.isin(self._batch, subset1_idxs), torch.isin(self._batch, subset2_idxs)
        subset1_mask, subset2_mask = subset1_mask & dropout_mask, subset2_mask & dropout_mask
        left_pc = xyz_mapped.clone()[subset1_mask]
        right_pc = xyz_mapped.clone()[subset2_mask]
        left_seg = self._segmentation.clone()[subset1_mask]
        right_seg = self._segmentation.clone()[subset2_mask]

        # get chamfer distance in those transported frames
        chamfer_loss = self._compute_chamfer(left_pc[None], right_pc[None], left_seg[None], right_seg[None])
        return chamfer_loss

    def _compute_chamfer(self, pc1, pc2, seg1=None, seg2=None):
        # get chamfer distance in those transported frames
        len1, len2 = torch.tensor([pc1.size(1)]).to(pc1.device), torch.tensor([pc2.size(1)]).to(pc2.device)
        (d1_forward, d2_forward), _ = chamfer_distance(pc1, pc2, x_lengths=len1, y_lengths=len2,
                                                       batch_reduction=None, point_reduction=None, norm=2)
        dists = torch.cat([d1_forward.flatten(), d2_forward.flatten()])
        dists = dists[dists > 0]
        chamfer_loss = torch.sqrt(dists).mean()

        # if segmentation, do mean of each segment for greater stability
        # if seg1 is not None and seg2 is not None:
        #     segs = torch.cat([seg1, seg2])
        #     segs = torch.unique(segs, return_inverse=True)[1]
        #     dists2 = torch_scatter.scatter_mean(dists**2, segs, dim=0)
        #     dists = torch_scatter.scatter_mean(dists, segs, dim=0)
        # else:
        #     dists, dists2 = dists, dists**2

        return chamfer_loss

    def compute_velocity_smoothness_loss(self, frame_idx, context=2, dropout_idxs=None):
        # each row of self.learned_motions corresponds to source, column corresponds to target
        frames_idxs_to_regularize = torch.nonzero(self.learned_motions[:, frame_idx].long()).flatten()
        points_mask = torch.isin(self._batch, frames_idxs_to_regularize)
        dropout_mask = torch.zeros_like(self._batch).bool()
        dropout_mask[dropout_idxs] = True
        points_mask &= dropout_mask

        # get deltas and deltas mask
        query_idxs = (torch.arange(context*2+1).to(self._batch.device) - context) + frame_idx
        query_idxs = query_idxs[(query_idxs >= min(self.active_frames)) & (query_idxs <= max(self.active_frames))]
        if len(query_idxs) < 3:
            return torch.zeros(1).to(self._xyz.device)

        # compute accelerations
        traj_offsets = torch.stack([self._delta_xyz[i][points_mask].clone().detach() if i != frame_idx else self._delta_xyz[i][points_mask].clone() for i in query_idxs], dim=0)  # T x N x 3
        velocities = traj_offsets[1:] - traj_offsets[:-1]
        accelerations = (velocities[1:] - velocities[:-1]).abs().sum(dim=2)  # context-2 x N

        # find out which accelerations are valid
        batch = self._batch[points_mask]
        valid_offsets = self.learned_motions[batch.repeat_interleave(query_idxs.size(0)), query_idxs.repeat(batch.size(0))]
        valid_offsets = valid_offsets.view(batch.size(0), query_idxs.size(0)).T  # size T x N
        valid_velocities = valid_offsets[1:] & valid_offsets[:-1]
        valid_acelerations = valid_velocities[1:] & valid_velocities[:-1]

        # loss is magnitude of accelerations
        if valid_acelerations.sum() == 0:
            return torch.zeros(1).to(self._xyz.device)
        else:
            return accelerations[valid_acelerations].mean()

    def compute_instance_isometry_loss(self, frameidx, pairs=4096):
        # get global isometry loss
        loss = torch.zeros(1).to(self._batch.device)
        if self.boundary != -1 and self._xyz.size(0) > 0:  #  and phase == 'motion-estimation':
            batch_global = (self._batch >= self.boundary).long()
            min_active, max_active = torch.min(self.active_frames).item(), torch.max(self.active_frames).item()

            # adjust for motion-expansion (i.e. the min-max boundaries will NOT have cached delta-xyzs)
            # this will only happend AFTER motion expansion!
            while self._delta_xyz_frozen[min_active].size(0) == 0:
                min_active += 1
            while self._delta_xyz_frozen[max_active].size(0) == 0:
                max_active -= 1

            # get frozen delta xyz into each fully-learned half
            sourceframe_left = torch.randint(min_active, self.boundary.item(), (1,)).to(self._batch.device)
            sourceframe_right = torch.randint(self.boundary.item(), max_active+1, (1,)).to(self._batch.device)
            xyz_mapped_left = self.get_mapped_means3D(sourceframe_left, frozen_offset=True)[batch_global == 0]
            xyz_mapped_right = self.get_mapped_means3D(sourceframe_right, frozen_offset=True)[batch_global == 1]
            xyz = self.get_xyz.clone()
            xyz[batch_global == 0] = xyz_mapped_left
            xyz[batch_global == 1] = xyz_mapped_right
            xyz = xyz.clone().detach()

            # compute isometry loss on difference from these original 'transformed halves'
            loss = self._compute_global_rigid_loss(xyz, batch_global, frameidx, pairs)

        return loss

    def _compute_global_rigid_loss(self, xyz, batch, frameidx, num_pairs):
        # Get xyz and transformed XYZ
        transformed_xyz = self.get_mapped_means3D(frameidx)

        # make batch segmentation-aware
        segments_u, segment_idxs = torch.unique(self._segmentation, return_inverse=True)
        batch_global_wseg = batch.clone() * segments_u.size(0) + segment_idxs  # however, this is unordered
        reorder = torch.argsort(batch_global_wseg)
        batch_global_wseg = batch_global_wseg[reorder]
        batch_global_wseg = torch.unique(batch_global_wseg, return_inverse=True)[1]
        xyz = xyz[reorder]
        transformed_xyz = transformed_xyz[reorder]

        # sample pairs of indices for each segment
        idxs1, idxs2 = [], []
        for i in range(batch_global_wseg.max() + 1):
            this_idxs = torch.where(batch_global_wseg == i)[0]
            idxs_shuffled_1 = this_idxs[torch.randperm(this_idxs.size(0))][:num_pairs]
            idxs_shuffled_2 = this_idxs[torch.randperm(this_idxs.size(0))][:num_pairs]
            idxs1.append(idxs_shuffled_1)
            idxs2.append(idxs_shuffled_2)
        idxs1, idxs2 = torch.cat(idxs1), torch.cat(idxs2)

        # compute original and transformed distances
        orig_dists = torch.norm(xyz[idxs1] - xyz[idxs2], dim=-1)
        new_dists = torch.norm(transformed_xyz[idxs1] - transformed_xyz[idxs2], dim=-1)

        # compute isometry loss
        l2_distance_diff = (orig_dists - new_dists)**2  # B x N x K=16
        l1_distance_diff = torch.abs(orig_dists - new_dists)  # B x N x K=16
        isometry_loss = l1_distance_diff.mean()

        return isometry_loss


