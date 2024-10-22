import gc
import numpy as np
import math

from torch import nn
import torch
from torch import Tensor
from torchtyping import TensorType

from nerfstudio.fields.base_field import Field
from src.utils.utils import PHASE_NAMES
from src.models.gaussians import GaussianModel
from src.utils.tracking_utils import get_means_2D


class GaussianSequence(nn.Module):

    def __init__(
        self,
        num_images: int,
        config,
        point_clouds: list[TensorType],
        maxpoints: int,
        seg_classes: Tensor = torch.tensor([0, 1, 2, 3]),
    ) -> None:
        super().__init__()

        self.num_images = num_images
        self.config = config
        self.register_buffer("phase", torch.tensor(0, dtype=torch.long))

        # set up gaussians
        self.nframes = len(point_clouds)
        self.gaussians, maxpoints = self.initialize_all_gaussian_sets(point_clouds, seg_classes, maxpoints)
        self.register_buffer("maxpoints",  torch.tensor(maxpoints))
        self.register_buffer("frameidx2gaussianidx",  torch.arange(len(self.gaussians)))
        self.register_buffer("frameidx2gaussianidxexpanded", -torch.ones(len(self.gaussians), dtype=torch.long))

        # temporary state -- keeps track of each iterative call to "iterate_sample_gaussians"
        self._time = 0
        self._viewpoint_camera = None
        self._gaussian_idxs = None
        self._dropout_idxs = None

    # ==================================== INITIALIZATION FUNCTIONS ====================================
    # ==================================================================================================
    def initialize_all_gaussian_sets(self, point_clouds, seg_classes, maxpoints):
        # first, estimate number of foreground points to sample, based on ratio of maxpoints
        num_points = []
        for i, pc in enumerate(point_clouds):
            foreground = torch.isin(pc[:, 3], seg_classes)
            sample_weights = torch.clip((1 / pc[:, 7]).float(), 0, 5)  # 7'th channel is depth
            foreground_weight, background_weight = sample_weights[foreground].sum(), sample_weights[~foreground].sum()
            n_frgrnd = int((foreground_weight / (foreground_weight + background_weight)) * maxpoints)
            num_points.append(n_frgrnd)
        foreground_npoints = int(sum(num_points) / len(num_points))
        print("Points for Each Frame of Gaussian Sequence: %s" % foreground_npoints)

        # initialize gaussian sets with previously estimated number of gaussians per frame
        gaussians = nn.ModuleList()
        for i, pc in enumerate(point_clouds):
            foreground = torch.isin(pc[:, 3], seg_classes)
            pc_foreground = pc[foreground].clone()
            g_foreground = self._initialize_gaussian_set(pc_foreground, i, foreground_npoints)
            gaussians.append(g_foreground)

        return gaussians, foreground_npoints

    def _initialize_gaussian_set(self, properties, frameidx, maxpnts, depth_downsample=True):
        """
        properties: [N, 8] tensor of properties [x, y, z, segmentation, r, g, b, depth]
        frameidx: frame timestep that this Gaussian set is initialized from
        """
        # perform depth-weighted random downsampling
        if properties.size(0) > maxpnts:
            if depth_downsample:
                probs = (1 / properties[:, 7]).float()  # 7'th channel is depth
                probs = torch.clip(probs, 0, 5)
                idxs = torch.multinomial(probs, maxpnts, replacement=False)
                properties = properties[idxs]
            else:
                idxs = torch.randperm(properties.size(0))[:maxpnts]
                properties = properties[idxs]

        # create gaussian splat of dynamic scene
        g = GaussianModel(self.nframes, freeze_frames_of_origin=self.config.freeze_frames_of_origin)
        g.create_from_input_pcd(properties[:, :3], properties[:, 4:7], frameidx=frameidx, segmentation=properties[:, 3])
        g.training_setup(self.config)
        g.optimize_all_except_motion()  # initializes to bundle adjustment learning rates
        print("Foreground Input size: %s" % properties.size(0))
        return g

    def prepare_load_weights(self, weights, field_name='foreground_field'):
        # prepare gaussian sets to have the correct size entries
        gaussians = nn.ModuleList()
        for i in range(self.nframes):
            key = '_model.field.{0}.gaussians.{1}'.format(field_name, i)
            g = self._prepare_gaussian_set(key, weights, self.nframes)
            if g is not None:
                gaussians.append(g)
        self.gaussians = gaussians

        # set to appropriate optimization phase
        phase = PHASE_NAMES[weights['_model.field.{0}.phase'.format(field_name)].item()]
        for i, g in enumerate(self.gaussians):
            g.optimize_all_except_motion() if phase == 'bundle-adjust' else self.gaussians[i].only_optimize_motion()

        # set which frames we've learned
        self.frameidx2gaussianidx = weights['_model.field.{0}.frameidx2gaussianidx'.format(field_name)]
        self.frameidx2gaussianidxexpanded = weights['_model.field.{0}.frameidx2gaussianidxexpanded'.format(field_name)]

    def _prepare_gaussian_set(self, key, weights, nframes):
        if key + "._xyz" not in weights:
            return None
        xyz = weights[key + "._xyz"]
        active_frames = weights[key + '.active_frames']
        g = GaussianModel(nframes, freeze_frames_of_origin=self.config.freeze_frames_of_origin)
        g.create_empty(num_gaussians=xyz.size(0), active_frames=active_frames)
        g.training_setup(self.config)
        return g

    # ==================================== RETRIEVE RENDERING STATE ====================================
    # ==================================================================================================
    def nextstep_get_gaussians(self, viewpoint_camera, gaussian_idx_overrides=None, dropout_override=None):
        # select a combination of Gaussian sets to render
        time_idx = viewpoint_camera.time
        if gaussian_idx_overrides is not None:
            sampled_gaussian_idxs = gaussian_idx_overrides
            detached = [False] + [True] * (len(gaussian_idx_overrides) - 1)
        else:
            sampled_gaussian_idxs, detached = self.select_gaussian_sets(time_idx)
        sampled_gaussian_sets = [self.gaussians[idx] for idx in sampled_gaussian_idxs]
        sampled_gaussian_times = [time_idx for _ in sampled_gaussian_sets]

        # for each gaussian set, compute dropout indices
        dropout_idxs = self.get_gaussian_dropouts(sampled_gaussian_sets, time_idx, dropout_override)

        # get properties of fused Gaussians sets
        means3D, rotations, opacity, shs, scales, segmentation = \
            self.fuse_gaussian_properties(sampled_gaussian_times, sampled_gaussian_sets, dropout_idxs, detached)

        # set up cache for this set of gaussians
        self._time = time_idx
        self._viewpoint_camera = viewpoint_camera
        self._gaussian_idxs = sampled_gaussian_idxs
        self._dropout_idxs = dropout_idxs

        return means3D, rotations, opacity, shs, scales, segmentation

    def select_gaussian_sets(self, time_idx):
        sampled_gaussian_sets, detached = [], []
        phase = PHASE_NAMES[self.phase.item()]

        # Motion Expansion - try to sample expanded gaussian set, and use original gaussians as supplement
        if phase == "motion-expansion":
            gaussian_idx = self.frameidx2gaussianidxexpanded[time_idx]
            if gaussian_idx == -1:  # means we're in validation or at sequence boundaries, so render normal gaussians
                assert (not self.train or time_idx == 0 or time_idx == (self.nframes - 1))
                sampled_gaussian_sets.append(self.frameidx2gaussianidx[time_idx])
                detached.append(False)
            else:
                sampled_gaussian_sets.append(gaussian_idx)
                detached.append(False)
                if torch.rand((1,)) < self.config.frame_transport_dropout:  # random dropout duplicate gaussians
                    sampled_gaussian_sets.append(self.frameidx2gaussianidx[time_idx])
                    detached.append(True)

        # Motion Estimation - only one gaussian set that can be chosen
        elif phase == "motion-estimation":
            sampled_gaussian_sets.append(self.frameidx2gaussianidx[time_idx])
            detached.append(False)

        # Bundle Adjust - randomly sample a valid gaussian set
        else:
            gaussian_idxs = [self.frameidx2gaussianidx[time_idx].item(), self.frameidx2gaussianidxexpanded[time_idx].item()]
            gaussian_idxs = [g for g in gaussian_idxs if g != -1]
            randselects = np.random.permutation(len(gaussian_idxs))
            sampled_gaussian_sets.append(gaussian_idxs[randselects[0]])
            detached.append(False)
            if len(gaussian_idxs) > 1:
                sampled_gaussian_sets.append(gaussian_idxs[randselects[1]])
                detached.append(True)

        return sampled_gaussian_sets, detached

    def get_gaussian_dropouts(self, sampled_gaussian_sets, time_idx, dropout_override):
        dropout_idxs = []
        sliding_window = PHASE_NAMES[self.phase.item()] == 'bundle-adjust' and len(sampled_gaussian_sets) > 1
        # training in motion estimation, bundle adjust, and motion expansion phases
        if self.training and not sliding_window:
            for i, g in enumerate(sampled_gaussian_sets):
                if dropout_override is not None and dropout_override[i] is not None:
                    drp_idxs = dropout_override[i]
                else:
                    drp_pct = self.config.frame_transport_dropout
                    drp_idxs = None if torch.rand((1,)) > drp_pct else self.dropout_frames(time_idx, g)
                dropout_idxs.append(drp_idxs)
        # evaluating before motion expansion
        elif not self.training and not sliding_window:
            for i, g in enumerate(sampled_gaussian_sets):
                dropout_idxs.append(None)
        # training and evaluating after fully learned motion expansion - uses sliding window
        else:
            # find an appropriate window size
            af1, af2 = sampled_gaussian_sets[0].active_frames.cpu(), sampled_gaussian_sets[1].active_frames.cpu()
            active_frames_intersect = np.intersect1d(af1.cpu().numpy(), af2.cpu().numpy())
            window1, window2 = len(af1) - len(active_frames_intersect), len(af2) - len(active_frames_intersect)
            window = max(window1, window2)

            # find the sliding window with frames closest to this target index
            active_frames_union = torch.unique(torch.cat([af1, af2]))
            sliding_window_frames = torch.argsort((active_frames_union - time_idx).abs())[:window + 1]
            sliding_window_frames = active_frames_union[sliding_window_frames].to(self.phase.device)

            # only keep Gaussians that belong to this sliding window
            for i, g in enumerate(sampled_gaussian_sets):
                drp_idxs = torch.isin(g._batch, sliding_window_frames)
                dropout_idxs.append(drp_idxs)

        return dropout_idxs

    def fuse_gaussian_properties(self, time_idxs, gaussians, dropout_idxs, detached):
        """
        Given a list of Gaussians objects, fuses together their properties.
        """

        fused_means3D, fused_rotations, fused_opacity, fused_shs, fused_scales, fused_seg = [], [], [], [], [], []
        for i, g in enumerate(gaussians):
            batch, means3D, rotations, opacity, shs, scales = g.get_properties(time_idxs[i])
            segmentation = g.get_segmentation

            if dropout_idxs[i] is not None:
                means3D = means3D[dropout_idxs[i]]
                rotations = rotations[dropout_idxs[i]]
                opacity = opacity[dropout_idxs[i]]
                scales = scales[dropout_idxs[i]]
                shs = shs[dropout_idxs[i]]
                segmentation = segmentation[dropout_idxs[i]]

            if detached[i]:
                means3D = means3D.detach()
                rotations = rotations.detach()
                opacity = opacity.detach()
                scales = scales.detach()
                shs = shs.detach()

            fused_means3D.append(means3D)
            fused_rotations.append(rotations)
            fused_opacity.append(opacity)
            fused_shs.append(shs)
            fused_scales.append(scales)
            fused_seg.append(segmentation)

        fused_means3D = torch.cat(fused_means3D, dim=0)
        fused_rotations = torch.cat(fused_rotations, dim=0)
        fused_opacity = torch.cat(fused_opacity, dim=0)
        fused_shs = torch.cat(fused_shs, dim=0)
        fused_scales = torch.cat(fused_scales, dim=0)
        fused_seg = torch.cat(fused_seg, dim=0)

        return fused_means3D, fused_rotations, fused_opacity, fused_shs, fused_scales, fused_seg

    def dropout_frames(self, time_idx, g):
        # filter to select random frame
        batch = g._batch
        learned_motions = g.learned_motions[g.active_frames]
        fully_learned = torch.prod(learned_motions, dim=0).to(bool)
        assert torch.any(fully_learned)
        fully_learned = torch.where(fully_learned)[0]

        # identify which frame indices to keep
        phase = PHASE_NAMES[self.phase.item()]
        if phase == 'bundle-adjust':
            randidxs = torch.randperm(batch.size(0))
            idxs = randidxs[:math.ceil(self.config.frame_transport_dropout * batch.size(0))]
        elif phase == 'motion-estimation':
            if g.boundary == -1:
                assert fully_learned.size(0) == 1
                keepframes = fully_learned
            else:
                learning_frames = g.active_frames
                tophalf = time_idx >= g.boundary
                if tophalf:
                    # todo Note: this used to be < time, but boundary seems more correct!
                    keepframes = [frame.item() for frame in learning_frames if frame < g.boundary]
                else:
                    keepframes = [frame.item() for frame in learning_frames if frame >= g.boundary]
                keepframes = torch.tensor(keepframes, device=fully_learned.device)
            idxs = torch.isin(batch.long(), keepframes.long())
        elif phase == 'motion-expansion':
            keepframes = fully_learned
            idxs = torch.isin(batch.long(), keepframes.long())
        else:
            raise RuntimeError("Must be in bundle adjust or motion estimation.")

        return idxs

    # ================================ ITERATIVE TRAINING CALLBACKS ====================================
    # ==================================================================================================
    def optimizer_step(self, step):
        # check for gradient error
        idx = self._gaussian_idxs[0]
        delta_xyz_grad = self.gaussians[idx].get_delta_xyz[self._time].grad
        if torch.any(torch.isnan(delta_xyz_grad)):
            print("----------- Gradient NAN Occured. Skipping Iteration ----------")
            self.gaussians[idx].optimizer.zero_grad(set_to_none=True)
            return

        # step optimizer
        self.gaussians[idx].optimizer.step()
        self.gaussians[idx].optimizer.zero_grad(set_to_none=True)

        # reset previously learned motions in motion estimation phase
        phase = PHASE_NAMES[self.phase.item()]
        if phase == 'motion-estimation' and self.config.freeze_previous_in_motion_estimation:
            self.gaussians[idx].reset_previously_learned_motions(self._time)

        # if we have a backup idx, reset its gradient
        if len(self._gaussian_idxs) > 1:
            for idx in self._gaussian_idxs[1:]:
                self.gaussians[idx].optimizer.zero_grad(set_to_none=True)

    def optimizer_zero_grad(self):
        # check for gradient error
        idx = self._gaussian_idxs[0]
        self.gaussians[idx].optimizer.zero_grad(set_to_none=True)

        # if we have a backup idx, reset its gradient
        if len(self._gaussian_idxs) > 1:
            for idx in self._gaussian_idxs[1:]:
                self.gaussians[idx].optimizer.zero_grad(set_to_none=True)

    def randselect_active_training_frames(self, motion_training_window):
        phase = PHASE_NAMES[self.phase.item()]
        if phase == 'bundle-adjust':
            return [i for i in range(self.nframes)]
        elif phase == 'motion-estimation':
            randidx = np.random.randint(0, len(self.gaussians))
            active_frames = self.gaussians[randidx].current_motion_estimation_frames(motion_training_window)
            return active_frames
        elif phase == 'motion-expansion':
            randidx = np.random.randint(0, len(self.gaussians))
            active_frames = self.gaussians[randidx].current_motion_estimation_frames(motion_training_window)
            if randidx == 0:  # don't sample first or last frame of sequence --> will cause error
                active_frames = active_frames[1:]
            if randidx == len(self.gaussians) - 1:
                active_frames = active_frames[:-1]
            return active_frames
        else:
            raise RuntimeError("Invalid phase specified.")

    def step_motion_scope(self):
        for i in range(len(self.gaussians)):
            self.gaussians[i].increase_scope_of_learned_motion()

    # ====================================== OPTIMIZATION PHASES =======================================
    # ==================================================================================================

    def step_stage(self, new_phase, scope_increase=None):
        phase = PHASE_NAMES[self.phase.item()]
        if phase == new_phase:
            return
        if phase == 'bundle-adjust' and new_phase == 'motion-estimation':
            if len(self.gaussians) == 1:
                return
            self.apply_merge()
            self.enter_motion_est_phase()
            self.phase += 1  # move to motion-estimation
        elif (phase == 'motion-estimation' or phase == 'motion-expansion') and new_phase == 'bundle-adjust':
            self.enter_bundle_adjust(reduce_scale=phase == 'motion-estimation')
            self.phase *= 0  # move to "bundle-adjust"
        elif phase == 'bundle-adjust' and new_phase == 'motion-expansion':
            if len(self.gaussians) == 1:
                return
            self.enter_motion_expansion_phase(scope_increase)
            self.phase += 2  # move to "motion-expansion"
        else:
            raise RuntimeError("Invalid phase specified.")

        # after transitioning phases, force clean up any old state
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()

    def apply_merge(self):
        # update frame to gaussian mappings
        self.frameidx2gaussianidx = self.frameidx2gaussianidx // 2

        # merge adjacent gaussians
        new_gaussians = nn.ModuleList()
        for i in range(0, len(self.gaussians), 2):
            g1 = self.gaussians[i]
            g2 = self.gaussians[i+1] if i + 1 < len(self.gaussians) else None
            if g2 is None:
                g_merge = g1
            else:
                # perform merge
                g_merge = GaussianModel(self.nframes, freeze_frames_of_origin=self.config.freeze_frames_of_origin)
                g_merge.merge_two_gaussian_sets(g1, g2)

                # interesting: doing this after global adjust, and NOT after motion est helps on DyCheck!
                if self.config.prune_points:
                    g_merge.prune_points(min_opacity=0.02, min_size=0.002)
                g_merge.downsample(self.maxpoints)

                # set up each gaussian
                g_merge.training_setup(self.config)
            new_gaussians.append(g_merge)
        self.gaussians = new_gaussians

    def enter_motion_est_phase(self):
        for i in range(len(self.gaussians)):
            self.gaussians[i].only_optimize_motion()

    def enter_bundle_adjust(self, reduce_scale=True):
        for i in range(len(self.gaussians)):
            self.gaussians[i].optimize_all_except_motion()
            if reduce_scale:
                self.gaussians[i].reduce_scale(factor=self.config.downsample_reducescale)

    def enter_motion_expansion_phase(self, scope_increase=1):
        # increase_active_frame_scope
        for i in range(len(self.gaussians)):
            # update gaussian
            orig_min, orig_max, new_min, new_max = self.gaussians[i].increase_active_frame_scope(scope_increase=scope_increase)
            self.gaussians[i].only_optimize_motion()
            self.gaussians[i].reduce_scale(factor=self.config.downsample_reducescale)

            # update frameidx2gaussianidxexpanded
            self.frameidx2gaussianidxexpanded[new_min:orig_min] = i
            self.frameidx2gaussianidxexpanded[orig_max+1:new_max+1] = i

        # start expansion off by increasing motion scope
        self.step_motion_scope()

    # ========================================= LOSS FUNCTIONS =========================================
    # ==================================================================================================

    def compute_isometry_loss(self, knn, knn_radius, per_segment, use_l2):
        gaussian_idx = self._gaussian_idxs[0]
        g = self.gaussians[gaussian_idx]
        phase = PHASE_NAMES[self.phase.item()]
        isometry_loss = g.compute_adjacent_isometry_loss(self._time, KNN=knn, knn_radius=knn_radius, phase=phase,
                                                         per_segment=per_segment, dropout_idxs=self._dropout_idxs[0],
                                                         use_l2=use_l2)
        return isometry_loss

    def compute_chamfer_loss(self, agg_group_ratio):
        gaussian_idx = self._gaussian_idxs[0]
        g = self.gaussians[gaussian_idx]
        chamfer_loss = g.compute_chamfer_loss(self._time, agg_group_ratio, dropout_idxs=self._dropout_idxs[0])
        return chamfer_loss

    def compute_scaling_loss(self):
        gaussian_idx = self._gaussian_idxs[0]
        g = self.gaussians[gaussian_idx]
        scaling_loss = g.compute_scaling_loss(self._dropout_idxs[0])
        return scaling_loss

    def compute_velocity_smoothing_loss(self):
        gaussian_idx = self._gaussian_idxs[0]
        g = self.gaussians[gaussian_idx]
        vel_loss = g.compute_velocity_smoothness_loss(self._time)
        return vel_loss

    def compute_instance_isometry_loss(self, num_pairs):
        gaussian_idx = self._gaussian_idxs[0]
        g = self.gaussians[gaussian_idx]
        loss = g.compute_instance_isometry_loss(self._time, pairs=num_pairs)
        return loss

    # ====================================== TRACKING LOSS HELPERS =====================================
    # ==================================================================================================
    def get_source_and_target_gaussians(self, window, target_fidx, src_fidx_override=None):
        # first, load appropriate gaussians
        gaussian_idx = self._gaussian_idxs[0]
        g = self.gaussians[gaussian_idx]

        # sample a source frame within the window
        assert window <= 15, "Window size must be less than or equal to 16 due to how we load tracks!"
        sampleable_frames, valid_gidxs = self.get_sampleable_tracking_frames(gaussian_idx)
        sampleable_frames = [frame for frame in sampleable_frames if abs(frame - self._time) <= window]
        if len(sampleable_frames) > 1 and self._time in sampleable_frames:  # at initialization, is len 1
            sampleable_frames.remove(self._time)
        if src_fidx_override is None:
            src_fidx = sampleable_frames[np.random.randint(0, len(sampleable_frames))]
        else:
            src_fidx = src_fidx_override
            if src_fidx not in sampleable_frames:
                return False, None

        # mask out gaussians that weren't rendered or do not have valid mappings into this frame
        gaussians_mask = torch.zeros_like(g.get_xyz[:, 0]).bool()
        gaussians_mask[self._dropout_idxs[0]] = True
        gaussians_mask[~valid_gidxs] = False

        # gather gaussian properties
        opacity = g.get_opacity.clone()[gaussians_mask]
        scales = g.get_scaling.clone()[gaussians_mask]
        segmentation = g.get_segmentation[gaussians_mask]

        # map means3D into source and target frames
        mapped_xyzs = []
        for idx in [src_fidx, target_fidx]:
            mapped_xyz = g.get_mapped_means3D(idx)[gaussians_mask].clone()
            mapped_xyzs.append(mapped_xyz)
        src_xyz, target_xyz = mapped_xyzs

        return True, (src_fidx, src_xyz, target_xyz, opacity, scales, segmentation)

    def get_sampleable_tracking_frames(self, gaussian_idx):
        g = self.gaussians[gaussian_idx]
        phase = PHASE_NAMES[self.phase.item()]

        # get the set of 'converged' frames, for which we can sample a source frame
        if phase == 'motion-estimation' and g.boundary != -1:  # bc learning half the Gaussians' motions
            tophalf = self._time >= g.boundary
            if tophalf:
                converged_frames = [frame.item() for frame in g.active_frames if frame < self._time]  # see this makes more sense to use self._time!
            else:
                converged_frames = [frame.item() for frame in g.active_frames if frame > self._time]
        elif phase == 'motion-expansion':  # bc learning all the Gaussians' motions
            converged_frames = torch.where(g.learned_motions.sum(dim=0))[0]
            if gaussian_idx == len(self.gaussians) - 1:
                converged_frames = converged_frames[1:]
            elif gaussian_idx == 0:
                converged_frames = converged_frames[:-1]
            else:
                converged_frames = converged_frames[1:-1]  # remove the first and last frames, bc currently being learned
            converged_frames = converged_frames.cpu().numpy().tolist()
        else:  # bc all motions already learned
            converged_frames = g.active_frames.cpu().numpy().tolist()

        converged_idxs = torch.isin(g._batch, torch.tensor(converged_frames).to(g._batch.device))
        return converged_frames, converged_idxs

    # ======================== ADDITIONAL STATE SETTING / RETRIEVING  ==================================
    # ==================================================================================================
    def get_means_2D(self, frameidx, camera):
        gaussian_idx = self.frameidx2gaussianidx[frameidx]
        g = self.gaussians[gaussian_idx]
        means3D = g.get_mapped_means3D(frameidx)
        return get_means_2D(means3D, camera)

    def upsample_gaussians(self, factor):
        # create new upsampled gaussians
        new_gaussians = nn.ModuleList()
        for i in range(0, len(self.gaussians)):
            g1 = self.gaussians[i]
            g_new = GaussianModel(self.nframes, freeze_frames_of_origin=self.config.freeze_frames_of_origin)
            g_new.create_upsampled_gaussians(g1, int(factor))
            g_new.training_setup(self.config)
            g_new.optimize_all_except_motion()
            new_gaussians.append(g_new)
        self.gaussians = new_gaussians
        torch.cuda.empty_cache()
        gc.collect()


