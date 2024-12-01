from time import time
from typing import Any, Dict, Optional, Literal, Type
import typing
from dataclasses import dataclass, field

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp.grad_scaler import GradScaler
import matplotlib.pyplot as plt
import subprocess
from pytorch3d.ops import knn_points
from torchvision.transforms.functional import gaussian_blur

# for visualizing best tracks
import cv2
import matplotlib
import numpy as np
import tqdm
import os
import wandb
import glob

import torch
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from nerfstudio.utils import profiler
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
    Pipeline,
)

from src.data.datamanager import GaussianSplattingDataManagerConfig, GaussianSplattingDataManager
from src.models.model import GaussianSplattingModelConfig
from src.utils.utils import PHASE_NAMES, PHASE_IDS
from src.utils.tracking_utils import get_tracking_eval_metrics
from src.models.ip2p import InstructPix2Pix

# torch.autograd.set_detect_anomaly(True)


@dataclass
class GaussianSplattingPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: GaussianSplattingPipeline)
    """target class to instantiate"""
    datamanager: GaussianSplattingDataManagerConfig = GaussianSplattingDataManagerConfig()
    """specifies the datamanager config"""
    model: ModelConfig = GaussianSplattingModelConfig()
    """specifies the model config"""
    stages: typing.List = None
    """specifies the optimization stages of the pipeline"""
    text_guided: bool = False
    """whether to use text-guided image editing"""
    prompt: str = "don't change the image"
    """prompt for InstructPix2Pix"""
    guidance_scale: float = 12.5
    """(text) guidance scale for InstructPix2Pix"""
    image_guidance_scale: float = 1.5
    """image guidance scale for InstructPix2Pix"""
    gs_steps: int = 2500
    """how many GS steps between dataset updates"""
    diffusion_steps: int = 20
    """Number of diffusion steps to take for InstructPix2Pix"""
    lower_bound: float = 0.7
    """Lower bound for diffusion timesteps to use for image editing"""
    upper_bound: float = 0.98
    """Upper bound for diffusion timesteps to use for image editing"""
    ip2p_device: Optional[str] = None
    """Second device to place InstructPix2Pix on. If None, will use the same device as the pipeline"""
    ip2p_use_full_precision: bool = False
    """Whether to use full precision for InstructPix2Pix"""



class GaussianSplattingPipeline(VanillaPipeline):
    """
    Pipeline for training and evaluating Gaussian Splatting on Nerfstudio data.
    """

    def __init__(
        self,
        config: GaussianSplattingPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
        
    ):
        Pipeline.__init__(self)
        self.config = config
        self.test_mode = test_mode

        # initialize data manager (which will use our scene flow dataset)
        self.datamanager: GaussianSplattingDataManager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
        )
        self.datamanager.to(device)
        assert self.datamanager.train_dataset is not None, "Missing input dataset"

        # get raw depth-map point cloud sequence for model initialization
        if hasattr(self.datamanager.train_dataset, 'get_full_depth_pc_sequence'):
            pc_seq = self.datamanager.train_dataset.get_full_depth_pc_sequence()
            config.model.point_cloud_sequence = pc_seq

        # initialize model
        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            datamanager=self.datamanager,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            device=device,
            grad_scaler=grad_scaler,
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(Model, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True))
            dist.barrier(device_ids=[local_rank])

        # indicates how long to train before transitioning into next stage
        # self.register_buffer("current_stage", torch.tensor(0))
        # self.register_buffer("stage_counter", torch.tensor(0))
        # self.register_buffer('updating-foreground', torch.tensor(0))
        # self.register_buffer('updating-background', torch.tensor(0))

        self.stages = self.config.stages
        self.register_buffer("stage", torch.tensor(-1))
        self.register_buffer("step_counter", torch.tensor(0))
        self.register_buffer("stage_num_steps", torch.tensor(0))
        self.register_buffer('updating-foreground', torch.tensor(True))
        self.register_buffer('updating-background', torch.tensor(True))
        self.register_buffer("foreground_sequence_length", torch.tensor(1))
        self.register_buffer("background_sequence_length", torch.tensor(1))

        self._step_stage()
        self.text_guided = self.config.text_guided

        if self.text_guided:
            self.ip2p_device = torch.device(device)
            self.ip2p = InstructPix2Pix(self.ip2p_device, ip2p_use_full_precision=self.config.ip2p_use_full_precision)

            self.text_embedding = self.ip2p.pipe._encode_prompt(
            self.config.prompt, device=self.ip2p_device, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=""
            )
            # which image index we are editing
            self.curr_edit_idx = 0
            # whether we are doing regular GS updates or editing images
            self.makeSquentialEdits = False

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        if ((step-1) % self.config.gs_steps) == 0:
                self.makeSquentialEdits = True

        if not self.text_guided or not self.makeSquentialEdits:
            # Update our motion's stage
            if self.training:
                self.step()

            # given the stage, find and train on activate frames
            active_frames = self.model.randselect_active_training_frames()
            ray_bundle, batch = self.datamanager.next_train(step, active_frames)

            # get model outputs, loss, and metrics
            model_outputs = self._model(ray_bundle)  # train distributed data parallel model if world_size > 1
            metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
            loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict, self.datamanager.iter_train_image_dataloader, self.datamanager.train_image_sampler, ray_bundle)
            return model_outputs, loss_dict, metrics_dict

        else:         
            # get index
            idx = self.curr_edit_idx
            cameras = self.datamanager.train_dataset.cameras.to(self.model.device)
            camera, data, image_idx = self.datamanager.next_train_idx(idx)
            camera_ray_bundle = cameras.generate_rays(camera_indices=image_idx)
            #camera_ray_bundle.metadata['nframes'][camera_ray_bundle.metadata['nframes'] == 132] = image_idx

            #camera_ray_bundle['metadata']['nframes'] 

            model_outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle, image_idx[0].item())
            #model_outputs = self.model(camera)
            metrics_dict = self.model.get_metrics_dict(model_outputs, data)

            original_image = self.datamanager.original_cached_train["image"][idx].unsqueeze(dim=0).permute(0, 3, 1, 2)
            rendered_image = model_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)

            edited_image = self.ip2p.edit_image(
                        self.text_embedding.to(self.ip2p_device),
                        rendered_image.to(self.ip2p_device),
                        original_image.to(self.ip2p_device),
                        guidance_scale=self.config.guidance_scale,
                        image_guidance_scale=self.config.image_guidance_scale,
                        diffusion_steps=self.config.diffusion_steps,
                        lower_bound=self.config.lower_bound,
                        upper_bound=self.config.upper_bound,
                    )

            # resize to original image size (often not necessary)
            if (edited_image.size() != rendered_image.size()):
                edited_image = torch.nn.functional.interpolate(edited_image, size=rendered_image.size()[2:], mode='bilinear')

            #from PIL import Image
            #Image.fromarray((edited_image.squeeze().permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)).save(f"render_debug/{idx}-{step}.png")
            # write edited image to dataloader
            edited_image = edited_image.to(original_image.dtype)
            self.datamanager.train_image_dataloader.cached_collated_batch["image"][idx] = edited_image.squeeze().permute(1,2,0)
            data["image"] = edited_image.squeeze().permute(1,2,0)
            data['depth_image'] = data['depth_image'].squeeze(0)
            data['segmentation'] = data['segmentation'].squeeze(0)
            data['tracks'] = data['tracks'].squeeze(0)
            data['track_mask'] = data['track_mask'].squeeze(0)
            data['eval_mask'] = data['eval_mask'].squeeze(0)
            data['track_segs'] = data['track_segs'].squeeze(0)
            #increment curr edit idx
            self.curr_edit_idx += 1
            if (self.curr_edit_idx >= len(self.datamanager.train_image_dataloader.cached_collated_batch['image'])):
                self.curr_edit_idx = 0
                self.makeSquentialEdits = False

            loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict, self.datamanager.iter_train_image_dataloader, self.datamanager.train_image_sampler)


            return model_outputs, loss_dict, metrics_dict

    def step(self):
        # identify if we need to expand our motion scope
        stage_name = self.stages[self.stage.item()]['type']
        stage_fgbg = self.stages[self.stage.item()].get('updating', 'foreground-and-background')
        if stage_name in ['motion-estimation-and-merge', 'expand-motion-sliding-window']:
            if stage_name == 'motion-estimation-and-merge':
                # default to most frames needed to learn
                nframes_to_learn = max(self.foreground_sequence_length, self.background_sequence_length)
            else:
                nframes_to_learn = self.stages[self.stage.item()]['window-size'] * 2
            iters_per_frame = int(self.stages[self.stage.item()]['steps'] * self.model.nframes / nframes_to_learn)
            if ((self.step_counter.item() + 1) % iters_per_frame) == 0:
                self.model.step_motion_scope()

        # check if we step to the next stage
        while self.step_counter > self.stage_num_steps:
            self._step_stage()
        self.step_counter += 1

    def _step_stage(self):
        # we evaluate if we just finished a bundle adjustment phase
        if self.stages[self.stage.item()]['type'] == 'global-adjust' and self.stage.item() >= 0:
            self._evaluate()

        # update stage counter
        if self.stage.item() + 1 < len(self.stages):
            next_stage = self.stages[self.stage.item() + 1]
        else:
            print("Repeating last stage.")
            next_stage = self.stages[self.stage.item()]

        # get whether to update foreground and background
        fgbg = next_stage.get('updating', 'foreground-and-background')

        if next_stage['type'] == 'global-adjust':
            print("Entering Bundle Adjustment Stage")
            self.model.step_stage('bundle-adjust', fgbg=fgbg)

        elif next_stage['type'] == 'motion-estimation-and-merge':
            print("Entering Motion Estimation Stage")
            self.model.step_stage('motion-estimation', fgbg=fgbg)
            if 'foreground' in fgbg:
                self.foreground_sequence_length *= 2
            if 'background' in fgbg:
                self.background_sequence_length *= 2

        elif next_stage['type'] == 'expand-motion-sliding-window':
            print("Entering Motion Expansion Stage")
            self.model.step_stage('motion-expansion', fgbg=fgbg, scope_increase=next_stage['window-size'])
            if 'foreground' in fgbg:
                self.foreground_sequence_length += next_stage['window-size'] * 2
            if 'background' in fgbg:
                self.background_sequence_length += next_stage['window-size'] * 2

        elif next_stage['type'] == 'upsample':
            print("Upsampling Gaussians")
            self.model.upsample_gaussians(factor=next_stage['factor'], fgbg=fgbg)
        else:
            raise RuntimeError("Invalid Learning Stage: {}".format(next_stage['type']))

        # update stage state
        self.stage_num_steps.data = torch.tensor(next_stage.get('steps', -1) * self.model.nframes).to(self.stage_num_steps)
        self.step_counter *= 0
        next_stage = self.stage + 1
        self.stage = next_stage if next_stage < len(self.stages) else self.stage

    def _evaluate(self):
        print("Evaluating Current Model")
        final_val_losses = self.get_average_eval_image_metrics()
        final_train_losses = self.get_average_eval_image_metrics(split="train")
        final_val_losses = {"FINAL_VAL_" + k: final_val_losses[k] for k in final_val_losses}
        final_train_losses = {"FINAL_TRAIN_" + k: final_train_losses[k] for k in final_train_losses}
        wandb.log(final_val_losses)
        wandb.log(final_train_losses)
        if self.stage > 2:
            dycheck_tracking_score = self.run_dycheck_tracking_eval(outdir='render_debug')
            wandb.log(dycheck_tracking_score)

    @profiler.time_function
    def get_average_eval_image_metrics(self, step: Optional[int] = None, split: str = "val"):
        self.eval()
        metrics_dict_list, metrics_dict_list_composited = [], []
        assert isinstance(self.datamanager, GaussianSplattingDataManager)
        if split == 'val':
            dataloader = self.datamanager.fixed_indices_eval_dataloader
        else:
            dataloader = self.datamanager.fixed_indices_train_dataloader

        num_images = len(dataloader)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
            moving_average = None
            for camera_ray_bundle, batch in dataloader:
                # run model
                inner_start = time()
                outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
                fps = 1 / (time() - inner_start)

                # simple empty-space inpainting technique that boosts metrics
                if not self.model.config.no_background:
                    inpainted_rgb, moving_average = self._inpaint_empty(
                        outputs['rgb'], outputs['segmentation'], moving_average=moving_average, moving_average_pct=0.4)
                    outputs['rgb'] = inpainted_rgb

                # get compute metrics
                metrics_dict, _ = self.model.get_image_metrics_and_images(outputs, batch)

                # add fps to metrics dict
                assert "fps" not in metrics_dict
                metrics_dict["fps"] = fps

                # add to list
                metrics_dict_list.append(metrics_dict)
                progress.advance(task)

        # average the metrics list
        metrics_dict = {}
        per_img_psnrs = [metrics_dict["psnr"] for metrics_dict in metrics_dict_list]
        print("========================================================")
        print("Per Image PSNRs:")
        print(per_img_psnrs)
        for key in metrics_dict_list[0].keys():
            metrics_dict[key] = float(
                torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list]))
            )

        self.train()
        return metrics_dict

    @staticmethod
    def _inpaint_empty(rgb, segmentation, empty_thresh=0.8, bkgrnd_thresh=0.83, bkgrnd_cls=0, moving_average=None,
                       moving_average_pct=1.0):
        # filter image to (a) empty pixels and (b) background pixels
        empty = segmentation.sum(dim=-1) < empty_thresh
        background = (segmentation.argmax(dim=-1) == bkgrnd_cls) & (segmentation.sum(dim=-1) > bkgrnd_thresh)
        empty_pixels, background_pixels = torch.where(empty), torch.where(background)

        # if we were too strict on the background filtering, set a softer threshold
        if background_pixels[0].size(0) == 0:
            background = (segmentation.argmax(dim=-1) == bkgrnd_cls) & (segmentation.sum(dim=-1) > bkgrnd_thresh / 2)
            background_pixels = torch.where(background)

        # copy closest background color to empty pixels
        if empty_pixels[0].size(0) > 0:
            # copy colors to background
            empty_pixels = torch.stack([empty_pixels[0], empty_pixels[1]], dim=1)
            background_pixels = torch.stack([background_pixels[0], background_pixels[1]], dim=1)
            _, idxs, __ = knn_points(empty_pixels[None].float(), background_pixels[None].float(), K=1)
            copied_colors = rgb[background_pixels[:, 0][idxs.flatten()], background_pixels[:, 1][idxs.flatten()]]
            rgb[empty_pixels[:, 0], empty_pixels[:, 1]] = copied_colors

            # blur for more visually appealing effect
            rgb_blurred = gaussian_blur(rgb.permute(2, 0, 1), kernel_size=59).permute(1, 2, 0)
            rgb_blurred = gaussian_blur(rgb_blurred.permute(2, 0, 1), kernel_size=59).permute(1, 2, 0)

            # create temporally smooth background
            if moving_average is None:
                rolling_background = rgb_blurred.detach().clone()
            else:
                rolling_background = moving_average * (1 - moving_average_pct) + rgb_blurred * moving_average_pct

            # finally, composite into new image
            inpainted_rgb = rgb * (1 - empty[:, :, None].float()) + rolling_background * empty[:, :, None].float()
        else:
            inpainted_rgb = rgb
            rolling_background = None

        return inpainted_rgb, rolling_background

    def get_tracking_eval_metrics(self, outdir=None, split='val', context_frames=3):
        self.eval()

        # --------------------- Compute 2D Tracking Loss ---------------------
        # if we have many gaussian collections, we need to identify correspondences between adjacent collections
        corresponding_idxs = self.model.glue_correspondences(context_frames)

        # select dataloader
        if split == 'val':
            dataloader = self.datamanager.fixed_indices_eval_dataloader
        else:
            dataloader = self.datamanager.fixed_indices_train_dataloader

        # load ground truth tracks
        gt_tracks_2d = dataloader[0]['gt_tracks_2D'].clone()
        gt_tracks_2d[:, :, [0, 1]] = gt_tracks_2d[:, :, [1, 0]]  # x and y are swapped in track files

        # load gaussian means2D, gt images, and renders
        means2D, images, renders = [], [], []
        for camera_ray_bundle, batch in tqdm.tqdm(dataloader):
            # get means2D
            frameidx = camera_ray_bundle.time
            frame_means2d = self.model.field.foreground_field.get_means_2D(frameidx, camera_ray_bundle)
            frame_means2d[:, [0, 1]] = frame_means2d[:, [1, 0]]  # x and y are swapped in projection code

            # collect appropriate means 2D based on correspondences with 1st frame points
            gaussian_idx = self.model.field.foreground_field.frameidx2gaussianidx[frameidx].detach().clone()
            frame_means2d_mapped = frame_means2d[corresponding_idxs[gaussian_idx]]
            means2D.append(frame_means2d_mapped)

            # get rendered image (for visualization)
            outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
            renders.append(outputs['rgb'].cpu().detach().numpy())
            images.append(batch['image'].cpu().numpy())

        # compute tracking metrics and visualizations
        metrics_dict = get_tracking_eval_metrics(means2D, gt_tracks_2d, images, renders, outdir, split)
        self.train()

        return metrics_dict

    def run_dycheck_tracking_eval(self, outdir=None, context_frames=3, candidates_per_gt=5, pckt_thresh=0.05):
        self.eval()

        # load ground truth tracks -- we can load any batch of data since gt tracks are always the same
        dataloader = self.datamanager.fixed_indices_train_dataloader
        data_batch = dataloader.get_data_from_image_idx(0)[1]
        gt_tracks_2d = data_batch['gt_tracks_2D'].clone()
        gt_tracks_2d[:, :, [0, 1]] = gt_tracks_2d[:, :, [1, 0]]  # x and y are swapped in track files
        # gt_tracks_2d is a (T, N, 3) tensor, given N ground truth tracks are labeled in total

        # keypoints are labeled very sparsely -- find which frames actually have keypoints
        labeled_frames = torch.any(gt_tracks_2d[:, :, 2].bool(), dim=1)
        labeled_frames = torch.nonzero(labeled_frames)

        # get image dimensions as well as a mapping from each frame to the gaussian set that "covers" the frame
        height, width, _ = data_batch['image'].shape
        frameidx2gaussiansetidx = self.model.field.foreground_field.frameidx2gaussianidx

        tracking_distances = []
        for i in range(len(labeled_frames) - 1):
            # iterate through adjacent pairs of labeled frames
            src_fidx, target_fidx = labeled_frames[i].item(), labeled_frames[i + 1].item()

            # find the gaussian set that maps into the source and target frames
            #     due to incomplete divide-and-conquer merging, this may be different gaussian sets!
            src_gaussianset_idx = frameidx2gaussiansetidx[src_fidx]
            target_gaussianset_idx = frameidx2gaussiansetidx[target_fidx]
            num_sets_traversed = target_gaussianset_idx - src_gaussianset_idx

            # get 2D means of gaussians in the source and target frames
            src_camera, src_batch = dataloader.get_data_from_time_idx(src_fidx)
            means2D_src = self.model.field.foreground_field.get_means_2D(src_fidx, src_camera)[:, [1, 0]]
            target_camera, target_batch = dataloader.get_data_from_time_idx(target_fidx)
            means2D_target = self.model.field.foreground_field.get_means_2D(target_fidx, target_camera)[:, [1, 0]]

            # "glue" together gaussian tracklets to identify corresponding gaussians across sets
            correspondence_mapping = self.model.glue_correspondences(context_frames, src_fidx)[num_sets_traversed]

            # use correspondences to assign a target gaussian to each source gaussian
            means2D_target = means2D_target[correspondence_mapping]

            # find source gaussians that are closest to each gt track in the source frame
            unoccluded_gt = gt_tracks_2d[src_fidx, :, 2].bool() & gt_tracks_2d[target_fidx, :, 2].bool()
            num_gt = unoccluded_gt.sum().item()
            src_gt = gt_tracks_2d[src_fidx, unoccluded_gt, :2].unsqueeze(1)  # (num_unoccluded, 1, 2)
            src_dists = (means2D_src[None, ...].repeat(num_gt, 1, 1) - src_gt).norm(dim=-1)
            closest_gidxs = torch.argsort(src_dists, dim=1)[:, :candidates_per_gt]  # (num_unoccluded, candidates)

            # find where each source gaussian is in the target frame, and compute the distance to GT in target frame
            closest_gaussians_in_target = means2D_target[closest_gidxs.flatten()].view(num_gt, candidates_per_gt, 2)
            target_gt = gt_tracks_2d[target_fidx, unoccluded_gt, :2].unsqueeze(1)
            target_dists = (closest_gaussians_in_target - target_gt).norm(dim=-1)

            # for each GT target, find minimum distances over our gaussian candidates
            min_dists = target_dists.min(dim=-1)[0]
            tracking_distances.append(min_dists)

            # debug visualize tracking
            if outdir is not None:
                if not os.path.exists(os.path.join(outdir, 'tracking')):
                    os.makedirs(os.path.join(outdir, 'tracking'))
                closest_gaussians_in_source = means2D_src[closest_gidxs.flatten()].view(num_gt, candidates_per_gt, 2)
                these_successes = min_dists < (pckt_thresh * max(height, width))
                src_image = self.model.get_outputs_for_camera_ray_bundle(src_camera)['rgb'].detach()
                target_image = self.model.get_outputs_for_camera_ray_bundle(src_camera)['rgb'].detach()
                self._visualize_pck_error(outdir, closest_gaussians_in_source,
                                          closest_gaussians_in_target, src_gt.squeeze(1), target_gt.squeeze(1),
                                          src_image, target_image, these_successes)

        # some scenes don't have labeled tracking data
        if len(tracking_distances) == 0:
            return {}

        # aggregate tracking stats
        tracking_distances = torch.cat(tracking_distances)
        success = tracking_distances < (pckt_thresh * max(height, width))
        mean_dist, mean_success = tracking_distances.mean(), success.float().mean()
        self.train()
        return {'Dycheck-MeanDist': mean_dist, 'Dycheck-MeanSuccess': mean_success}

    @staticmethod
    def _visualize_pck_error(outdir, src_means2d, target_means2d, src_gt, target_gt, src_image, target_image, success):
        # create blurred composited image for visualization
        src_image = np.mean(src_image.cpu().numpy(), axis=-1)
        target_image = np.mean(target_image.cpu().numpy(), axis=-1)
        image = (src_image + target_image) / 2
        image = np.stack([image, image, image], axis=-1) * 255

        cmap = matplotlib.colormaps['Paired']
        for track_idx in range(src_gt.size(0)):
            # visualize source and target gt
            ty_1, tx_1 = src_gt[track_idx].cpu().numpy().tolist()
            ty_2, tx_2 = target_gt[track_idx].cpu().numpy().tolist()
            tracks = [(ty_1, tx_1), (ty_2, tx_2)]
            color = (np.array(cmap(track_idx % 12))[:3] * 255).astype(int)[::-1]
            thickness = 5
            for ty, tx in tracks:
                cv2.line(image, (int(tx) - 6, int(ty) - 6), (int(tx) + 6, int(ty) + 6), color.tolist(), thickness)
                cv2.line(image, (int(tx) - 6, int(ty) + 6), (int(tx) + 6, int(ty) - 6), color.tolist(), thickness)

            # visualize source and target means
            if not success[track_idx]:
                color = np.array([0, 0, 255])
            for candidate_idx in range(src_means2d.size(1)):
                ty_1, tx_1 = src_means2d[track_idx, candidate_idx, :2].detach().cpu().numpy().tolist()
                ty_2, tx_2 = target_means2d[track_idx, candidate_idx, :2].detach().cpu().numpy().tolist()
                cv2.circle(image, (int(tx_1), int(ty_1)), 3, tuple(color.tolist()), -1)
                cv2.circle(image, (int(tx_2), int(ty_2)), 3, tuple(color.tolist()), -1)

            cv2.imwrite(os.path.join(outdir, 'tracking', 'DYCHECK_VIZ_%s.png' % track_idx), image)

    def load_pipeline(self, loaded_state: Dict[str, Any], step: int) -> None:
        """Load the checkpoint from the given path

        Args:
            loaded_state: pre-trained model state dict
            step: training step of the loaded checkpoint
        """
        state = {
            (key[len("module.") :] if key.startswith("module.") else key): value for key, value in loaded_state.items()
        }
        self.model.update_to_step(step)
        self.model.prepare_load_weights(state)
        self.load_state_dict(state)

    def load_additional_model(self, load_path: str) -> None:
        """Load the additional model from the given path

        Args:
            load_path: path to the pre-trained model directory
        """
        latest_checkpoint = sorted(os.listdir(load_path))[-1]
        model_path = os.path.join(load_path, latest_checkpoint)
        loaded_state = torch.load(model_path, map_location=self.device)['pipeline']
        self.model.populate_background_field(loaded_state)

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
        self._model.eval()
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
        self._model.train(mode)
        return self
    

