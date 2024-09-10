import gc
from typing import Dict
import cv2
import time
import numpy as np
import math
import tqdm
import os
import subprocess
import glob

import torch
import torch_scatter
import torch.nn as nn
import frnn
from gsplat.rendering import rasterization

# for debug visualizations
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib

VIS_IDX = 0


class TrackingLoss(nn.Module):
    def __init__(self, K=8, r=4, per_segment=True):
        super(TrackingLoss, self).__init__()
        self.K = K
        self.r = r
        self.per_segment = per_segment

    def forward(self, src_camera, target_camera,  src_means3D, target_means3D, opacity, scales, segmentation,
                src_particles, target_particles, particles_seg):

        # get means and depths for the target frame
        means2D_target, depths_target = get_means_2D(target_means3D, target_camera, return_depths=True)
        depths_target = depths_target.detach()

        # get means and depths for the source frame
        means2D_src, depths_src, conical_opacity = get_means_2D(src_means3D, src_camera, opacity, scales,
                                                                      return_depths=True, return_conical_opacity=True)
        means2D_src = means2D_src.detach()
        depths_src = depths_src.detach().flatten()
        conical_opacity = conical_opacity.detach()

        # append segmentation categories if we want to apply loss per-instance category
        if self.per_segment:
            segmentation = 2 * self.r * segmentation.flatten().float()
            means2D_src_wseg = torch.cat([means2D_src, segmentation.unsqueeze(1)], dim=1)
            src_track_seg = 2 * self.r * particles_seg.flatten().float()
            src_particles_wseg = torch.cat([src_particles, src_track_seg.unsqueeze(1)], dim=1)
        else:
            means2D_src_wseg = means2D_src.clone()
            src_particles_wseg = src_particles.clone()

        # get KNN closest gaussians for each tracked particle
        device = means2D_src.device
        lengths1 = torch.tensor([src_particles_wseg.size(0)]).to(device)
        lengths2 = torch.tensor([means2D_src_wseg.size(0)]).to(device)
        dists2, raw_knn_idxs, nn, grid = \
            frnn.frnn_grid_points(src_particles_wseg.unsqueeze(0), means2D_src_wseg.unsqueeze(0), lengths1, lengths2,
                                  self.K, self.r, grid=None, return_nn=False, return_sorted=True)

        # gather source depths by knn, and then densely order them
        src_particles_gather, means2D_src_gather, depths_src_gather, conical_opacity, mask, knn_idxs, particles_idxs = \
            self._gather_knn(raw_knn_idxs, depths_src, src_particles, means2D_src, conical_opacity, self.K)

        # compute per-particle influences and distances in the source frame
        dists = (means2D_src_gather - src_particles_gather)  # MK x 2
        influences, dists = self._accumulate_opacity_into_influence(particles_idxs, dists, conical_opacity)
        dists_source = torch.norm(dists, dim=-1)

        # get particles and means2D in target frame
        target_particles_gather = torch.gather(target_particles, 0, torch.stack([particles_idxs.flatten(), particles_idxs.flatten()], dim=-1)).reshape(-1, 2)  # MK' x 2
        means2D_target_gather = torch.gather(means2D_target, 0, torch.stack([knn_idxs.flatten(), knn_idxs.flatten()], dim=-1)).reshape(-1, 2)  # MK' x 2
        depths_target_gather = torch.gather(depths_target.flatten(), 0, knn_idxs.flatten()).flatten()
        dists_target = torch.norm(means2D_target_gather - target_particles_gather, dim=-1)

        # compute errors
        errors = torch.abs(dists_target * depths_target_gather - dists_source * depths_src_gather) * influences
        particles_idxs_contiguous = torch.unique(particles_idxs, return_inverse=True)[1]
        errors = torch_scatter.scatter_sum(errors, index=particles_idxs_contiguous)
        return torch.mean(errors)

    @staticmethod
    def _gather_knn(raw_knn_idxs, depths_source, src_particles, means2D_source, conical_opacity, K: int):
        device = raw_knn_idxs.device
        M = src_particles.shape[0]

        # gather source depths by knn, and then densely order them
        raw_knn_idxs[raw_knn_idxs == -1] = 0
        depths_source_knn = torch.gather(depths_source, 0, raw_knn_idxs.flatten()).reshape(-1, K)
        depth_knn_order = torch.argsort(depths_source_knn, dim=-1)  # depth-order each pixel's KNN
        knn_idxs = torch.gather(raw_knn_idxs.squeeze(0), 1, depth_knn_order)  # shape (batch_size, K)
        mask = knn_idxs != -1
        knn_idxs = knn_idxs[mask]  # MK' x 1

        # gather particles
        particles_idxs = torch.arange(M, device='cuda').repeat_interleave(K).to(device)[mask.flatten()]
        src_particles_gathered = torch.gather(src_particles, 0, torch.stack([particles_idxs.flatten(), particles_idxs.flatten()], dim=-1)).reshape(-1, 2)  # MK' x 2

        # Gather gaussian properties
        means2D_source_gathered = torch.gather(means2D_source, 0, torch.stack([knn_idxs.flatten(), knn_idxs.flatten()], dim=-1)).reshape(-1, 2)  # MK' x 2
        depths_source_gathered = torch.gather(depths_source, 0, knn_idxs.flatten())  # MK' x 2
        conical_opacity = torch.gather(conical_opacity, 0, torch.stack( [knn_idxs.flatten(), knn_idxs.flatten(), knn_idxs.flatten(), knn_idxs.flatten()], dim=-1)).reshape(-1, 4)  # MK' x 2

        return src_particles_gathered, means2D_source_gathered, depths_source_gathered, conical_opacity, mask, knn_idxs, particles_idxs

    @staticmethod
    def _accumulate_opacity_into_influence(batch, dists, conical_opacity):
        # compute per-particle opacities
        power = -0.5 * (conical_opacity[:, 0] + dists[:, 0] ** 2 + conical_opacity[:, 2] * dists[:, 1] ** 2)
        power = power - conical_opacity[:, 1] * dists[:, 0] * dists[:, 1]
        power = torch.clip(power, max=0)
        alpha = torch.clip(conical_opacity[:, 3] * torch.exp(power), 0, 0.99)

        # use log trick to get cumulative product
        _, track_idxs_local = torch.unique(batch, return_inverse=True)
        ptr = torch.where((track_idxs_local[1:] - track_idxs_local[:-1]) > 0)[0] + 1
        log_transparencies = torch.log2(1 - alpha)
        log_agg_transparencies = torch.cumsum(log_transparencies, dim=0)
        diff_corrections = torch.cat([torch.zeros(1).to(log_transparencies), log_agg_transparencies[ptr - 1]])
        diff_corrections = diff_corrections[track_idxs_local]
        log_agg_transparencies = log_agg_transparencies - diff_corrections
        agg_transparencies = torch.exp2(log_agg_transparencies)

        # correct for 1st element being less than 1
        ptr_wzero = torch.cat([torch.zeros(1).to(ptr), ptr])
        init_transparencies = agg_transparencies[ptr_wzero]
        init_transparencies = init_transparencies[track_idxs_local]
        agg_transparencies = agg_transparencies / init_transparencies

        # finally, get influences
        influences = alpha * agg_transparencies
        influences = influences.detach().clone()
        return influences, dists

    @staticmethod
    def vis_particles(means2D_sorted, track_locs, means2D_gathered, influences, sub_idx):
        global VIS_IDX
        fig, ax = plt.subplots()
        gaussians_2d_x = means2D_sorted[:, 0].detach().cpu().numpy().tolist()
        gaussians_2d_y = (-means2D_sorted[:, 1]).detach().cpu().numpy().tolist()
        c = np.array([[0.5, 0.5, 0.5] for i in range(len(gaussians_2d_x))])
        tracks_2d_x = track_locs[:, 0].detach().cpu().numpy().tolist()
        tracks_2d_y = (-track_locs[:, 1]).detach().cpu().numpy().tolist()
        c_tracks = np.array([[1.0, 0, 0] for i in range(len(tracks_2d_x))])

        # get gaussians 2D
        keygaussians, keyinfluences = means2D_gathered.clone(), influences.clone()
        keygaussians_x = keygaussians[:, 0].flatten().detach().cpu().numpy().tolist()
        keygaussians_y = (-keygaussians[:, 1]).flatten().detach().cpu().numpy().tolist()
        keyinfluences = keyinfluences.flatten().detach().cpu().numpy().tolist()
        cmap = plt.get_cmap('coolwarm')
        c_influence = cmap(keyinfluences)

        # add line segments
        tracks_gathered = track_locs.clone()
        line_segs = torch.zeros(tracks_gathered.size(0), 2, 2)
        line_segs[:, 0] = keygaussians.cpu()
        line_segs[:, 1] = tracks_gathered.cpu()
        line_segs[:, :, 1] *= -1
        line_segs = line_segs.detach().numpy()
        # colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        line_segments = LineCollection(line_segs, linewidths=(0.5, 0.5, 0.5, 0.5), linestyle='solid')

        ax.scatter(gaussians_2d_x, gaussians_2d_y, marker='.', c=c, s=2.0, alpha=0.5)
        ax.scatter(tracks_2d_x, tracks_2d_y, marker='x', c=c_tracks, s=1.0)
        ax.scatter(keygaussians_x, keygaussians_y, marker='o', c=c_influence, s=1.0, alpha=0.5)
        ax.add_collection(line_segments)

        ax.set_xlim([0, 1000])
        ax.set_ylim([-600, 0])
        # fig.show()
        fig.savefig('test_%s_%s.svg' % (VIS_IDX, sub_idx))
        if sub_idx == 2:
            VIS_IDX += 1
        # wandb.log({"Gaussians": ax})

    def compute_rendered_particle_loss(self, viewpoint_camera, gaussians, flow3d, tracks_src, tracks_target,
                                       dropout_idxs, per_segment=True):
        # get image to render into
        viewmat = viewpoint_camera.world_view_transform.clone().detach().cuda().T  # or nstudio_c2w
        fx, fy, cx, cy = viewpoint_camera.fx, viewpoint_camera.fy, viewpoint_camera.cx, viewpoint_camera.cy
        Ks = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).float().cuda()
        backgrounds = torch.zeros(1, 3).cuda()

        # get properties to render
        means3D, rotations, opacity, shs, scales, segmentation = gaussians
        means3D = means3D[dropout_idxs].detach().clone()
        rotations = rotations[dropout_idxs].detach().clone()
        opacity = opacity[dropout_idxs].detach().clone()
        scales = scales[dropout_idxs].detach().clone()
        flow3d = flow3d[dropout_idxs]

        # render 3D flow
        rendered_flow3d, render_alphas, info = rasterization(
            means=means3D,
            quats=rotations,
            scales=scales,
            opacities=opacity.flatten(),
            colors=flow3d,
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

        # convert from 3D scene flow to 2D pixel flow
        flow3d, depths = rendered_flow3d[0, :, :, :3].reshape(-1, 3), rendered_flow3d[0, :, :, 3].reshape(-1, 1)
        # flow3d_affine = torch.cat([flow3d, torch.ones(flow3d.size(0), 1).to(flow3d)], dim=-1)  # N x 4

        # rotate flow3d into camera space
        h, w = viewpoint_camera.image_height, viewpoint_camera.image_width
        flow3d_camspace = flow3d.double() @ viewpoint_camera.world_view_transform.double()[0:3, 0:3]
        flow3d_homog = (flow3d_camspace / (depths.float() + 0.00001)).float()

        # visualize camspace flow
        visualize = np.random.randint(0, 40) == 0
        randval = np.random.randint(0, 30)
        # visualize 2D flow
        if flow3d.sum() > 0 and visualize:
            import flow_vis
            viz_flow = flow3d_homog.reshape(h, w, 3).data.cpu().numpy()
            viz_flow = flow_vis.flow_to_color(viz_flow[:, :, :2], convert_to_bgr=True)
            # viz_flow = viz_flow / viz_flow.max()
            viz_flow = (viz_flow * 255).astype(np.uint8)
            cv2.imwrite("%s_flow_viz.png" % randval, viz_flow)

        w_flow = flow3d_homog[:, 0] * viewpoint_camera.fx  # NDC to pixel space
        h_flow = flow3d_homog[:, 1] * viewpoint_camera.fy
        wh_flow = torch.stack([w_flow, h_flow], dim=1)  # hw x 2

        # filter cotracker tracks for segmentation class
        tracks = tracks.float().to(device).detach()
        # todo: add in background track removal
        # track_seg = track_segmentation.float().to(device)
        # if self.config.static_background:  # remove background tracks
        #     tracks = src_trackstracks[:, track_seg != 0]
        #     track_seg = track_seg[track_seg != 0]

        # get 2D cotracker flow
        target_relative, source_relative = self.get_local_cotracker_index_v2(self._time, src_fidx)
        src_particles = tracks[source_relative, :, :2]
        target_particles = tracks[target_relative, :, :2]
        flow2d = src_particles - target_particles
        lookup_idxs = target_particles[:, 1].long() * w + target_particles[:, 0].long()

        # visualize flow 2D
        # visualize camspace flow
        ys, xs = -lookup_idxs // w, lookup_idxs % w
        if flow2d.sum() > 0 and visualize:
            clrs = flow2d.cpu().numpy()
            clrs = clrs - np.min(clrs)
            clrs = clrs / np.max(clrs)
            clrs = np.concatenate([clrs, np.zeros((clrs.shape[0], 1))], axis=1)
            clrs = np.clip(clrs, 0, 1)
            plt.scatter(xs.cpu().numpy(), ys.cpu().numpy(), c=clrs)
            plt.savefig("%s_track_viz.png" % randval)
            plt.close()

        # if per_segment:
        #     g_segmentation = g.get_segmentation[dropout_idxs][valid_gidxs].detach().clone().flatten()
        #     g_segmentation = g_segmentation * r * 2
        #     means2D_source_wseg = torch.cat([means2D_source, g_segmentation.unsqueeze(1)], dim=1)
        #     src_track_seg = src_track_seg[batch_idxs].flatten().float() * r * 2
        #     src_particles_wseg = torch.cat([src_particles, src_track_seg.unsqueeze(1)], dim=1)
        # else:
        #     means2D_source_wseg = means2D_source.clone()
        #     src_particles_wseg = src_particles.clone()

        # compute 2d flow error
        sparse_render_flow = wh_flow[lookup_idxs]
        flow_err = torch.abs(sparse_render_flow - flow2d)

        # visualize flow error
        if flow_err.sum() > 0 and visualize:
            clrs = torch.norm(flow_err, dim=-1).data.cpu().numpy()
            clrs = clrs / clrs.max()
            clrs = np.stack([clrs, clrs, clrs], axis=1)
            clrs = np.clip(clrs, 0, 1)
            plt.scatter(xs.cpu().numpy(), ys.cpu().numpy(), c=clrs)
            plt.savefig("%s_track_error.png" % randval)
            plt.close()

        return flow_err.mean()


def get_local_cotracker_index(nframes, source_frameidx, target_frameidx):
    if source_frameidx < 16:  # we always have the first 33 frames
        source_relative = source_frameidx
        target_relative = target_frameidx
    elif source_frameidx >= (nframes - 15):  # if nframe=100, 84 or higher inclusive;
        from_end_idx = (nframes - 1) - source_frameidx
        source_relative = 30 - from_end_idx  # [0,32] inclusive, and then subtract off from end
        target_relative = (nframes - 1) - target_frameidx
        target_relative = 30 - target_relative  # [0,32] inclusive, and then subtract off from end
    else:
        source_relative = 15
        target_relative = target_frameidx - source_frameidx + 15
    return source_relative, target_relative


def get_means_2D(xyz, camera, opacity=None, scales=None, return_depths=False, return_conical_opacity=False):
    # project Gaussians to 2D
    projected_unhomog = xyz @ camera.full_proj_transform[:3, :3]
    projected_unhomog = projected_unhomog + camera.full_proj_transform[3:4, :3]
    reg = torch.sum(xyz * camera.full_proj_transform[:3, 3].reshape(1, 3), dim=-1) + camera.full_proj_transform[3, 3]
    projected_homogenous = projected_unhomog / (reg.unsqueeze(-1) + 0.0001)
    depths = projected_unhomog[:, 2:3]
    h, w = camera.image_height, camera.image_width
    projected_homogenous[:, 0] = ((projected_homogenous[:, 0] + 1.0) * w - 1.0) * 0.5  # convert from NDC coords to pixel space
    projected_homogenous[:, 1] = ((projected_homogenous[:, 1] + 1.0) * h - 1.0) * 0.5
    means2D_differentiable = projected_homogenous[:, :2]

    if return_depths and not return_conical_opacity:
        return means2D_differentiable, depths

    if return_conical_opacity:
        scales_1d = scales[:, 0].clone()
        cov_1d = scales_1d ** 2
        conical_x = cov_1d * (camera.fx / (reg + 0.0001)) ** 2 + 0.3
        conical_y = cov_1d * (camera.fy / (reg + 0.0001)) ** 2 + 0.3
        det = conical_x * conical_y
        conical_x, conical_y = conical_x / det, conical_y / det
        conical_xy = torch.zeros_like(conical_x)
        conical_opacity = torch.stack([conical_x, conical_xy, conical_y, opacity.flatten()], dim=-1)
        return means2D_differentiable, depths, conical_opacity

    return means2D_differentiable


# ============================================================================================================
# ====================================== Tracking Evaluation Scripts =========================================
# ============================================================================================================

def get_tracking_eval_metrics(means2D, gt_tracks, images, renders, outdir=None, split='val'):

    means2D = torch.stack(means2D, dim=0)  # size (nframes, ngaussians, 2)

    # iterate through gt tracks and find the closest gaussian trajectory for each track
    ntracks = gt_tracks.size(1)  # size (nframes, ntracks, 3)
    min_dists, g_idxs, track_idxs = [], [], []
    for i in tqdm.tqdm(range(ntracks)):
        observed_mask = gt_tracks[:, i, 2].bool()
        if observed_mask.sum() == 0:
            continue

        track_filtered = gt_tracks[:, i:i+1, :2][observed_mask]  # size (T', 1, 2)
        means2D_filtered = means2D[observed_mask]   # size (T', ngaussians, 2)
        dists = torch.norm(means2D_filtered - track_filtered, dim=-1)  # size (T', ngaussians)
        mean_dists = dists.mean(dim=0)
        g_idx = torch.argmin(mean_dists)
        min_dist = mean_dists[g_idx]
        g_idxs.append(g_idx)
        min_dists.append(min_dist)
        track_idxs.append(i)

    # get mean distance
    if len(min_dists) == 0:
        return {"mean_track_dist": 0}

    track_error = sum(min_dists) / len(min_dists)
    metrics_dict = {}
    metrics_dict["mean_track_dist"] = track_error

    if outdir is None:
        return metrics_dict

    # visualize tracks
    if not os.path.exists(os.path.join(outdir, 'tracking')):
        os.makedirs(os.path.join(outdir, 'tracking'))
    cmap = matplotlib.colormaps['Paired']

    # quick track visualization for debugging
    for j in range(len(track_idxs)):
        t_start = torch.nonzero(gt_tracks[:, track_idxs[j], 2])
        if len(t_start) == 0:
            continue
        else:
            t_start = t_start[0].item()

        img = images[t_start]
        img = np.mean(img, axis=-1)
        img = np.stack([img, img, img], axis=-1)
        img *= 255

        track_sequence, gaussian_sequence = [], []
        for i in range(len(renders)):
            t_valid = bool(gt_tracks[i, track_idxs[j], 2].item())
            if t_valid:
                gy, gx = means2D[i, g_idxs[j], :2].cpu().detach().numpy().tolist()
                ty, tx = gt_tracks[i, track_idxs[j], :2].cpu().numpy().tolist()
                gaussian_sequence.append((int(gx), int(gy)))
                track_sequence.append((int(tx), int(ty)))

        # draw gt sequence on image
        gt_cmap = plt.get_cmap("viridis")
        for i in range(len(track_sequence) - 1):
            pt1, pt2 = track_sequence[i], track_sequence[i + 1]
            color = (np.array(gt_cmap(i / len(track_sequence)))[:3] * 255).astype(int)[::-1]
            cv2.line(img, pt1, pt2, color.tolist(), 3)

        # draw pred sequence on image
        pred_cmap = plt.get_cmap("magma")
        for i in range(len(gaussian_sequence) - 1):
            pt1, pt2 = gaussian_sequence[i], gaussian_sequence[i + 1]
            color = (np.array(pred_cmap(i / len(gaussian_sequence)))[:3] * 255).astype(int)[::-1]
            cv2.line(img, pt1, pt2, color.tolist(), 3)

        # write image
        cv2.imwrite(os.path.join(outdir, 'tracking', '%s_track_%s_quickviz.png' % (split, j)), img)

    # detailed track visualization
    WINDOW = 5
    images_annot = []
    for i in range(len(renders)):
        imgs = []
        keep_img = False
        for kk in range(2):
            if kk == 0:
                img = renders[i]
            else:
                img = images[i]
            img = np.mean(img, axis=-1)
            img = np.stack([img, img, img], axis=-1)
            img *= 255
            for j in range(len(track_idxs)):
                gy, gx = means2D[i, g_idxs[j], :2].cpu().detach().numpy().tolist()
                ty, tx = gt_tracks[i, track_idxs[j], :2].cpu().numpy().tolist()
                t_valid = bool(gt_tracks[i, track_idxs[j], 2].item())

                # draw gaussian as a point and track as an X
                if t_valid:
                    keep_img = True
                    color = (np.array(cmap(j % 12))[:3] * 255).astype(int)[::-1]
                    cv2.circle(img, (int(gx), int(gy)), 4, color.tolist(), -1)
                    for ii in range(WINDOW):
                        pt1 = means2D[max(0, i-ii), g_idxs[j], :2].cpu().detach().numpy().tolist()
                        pt2 = means2D[max(0, i-ii-1), g_idxs[j], :2].cpu().detach().numpy().tolist()
                        cv2.line(img, (int(pt1[1]), int(pt1[0])), (int(pt2[1]), int(pt2[0])), color.tolist(), 2)

                    thickness = 4
                    cv2.line(img, (int(tx)-4, int(ty)-4), (int(tx)+4, int(ty)+4), color.tolist(), thickness)
                    cv2.line(img, (int(tx)-4, int(ty)+4), (int(tx)+4, int(ty)-4), color.tolist(), thickness)
                    for ii in range(WINDOW):
                        is_valid = gt_tracks[max(0, i-ii), track_idxs[j], 2] and gt_tracks[max(0, i-ii-1), track_idxs[j], 2]
                        if is_valid:
                            pt1 = gt_tracks[max(0, i-ii), track_idxs[j], :2].cpu().detach().numpy().tolist()
                            pt2 = gt_tracks[max(0, i-ii-1), track_idxs[j], :2].cpu().detach().numpy().tolist()
                            cv2.line(img, (int(pt1[1]), int(pt1[0])), (int(pt2[1]), int(pt2[0])), color.tolist(), 2)
            imgs.append(img)
        img = np.concatenate(imgs, axis=1)
        if keep_img:
            images_annot.append(img)
            cv2.imwrite(os.path.join(outdir, 'tracking', '%s_track_eval_%04d.png' % (split, i)), img)
    if len(images_annot) > 2:
        framerate = 5
        command = "ffmpeg -framerate {0} -pattern_type glob -i '{1}/{2}' -c:v libx264 -pix_fmt yuv420p {3}/{4}_tracking.mp4".format(
            framerate, os.path.join(outdir, 'tracking'), '%s_track_eval_*.png' % split, os.path.join(outdir, 'tracking'), split)
        subprocess.run(command, shell=True)
        for f in glob.glob(os.path.join(outdir, 'tracking', '%s_track_eval_*.png' % split)):
            os.remove(f)

    return metrics_dict
