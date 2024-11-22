import json
import shutil

import torch
import tqdm
from src.utils.shutils import SH2RGB
import numpy as np
import time
import numpy as np
import os
import copy
import argparse


HEADER_SPARSE = ('ply\n'
          'format ascii 1.0\n'
          'element vertex %d\n'
          'property float x\n'
          'property float y\n'
          'property float z\n'
          'property uint8 red\n'
          'property uint8 green\n'
          'property uint8 blue\n'
          'property float scale\n'
          'property float opacity\n'
          'property uint8 segmentation\n'
          'property uint8 origin\n'
          'property list uchar uint8 frame_index\n'
          'property list uchar float x_offset\n'
          'property list uchar float y_offset\n'
          'property list uchar float z_offset\n'
          'end_header\n'
          )


HEADER_DENSE = ('ply\n'
          'format ascii 1.0\n'
          'element vertex %d\n'
          'property float x\n'
          'property float y\n'
          'property float z\n'
          'property uint8 red\n'
          'property uint8 green\n'
          'property uint8 blue\n'
          'property float scale\n'
          'property float opacity\n'
          'property uint8 segmentation\n'
          'end_header\n'
          )


def extract_pipeline_data(pipeline):
    all_gaussians = []
    all_frames_covered = []
    fg_bg = []
    print("Extracting Data for PLY Writing.")
    for field_type in ['foreground_field',  'background_field']:
        if not '_model.field.%s.frameidx2gaussianidx' % field_type in pipeline:
            continue  # background field may not be in keys!

        # get frame to gaussian set mapping
        frameidx2gaussianidx = pipeline['_model.field.%s.frameidx2gaussianidx' % field_type]
        num_sets = torch.unique(frameidx2gaussianidx).size(0)

        for set_idx in tqdm.tqdm(range(num_sets)):
            active_frames = pipeline['_model.field.%s.gaussians.%d.active_frames' % (field_type, set_idx)]
            num_gaussians = pipeline['_model.field.%s.gaussians.%d._xyz' % (field_type, set_idx)].size(0)

            # get properties
            xyzs = pipeline['_model.field.%s.gaussians.%d._xyz' % (field_type, set_idx)]
            segmentations = pipeline['_model.field.%s.gaussians.%d._segmentation' % (field_type, set_idx)].view(-1, 1)
            shs = pipeline['_model.field.%s.gaussians.%d._features_dc' % (field_type, set_idx)]
            colors = torch.clip(SH2RGB(shs).reshape(-1, 3) * 255, 0, 255).view(-1, 3).long()
            scalings = torch.exp(pipeline['_model.field.%s.gaussians.%d._scaling' % (field_type, set_idx)])
            opacities = torch.sigmoid(pipeline['_model.field.%s.gaussians.%d._opacity' % (field_type, set_idx)])
            batch = pipeline['_model.field.%s.gaussians.%d._batch' % (field_type, set_idx)].view(-1, 1)

            # get per-frame offsets
            xyz_offsets = []
            for j in active_frames:
                offset = pipeline['_model.field.%s.gaussians.%d._delta_xyz.%d' % (field_type, set_idx, j)]
                xyz_offsets.append(offset)
            xyz_offsets = torch.stack(xyz_offsets, dim=-1).view(num_gaussians, -1)  # N x 3 X T --> N x 3*T
            frame_indices = active_frames.unsqueeze(0).repeat(num_gaussians, 1)  # N x T

            # concatenate everything
            xyzs = np.round(xyzs.cpu().numpy(), decimals=5).astype(str)
            segmentations = np.round(segmentations.cpu().numpy(), decimals=0).astype(int).astype(str)
            colors = colors.cpu().numpy().astype(str)
            scalings = np.round(scalings.cpu().numpy(), decimals=5).astype(str)
            opacities = np.round(opacities.cpu().numpy(), decimals=5).astype(str)
            xyz_offsets = np.round(xyz_offsets.cpu().numpy(), decimals=5).astype(str)
            frame_indices = np.round(frame_indices.cpu().numpy(), decimals=0).astype(str)
            batch = batch.long().cpu().numpy().astype(str)
            gaussians = np.concatenate([xyzs, colors, scalings, opacities, segmentations, batch, frame_indices, xyz_offsets], axis=-1)

            # record to all gaussians
            all_gaussians.append(gaussians.astype(bytes))
            all_frames_covered.append(active_frames.cpu().numpy())
            fg_bg.append(np.ones(gaussians.shape[0], dtype=bool) if field_type == 'foreground_field' else np.zeros(gaussians.shape[0], dtype=bool))

    return all_gaussians, all_frames_covered, fg_bg


def format_to_sparse_ply(pipeline, outfile):
    all_gaussians, _, _ = extract_pipeline_data(pipeline)  # , field_type, set_idx)
    print("Writing Data to PLY file.")
    with open(outfile, 'a') as f:
        # prepend header
        num_gaussians = sum([g.shape[0] for g in all_gaussians])
        f.writelines(HEADER_SPARSE % num_gaussians)
        print("Writing to PLY file.")

        for section in tqdm.tqdm(all_gaussians):
            for gaussian in section:
                writeline = b' '.join(gaussian.tolist()) + b'\n'
                f.write(str(writeline))



def format_to_dense_ply(pipeline, outfolder):
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    all_gaussians, all_frames_covered, _ = extract_pipeline_data(pipeline)
    numframes = len(pipeline['_model.field.foreground_field.frameidx2gaussianidx'])

    print("Writing to PLY files.")
    for frameidx in tqdm.tqdm(range(numframes)):
        valid_sections = [frameidx in all_frames_covered[i] for i in range(len(all_frames_covered))]

        with open(os.path.join(outfolder, "%04d.ply" % frameidx), 'w') as f:
            # write header
            num_gaussians = sum([all_gaussians[i].shape[0] for i in range(len(all_gaussians)) if valid_sections[i]])
            f.writelines(HEADER_DENSE % num_gaussians)

            # write gaussians
            for i, section in enumerate(all_gaussians):
                if not valid_sections[i]:
                    continue

                # transform section to correct ref frame
                xyzs = torch.from_numpy(section[:, :3].copy().astype(float)).clone()
                N = xyzs.shape[0]
                xyz_offsets = section[:, 10:]  # x 3*T
                xyz_offsets = xyz_offsets.reshape(N, 4, -1)
                offset_idx = np.where(all_frames_covered[i] == frameidx)
                offset_idx = offset_idx[0][0]
                xyz_offsets = xyz_offsets[:, 1:, offset_idx]
                xyz_offsets = xyz_offsets.astype(float)
                batch = section[:, 9]
                xyz_offsets[batch.astype(int) == frameidx] *= 0
                xyzs_copy = np.round((xyzs.numpy() + xyz_offsets), decimals=5).astype(str).astype(bytes)

                for jj in range(section.shape[0]):
                    writebytes = xyzs_copy[jj].tolist() + section[jj, 3:9].tolist()
                    writeline = b' '.join(writebytes) + b'\n'
                    writeline = writeline.decode('utf-8')
                    f.write(writeline)


def format_to_dense_ply_sliding_window(pipeline, outfolder):
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    all_gaussians, all_frames_covered, fg_bg = extract_pipeline_data(pipeline)
    numframes = len(pipeline['_model.field.foreground_field.frameidx2gaussianidx'])
    foreground_wmotexpand = torch.any(pipeline['_model.field.foreground_field.frameidx2gaussianidxexpanded'] != -1).item()
    background_wmotexpand = torch.any(pipeline['_model.field.background_field.frameidx2gaussianidxexpanded'] != -1).item()
    foreground_seqlen = pipeline['foreground_sequence_length'].item()
    background_seqlen = pipeline['background_sequence_length'].item()
    fg_dist = foreground_seqlen / (2 if foreground_wmotexpand else 1)
    bg_dist = background_seqlen / (2 if background_wmotexpand else 1)

    print("Writing to PLY files.")
    for frameidx in tqdm.tqdm(range(numframes)):
        valid_sections = [frameidx in all_frames_covered[i] for i in range(len(all_frames_covered))]

        with open(os.path.join(outfolder, "%04d.ply" % frameidx), 'w') as f:
            # write header
            # num_gaussians = sum([all_gaussians[i].shape[0] for i in range(len(all_gaussians)) if valid_sections[i]])
            # f.writelines(HEADER_DENSE % num_gaussians)

            num_gaussians = 0
            # write gaussians
            for i, section in enumerate(all_gaussians):
                if not valid_sections[i]:
                    continue

                # transform section to correct ref frame
                xyzs = torch.from_numpy(section[:, :3].copy().astype(float)).clone()
                N = xyzs.shape[0]
                xyz_offsets = section[:, 10:]  # x 3*T
                xyz_offsets = xyz_offsets.reshape(N, 4, -1)
                offset_idx = np.where(all_frames_covered[i] == frameidx)
                offset_idx = offset_idx[0][0]
                xyz_offsets = xyz_offsets[:, 1:, offset_idx]
                xyz_offsets = xyz_offsets.astype(float)
                batch = section[:, 9]
                xyz_offsets[batch.astype(int) == frameidx] *= 0
                xyzs_copy = np.round((xyzs.numpy() + xyz_offsets), decimals=5).astype(str).astype(bytes)

                # filter section based on frame-of-origin distance to target-frame
                this_fgbg = fg_bg[i]
                fgbg_distthresh = fg_dist * this_fgbg + bg_dist * ~this_fgbg
                dists = np.abs(frameidx - batch.astype(int))
                valid = dists <= fgbg_distthresh
                section_valid = section[valid]
                xyzs_copy_valid = xyzs_copy[valid]
                num_gaussians += section_valid.shape[0]

                for jj in range(section_valid.shape[0]):
                    writebytes = xyzs_copy_valid[jj].tolist() + section_valid[jj, 3:9].tolist()
                    writeline = b' '.join(writebytes) + b'\n'
                    writeline = writeline.decode('utf-8')
                    f.write(writeline)

        # add header after-the-fact
        with open(os.path.join(outfolder, "%04d.ply" % frameidx), 'r+') as f:
            content = f.read()
            f.seek(0, 0)
            f.write(HEADER_DENSE % num_gaussians + content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        default="convert-to-video",
        help="one of ['sparse', 'dense', 'sliding-window']",
    )
    parser.add_argument(
        "--weights",
        default="./",
        help="path of weights to convert into ply",
    )
    parser.add_argument(
        "--output",
        default="./out.mp4",
        help="name of output folder or filename",
    )

    args = parser.parse_args()

    if torch.cuda.is_available():
        weights = torch.load(args.weights, map_location='cuda:0')
    else:
        weights = torch.load(args.weights, map_location='cpu')
    pipeline = weights['pipeline']
    if args.command == 'sparse':
        format_to_sparse_ply(pipeline, args.output.replace('.ply', '') + '.ply')
    elif args.command == 'dense':
        format_to_dense_ply(pipeline, args.output)
    else:
        format_to_dense_ply_sliding_window(pipeline, args.output)

