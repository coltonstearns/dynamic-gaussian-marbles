# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import cv2
import torch
import argparse
import numpy as np
import re
import tqdm

from PIL import Image
from cotracker.utils.visualizer import read_video_from_path, Visualizer
from cotracker.predictor import CoTrackerPredictor

_COTRACKER_URL = "https://huggingface.co/facebook/cotracker/resolve/main/cotracker2.pth"
DEFAULT_DEVICE = ('cuda' if torch.cuda.is_available() else
                  'mps' if torch.backends.mps.is_available() else
                  'cpu')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        default="./data",
        help="directory to output video and tracked points",
    )
    parser.add_argument(
        "--video_path",
        default="",
        help="path to a video",
    )
    parser.add_argument(
        "--image_folder_path",
        default="",
        help="path to folder containing ordered rgb images",
    )
    parser.add_argument(
        "--mask_folder_path",
        default="",
        type=str,
        help="path to per-frame masks",
    )
    parser.add_argument(
        "--path_regex",
        default="",
        help="regex pattern to match rgb image filenames",
    )
    parser.add_argument(
        "--viz",
        action="store_true",
    )
    parser.add_argument(
        "--frame_span",
        default=-1,
        type=int,
        help="how many frames forward and backward to track across",
    )

    parser.add_argument("--grid_size", type=int, default=0, help="Regular grid size")

    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    assert args.video_path or args.image_folder_path
    if args.video_path:
        # load the input video frame by frame
        frames = read_video_from_path(args.video_path)
        masks = None
        imnames = None
    else:  # args.image_folder_path
        ims = sorted(os.listdir(args.image_folder_path))
        frames = []
        masks = []
        imnames = []
        for impath in ims:
            if re.search(args.path_regex, impath) is None:
                continue
            frame = cv2.imread(os.path.join(args.image_folder_path, impath))
            frames.append(np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            imnames.append(impath.replace(".png", ""))

            # check masks
            if args.mask_folder_path != "":
                maskpath = impath.replace(".png", ".npy")
                mask = np.load(os.path.join(args.mask_folder_path, maskpath))
                mask = mask > 0
                masks.append(mask)

        frames = np.stack(frames)
        masks = np.stack(masks) if len(masks) > 0 else None

    nframes = frames.shape[0]
    video = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float()

    # Cotracker V1
    # CHECKPOINT="/home/colton/Documents/pointflownerf/src/data/checkpoints/cotracker_stride_8_wind_16.pth"
    # model = CoTrackerPredictor(checkpoint=CHECKPOINT)
    # model = model.to(DEFAULT_DEVICE)

    # Cotracker V2
    model = CoTrackerPredictor(checkpoint=None)
    state_dict = torch.hub.load_state_dict_from_url(_COTRACKER_URL, map_location="cpu")
    model.model.load_state_dict(state_dict)
    model = model.to(DEFAULT_DEVICE)

    for i in tqdm.tqdm(range(nframes)):
        # filter to subsequence
        start, end, grid_query_frame = 0, nframes, i
        if args.frame_span > 0:
            start = max(0, i - args.frame_span)
            end = min(nframes, i + args.frame_span + 1)
            this_video = video.clone()[:, start:end]
            grid_query_frame = i - start
        else:
            this_video = video.clone()
        this_video = this_video.to(DEFAULT_DEVICE)

        # get mask if applicable
        segm_mask = None
        if masks is not None:
            segm_mask = torch.from_numpy(masks[i]).to(DEFAULT_DEVICE)
            segm_mask = segm_mask[None, None, ...].float()

        # run predictions
        pred_tracks, pred_visibility = model(
            this_video,
            grid_size=args.grid_size,
            grid_query_frame=grid_query_frame,
            backward_tracking=True,
            segm_mask=segm_mask
        )

        # save the tracks
        pred = torch.cat([pred_tracks.squeeze(0), pred_visibility.squeeze(0).unsqueeze(-1)], dim=-1)
        pred = pred.cpu().numpy()

        # broadcast to entire sequence
        pred_all = np.zeros((nframes, pred.shape[1], pred.shape[2]))
        pred_all[start:end] = pred
        pred = pred_all

        if imnames is not None:
            if imnames[i].startswith("0_"):
                name = imnames[i][2:]
            else:
                name = imnames[i]
            outpath = os.path.join(args.outdir, 'track_%s.npy' % name)
        else:
            outpath = os.path.join(args.outdir, 'track_%05d.npy' % i)
        np.save(outpath, pred.astype(np.float16))
        # print(pred[:, :, 2])
        print(pred.shape)
        print("computed %s" % i)

        if args.viz:
            # save a video with predicted tracks
            seq_name = args.video_path.split("/")[-1]
            vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=2, tracks_leave_trace=4)
            vis.visualize(this_video, pred_tracks, pred_visibility, query_frame=grid_query_frame, filename="video_%s" % i)