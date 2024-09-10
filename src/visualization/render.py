import tempfile
from nerfstudio.scripts.render import CropData, get_crop_from_json, get_path_from_json, _render_trajectory_video
import os
import sys
import cv2
import os.path as osp
import json
import tqdm
import shutil
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
import subprocess
import mediapy as media
import torch
import numpy as np
from nerfstudio.utils import colormaps, install_checks


def render_images_in_split(pipeline, tmp_output, final_output, split="train"):
    if not osp.exists(tmp_output):
        os.makedirs(tmp_output)
    if not osp.exists(final_output):
        os.makedirs(final_output)

    # load cameras
    if split == "train":
        cameras = pipeline.datamanager.train_dataparser_outputs.cameras
    elif split == "val":
        cameras = pipeline.datamanager.dataparser.get_dataparser_outputs(split='val').cameras
    else:
        raise RuntimeError("Rendering must be from train or test split. Got %s!" % split)

    # because we're in nerfstudio camera format, divide times by MAXTIME in order to get correct indices
    nframes = pipeline.model.nframes
    cameras.times /= nframes
    cameras.metadata['viz_tracking'] = torch.ones_like(cameras.times) * 0  # causes us to render point trajectories

    # run nerfstudio rendering
    _render_trajectory_video(
        pipeline,
        cameras,
        output_filename=Path(tmp_output + "_renders"),
        rendered_output_names=['rgb'],
        rendered_resolution_scaling_factor=1.0,
        output_format="images",
        image_format="jpeg",
        jpeg_quality=100,
        colormap_options=colormaps.ColormapOptions(),
    )

    # obtain which camera ID and frame number corresponds to which image-index
    if split == "train":
        image_filenames = pipeline.datamanager.train_dataparser_outputs.image_filenames
    else:  # render_type == "val":
        image_filenames = pipeline.datamanager.dataparser.get_dataparser_outputs(split='val').image_filenames

    # iterate through directory (sorted by img name), and reassign to appropriate name
    render_fnames = os.listdir(tmp_output + "_renders")
    render_fnames = [fname for fname in render_fnames if fname.endswith("jpg")]
    render_fnames = sorted(render_fnames)
    for i, render_fname in enumerate(render_fnames):
        render = cv2.imread(osp.join(tmp_output + "_renders", render_fname))
        img = cv2.imread(str(image_filenames[i]))
        if render is None:
            print("Error opening rendered file!...")
            continue
        if render.shape != img.shape:
            h, w, _ = render.shape
            img = cv2.resize(img, (w, h))
        combined = np.concatenate([img, render], axis=1)
        outpath = osp.join(tmp_output, image_filenames[i].name)
        outpath = outpath[:-3] + 'jpg'  # in case it's a png file
        cv2.imwrite(outpath, combined)
    shutil.rmtree(osp.join(tmp_output + "_renders"))

    # convert images into videos (if there's more than 1 frame)
    frames = sorted([out_fname for out_fname in render_fnames])
    frames = [osp.join(tmp_output, out_fname) for out_fname in frames]
    if len(frames) > 2:
        framerate = min(int(round(len(frames) / 5)), 30)
        command = "ffmpeg -framerate {0} -pattern_type glob -i '{1}/*.jpg' -c:v libx264 -pix_fmt yuv420p {2}/train.mp4".format(framerate, tmp_output, tmp_output)
        subprocess.run(command, shell=True)

    # copy from output directory to artifact directory
    out_fnames = [fname for fname in os.listdir(tmp_output) if (fname.endswith("jpg") or fname.endswith("mp4"))]
    for i, out_fname in enumerate(out_fnames):
        shutil.copy(osp.join(tmp_output, out_fname), osp.join(final_output, out_fname))


def render_video_trajectory(pipeline, output, outnames, camera_paths, viz_tracking=False):
    if not osp.exists(output):
        os.makedirs(output)
    assert len(outnames) == len(camera_paths)
    for i, camera_path in enumerate(camera_paths):
        # load nerfstudio camera trajectory
        with open(camera_path, "r", encoding="utf-8") as f:
            camera_path = json.load(f)
        # camera_path['render_width'] = 540
        # camera_path['render_height'] = 360

        seconds = camera_path["seconds"]
        crop_data = get_crop_from_json(camera_path)
        cameras = get_path_from_json(camera_path)
        output_filename = Path(osp.join(output, outnames[i]))

        # also render tracked points
        if viz_tracking:
            # because we're in nerfstudio camera format, divide times by MAXTIME in order to get correct indices
            cameras.metadata = {}
            cameras.metadata['viz_tracking'] = torch.ones_like(
                cameras.times) * 0  # causes us to render point trajectories

        # run nerfstudio rendering
        _render_trajectory_video(
            pipeline,
            cameras,
            output_filename=output_filename,
            rendered_output_names=['rgb'],
            rendered_resolution_scaling_factor=1.0,
            crop_data=crop_data,
            output_format="video",
            seconds=seconds,
            image_format="jpeg",
            jpeg_quality=100,
            colormap_options=colormaps.ColormapOptions()
        )


def render_trainval(trainer, output, timestamp):
    # tmp = tempfile.TemporaryDirectory()
    tmp = "./tmp%05d" % np.random.randint(99999)
    print("Temporary renders stored at %s" % tmp)

    # render training images
    trainer.pipeline.eval()
    render_images_in_split(trainer.pipeline,
                           osp.join(tmp, "render", timestamp, "train"),
                           osp.join(output, "render", "train"),
                           split="train")

    # render validation images
    trainer.pipeline.eval()
    render_images_in_split(trainer.pipeline,
                           osp.join(tmp, "render", timestamp, "val"),
                           osp.join(output, "render", "val"),
                           split="val")
    shutil.rmtree(tmp)