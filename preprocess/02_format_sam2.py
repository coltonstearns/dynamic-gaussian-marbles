
import os
import shutil
import argparse
import subprocess
import fnmatch
import cv2
import numpy as np
import torch.nn.functional as F
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        default="./",
        help="directory to output segmentation npy files",
    )
    parser.add_argument(
        "--video_folder_path",
        default="",
        help="path to folder containing ordered SAM2 mask videos",
    )
    parser.add_argument(
        "--image_folder_path",
        default="",
        help="path to folder containing ordered rgb images",
    )
    parser.add_argument(
        "--path_fnmatch",
        default="",
        help="glob pattern to match valid filenames",
    )

    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # load rgb names and format them
    fnames = os.listdir(args.image_folder_path)
    fnames = [name for name in fnames if fnmatch.fnmatch(name, args.path_fnmatch)]
    fnames = sorted(fnames)

    # iterate through videos in SAM2 folder
    sam_pred_fnames = os.listdir(args.video_folder_path)
    sam_pred_fnames = [name for name in sam_pred_fnames if name.endswith(".mp4")]
    frame_segmentations = [[] for _ in range(len(fnames))]
    for sam_vidname in sam_pred_fnames:
        # use cv2 to load this video and iterate through the frames
        cap = cv2.VideoCapture(os.path.join(args.video_folder_path, sam_vidname))
        success, img = cap.read()
        fno = 0
        while success:
            img = img.mean(axis=-1) > 235.0
            frame_segmentations[fno].append(img)
            success, img = cap.read()
            fno += 1

    # iterate through each frame's segmentation prediction
    for i, segmentations in enumerate(frame_segmentations):
        h, w = segmentations[0].shape
        all_segs = np.stack(segmentations, axis=-1).astype(float)
        background = np.ones((h, w, 1)) * 0.5
        all_segs = np.concatenate((background, all_segs), axis=-1)
        segmentation = np.argmax(all_segs, axis=-1)
        rgb = cv2.imread(os.path.join(args.image_folder_path, fnames[i]))
        h, w = rgb.shape[0], rgb.shape[1]
        segmentation = cv2.resize(segmentation, (w, h), 0, 0, interpolation=cv2.INTER_NEAREST)
        np.save(os.path.join(args.outdir, fnames[i].replace('png', 'npy')), segmentation)

    # if args.command == 'convert-to-video':
    #     glob_fname = os.path.join(args.image_folder_path, args.path_fnmatch)
    #     ffmpeg_cmd = "ffmpeg -framerate 30 -pattern_type glob -i '{0}' -c:v libx264 -pix_fmt yuv420p {1}".format(glob_fname, os.path.join(args.outdir, args.output_video_name))
    #     print("Running the following ffmpeg command:")
    #     print(ffmpeg_cmd)
    #     subprocess.run(ffmpeg_cmd, shell=True)
    #
    # elif args.command == 'format-masks':
    #     masks = sorted(os.listdir(args.mask_folder_path))
    #     newnames = [name.replace('png', 'npy') for name in fnames]
    #     for i, name in enumerate(newnames):
    #         shutil.copy(os.path.join(args.mask_folder_path, masks[i]), os.path.join(args.outdir, name))
